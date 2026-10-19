"""The sentinel HTTP service.

Runs alongside NVR rather than inside it. It exposes what an operator or a UI
needs: the refined verdicts, the investigation traces behind them, the
long-term pattern library, and the endpoint that closes the learning loop.

It also exposes ``POST /alerts/simulate``, which injects a synthetic alert
through the real cascade. That is how the whole system is exercised on a box
with no cameras attached.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from sentinel import __version__
from sentinel.config import SentinelConfig
from sentinel.nvr.client import NVRClient
from sentinel.nvr.source import review_payload_to_context
from sentinel.tools.registry import tool_catalog
from sentinel.types import AlertContext, Severity
from sentinel.wiring import System, build_system

logger = logging.getLogger(__name__)


class SimulatedAlert(BaseModel):
    camera: str
    review_id: str | None = None
    labels: list[str] = Field(default_factory=lambda: ["person"])
    zones: list[str] = Field(default_factory=list)
    sub_labels: list[str] = Field(default_factory=list)
    start_time: float | None = None
    duration_s: float = 8.0
    severity: str = "alert"
    event_ids: list[str] = Field(default_factory=list)
    wait_for_investigation: bool = False


class FeedbackIn(BaseModel):
    review_id: str
    verdict: str = Field(description="benign | suspicious | uncertain")
    note: str = ""
    reviewer: str = "unknown"


class ArmingIn(BaseModel):
    state: str = Field(description="armed_away | armed_night | armed_home | disarmed")


def create_app(
    config: SentinelConfig | None = None,
    *,
    nvr_client: NVRClient | None = None,
    start_sources: bool = True,
) -> FastAPI:
    cfg = config or SentinelConfig.load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        system = build_system(cfg, nvr_client=nvr_client, start_sources=start_sources)
        app.state.system = system
        app.state.started_at = time.time()
        await system.start()
        logger.info(
            "sentinel up: vlm=%s/%s agent=%s/%s",
            cfg.vlm.backend,
            cfg.vlm.model,
            cfg.agent.backend,
            cfg.agent.model,
        )
        try:
            yield
        finally:
            await system.stop()

    app = FastAPI(
        title="Sentinel",
        version=__version__,
        description="VLM screening + agentic investigation layered over an NVR.",
        lifespan=lifespan,
    )

    def sys_() -> System:
        system: System | None = getattr(app.state, "system", None)
        if system is None:
            raise HTTPException(status_code=503, detail="system not started")
        return system

    # ---------------- health & introspection ----------------

    @app.get("/health")
    async def health() -> dict[str, Any]:
        system = sys_()
        nvr_ok, nvr_detail = True, "ok"
        try:
            await system.nvr.version()
        except Exception as exc:
            nvr_ok, nvr_detail = False, str(exc)
        return {
            "status": "ok" if nvr_ok else "degraded",
            "uptime_s": round(time.time() - app.state.started_at, 1),
            "nvr": {"reachable": nvr_ok, "detail": nvr_detail,
                    "base_url": system.config.nvr.base_url},
            "tier2": {
                "backend": system.config.vlm.backend,
                "model": system.config.vlm.model,
                "stats": system.screener.stats,
            },
            "tier3": {
                "backend": system.investigations.investigator.backend.name,
                "model": system.investigations.investigator.backend.model,
                "queue_depth": system.investigations.queue.qsize(),
                "stats": system.investigations.stats,
            },
            "sources": [type(s).__name__ for s in system.sources],
        }

    @app.get("/config")
    async def get_config() -> dict[str, Any]:
        system = sys_()
        return {
            "vlm": system.config.vlm.model_dump(),
            "agent": system.config.agent.model_dump(),
            "gate": system.config.gate.model_dump(),
            "decision": system.policy.describe(),
            "memory": system.config.memory.model_dump(),
        }

    @app.get("/tools")
    async def tools() -> dict[str, Any]:
        return {"tools": tool_catalog()}

    @app.post("/tools/{name}")
    async def run_tool(name: str, args: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        """Run one forensic tool directly. Useful for debugging a case by hand."""
        from sentinel.tools.registry import TOOLS_BY_NAME

        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"unknown tool {name}")
        return await tool.run(sys_().tool_ctx, args)

    # ---------------- alerts & decisions ----------------

    @app.get("/decisions")
    async def decisions(limit: int = Query(50, le=500)) -> dict[str, Any]:
        return {"decisions": sys_().memory_sink.recent(limit)}

    @app.get("/alerts/{review_id}")
    async def get_alert(review_id: str) -> dict[str, Any]:
        system = sys_()
        row = system.store.get_alert(review_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown review_id")
        record = dict(row)
        record["labels"] = json.loads(record.get("labels") or "[]")
        record["zones"] = json.loads(record.get("zones") or "[]")
        record["screen"] = json.loads(record["screen_json"]) if record.get("screen_json") else None
        record.pop("screen_json", None)
        record["decision_history"] = [
            d.to_dict() for d in system.memory_sink.all_for(review_id)
        ]
        return record

    @app.post("/alerts/simulate")
    async def simulate(alert: SimulatedAlert) -> dict[str, Any]:
        """Push a synthetic alert through the real cascade."""
        system = sys_()
        start = alert.start_time if alert.start_time is not None else time.time()
        ctx = AlertContext(
            review_id=alert.review_id or f"sim-{int(start * 1000)}",
            camera=alert.camera,
            start_time=start,
            end_time=start + alert.duration_s,
            severity=Severity(alert.severity),
            labels=alert.labels,
            sub_labels=alert.sub_labels,
            zones=alert.zones,
            event_ids=alert.event_ids,
        )
        decision = await system.pipeline.handle(ctx)
        if alert.wait_for_investigation:
            await system.investigations.drain(timeout=system.config.agent.timeout_s + 10)
            latest = system.memory_sink.latest(ctx.review_id)
            if latest is not None:
                decision = latest
        return {"review_id": ctx.review_id, "decision": decision.to_dict()}

    @app.post("/alerts/ingest")
    async def ingest(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Accept an NVR review payload over HTTP instead of MQTT."""
        ctx = review_payload_to_context(payload)
        if ctx is None:
            raise HTTPException(status_code=400, detail="not a review payload")
        await sys_().alert_queue.put(ctx)
        return {"queued": ctx.review_id}

    # ---------------- investigations ----------------

    @app.get("/investigations/{investigation_id}")
    async def get_investigation(investigation_id: str) -> dict[str, Any]:
        row = sys_().store.get_investigation(investigation_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown investigation")
        record = dict(row)
        record["trace"] = json.loads(record.pop("trace_json") or "{}")
        return record

    @app.get("/investigations")
    async def list_investigations(limit: int = Query(25, le=200)) -> dict[str, Any]:
        rows = sys_().store.query(
            "SELECT id, review_id, camera, status, verdict, confidence, iterations, "
            "backend, model, cost_usd, started_at, finished_at "
            "FROM investigations ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return {"investigations": [dict(r) for r in rows]}

    # ---------------- memory & the learning loop ----------------

    @app.post("/feedback")
    async def submit_feedback(fb: FeedbackIn) -> dict[str, Any]:
        system = sys_()
        decision = system.memory_sink.latest(fb.review_id)
        influenced = decision.matched_memory_ids if decision else []
        try:
            result = system.feedback.submit(
                fb.review_id,
                fb.verdict,
                note=fb.note,
                reviewer=fb.reviewer,
                influenced_pattern_ids=influenced,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "review_id": result.review_id,
            "human_verdict": result.human_verdict,
            "system_verdict": result.system_verdict,
            "agreed": result.agreed,
            "pattern_id": result.pattern_id,
            "patterns_scored": result.patterns_scored,
        }

    @app.get("/patterns")
    async def patterns(kind: str | None = None) -> dict[str, Any]:
        return {"patterns": sys_().patterns.all_patterns(kind)}

    @app.get("/patterns/search")
    async def search_patterns(
        q: str, signature: str | None = None, limit: int = 5
    ) -> dict[str, Any]:
        matches = sys_().patterns.search(q, signature=signature, top_k=limit)
        return {"matches": [m.to_dict() for m in matches]}

    @app.delete("/patterns/{pattern_id}")
    async def delete_pattern(pattern_id: str) -> dict[str, Any]:
        sys_().patterns.delete(pattern_id)
        return {"deleted": pattern_id}

    # ---------------- arming ----------------

    @app.get("/arming")
    async def get_arming() -> dict[str, Any]:
        return sys_().arming.get()

    @app.post("/arming")
    async def set_arming(body: ArmingIn) -> dict[str, Any]:
        try:
            return sys_().arming.set(body.state)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ---------------- metrics ----------------

    @app.get("/metrics")
    async def metrics(window_hours: float = Query(24.0)) -> dict[str, Any]:
        system = sys_()
        since = time.time() - window_hours * 3600
        return {
            "window_hours": window_hours,
            "cost_funnel": system.pipeline.cost_funnel(),
            "review_load": system.feedback.review_load(since),
            "accuracy": system.feedback.accuracy_report(since),
            "screener": system.screener.stats,
            "investigations": system.investigations.stats,
            "stage_counts": system.store.funnel_counts(since),
        }

    return app
