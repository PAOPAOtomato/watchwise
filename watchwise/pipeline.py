"""The cascade: source -> gate -> tier 2 -> decide -> dispatch -> tier 3.

One alert takes at most two passes through this module:

  pass 1 (synchronous, on the alert path)
      gate, then optionally screen, then decide and dispatch. This is what a
      human sees, and it completes in gate-time or gate+VLM time.

  pass 2 (asynchronous, off the alert path)
      an investigation refines the record and may downgrade or escalate it.
      A revised decision is dispatched as an update.

The split is what makes the latency budget achievable at all: nothing that
calls a large model is allowed to sit between tier 1 and the alert.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

from watchwise.decision import DecisionPolicy
from watchwise.memory.store import Store
from watchwise.tier2.screener import Screener
from watchwise.tier3.investigator import InvestigationQueue
from watchwise.types import (
    AlertContext,
    Decision,
    Investigation,
    ScreenResult,
    Stage,
)

logger = logging.getLogger(__name__)

DecisionSink = Callable[[AlertContext, Decision], Awaitable[None]]


class Pipeline:
    def __init__(
        self,
        alert_queue: "asyncio.Queue[AlertContext]",
        screener: Screener,
        policy: DecisionPolicy,
        store: Store,
        investigations: InvestigationQueue | None = None,
        sinks: list[DecisionSink] | None = None,
    ):
        self.alert_queue = alert_queue
        self.screener = screener
        self.policy = policy
        self.store = store
        self.investigations = investigations
        self.sinks = sinks or []
        self._task: asyncio.Task[None] | None = None
        # review_id -> what tier 2 concluded, so the tier-3 callback can
        # produce a revised decision with the full picture.
        self._pending: dict[str, tuple[AlertContext, ScreenResult | None, list[str]]] = {}
        self.metrics = {
            "ingested": 0,
            "gate_resolved": 0,
            "vlm_calls": 0,
            "investigations_queued": 0,
            "screen_latency_ms_total": 0.0,
        }

    # ---------------- lifecycle ----------------

    async def start(self) -> None:
        if self.investigations is not None:
            self.investigations.on_done = self._on_investigation_done
            await self.investigations.start()
        self._task = asyncio.create_task(self._run(), name="pipeline")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self.investigations is not None:
            await self.investigations.stop()

    async def _run(self) -> None:
        while True:
            ctx = await self.alert_queue.get()
            try:
                await self.handle(ctx)
            except Exception:
                logger.exception("pipeline failed on %s", ctx.review_id)
            finally:
                self.alert_queue.task_done()

    # ---------------- the cascade ----------------

    async def handle(self, ctx: AlertContext) -> Decision:
        """Process one alert through pass 1 and queue pass 2 if needed."""
        started = time.perf_counter()
        self.metrics["ingested"] += 1

        record = ctx.to_dict()
        record["signature"] = ctx.signature()
        self.store.upsert_alert(record, Stage.INGESTED.value)

        gate_result, screen = await self.screener.screen(ctx)

        if screen is None:
            # Resolved by the gate: no model was called at all.
            self.metrics["gate_resolved"] += 1
            decision = self.policy.from_gate(ctx, gate_result)
            self.store.upsert_alert(record, Stage.GATED.value)
            self.store.record_decision(decision.to_dict())
            await self._dispatch(ctx, decision)
            return decision

        self.metrics["vlm_calls"] += 1
        self.metrics["screen_latency_ms_total"] += screen.latency_ms
        self.store.upsert_alert(record, Stage.SCREENED.value)

        decision, needs_investigation = self.policy.from_screening(ctx, screen)
        self.store.record_decision(decision.to_dict())
        await self._dispatch(ctx, decision)

        if needs_investigation and self.investigations is not None:
            self._pending[ctx.review_id] = (ctx, screen, gate_result.matched_pattern_ids)
            if self.investigations.submit(ctx, screen):
                self.metrics["investigations_queued"] += 1
            else:
                self._pending.pop(ctx.review_id, None)
                logger.warning(
                    "could not queue investigation for %s; alert stands as dispatched",
                    ctx.review_id,
                )

        logger.debug(
            "pass 1 for %s took %.1fms -> %s",
            ctx.review_id,
            (time.perf_counter() - started) * 1000.0,
            decision.disposition.value,
        )
        return decision

    async def _on_investigation_done(
        self, ctx: AlertContext, inv: Investigation
    ) -> None:
        """Pass 2: revise the record once the agent has finished."""
        _, screen, memory_ids = self._pending.pop(
            ctx.review_id, (ctx, None, [])
        )
        decision = self.policy.from_investigation(
            ctx, screen, inv, matched_memory_ids=memory_ids
        )
        self.store.record_decision(decision.to_dict())
        await self._dispatch(ctx, decision)

    async def _dispatch(self, ctx: AlertContext, decision: Decision) -> None:
        for sink in self.sinks:
            try:
                await sink(ctx, decision)
            except Exception:
                logger.exception("decision sink %r failed", getattr(sink, "__name__", sink))

    # ---------------- reporting ----------------

    def cost_funnel(self) -> dict[str, float | int | None]:
        """The measured cost funnel: how many alerts reached each model tier.

        ``vlm_call_rate`` is the fraction of alerts that reached a model at
        all; ``agent_call_rate`` the fraction that reached the expensive one.
        These are counted, not asserted.
        """
        ingested = self.metrics["ingested"]
        vlm = self.metrics["vlm_calls"]
        agent = self.metrics["investigations_queued"]
        return {
            "alerts_ingested": ingested,
            "gate_resolved_without_model": self.metrics["gate_resolved"],
            "vlm_calls": vlm,
            "investigations_queued": agent,
            "vlm_call_rate": round(vlm / ingested, 4) if ingested else None,
            "agent_call_rate": round(agent / ingested, 4) if ingested else None,
            "mean_screen_latency_ms": round(
                self.metrics["screen_latency_ms_total"] / vlm, 1
            )
            if vlm
            else None,
        }
