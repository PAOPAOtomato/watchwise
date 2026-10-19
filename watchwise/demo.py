"""A simulated day of alerts through the full cascade (``watchwise demo``).

Everything is real except the NVR and, by default, the models: a fake NVR
serves real frames over real HTTP, and the gate, screener, decision policy,
investigation loop, tools and memory are the production code paths. With the
default stand-in models, the numbers measure *routing*, not perception.

The scenario is shaped like a small real deployment: a repeating false-alarm
source (headlights on the drive), routine daytime traffic, animals at night,
and a few genuine after-hours person alerts. It runs the day twice -- cold,
then again after a simulated human has corrected what reached them -- so the
learning loop is visible in the output.
"""

from __future__ import annotations

import asyncio
import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from watchwise.config import (
    AgentSettings,
    DecisionSettings,
    GateSettings,
    MemorySettings,
    MQTTSettings,
    NVRSettings,
    WatchwiseConfig,
    VLMSettings,
)
from watchwise.fakes.nvr_server import FakeNVRState, build_fake_nvr
from watchwise.nvr.client import NVRClient
from watchwise.types import AlertContext, Disposition, Severity
from watchwise.wiring import build_system

DAY_START = datetime(2026, 3, 10, 0, 0, 0)
CAMERAS = ("cam_drive", "cam_door", "cam_yard")


@dataclass
class DemoResult:
    alerts: int
    cold_funnel: dict[str, Any]
    warm_funnel: dict[str, Any]
    cold_surfaced: int
    warm_surfaced: int
    corrections: int
    patterns_learned: int
    genuine_hidden: list[str]


def build_day(seed: int = 7) -> list[tuple[AlertContext, str]]:
    """Return (alert, ground_truth_verdict) pairs for one simulated day."""
    rng = random.Random(seed)
    alerts: list[tuple[AlertContext, str]] = []

    def add(minute: int, camera: str, labels: list[str], zones: list[str], truth: str) -> None:
        ts = (DAY_START + timedelta(minutes=minute)).timestamp()
        alerts.append(
            (
                AlertContext(
                    review_id=f"rev-{len(alerts):03d}",
                    camera=camera,
                    start_time=ts,
                    end_time=ts + rng.uniform(4, 20),
                    severity=Severity.ALERT,
                    labels=labels,
                    zones=zones,
                ),
                truth,
            )
        )

    # 1. The recurring false-alarm source: headlights on the drive at night,
    #    labelled as a car. 20 of them, in bursts.
    minute = 5
    for _ in range(20):
        add(minute, "cam_drive", ["car"], ["driveway"], "benign")
        minute += rng.choice([1, 2, 14, 31])
        if minute > 330:
            minute = rng.randint(5, 300)

    # 2. Routine daytime traffic: deliveries and residents. 22 of them.
    for i in range(22):
        add(540 + i * 20, "cam_door", ["person"], ["porch"], "benign")

    # 3. Animals at night. 12.
    for i in range(12):
        add(1350 + i * 5, "cam_yard", [rng.choice(["cat", "dog"])], [], "benign")

    # 4. Genuine after-hours person alerts on the perimeter. 6.
    for i in range(6):
        add(1400 + i * 7, "cam_yard", ["person"], ["perimeter"], "suspicious")

    alerts.sort(key=lambda pair: pair[0].start_time)
    for i, (alert, _) in enumerate(alerts):
        alert.review_id = f"rev-{i:03d}"
    return alerts


def demo_config(db_path: str, models_from: WatchwiseConfig | None = None) -> WatchwiseConfig:
    """The demo's settings; with ``models_from``, its models and roles are used."""
    config = WatchwiseConfig(
        nvr=NVRSettings(base_url="http://fake-nvr"),
        mqtt=MQTTSettings(enabled=False),
        gate=GateSettings(dedupe_window_s=120.0, never_screen_labels=["cat", "dog"]),
        vlm=VLMSettings(backend="stub", model="stub-vlm", frames=2),
        agent=AgentSettings(backend="scripted", max_concurrency=2, timeout_s=30.0),
        memory=MemorySettings(db_path=db_path, embedding_dim=256),
        decision=DecisionSettings(),
    )
    if models_from is not None:
        config.models = dict(models_from.models)
        config.roles = models_from.roles.model_copy()
        config.vlm = models_from.vlm.model_copy()
        config.agent = models_from.agent.model_copy(update={"timeout_s": max(models_from.agent.timeout_s, 120.0)})
    return config


async def run_demo(
    models_from: WatchwiseConfig | None = None,
    *,
    out: Callable[[str], None] = print,
    seed: int = 7,
) -> DemoResult:
    tmp = Path(tempfile.mkdtemp(prefix="watchwise-demo-"))
    state = FakeNVRState()
    state.cameras = {c: {"enabled": True} for c in CAMERAS}
    state.camera_stats = {
        c: {"camera_fps": 5.0, "process_fps": 5.0, "detection_fps": 1.0, "pid": 1}
        for c in state.cameras
    }
    app, _ = build_fake_nvr(state)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fake-nvr")
    nvr = NVRClient(NVRSettings(base_url="http://fake-nvr"), client=http)

    config = demo_config(str(tmp / "watchwise.db"), models_from)
    system = build_system(config, nvr_client=nvr, start_sources=False)
    await system.start()
    try:
        day = build_day(seed)
        out(f"Simulating {len(day)} alerts across one day on {len(CAMERAS)} cameras.\n")

        # ---- pass 1: a cold system, no learned patterns ----
        results: list[tuple[str, str, str]] = []
        for alert, truth in day:
            decision = await system.pipeline.handle(alert)
            results.append((alert.review_id, truth, decision.disposition.value))
        await system.investigations.drain(timeout=max(120.0, config.agent.timeout_s * 4))
        await asyncio.sleep(0.2)
        cold_funnel = dict(system.pipeline.cost_funnel())
        cold_surfaced = _surfaced(system)

        # ---- human review: correct everything a human would have seen ----
        corrected = 0
        pattern_ids: dict[str, str] = {}
        for review_id, truth, _ in results:
            decision = system.memory_sink.latest(review_id)
            if decision is None or decision.disposition not in (
                Disposition.ALERTED,
                Disposition.ESCALATED_HUMAN,
            ):
                continue
            signature = system.store.get_alert(review_id)["signature"]
            result = system.feedback.submit(
                review_id,
                truth,
                note=_note(signature, truth),
                reviewer="operator",
                influenced_pattern_ids=[pattern_ids[signature]] if signature in pattern_ids else [],
            )
            if result.pattern_id:
                pattern_ids[signature] = result.pattern_id
            corrected += 1

        # ---- pass 2: the same day again, now with memory ----
        system.pipeline.metrics = {
            k: 0 if isinstance(v, int) else 0.0 for k, v in system.pipeline.metrics.items()
        }
        warm_results = []
        for alert, truth in build_day(seed):
            alert.review_id = f"warm-{alert.review_id}"
            # Shift a day forward so nothing is deduped against yesterday.
            alert.start_time += 86400
            if alert.end_time:
                alert.end_time += 86400
            decision = await system.pipeline.handle(alert)
            warm_results.append((alert.review_id, truth, decision.disposition.value))
        await system.investigations.drain(timeout=max(120.0, config.agent.timeout_s * 4))
        await asyncio.sleep(0.2)
        warm_funnel = dict(system.pipeline.cost_funnel())
        warm_surfaced = _surfaced(system, prefix="warm-")

        truths = {a.review_id: t for a, t in day}
        hidden = [
            rid
            for rid, _, disp in warm_results
            if truths.get(rid.replace("warm-", "")) == "suspicious"
            and disp in ("suppressed", "downgraded")
        ]
        result = DemoResult(
            alerts=len(day),
            cold_funnel=cold_funnel,
            warm_funnel=warm_funnel,
            cold_surfaced=len(set(cold_surfaced)),
            warm_surfaced=len(set(warm_surfaced)),
            corrections=corrected,
            patterns_learned=len(pattern_ids),
            genuine_hidden=hidden,
        )
        _report(result, out)
        return result
    finally:
        await system.stop()
        await http.aclose()


def _note(signature: str, truth: str) -> str:
    if "car" in signature:
        return "headlights sweeping the gate get labelled as a vehicle"
    if "porch" in signature:
        return "routine deliveries and residents at the front door"
    if truth == "suspicious":
        return "genuine after-hours person on the perimeter"
    return "routine activity"


def _surfaced(system: Any, prefix: str = "rev-") -> list[str]:
    return [
        d.review_id
        for _, d in system.memory_sink.decisions
        if d.review_id.startswith(prefix)
        and d.disposition in (Disposition.ALERTED, Disposition.ESCALATED_HUMAN)
    ]


def _report(r: DemoResult, out: Callable[[str], None]) -> None:
    line = "=" * 68
    out(line)
    out("PASS 1 - cold system, no learned patterns")
    out(line)
    _print_funnel(r.cold_funnel, out)
    out(f"  surfaced to a human       : {r.cold_surfaced} of {r.alerts}")
    out("")
    out(f"Human review: {r.corrections} alerts judged, {r.patterns_learned} patterns learned.")
    out("")
    out(line)
    out("PASS 2 - same day replayed, with the learned patterns in memory")
    out(line)
    _print_funnel(r.warm_funnel, out)
    out(f"  surfaced to a human       : {r.warm_surfaced} of {r.alerts}")
    out("")
    out("-" * 68)
    if r.cold_surfaced:
        out(
            f"Manual review volume      : {r.cold_surfaced} -> {r.warm_surfaced} "
            f"({100 * (1 - r.warm_surfaced / r.cold_surfaced):.0f}% reduction, simulated)"
        )
    vlm_cold, vlm_warm = r.cold_funnel["vlm_calls"], r.warm_funnel["vlm_calls"]
    if vlm_cold:
        out(
            f"VLM calls                 : {vlm_cold} -> {vlm_warm} "
            f"({100 * (1 - vlm_warm / vlm_cold):.0f}% reduction, simulated)"
        )
    out(
        f"Model calls avoided       : {r.warm_funnel['gate_resolved_without_model']} "
        f"of {r.alerts} alerts resolved for free by the gate"
    )
    out(
        f"Genuine alerts hidden     : {len(r.genuine_hidden)} "
        "(these are missed detections, and must stay at 0)"
    )
    if r.genuine_hidden:
        out(f"  -> {r.genuine_hidden}")
    out("-" * 68)


def _print_funnel(funnel: dict[str, Any], out: Callable[[str], None]) -> None:
    out(f"  alerts ingested           : {funnel['alerts_ingested']}")
    out(f"  resolved free by the gate : {funnel['gate_resolved_without_model']}")
    out(
        f"  reached the VLM (tier 2)  : {funnel['vlm_calls']} "
        f"({100 * (funnel['vlm_call_rate'] or 0):.0f}%)"
    )
    out(
        f"  reached the agent (tier 3): {funnel['investigations_queued']} "
        f"({100 * (funnel['agent_call_rate'] or 0):.0f}%)"
    )
    if funnel["mean_screen_latency_ms"] is not None:
        out(f"  mean screening latency    : {funnel['mean_screen_latency_ms']:.1f} ms")


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
