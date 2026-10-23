"""Layer 6: the whole cascade, and the HTTP service.

These are the tests that answer "does the thing described in the README
actually work end to end" -- including the closed learning loop, where a
repeated false alarm stops reaching a human after enough human corrections.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from watchwise.config import (
    AgentSettings,
    DecisionSettings,
    GateSettings,
    MQTTSettings,
    MemorySettings,
    NVRSettings,
    WatchwiseConfig,
    VLMSettings,
)
from watchwise.fakes.nvr_server import FakeNVRState, build_fake_nvr
from watchwise.nvr.client import NVRClient
from watchwise.tier2.vlm.base import VLMBackend
from watchwise.types import AlertContext, Disposition, Severity, Stage, Verdict
from watchwise.wiring import build_system
from tests.conftest import DAY_TS, NIGHT_TS


def _config(tmp_path) -> WatchwiseConfig:
    return WatchwiseConfig(
        nvr=NVRSettings(base_url="http://fake-nvr"),
        mqtt=MQTTSettings(enabled=False),
        gate=GateSettings(dedupe_window_s=30.0),
        vlm=VLMSettings(backend="stub", model="stub-vlm", frames=2),
        agent=AgentSettings(backend="scripted", max_concurrency=1, timeout_s=30.0),
        memory=MemorySettings(db_path=str(tmp_path / "watchwise.db"), embedding_dim=128),
        decision=DecisionSettings(),
    )


def _fake_client(state: FakeNVRState) -> tuple[NVRClient, httpx.AsyncClient]:
    app, _ = build_fake_nvr(state)
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://fake-nvr"
    )
    return NVRClient(NVRSettings(base_url="http://fake-nvr"), client=http), http


@pytest.fixture
async def system(tmp_path):
    state = FakeNVRState()
    client, http = _fake_client(state)
    sys_ = build_system(_config(tmp_path), nvr_client=client, start_sources=False)
    sys_.nvr_state = state  # type: ignore[attr-defined]
    await sys_.start()
    yield sys_
    await sys_.stop()
    await http.aclose()


def _alert(review_id: str, *, camera="cam_drive", labels=None, zones=None, ts=DAY_TS,
           duration=8.0) -> AlertContext:
    return AlertContext(
        review_id=review_id,
        camera=camera,
        start_time=ts,
        end_time=ts + duration,
        severity=Severity.ALERT,
        labels=labels if labels is not None else ["person"],
        zones=zones or [],
    )


# ---------------- the cascade ----------------


BENIGN_REPLY = '{"verdict": "benign", "confidence": 0.9, "reason": "a cat crossing"}'


class SlowVLM(VLMBackend):
    """A screener that takes its time, and counts its calls."""

    name = "slow"

    def __init__(self, delay_s: float):
        super().__init__("slow-vlm")
        self.delay_s = delay_s
        self.calls = 0

    async def infer(self, frames, prompt, system):
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return BENIGN_REPLY


class TogetherVLM(VLMBackend):
    """Answers only once ``parties`` screens are running at the same time."""

    name = "together"

    def __init__(self, parties: int):
        super().__init__("together-vlm")
        self.parties = parties
        self.prompts: list[str] = []
        self._all_in = asyncio.Event()

    async def infer(self, frames, prompt, system):
        self.prompts.append(prompt)
        if len(self.prompts) >= self.parties:
            self._all_in.set()
        await self._all_in.wait()
        return BENIGN_REPLY


async def test_a_slow_screen_sends_an_analyzing_decision_first(system):
    """B-002: with a slow screener the phone used to stay silent until the verdict."""
    await system.pipeline.handle(_alert("quick", labels=["cat"]))
    assert [d.stage for d in system.memory_sink.all_for("quick")] == [Stage.SCREENED]

    system.screener.vlm = SlowVLM(0.7)
    system.arming.set("armed_away")
    await system.pipeline.handle(_alert("slow-armed", labels=["cat"], ts=DAY_TS + 3600))
    first, final = system.memory_sink.all_for("slow-armed")
    assert (first.stage, first.disposition) == (Stage.INGESTED, Disposition.ALERTED)
    assert first.rationale.startswith("analyzing")
    assert (final.stage, final.disposition) == (Stage.SCREENED, Disposition.DOWNGRADED)

    # Disarmed, the first card may wait up to 10 s for the verdict.
    system.arming.set("disarmed")
    await system.pipeline.handle(_alert("slow-home", labels=["cat"], ts=DAY_TS + 7200))
    assert [d.stage for d in system.memory_sink.all_for("slow-home")] == [Stage.SCREENED]
    assert system.pipeline.latency()["provisional_dispatches"] == 1
    assert system.pipeline.latency()["first_dispatches"] == 3


async def test_an_events_end_costs_a_model_call_only_if_it_shows_something_new(system):
    """B-004: an event is screened when it starts; its end report calls the
    model again only for a new label or zone."""
    vlm = SlowVLM(0.0)
    system.screener.vlm = vlm
    start = _alert("walk", labels=["car"], ts=DAY_TS + 9000)
    start.end_time = None  # still in progress
    first = await system.pipeline.handle(start)

    same = await system.pipeline.handle(_alert("walk", labels=["car"], ts=DAY_TS + 9000, duration=20.0))
    assert vlm.calls == 1 and same is first
    assert system.store.get_alert("walk")["end_time"] == DAY_TS + 9000 + 20.0

    grown = await system.pipeline.handle(
        _alert("walk", labels=["car", "person"], ts=DAY_TS + 9000, duration=25.0)
    )
    assert vlm.calls == 2  # and not taken for a duplicate of its own start
    assert system.memory_sink.latest("walk") is grown
    latency = system.pipeline.latency()
    assert (latency["rescreens"], latency["reports_without_model"]) == (1, 1)
    assert system.pipeline.cost_funnel()["alerts_ingested"] == 1


async def test_one_reviews_reports_go_in_order_while_other_alerts_run_alongside(system):
    vlm = TogetherVLM(parties=2)  # deadlocks unless two screens overlap
    system.screener.vlm = vlm
    a_start = _alert("a", camera="cam_door", labels=["dog"], ts=DAY_TS + 12000)
    a_end = _alert("a", camera="cam_door", labels=["dog"], ts=DAY_TS + 12000, duration=15.0)
    b_start = _alert("b", camera="cam_yard", labels=["dog"], ts=DAY_TS + 12000)
    for ctx in (a_start, a_end, b_start):
        system.alert_queue.put_nowait(ctx)
    async with asyncio.timeout(5):
        await system.alert_queue.join()

    # a's end waited for a's start and found nothing new; b ran alongside a.
    assert len(vlm.prompts) == 2
    assert sum('"cam_door"' in p for p in vlm.prompts) == 1 and sum('"cam_yard"' in p for p in vlm.prompts) == 1
    assert system.pipeline.latency()["reports_without_model"] == 1


async def test_a_benign_alert_is_downgraded_by_tier_2_alone(system):
    decision = await system.pipeline.handle(_alert("r1", labels=["cat"]))
    assert decision.disposition is Disposition.DOWNGRADED
    assert decision.stage is Stage.SCREENED
    assert system.pipeline.metrics["vlm_calls"] == 1
    assert system.pipeline.metrics["investigations_queued"] == 0


async def test_an_uncertain_alert_escalates_to_an_investigation(system):
    decision = await system.pipeline.handle(_alert("r2", labels=["car"], ts=NIGHT_TS))
    assert decision.disposition is Disposition.ALERTED
    assert "not confident" in decision.rationale
    assert system.pipeline.metrics["investigations_queued"] == 1

    await system.investigations.drain(timeout=30)
    await asyncio.sleep(0.05)
    final = system.memory_sink.latest("r2")
    assert final.stage is Stage.INVESTIGATED
    assert final.investigation_id is not None


async def test_the_alert_is_dispatched_before_the_investigation_finishes(system):
    """The non-blocking property."""
    await system.pipeline.handle(_alert("r3", labels=["car"], ts=NIGHT_TS))
    dispatched = system.memory_sink.all_for("r3")
    assert len(dispatched) == 1
    assert dispatched[0].stage is Stage.SCREENED

    await system.investigations.drain(timeout=30)
    await asyncio.sleep(0.05)
    assert len(system.memory_sink.all_for("r3")) == 2
    assert system.memory_sink.all_for("r3")[1].stage is Stage.INVESTIGATED


async def test_a_duplicate_burst_costs_exactly_one_model_call(system):
    """The dedupe half of the cost story."""
    for i in range(6):
        await system.pipeline.handle(_alert(f"dup-{i}", labels=["person"], ts=DAY_TS))
    funnel = system.pipeline.cost_funnel()
    assert funnel["alerts_ingested"] == 6
    assert funnel["vlm_calls"] == 1
    assert funnel["gate_resolved_without_model"] == 5
    assert funnel["vlm_call_rate"] == pytest.approx(1 / 6, abs=1e-3)


async def test_every_alert_is_persisted_with_its_decision(system):
    await system.pipeline.handle(_alert("p1", labels=["cat"]))
    row = system.store.get_alert("p1")
    assert row is not None
    assert row["disposition"] == "downgraded"
    assert row["signature"] == "cam_drive|cat|nozone"
    assert row["screen_json"] is not None


async def test_the_pipeline_survives_a_dead_nvr(tmp_path):
    """Everything downstream must degrade, not crash, if tier 1 is unreachable."""
    client = NVRClient(NVRSettings(base_url="http://127.0.0.1:1", timeout_s=0.3))
    sys_ = build_system(_config(tmp_path), nvr_client=client, start_sources=False)
    await sys_.start()
    try:
        decision = await sys_.pipeline.handle(_alert("dead", labels=["person"]))
        assert decision.disposition is Disposition.ALERTED
        assert decision.screen is not None
        assert decision.screen.error == "no_frames"
    finally:
        await sys_.stop()


# ---------------- the closed learning loop ----------------


async def test_repeated_false_alarms_stop_reaching_a_human_after_corrections(system):
    """judge -> correct -> feed back into memory -> auto-downgrade.

    This is the central claim, exercised end to end: the same alert,
    corrected by a human four times, is thereafter resolved for free by the
    gate instead of costing a model call or a human's attention.
    """
    signature_alert = dict(camera="cam_drive", labels=["car"], zones=["driveway"])
    pattern_id = None

    for i in range(4):
        # Space them out so the dedupe window is not what is doing the work.
        ts = DAY_TS + i * 3600
        await system.pipeline.handle(_alert(f"fa-{i}", ts=ts, **signature_alert))
        result = system.feedback.submit(
            f"fa-{i}",
            "benign",
            note="headlights sweeping the gate get labelled as a car",
            reviewer="operator",
            influenced_pattern_ids=[pattern_id] if pattern_id else [],
        )
        pattern_id = result.pattern_id

    before = dict(system.pipeline.metrics)
    decision = await system.pipeline.handle(
        _alert("fa-final", ts=DAY_TS + 10 * 3600, **signature_alert)
    )

    assert decision.stage is Stage.GATED, "must be resolved before any model call"
    assert decision.matched_memory_ids == [pattern_id]
    assert "known false-alarm pattern" in decision.rationale
    assert system.pipeline.metrics["vlm_calls"] == before["vlm_calls"], (
        "a learned false alarm must cost zero inference"
    )
    # Four corrections earns a demotion off the alert queue, but not yet the
    # right to hide the alert entirely -- suppression needs a longer record.
    assert decision.disposition is Disposition.DOWNGRADED
    assert decision.severity is Severity.DETECTION

    # With a longer track record of humans agreeing, it graduates to full
    # suppression. This is the auto-downgrade escalation curve.
    for i in range(8):
        ts = DAY_TS + (20 + i) * 3600
        await system.pipeline.handle(_alert(f"fa2-{i}", ts=ts, **signature_alert))
        system.feedback.submit(
            f"fa2-{i}", "benign", note="headlights again",
            influenced_pattern_ids=[pattern_id],
        )

    graduated = await system.pipeline.handle(
        _alert("fa-graduated", ts=DAY_TS + 40 * 3600, **signature_alert)
    )
    assert graduated.stage is Stage.GATED
    assert graduated.disposition is Disposition.SUPPRESSED
    assert graduated.severity is Severity.SUPPRESSED


async def test_corrections_from_the_phone_alone_teach_the_gate(system):
    """B-017: a correction scores only the patterns on the alert's decision
    (that is all a button press knows). The patterns shown to the screener
    must be among them, or a new pattern never earns the trust the gate
    requires and the learning loop never closes outside this file."""
    alert_kwargs = dict(camera="cam_yard", labels=["cat"], zones=["lawn"])
    for i in range(3):
        review_id = f"learn-{i}"
        await system.pipeline.handle(_alert(review_id, ts=DAY_TS + i * 3600, **alert_kwargs))
        latest = system.memory_sink.latest(review_id)
        system.feedback.submit(
            review_id, "benign", reviewer="telegram:tester",
            influenced_pattern_ids=list(latest.matched_memory_ids),
        )

    held = await system.pipeline.handle(_alert("learn-3", ts=DAY_TS + 3 * 3600, **alert_kwargs))
    assert held.stage is Stage.GATED, "the fourth one must be settled from memory, for free"
    assert "known false-alarm pattern" in held.rationale
    assert held.disposition in (Disposition.DOWNGRADED, Disposition.SUPPRESSED)


async def test_a_pattern_that_starts_being_wrong_loses_its_authority(system):
    """The safety valve on auto-suppression."""
    alert_kwargs = dict(camera="cam_yard", labels=["person"], zones=["perimeter"])
    pattern_id = None
    for i in range(4):
        await system.pipeline.handle(_alert(f"x-{i}", ts=DAY_TS + i * 3600, **alert_kwargs))
        result = system.feedback.submit(
            f"x-{i}", "benign", note="the neighbour cuts through the yard",
            influenced_pattern_ids=[pattern_id] if pattern_id else [],
        )
        pattern_id = result.pattern_id

    suppressed = await system.pipeline.handle(
        _alert("x-live", ts=DAY_TS + 10 * 3600, **alert_kwargs)
    )
    assert suppressed.stage is Stage.GATED

    # Humans now disagree repeatedly: those were real intrusions.
    for i in range(8):
        await system.pipeline.handle(_alert(f"y-{i}", ts=DAY_TS + (20 + i) * 3600,
                                            **alert_kwargs))
        system.feedback.submit(f"y-{i}", "suspicious",
                               influenced_pattern_ids=[pattern_id])

    row = system.patterns.get(pattern_id)
    assert row["disagreements"] >= 8

    after = await system.pipeline.handle(
        _alert("y-live", ts=DAY_TS + 40 * 3600, **alert_kwargs)
    )
    assert after.stage is not Stage.GATED, (
        "a pattern humans keep overruling must stop suppressing on its own"
    )


async def test_metrics_report_measured_numbers_not_asserted_ones(system):
    for i in range(3):
        await system.pipeline.handle(_alert(f"m-{i}", labels=["cat"], ts=DAY_TS + i * 3600))
    funnel = system.pipeline.cost_funnel()
    assert funnel["alerts_ingested"] == 3
    assert funnel["mean_screen_latency_ms"] is not None
    load = system.feedback.review_load()
    assert load["alerts_ingested"] == 3
    accuracy = system.feedback.accuracy_report()
    assert accuracy["auto_downgrade_accuracy"] is None, (
        "accuracy must stay null until humans have actually reviewed auto decisions"
    )


# ---------------- the HTTP service ----------------


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from watchwise.service import create_app

    state = FakeNVRState()
    nvr_client, _http = _fake_client(state)
    app = create_app(_config(tmp_path), nvr_client=nvr_client, start_sources=False)
    with TestClient(app) as c:
        c.nvr_state = state  # type: ignore[attr-defined]
        yield c


def test_health_reports_every_tier(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["nvr"]["reachable"] is True
    assert body["tier2"]["backend"] == "stub"
    assert body["tier3"]["backend"] == "scripted"
    assert body["warmup"] == {}  # stand-in models: nothing to load


def test_the_service_names_the_models_in_use(tmp_path, caplog):
    """B-012: generated configs set the screener in models/roles, which leaves
    vlm.backend and agent.model at defaults that nothing uses."""
    import logging

    from fastapi.testclient import TestClient

    from watchwise.config import ModelEndpoint, RolesSettings
    from watchwise.service import create_app

    config = _config(tmp_path)
    config.models = {"vision": ModelEndpoint(provider="ollama", model="qwen3-vl:8b", base_url="http://unused:11434")}
    config.roles = RolesSettings(screener="vision")
    config.service.warm_up_models = False  # no Ollama here
    nvr_client, _http = _fake_client(FakeNVRState())
    with caplog.at_level(logging.INFO, logger="watchwise.service"):
        with TestClient(create_app(config, nvr_client=nvr_client, start_sources=False)) as c:
            health = c.get("/health").json()
            settings = c.get("/config").json()["vlm"]
    assert (health["tier2"]["backend"], health["tier2"]["model"]) == ("ollama", "qwen3-vl:8b")
    assert (settings["backend"], settings["model"]) == ("ollama", "qwen3-vl:8b")
    assert "screener=ollama/qwen3-vl:8b investigator=scripted/" in caplog.text


def test_tools_endpoint_lists_all_four_categories(client):
    tools = client.get("/tools").json()["tools"]
    assert {t["category"] for t in tools} >= {
        "vector_search", "device_status", "cross_camera", "duty_roster"
    }


def test_a_tool_can_be_invoked_directly_for_debugging(client):
    body = client.post("/tools/get_device_status", json={"camera": "cam_drive"}).json()
    assert body["ok"] is True
    assert body["cameras"]["cam_drive"]["healthy"] is True


def test_unknown_tool_is_404(client):
    assert client.post("/tools/nope", json={}).status_code == 404


def test_simulate_runs_the_real_cascade(client):
    body = client.post(
        "/alerts/simulate",
        json={"camera": "cam_drive", "labels": ["cat"], "review_id": "sim-1"},
    ).json()
    assert body["decision"]["disposition"] == "downgraded"
    assert body["decision"]["stage"] == "screened"


def test_simulate_can_wait_for_the_investigation(client):
    body = client.post(
        "/alerts/simulate",
        json={
            "camera": "cam_drive",
            "labels": ["person"],
            "review_id": "sim-2",
            "start_time": NIGHT_TS,
            "wait_for_investigation": True,
        },
    ).json()
    assert body["decision"]["stage"] == "investigated"
    assert body["decision"]["investigation_id"]

    inv = client.get(f"/investigations/{body['decision']['investigation_id']}").json()
    assert inv["status"] == "done"
    assert inv["trace"]["tool_calls"]


def test_alert_lookup_returns_the_full_decision_history(client):
    client.post(
        "/alerts/simulate",
        json={"camera": "cam_drive", "labels": ["person"], "review_id": "sim-3",
              "start_time": NIGHT_TS, "wait_for_investigation": True},
    )
    body = client.get("/alerts/sim-3").json()
    assert body["camera"] == "cam_drive"
    assert body["labels"] == ["person"]
    assert len(body["decision_history"]) == 2


def test_ingest_accepts_a_raw_nvr_review_payload(client):
    payload = {
        "type": "end",
        "after": {
            "id": "rev-http", "camera": "cam_door", "start_time": DAY_TS,
            "end_time": DAY_TS + 5, "severity": "alert",
            "data": {"objects": ["person"], "zones": [], "detections": []},
        },
    }
    assert client.post("/alerts/ingest", json=payload).json() == {"queued": "rev-http"}


def test_ingest_rejects_nonsense(client):
    assert client.post("/alerts/ingest", json={"type": "end"}).status_code == 400


def test_feedback_creates_a_pattern_and_is_visible(client):
    client.post(
        "/alerts/simulate",
        json={"camera": "cam_drive", "labels": ["car"], "zones": ["driveway"],
              "review_id": "sim-4"},
    )
    body = client.post(
        "/feedback",
        json={"review_id": "sim-4", "verdict": "benign", "note": "headlights",
              "reviewer": "sam"},
    ).json()
    assert body["pattern_id"]

    patterns = client.get("/patterns").json()["patterns"]
    assert any(p["id"] == body["pattern_id"] for p in patterns)

    hits = client.get("/patterns/search", params={"q": "headlights"}).json()["matches"]
    assert hits and hits[0]["description"] == "headlights"


def test_feedback_on_an_unknown_alert_is_404(client):
    assert client.post(
        "/feedback", json={"review_id": "nope", "verdict": "benign"}
    ).status_code == 404


def test_feedback_with_a_bad_verdict_is_400(client):
    client.post("/alerts/simulate",
                json={"camera": "cam_drive", "labels": ["cat"], "review_id": "sim-5"})
    assert client.post(
        "/feedback", json={"review_id": "sim-5", "verdict": "banana"}
    ).status_code == 400


def test_pattern_can_be_deleted(client):
    client.post("/alerts/simulate",
                json={"camera": "cam_drive", "labels": ["cat"], "review_id": "sim-6"})
    pid = client.post(
        "/feedback", json={"review_id": "sim-6", "verdict": "benign", "note": "a cat"}
    ).json()["pattern_id"]
    assert client.delete(f"/patterns/{pid}").json() == {"deleted": pid}
    assert not client.get("/patterns").json()["patterns"]


def test_arming_state_round_trips(client):
    assert client.get("/arming").json()["state"] == "armed_away"
    assert client.post("/arming", json={"state": "disarmed"}).json()["state"] == "disarmed"
    assert client.get("/arming").json()["state"] == "disarmed"
    assert client.post("/arming", json={"state": "banana"}).status_code == 400


def test_metrics_expose_the_cost_funnel(client):
    for i in range(3):
        client.post(
            "/alerts/simulate",
            json={"camera": "cam_drive", "labels": ["cat"], "review_id": f"met-{i}",
                  "start_time": DAY_TS + i * 3600},
        )
    # A wide window, because the fixtures use a fixed historical timestamp;
    # the default 24h window would correctly exclude them.
    body = client.get("/metrics", params={"window_hours": 24 * 365 * 5}).json()
    assert body["cost_funnel"]["alerts_ingested"] == 3
    assert body["review_load"]["alerts_ingested"] == 3
    assert "screener" in body and "investigations" in body


def test_config_endpoint_exposes_the_risk_posture(client):
    body = client.get("/config").json()
    assert body["decision"]["human_fallback_confidence"] == 0.60
    assert body["vlm"]["backend"] == "stub"
