"""Layer 2: the gate, the frame sampler, the VLM contract, and the screener.

The gate tests matter most -- that component decides how much the system
costs to run, and a bug there is silent (you just pay more, or you suppress
something you should not have).
"""

from __future__ import annotations

import pytest

from watchwise.config import GateSettings, VLMSettings
from watchwise.memory.long_term import PatternLibrary
from watchwise.tier2 import prompts
from watchwise.tier2.gate import ScreeningGate
from watchwise.tier2.sampling import FrameSampler, sample_timestamps
from watchwise.tier2.screener import Screener
from watchwise.tier2.vlm.base import VLMBackend, parse_screen_reply
from watchwise.tier2.vlm.stub import StubVLM
from watchwise.types import Frame, Verdict
from tests.conftest import DAY_TS, NIGHT_TS, make_alert


# ---------------- the gate ----------------


def test_gate_passes_through_when_nothing_matches(patterns):
    gate = ScreeningGate(GateSettings(), patterns)
    result = gate.evaluate(make_alert())
    assert result.passed_through
    assert result.rule == "no_rule_matched"


def test_gate_always_alert_label_skips_the_model(patterns):
    gate = ScreeningGate(GateSettings(always_alert_labels=["fire"]), patterns)
    result = gate.evaluate(make_alert(labels=["fire", "person"]))
    assert not result.passed_through
    assert result.verdict is Verdict.SUSPICIOUS
    assert result.confidence == 1.0
    assert result.rule == "always_alert_label"


def test_gate_never_screen_label_requires_all_labels_to_match(patterns):
    gate = ScreeningGate(GateSettings(never_screen_labels=["cat", "dog"]), patterns)
    assert not gate.evaluate(make_alert(labels=["cat"])).passed_through
    # A person alongside the cat must NOT be suppressed.
    assert gate.evaluate(make_alert(review_id="r2", labels=["cat", "person"])).passed_through


def test_gate_dedupes_within_the_window_but_not_outside_it(patterns):
    gate = ScreeningGate(GateSettings(dedupe_window_s=60.0), patterns)
    assert gate.evaluate(make_alert(start_time=1000.0)).passed_through
    second = gate.evaluate(make_alert(review_id="r2", start_time=1030.0))
    assert not second.passed_through
    assert second.rule == "dedupe_window"
    assert gate.evaluate(make_alert(review_id="r3", start_time=1200.0)).passed_through


def test_gate_score_floor(patterns):
    gate = ScreeningGate(GateSettings(min_object_score=0.7), patterns)
    alert = make_alert()
    alert.detections = [{"score": 0.4}, {"score": 0.55}]
    result = gate.evaluate(alert)
    assert not result.passed_through
    assert result.rule == "low_score"


def test_gate_suppresses_only_a_trusted_repeat_pattern(store, patterns):
    """The auto-downgrade path: enough corrections AND enough trust."""
    settings = GateSettings(memory_min_corrections=3, memory_suppress_similarity=0.86)
    gate = ScreeningGate(settings, patterns)
    alert = make_alert(camera="cam_drive", labels=["car"], zones=["driveway"])
    sig = alert.signature()

    # One correction: not yet trusted, must still be screened.
    patterns.upsert(kind="false_alarm", signature=sig, description="headlights sweep the gate")
    assert gate.evaluate(alert).passed_through

    # Three corrections plus agreeing outcomes: now it may suppress.
    pid = patterns.upsert(kind="false_alarm", signature=sig, description="headlights again")
    patterns.upsert(kind="false_alarm", signature=sig, description="headlights again")
    for _ in range(3):
        patterns.record_outcome(pid, agreed=True)

    result = gate.evaluate(make_alert(review_id="r9", camera="cam_drive",
                                      labels=["car"], zones=["driveway"],
                                      start_time=alert.start_time + 4000.0))
    assert not result.passed_through
    assert result.rule == "memory_pattern"
    assert result.verdict is Verdict.BENIGN
    assert result.matched_pattern_ids == [pid]


def test_gate_pattern_loses_the_right_to_suppress_when_it_disagrees(store, patterns):
    settings = GateSettings(memory_min_corrections=1)
    gate = ScreeningGate(settings, patterns, min_pattern_trust=0.7)
    alert = make_alert(labels=["car"], zones=["driveway"])
    pid = patterns.upsert(kind="false_alarm", signature=alert.signature(),
                          description="headlights")
    for _ in range(6):
        patterns.record_outcome(pid, agreed=False)

    result = gate.evaluate(alert)
    assert result.passed_through, "a pattern humans keep overruling must stop suppressing"


# ---------------- frame sampling ----------------


def test_sample_timestamps_are_inside_the_segment_and_front_biased():
    alert = make_alert(start_time=1000.0, duration=10.0)
    ts = sample_timestamps(alert, 3)
    assert len(ts) == 3
    assert all(1000.0 <= t <= 1010.0 for t in ts)
    assert ts == sorted(ts)
    # Front bias: the midpoint sample sits before the true middle.
    assert ts[1] < 1005.0


def test_sample_timestamps_degenerate_cases():
    assert sample_timestamps(make_alert(), 0) == []
    assert len(sample_timestamps(make_alert(duration=0.1), 3)) == 1


async def test_sampler_falls_back_when_recordings_are_missing(nvr_client, nvr_state):
    nvr_state.recordings_available = False
    nvr_state.add_event(event_id="ev-1", camera="cam_drive")
    sampler = FrameSampler(nvr_client)
    frames = await sampler.for_alert(
        make_alert(camera="cam_drive"), 3
    )
    assert frames, "must fall back rather than return nothing"
    assert frames[0].source in ("snapshot",)


async def test_sampler_falls_all_the_way_back_to_the_live_frame(nvr_client, nvr_state):
    nvr_state.recordings_available = False
    sampler = FrameSampler(nvr_client)
    frames = await sampler.for_alert(make_alert(camera="cam_drive"), 2)
    assert len(frames) == 1
    assert frames[0].source == "snapshot"


async def test_sampler_returns_recording_frames_when_available(nvr_client):
    sampler = FrameSampler(nvr_client)
    frames = await sampler.for_alert(make_alert(camera="cam_drive"), 3)
    assert len(frames) == 3
    assert all(f.source == "recording" for f in frames)


# ---------------- the VLM contract ----------------


def test_parse_accepts_bare_json():
    r = parse_screen_reply('{"verdict":"benign","confidence":0.9,"reason":"a cat"}')
    assert r.verdict is Verdict.BENIGN and r.confidence == 0.9


def test_parse_accepts_fenced_json_with_prose():
    raw = 'Sure!\n```json\n{"verdict": "suspicious", "confidence": 0.7, "reason": "x"}\n```\n'
    assert parse_screen_reply(raw).verdict is Verdict.SUSPICIOUS


def test_parse_accepts_json_embedded_in_prose():
    raw = 'Looking at the frames: {"verdict":"benign","confidence":0.6,"reason":"y"} done.'
    assert parse_screen_reply(raw).verdict is Verdict.BENIGN


def test_parse_maps_aliases_and_clamps_confidence():
    r = parse_screen_reply('{"verdict":"THREAT","confidence":5,"reason":"z"}')
    assert r.verdict is Verdict.SUSPICIOUS and r.confidence == 1.0


def test_parse_fails_open_to_uncertain():
    """The safety property: unparseable must escalate, never silently drop."""
    for raw in ("", "I cannot help with that", "{not json"):
        r = parse_screen_reply(raw)
        assert r.verdict is Verdict.UNCERTAIN
        assert r.confidence == 0.0


async def test_backend_error_becomes_uncertain_not_an_exception():
    class Exploding(VLMBackend):
        name = "boom"

        async def infer(self, frames, prompt, system):
            raise RuntimeError("model offline")

    result = await Exploding("m").screen([], "p", "s")
    assert result.verdict is Verdict.UNCERTAIN
    assert result.error == "model offline"
    assert result.backend == "boom"


async def test_stub_vlm_is_deterministic_and_reads_the_prompt():
    vlm = StubVLM()
    night = make_alert(labels=["person"], zones=["perimeter"], start_time=NIGHT_TS)
    prompt = prompts.build_prompt(night, 3)
    a = await vlm.screen([], prompt, prompts.SYSTEM)
    b = await vlm.screen([], prompt, prompts.SYSTEM)
    assert a.verdict is Verdict.SUSPICIOUS
    assert a.to_dict() | {"latency_ms": 0} == b.to_dict() | {"latency_ms": 0}


@pytest.mark.parametrize(
    "labels,zones,ts,expected",
    [
        (["cat"], [], DAY_TS, Verdict.BENIGN),
        (["person"], ["perimeter"], NIGHT_TS, Verdict.SUSPICIOUS),
        (["car"], [], DAY_TS, Verdict.BENIGN),
        (["car"], [], NIGHT_TS, Verdict.UNCERTAIN),
        ([], [], DAY_TS, Verdict.UNCERTAIN),
    ],
)
async def test_stub_vlm_verdicts(labels, zones, ts, expected):
    alert = make_alert(labels=labels, zones=zones, start_time=ts)
    result = await StubVLM().screen([], prompts.build_prompt(alert, 2), prompts.SYSTEM)
    assert result.verdict is expected


# ---------------- the screener ----------------


async def test_screener_skips_the_model_when_the_gate_resolves(nvr_client, patterns):
    gate = ScreeningGate(GateSettings(never_screen_labels=["cat"]), patterns)
    screener = Screener(gate, FrameSampler(nvr_client), StubVLM(), VLMSettings(frames=2))
    gate_result, screen = await screener.screen(make_alert(labels=["cat"]))
    assert not gate_result.passed_through
    assert screen is None, "a gated alert must not reach the model"
    assert screener.stats == {"gated": 1, "screened": 0, "errors": 0}


async def test_screener_calls_the_model_when_the_gate_passes(nvr_client, patterns):
    gate = ScreeningGate(GateSettings(), patterns)
    screener = Screener(gate, FrameSampler(nvr_client), StubVLM(), VLMSettings(frames=2))
    gate_result, screen = await screener.screen(
        make_alert(camera="cam_drive", labels=["person"], start_time=NIGHT_TS,
                   zones=["perimeter"])
    )
    assert gate_result.passed_through
    assert screen is not None
    assert screen.frames_used == 2
    assert screen.verdict is Verdict.SUSPICIOUS
    assert screener.stats["screened"] == 1


async def test_screener_reports_no_frames_as_uncertain(nvr_client, nvr_state, patterns):
    nvr_state.recordings_available = False
    nvr_state.cameras.pop("cam_drive")  # kill every fallback too
    gate = ScreeningGate(GateSettings(), patterns)
    screener = Screener(gate, FrameSampler(nvr_client), StubVLM(), VLMSettings(frames=2))
    _, screen = await screener.screen(make_alert(camera="cam_drive"))
    assert screen is not None
    assert screen.verdict is Verdict.UNCERTAIN
    assert screen.error == "no_frames"


async def test_screener_injects_memory_context_into_the_prompt(nvr_client, patterns):
    """Patterns too weak for the gate still inform the model."""
    alert = make_alert(camera="cam_drive", labels=["car"], zones=["driveway"])
    patterns.upsert(
        kind="false_alarm",
        signature=alert.signature(),
        description="headlights sweeping the gate get labelled as a person",
    )
    gate = ScreeningGate(GateSettings(), patterns)
    screener = Screener(gate, FrameSampler(nvr_client), StubVLM(), VLMSettings(frames=1),
                        patterns)
    context = screener._memory_context(alert)
    assert "headlights sweeping the gate" in context
    assert "priors from past human review" in context
