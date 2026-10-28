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
from watchwise.tier2.vlm.base import VLMBackend, VLMReply, parse_screen_reply
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


def test_a_review_is_never_a_duplicate_of_itself(patterns):
    """Frigate reports one event when it starts and again when it ends."""
    gate = ScreeningGate(GateSettings(dedupe_window_s=60.0), patterns)
    assert gate.evaluate(make_alert(review_id="r1", start_time=1000.0)).passed_through
    assert gate.evaluate(make_alert(review_id="r1", start_time=1000.0)).passed_through
    assert gate.evaluate(make_alert(review_id="r2", start_time=1010.0)).rule == "dedupe_window"


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


class Raising(VLMBackend):
    name = "raising"

    def __init__(self, exc: Exception):
        super().__init__("m")
        self.exc = exc

    async def infer(self, frames, prompt, system):
        raise self.exc


async def test_an_exception_without_a_message_is_still_an_error():
    """B-010: httpx's ReadTimeout has an empty message, so a timeout used to
    look like success (``error == ""``) and `check` printed [ok]."""
    import httpx

    timeout = await Raising(httpx.ReadTimeout("")).screen([], "p", "s")
    assert (timeout.verdict, timeout.confidence, timeout.error) == (Verdict.UNCERTAIN, 0.0, "timeout")
    assert "did not answer in time (ReadTimeout)" in timeout.reason

    try:  # a provider that re-raises a timeout as its own error keeps the cause
        try:
            raise httpx.ReadTimeout("")
        except httpx.ReadTimeout as exc:
            raise RuntimeError("cannot reach Ollama") from exc
    except RuntimeError as wrapped:
        assert (await Raising(wrapped).screen([], "p", "s")).error == "timeout"

    bare = await Raising(RuntimeError()).screen([], "p", "s")
    assert bare.error == "RuntimeError" and bare.reason == "screening backend error: RuntimeError"


class Replying(VLMBackend):
    name = "replying"

    def __init__(self, reply):
        super().__init__("m")
        self.reply = reply

    async def infer(self, frames, prompt, system):
        return self.reply


async def test_no_usable_reply_is_an_error_but_a_complete_answer_is_kept():
    empty = await Replying("").screen([], "p", "s")
    assert (empty.verdict, empty.error) == (Verdict.UNCERTAIN, "empty_reply")

    cut = await Replying(VLMReply("", truncated=True, max_tokens=256, thinking=True)).screen([], "p", "s")
    assert (cut.verdict, cut.error) == (Verdict.UNCERTAIN, "truncated")
    assert "256-token output cap while it was still thinking" in cut.reason
    half = await Replying(VLMReply('{"verdict": "ben', truncated=True, max_tokens=64)).screen([], "p", "s")
    assert half.error == "truncated" and half.reason.endswith("64-token output cap mid-answer")

    # Ollama's JSON mode can pad a finished answer with whitespace up to the
    # cap: the reply is "truncated", but the verdict in it is complete.
    padded = VLMReply('{"verdict": "benign", "confidence": 0.9, "reason": "cat"}\n\n\n', truncated=True)
    kept = await Replying(padded).screen([], "p", "s")
    assert (kept.verdict, kept.confidence, kept.error) == (Verdict.BENIGN, 0.9, None)


def test_parse_ignores_inline_thinking():
    raw = '<think>Is {that} a cat? It has a tail.</think>{"verdict":"benign","confidence":0.8,"reason":"cat"}'
    assert parse_screen_reply(raw).verdict is Verdict.BENIGN
    assert parse_screen_reply("<think>still deciding {").error == "empty_reply"


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
    assert screener.stats == {"gated": 1, "screened": 0, "errors": 0, "summaries": 0}


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


async def test_the_screener_names_the_close_patterns_it_was_shown(nvr_client, patterns):
    """They shaped the verdict, so a correction of it must score them."""
    alert = make_alert(camera="cam_drive", labels=["car"], zones=["driveway"])
    pattern_id = patterns.upsert(
        kind="false_alarm", signature=alert.signature(), description="headlights on the gate"
    )
    gate = ScreeningGate(GateSettings(), patterns)
    screener = Screener(gate, FrameSampler(nvr_client), StubVLM(), VLMSettings(frames=1), patterns)
    _, screen = await screener.screen(alert)
    assert screen.memory_ids == [pattern_id]
    _, unrelated = await screener.screen(make_alert(review_id="r2", camera="cam_door", labels=["dog"]))
    assert unrelated.memory_ids == []


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


def test_the_prompt_says_who_should_be_home():
    """B-015: the screener used to judge a person indoors without knowing the house was armed."""
    alert = make_alert(labels=["person"])
    assert "nobody is expected on site" in prompts.build_prompt(alert, 1, arming="armed_away")
    assert "asleep" in prompts.build_prompt(alert, 1, arming="armed_night")
    assert "Household" not in prompts.build_prompt(alert, 1)


async def test_the_screener_tells_the_model_the_arming_state(nvr_client, patterns):
    seen: list[str] = []

    class Recording(VLMBackend):
        name = "recording"

        async def infer(self, frames, prompt, system):
            seen.append(prompt)
            return '{"verdict": "benign", "confidence": 0.9, "reason": "x"}'

    class Armed:
        def get(self):
            return {"state": "armed_away"}

    screener = Screener(ScreeningGate(GateSettings(), patterns), FrameSampler(nvr_client),
                        Recording("m"), VLMSettings(frames=1), arming=Armed())
    await screener.screen(make_alert(camera="cam_drive"))
    assert "Household: armed away" in seen[0]


# ---------------- the whole-event summary (M1-18, D-046) ----------------


def test_a_summary_samples_the_whole_event_evenly():
    alert = make_alert(start_time=1000.0, duration=50.0)
    assert sample_timestamps(alert, 4, front_bias=False) == [1010.0, 1020.0, 1030.0, 1040.0]
    assert max(sample_timestamps(alert, 4)) < 1040.0, "screening frames stay front-biased"


def test_summary_replies_are_read_forgivingly_and_never_guessed():
    from watchwise.tier2.vlm.base import parse_summary_reply

    good = parse_summary_reply('Sure: ```json\n{"title": "Walks past", "summary": "A person walks left to right '
                               'and leaves.", "threat_level": 5, "confidence": 0.8, "concerns": "none"}\n```')
    assert (good.title, good.threat_level, good.confidence, good.concerns) == ("Walks past", 2, 0.8, ["none"])
    assert good.text() == "Walks past. A person walks left to right and leaves."
    for bad in (None, "", "no idea", '{"title": "x"}', "<think>{</think>"):
        assert parse_summary_reply(bad) is None


def test_the_summary_prompt_names_the_concerns_and_the_household():
    alert = make_alert(labels=["person"], duration=42.0)
    prompt = prompts.build_summary_prompt(alert, 4, arming="armed_night", concerns=["unusual gestures"])
    assert "4 frame(s) spread across one event" in prompt and '"duration_s": 42.0' in prompt
    assert "- unusual gestures" in prompt and "asleep" in prompt
    assert "start to finish" in prompts.SUMMARY_SYSTEM
    assert "marked earlier events like this one real" not in prompt


def test_the_summary_is_shown_what_the_owner_marked_real():
    """D-053: the summary can only raise, so it sees the Real examples."""
    prompt = prompts.build_summary_prompt(make_alert(labels=["person"]), 4,
                                          owner_real=["waving both arms at the camera"])
    assert '- "waving both arms at the camera"' in prompt
    # B-026: the key the reply is parsed with (parse_summary_reply), not Frigate's.
    assert "rate threat_level at least 1" in prompt and "potential_threat_level" not in prompt


async def test_the_screener_summarizes_the_whole_event(nvr_client, patterns):
    asked: list[tuple[int, str]] = []

    class Describing(VLMBackend):
        name = "describing"

        async def infer(self, frames, prompt, system):
            asked.append((len(frames), system))
            return '{"title": "Delivery", "summary": "A courier leaves a box and goes.", "threat_level": 0}'

    gate = ScreeningGate(GateSettings(), patterns)
    screener = Screener(gate, FrameSampler(nvr_client), Describing("m"), VLMSettings(summary_frames=3))
    summary = await screener.summarize(make_alert(camera="cam_drive", duration=30.0))
    assert summary.short_summary == "A courier leaves a box and goes."
    assert asked == [(3, prompts.SUMMARY_SYSTEM)]

    off = Screener(gate, FrameSampler(nvr_client), Describing("m"), VLMSettings(summary_frames=0))
    assert await off.summarize(make_alert(camera="cam_drive")) is None and len(asked) == 1

    class Failing(VLMBackend):
        name = "failing"

        async def infer(self, frames, prompt, system):
            raise RuntimeError("model unavailable")

    broken = Screener(gate, FrameSampler(nvr_client), Failing("m"), VLMSettings(summary_frames=3))
    assert await broken.summarize(make_alert(camera="cam_drive")) is None


def test_model_checked_memory_never_holds_a_routine_back_at_the_gate(store, patterns):
    """D-051: the gate can't see behaviour, so on GPU profiles it doesn't decide routines."""
    alert = make_alert(camera="cam_drive", labels=["car"], zones=["driveway"])
    pattern_id = patterns.upsert(kind="false_alarm", signature=alert.signature(), description="headlights")
    for _ in range(4):
        patterns.upsert(kind="false_alarm", signature=alert.signature(), description="headlights")
        patterns.record_outcome(pattern_id, agreed=True)
    free = ScreeningGate(GateSettings(), patterns, store=store)
    assert free.evaluate(alert).rule == "memory_pattern"
    checked = ScreeningGate(GateSettings(memory_mode="model_checked"), patterns, store=store)
    assert checked.evaluate(make_alert(review_id="r2", camera="cam_drive", labels=["car"], zones=["driveway"],
                                       start_time=DAY_TS + 999)).passed_through


async def test_a_match_to_what_the_owner_marked_real_is_never_benign(nvr_client, store, patterns):
    """D-055: live, the model said "waving both arms at the camera, which matches a
    previously marked real event" and still answered benign."""
    import json
    import time as _time

    alert = make_alert(camera="cam_door", labels=["person"], zones=["hall"])
    store.upsert_alert({"review_id": "old", "camera": "cam_door", "start_time": 1.0, "signature": alert.signature(),
                        "labels": ["person"], "zones": ["hall"]}, "screened")
    store.execute("INSERT INTO corrections(review_id, signature, human_verdict, note, created_at) VALUES (?,?,?,?,?)",
                  ("old", alert.signature(), "suspicious", "waving both arms at the camera", _time.time()))
    prompts_seen: list[str] = []

    class Replying(VLMBackend):
        name = "replying"

        def __init__(self, tags):
            super().__init__("m")
            self.tags = tags

        async def infer(self, frames, prompt, system):
            prompts_seen.append(prompt)
            return json.dumps({"verdict": "benign", "confidence": 0.95, "reason": "waving", "tags": self.tags})

    gate = ScreeningGate(GateSettings(), patterns, store=store)
    for tags, expected in ((["OWNER_REAL"], Verdict.SUSPICIOUS), (["expected"], Verdict.BENIGN)):
        screener = Screener(gate, FrameSampler(nvr_client), Replying(tags), VLMSettings(), patterns=patterns)
        assert (await screener.screen_frames(alert)).verdict is expected
    assert 'marked real: "waving both arms at the camera"' in prompts_seen[-1]
    assert 'add the tag "owner_real"' in prompts_seen[-1]


# ---------------- what the owner said in /setup (D-060) ----------------


def test_the_prompts_carry_the_camera_notes_and_the_always_alert_rules():
    alert = make_alert(camera="webcam", labels=["person"])
    plain = prompts.build_prompt(alert, 2)
    assert "the owner" not in plain.lower()
    screen = prompts.build_prompt(alert, 2, camera_notes="my bedroom workspace",
                                  always_alert=["anyone at the window"])
    assert "About this camera, from the owner: my bedroom workspace" in screen
    assert 'The owner always wants an alert for:\n- "anyone at the window"' in screen
    assert 'answer "suspicious" and add the tag "owner_real"' in screen
    summary = prompts.build_summary_prompt(alert, 4, camera_notes="my bedroom workspace",
                                           always_alert=["anyone at the window"])
    assert "About this camera, from the owner: my bedroom workspace" in summary
    assert '- "anyone at the window"' in summary and "rate threat_level at least 1" in summary


async def test_the_screener_reads_what_setup_changes_while_it_runs(nvr_client, patterns):
    """/setup updates the service's config objects in place; the next screen
    uses them, with no restart."""
    from watchwise.config import AlertRules, CameraSettings

    seen: list[str] = []

    class Recording(VLMBackend):
        name = "recording"

        async def infer(self, frames, prompt, system):
            seen.append(prompt)
            return '{"verdict": "benign", "confidence": 0.95, "reason": "someone at the window", "tags": ["owner_real"]}'

    cameras: dict[str, CameraSettings] = {}
    alerts = AlertRules()
    screener = Screener(ScreeningGate(GateSettings(), patterns), FrameSampler(nvr_client), Recording("m"),
                        VLMSettings(frames=1), cameras=cameras, alerts=alerts)
    alert = make_alert(camera="cam_drive")
    await screener.screen_frames(alert)
    assert "owner" not in seen[0]

    cameras["cam_drive"] = CameraSettings(notes="the driveway; the bins go out on Tuesdays")
    alerts.always = ["anyone at the window"]
    result = await screener.screen_frames(alert)
    assert "About this camera, from the owner: the driveway" in seen[1]
    assert '- "anyone at the window"' in seen[1]
    assert result.verdict is Verdict.SUSPICIOUS, "a rule the owner set can only raise"


def test_a_watch_answer_counts_only_with_evidence():
    """B-029: a watch is its own strict question; a bare yes isn't enough."""
    from watchwise.tier2.vlm.base import parse_watch_reply

    assert parse_watch_reply('{"seen": true, "evidence": "a person lying on the floor"}') == (True, "a person lying on the floor")
    assert parse_watch_reply('```json\n{"seen": "yes", "evidence": "the parcel is gone"}\n```')[0] is True
    assert parse_watch_reply('{"seen": true, "evidence": ""}') == (False, "")
    assert parse_watch_reply('{"seen": false, "evidence": "a person seated at the desk"}')[0] is False
    for unreadable in (None, "", "yes", "<think>{</think>"):
        assert parse_watch_reply(unreadable) is None


def test_only_rules_worded_for_anyone_cover_recognized_residents():
    """D-067: the owner's wording decides ("anyone", "any person")."""
    from watchwise.config import AlertRules

    rules = AlertRules(always=["anyone at the window", "Any person in the kitchen at night", "anybody climbing",
                               "everyone leaving", "someone at the window", "a ladder", "company van", "many people"])
    assert rules.for_everyone() == ["anyone at the window", "Any person in the kitchen at night", "anybody climbing",
                                    "everyone leaving"]
