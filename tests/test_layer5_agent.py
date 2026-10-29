"""Layer 5: the investigation loop, the decision policy, and the queue.

What is actually being asserted here is the intended behaviour of
the agent: that it plans its own path rather than running a fixed script,
that it reaches a conclusion in a small number of rounds, that it gathers
more evidence instead of guessing when confidence is low, and that it never
blocks the real-time alert path.
"""

from __future__ import annotations

import asyncio

import pytest

from watchwise.config import AgentSettings, DecisionSettings, MemorySettings
from watchwise.decision import DecisionPolicy
from watchwise.memory.short_term import ShortTermMemory
from watchwise.tier2.gate import GateResult
from watchwise.tier3.backends.base import (
    AgentBackend,
    Conclusion,
    ToolRunner,
    parse_conclusion,
)
from watchwise.tier3.backends.scripted import ScriptedBackend
from watchwise.tier3.investigator import InvestigationQueue, Investigator
from watchwise.tools.base import ToolContext
from watchwise.tools.duty_roster import DutyRoster
from watchwise.types import (
    Decision,
    Disposition,
    Investigation,
    ScreenResult,
    Severity,
    Stage,
    Verdict,
)
from tests.conftest import NIGHT_TS, make_alert


@pytest.fixture
def tool_ctx(nvr_client, store, patterns, tmp_path):
    from watchwise.tier2.vlm.stub import StubVLM

    return ToolContext(
        nvr=nvr_client,
        store=store,
        patterns=patterns,
        roster=None,
        config_cache={"vlm": StubVLM(), "arming_path": str(tmp_path / "arming.json")},
    )


@pytest.fixture
def investigator(tool_ctx, store):
    return Investigator(
        ScriptedBackend(max_iterations=6, stop_confidence=0.85),
        tool_ctx,
        store,
        AgentSettings(max_concurrency=1, timeout_s=30.0),
        MemorySettings(),
    )


UNCERTAIN = ScreenResult(
    verdict=Verdict.UNCERTAIN, confidence=0.4, reason="could not tell",
    backend="stub", model="stub-vlm",
)


# ---------------- the ReAct loop ----------------


async def test_investigation_runs_tools_and_reaches_a_conclusion(investigator):
    inv = await investigator.investigate(make_alert(labels=["person"]), UNCERTAIN)
    assert inv.status == "done"
    assert inv.iterations >= 1
    assert inv.tool_calls, "an investigation with no tool calls is not an investigation"
    assert inv.conclusion
    assert 0.0 <= inv.confidence <= 1.0


async def test_investigation_converges_in_a_small_number_of_rounds(investigator):
    """The 2-3 iteration target, as a bound rather than an exact count."""
    inv = await investigator.investigate(make_alert(labels=["person"]), UNCERTAIN)
    assert 1 <= inv.iterations <= 6


async def test_the_plan_adapts_to_the_alert_rather_than_being_fixed(investigator):
    """A person alert consults the roster; a vehicle alert does not."""
    person = await investigator.investigate(
        make_alert(review_id="p", labels=["person"]), UNCERTAIN
    )
    vehicle = await investigator.investigate(
        make_alert(review_id="v", labels=["car"]), UNCERTAIN
    )
    person_tools = {c.name for c in person.tool_calls}
    vehicle_tools = {c.name for c in vehicle.tool_calls}
    assert "check_duty_roster" in person_tools
    assert "check_duty_roster" not in vehicle_tools


async def test_prior_human_judgement_shifts_the_conclusion(investigator, patterns):
    """Long-term memory feeding tier 3, not just the gate."""
    alert = make_alert(camera="cam_drive", labels=["car"], zones=["driveway"])
    pid = patterns.upsert(
        kind="false_alarm",
        signature=alert.signature(),
        description="headlights sweeping the gate",
    )
    for _ in range(5):
        patterns.upsert(kind="false_alarm", signature=alert.signature(),
                        description="headlights")
        patterns.record_outcome(pid, agreed=True)

    inv = await investigator.investigate(alert, UNCERTAIN)
    assert inv.verdict is Verdict.BENIGN
    assert "false-alarm pattern" in inv.conclusion


async def test_multi_camera_activity_pushes_toward_suspicious(investigator, nvr_state):
    import time

    t0 = time.time()
    for cam in ("cam_yard", "cam_door"):
        nvr_state.add_event(event_id=f"e-{cam}", camera=cam, start_time=t0 + 20)
    alert = make_alert(camera="cam_drive", labels=["person"], start_time=t0)
    inv = await investigator.investigate(alert, UNCERTAIN)
    assert inv.verdict is Verdict.SUSPICIOUS
    assert any("cameras recorded activity" in t for t in [inv.conclusion])


async def test_a_disarmed_system_pushes_toward_benign(investigator, tool_ctx, tmp_path):
    from watchwise.tools.device_status import ArmingState

    ArmingState(tool_ctx.config_cache["arming_path"]).set("disarmed")
    inv = await investigator.investigate(make_alert(labels=["person"]), UNCERTAIN)
    assert "disarmed" in inv.conclusion


async def test_roster_match_explains_night_activity(investigator, tool_ctx, tmp_path):
    roster = tmp_path / "roster.yml"
    roster.write_text(
        'timezone: UTC\nentries:\n  - name: night_patrol\n    days: [mon, tue, wed, thu, '
        'fri, sat, sun]\n    start: "00:00"\n    end: "23:59"\n    cameras: [cam_drive]\n'
        '    labels: [person]\n    expect: rostered guard\n    confidence: 0.9\n',
        encoding="utf-8",
    )
    tool_ctx.roster = DutyRoster(roster, "UTC")
    inv = await investigator.investigate(
        make_alert(camera="cam_drive", labels=["person"], start_time=NIGHT_TS), UNCERTAIN
    )
    assert "night_patrol" in inv.conclusion


async def test_low_confidence_triggers_another_evidence_round(tool_ctx, store):
    """The 'gather more rather than guess' behaviour."""
    backend = ScriptedBackend(max_iterations=8, stop_confidence=0.99)
    investigator = Investigator(
        backend, tool_ctx, store, AgentSettings(timeout_s=30.0), MemorySettings()
    )
    inv = await investigator.investigate(make_alert(labels=["person"]), UNCERTAIN)
    names = [c.name for c in inv.tool_calls]
    assert "describe_historical_frames" in names, (
        "with an unreachable stop threshold the agent must escalate to gathering "
        "more visual evidence rather than concluding"
    )


async def test_tool_budget_is_enforced(tool_ctx, store):
    inv = Investigation(id="i", review_id="r", camera="c")
    memory = ShortTermMemory()
    runner = ToolRunner(tool_ctx, memory, inv, max_calls=2)
    await runner.call("get_device_status", {"camera": "cam_drive"})
    await runner.call("cross_reference_cameras", {"start_time": 1_700_000_000.0})
    assert runner.budget_exhausted
    blocked = await runner.call("get_device_status", {"camera": "cam_yard"})
    assert blocked["ok"] is False and "budget" in blocked["error"]


async def test_identical_calls_are_cached_not_repeated(tool_ctx):
    inv = Investigation(id="i", review_id="r", camera="c")
    runner = ToolRunner(tool_ctx, ShortTermMemory(), inv, max_calls=10)
    await runner.call("get_device_status", {"camera": "cam_drive"})
    again = await runner.call("get_device_status", {"camera": "cam_drive"})
    assert "cached result" in again["note"]
    assert len(inv.tool_calls) == 1, "a repeated call must not cost a second round trip"


async def test_unknown_tool_is_reported_not_raised(tool_ctx):
    runner = ToolRunner(tool_ctx, ShortTermMemory(), Investigation(id="i", review_id="r",
                                                                   camera="c"))
    result = await runner.call("teleport", {})
    assert result["ok"] is False and "unknown tool" in result["error"]


# ---------------- failure handling ----------------


async def test_a_timeout_becomes_uncertain_and_escalates(tool_ctx, store):
    class Slow(AgentBackend):
        name = "slow"

        async def run(self, ctx, screen, memory, runner):
            await asyncio.sleep(5)
            return Conclusion(Verdict.BENIGN, 1.0, "never gets here")

    investigator = Investigator(
        Slow(), tool_ctx, store, AgentSettings(timeout_s=0.05), MemorySettings()
    )
    inv = await investigator.investigate(make_alert(), UNCERTAIN)
    assert inv.status == "failed"
    assert inv.verdict is Verdict.UNCERTAIN
    assert inv.confidence == 0.0
    assert "timed out" in inv.error


async def test_a_crashing_backend_does_not_lose_the_alert(tool_ctx, store):
    class Broken(AgentBackend):
        name = "broken"

        async def run(self, ctx, screen, memory, runner):
            raise RuntimeError("model exploded")

    investigator = Investigator(
        Broken(), tool_ctx, store, AgentSettings(timeout_s=5.0), MemorySettings()
    )
    inv = await investigator.investigate(make_alert(), UNCERTAIN)
    assert inv.status == "failed"
    assert inv.verdict is Verdict.UNCERTAIN
    assert "model exploded" in inv.error


async def test_the_trace_is_persisted_for_audit(investigator, store):
    inv = await investigator.investigate(make_alert(labels=["person"]), UNCERTAIN)
    import json

    row = store.get_investigation(inv.id)
    assert row is not None
    trace = json.loads(row["trace_json"])
    assert trace["tool_calls"], "a conclusion with no recorded evidence is not auditable"
    assert trace["thoughts"]


# ---------------- conclusion parsing ----------------


def test_parse_conclusion_accepts_well_formed_json():
    c = parse_conclusion(
        '{"verdict":"suspicious","confidence":0.9,"conclusion":"x",'
        '"recommended_action":"look","evidence":["a","b"]}'
    )
    assert c.verdict is Verdict.SUSPICIOUS and c.confidence == 0.9
    assert c.evidence == ["a", "b"]


def test_parse_conclusion_fails_safe():
    for raw in ("", "nope", "{broken"):
        c = parse_conclusion(raw)
        assert c.verdict is Verdict.UNCERTAIN and c.confidence == 0.0


def test_parse_conclusion_handles_a_fenced_block_with_prose():
    c = parse_conclusion(
        'Here is my finding:\n```json\n{"verdict":"benign","confidence":0.88,'
        '"conclusion":"routine"}\n```'
    )
    assert c.verdict is Verdict.BENIGN and c.confidence == 0.88


# ---------------- the queue ----------------


async def test_queue_processes_items_and_reports_stats(investigator):
    queue = InvestigationQueue(investigator, AgentSettings(max_concurrency=2, timeout_s=30))
    done: list[str] = []

    async def on_done(ctx, inv):
        done.append(inv.review_id)

    queue.on_done = on_done
    await queue.start()
    for i in range(4):
        assert queue.submit(make_alert(review_id=f"r{i}"), UNCERTAIN)
    await queue.drain(timeout=30)
    await queue.stop()
    assert sorted(done) == ["r0", "r1", "r2", "r3"]
    assert queue.stats["completed"] == 4


async def test_queue_never_blocks_the_caller_when_full(investigator):
    """Backpressure must not propagate up into the alert path."""
    queue = InvestigationQueue(investigator, AgentSettings(queue_size=2, max_concurrency=1))
    # Deliberately not started: nothing drains it.
    for i in range(10):
        accepted = queue.submit(make_alert(review_id=f"r{i}"), UNCERTAIN)
        assert accepted is True, "submit must always accept, dropping the oldest instead"
    assert queue.queue.qsize() == 2
    assert queue.stats["dropped"] == 8


async def test_stop_returns_even_if_a_worker_absorbs_its_cancellation(tool_ctx, store):
    """Shutdown must never hang. It did on Python 3.11, where wait_for could
    swallow a cancellation that arrived just as an investigation finished,
    leaving the worker alive and stop() awaiting it forever."""
    started = asyncio.Event()

    class Stubborn(AgentBackend):
        name = "stubborn"

        async def run(self, ctx, screen, memory, runner):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                pass  # absorb the cancellation, as the 3.11 bug did
            return Conclusion(Verdict.UNCERTAIN, 0.0, "absorbed")

    investigator = Investigator(
        Stubborn(), tool_ctx, store, AgentSettings(timeout_s=60.0), MemorySettings()
    )
    queue = InvestigationQueue(investigator, AgentSettings(max_concurrency=1))
    await queue.start()
    queue.submit(make_alert(), UNCERTAIN)
    await started.wait()
    async with asyncio.timeout(10):
        await queue.stop(grace_s=0.2)
    assert queue._workers == []


async def test_a_failing_investigation_does_not_kill_the_worker(tool_ctx, store):
    class Flaky(AgentBackend):
        name = "flaky"
        calls = 0

        async def run(self, ctx, screen, memory, runner):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise RuntimeError("first one explodes")
            return Conclusion(Verdict.BENIGN, 0.9, "fine")

    investigator = Investigator(
        Flaky(), tool_ctx, store, AgentSettings(timeout_s=5.0), MemorySettings()
    )
    queue = InvestigationQueue(investigator, AgentSettings(max_concurrency=1))
    await queue.start()
    queue.submit(make_alert(review_id="bad"), UNCERTAIN)
    queue.submit(make_alert(review_id="good"), UNCERTAIN)
    await queue.drain(timeout=20)
    await queue.stop()
    assert queue.stats["failed"] == 1
    assert queue.stats["completed"] == 1


# ---------------- the decision policy ----------------


@pytest.fixture
def policy():
    return DecisionPolicy(DecisionSettings())


def test_gate_suspicious_alerts_immediately(policy):
    gate = GateResult(passed_through=False, verdict=Verdict.SUSPICIOUS, confidence=1.0,
                      reason="fire label", rule="always_alert_label")
    d = policy.from_gate(make_alert(), gate)
    assert d.disposition is Disposition.ALERTED
    assert d.stage is Stage.GATED


def test_gate_memory_match_suppresses_but_a_weak_rule_only_downgrades(policy):
    strong = GateResult(passed_through=False, verdict=Verdict.BENIGN, confidence=0.9,
                        reason="known pattern", rule="memory_pattern")
    assert policy.from_gate(make_alert(), strong).disposition is Disposition.SUPPRESSED

    weak = GateResult(passed_through=False, verdict=Verdict.BENIGN, confidence=0.7,
                      reason="duplicate", rule="dedupe_window")
    weak_decision = policy.from_gate(make_alert(), weak)
    assert weak_decision.disposition is Disposition.DOWNGRADED
    assert weak_decision.severity is Severity.DETECTION


def test_confident_benign_screening_downgrades_without_investigating(policy):
    screen = ScreenResult(verdict=Verdict.BENIGN, confidence=0.92, reason="a cat")
    decision, needs = policy.from_screening(make_alert(), screen)
    assert decision.disposition is Disposition.DOWNGRADED
    assert needs is False


def test_confident_suspicious_screening_alerts_and_still_investigates(policy):
    screen = ScreenResult(verdict=Verdict.SUSPICIOUS, confidence=0.9, reason="prowler")
    decision, needs = policy.from_screening(make_alert(), screen)
    assert decision.disposition is Disposition.ALERTED
    assert needs is True, "a human wants the context, not just the alert"


def test_uncertain_screening_alerts_now_and_escalates(policy):
    decision, needs = policy.from_screening(make_alert(), UNCERTAIN)
    assert decision.disposition is Disposition.ALERTED
    assert needs is True
    assert "not confident" in decision.rationale


@pytest.mark.parametrize(
    "verdict,confidence,expected",
    [
        (Verdict.UNCERTAIN, 0.9, Disposition.ESCALATED_HUMAN),
        (Verdict.BENIGN, 0.4, Disposition.ESCALATED_HUMAN),
        (Verdict.SUSPICIOUS, 0.4, Disposition.ESCALATED_HUMAN),
        (Verdict.SUSPICIOUS, 0.9, Disposition.ALERTED),
        (Verdict.BENIGN, 0.7, Disposition.DOWNGRADED),
        (Verdict.BENIGN, 0.95, Disposition.SUPPRESSED),
    ],
)
def test_investigation_dispositions(policy, verdict, confidence, expected):
    inv = Investigation(id="i", review_id="r", camera="c", verdict=verdict,
                        confidence=confidence, conclusion="because")
    decision = policy.from_investigation(make_alert(), UNCERTAIN, inv)
    assert decision.disposition is expected


def test_an_escalation_gives_its_real_reason(policy):
    """B-019: an uncertain verdict escalates at any confidence; the text used
    to claim that 0.75 was below the 0.60 threshold."""
    unsure = Investigation(id="i", review_id="r", camera="c", verdict=Verdict.UNCERTAIN,
                           confidence=0.75, conclusion="blurred frames")
    assert policy.from_investigation(make_alert(), None, unsure).rationale == (
        "the investigation could not decide. blurred frames"
    )
    thin = Investigation(id="i", review_id="r", camera="c", verdict=Verdict.BENIGN,
                         confidence=0.4, conclusion="thin")
    assert "0.40, below the 0.60 autonomy threshold" in policy.from_investigation(make_alert(), None, thin).rationale


def test_low_confidence_never_auto_resolves(policy):
    """The over-autonomy guard, stated as a property over the whole range."""
    for conf in (0.0, 0.1, 0.3, 0.59):
        for verdict in (Verdict.BENIGN, Verdict.SUSPICIOUS, Verdict.UNCERTAIN):
            inv = Investigation(id="i", review_id="r", camera="c", verdict=verdict,
                                confidence=conf, conclusion="thin")
            d = policy.from_investigation(make_alert(), None, inv)
            assert d.disposition is Disposition.ESCALATED_HUMAN
            assert d.severity is Severity.ALERT


@pytest.mark.parametrize(
    "disposition, level, expected",
    [
        (Disposition.DOWNGRADED, 1, Disposition.ALERTED),
        (Disposition.SUPPRESSED, 2, Disposition.ALERTED),
        (Disposition.DOWNGRADED, 0, Disposition.DOWNGRADED),
        (Disposition.ALERTED, 0, Disposition.ALERTED),
        (Disposition.ESCALATED_HUMAN, 0, Disposition.ESCALATED_HUMAN),
    ],
)
def test_frigates_summary_can_raise_but_never_lower(policy, disposition, level, expected):
    """D-045: Frigate's review of the whole event is evidence, not a second judge."""
    from watchwise.types import ReviewSummary

    decision = Decision(review_id="r", disposition=disposition, severity=Severity.DETECTION,
                        confidence=0.9, rationale="tier-2 screening: fine", stage=Stage.SCREENED)
    summary = ReviewSummary(title="Person at the window", short_summary="Someone tries the window.",
                            threat_level=level, confidence=0.7)
    result = policy.with_summary(decision, summary)
    assert result.disposition is expected
    assert result.summary == "Person at the window. Someone tries the window."
    if expected is not disposition:
        assert result.stage is Stage.SUMMARIZED and "rated it" in result.rationale


@pytest.mark.parametrize(
    "arming, labels, expected",
    [
        ("armed_away", ["person"], Disposition.ALERTED),
        ("armed_night", ["person", "car"], Disposition.ALERTED),
        ("armed_home", ["person"], Disposition.DOWNGRADED),
        ("disarmed", ["person"], Disposition.DOWNGRADED),
        ("armed_away", ["cat"], Disposition.DOWNGRADED),
        (None, ["person"], Disposition.DOWNGRADED),
    ],
)
def test_a_person_while_armed_is_never_dismissed_by_screening(policy, arming, labels, expected):
    """D-047: only an investigation (which reads arming and the roster) may clear it."""
    screen = ScreenResult(verdict=Verdict.BENIGN, confidence=0.98, reason="a person at a desk")
    decision, investigate = policy.from_screening(make_alert(labels=labels), screen, arming=arming)
    assert decision.disposition is expected
    assert investigate is (expected is Disposition.ALERTED)
    if expected is Disposition.ALERTED:
        assert decision.rationale.startswith("a person in view while armed")


@pytest.mark.parametrize(
    "arming, labels, roster_expects, expected",
    [
        ("armed_away", ["person"], False, Disposition.ESCALATED_HUMAN),
        ("armed_night", ["person"], False, Disposition.ESCALATED_HUMAN),
        ("armed_away", ["person"], True, Disposition.SUPPRESSED),
        ("armed_home", ["person"], False, Disposition.SUPPRESSED),
        ("armed_away", ["cat"], False, Disposition.SUPPRESSED),
    ],
)
def test_while_armed_only_the_roster_clears_a_person(policy, arming, labels, roster_expects, expected):
    """D-049 (B-022): live, the investigator cleared the user walking past while
    armed away, on the history of desk walks alone."""
    inv = Investigation(id="i", review_id="r", camera="c", verdict=Verdict.BENIGN,
                        confidence=0.98, conclusion="matches a recurring false alarm pattern")
    decision = policy.from_investigation(make_alert(labels=labels), None, inv,
                                         arming=arming, roster_expects=roster_expects)
    assert decision.disposition is expected
    if expected is Disposition.ESCALATED_HUMAN:
        assert "duty roster expects nobody" in decision.rationale
    if roster_expects:
        assert decision.rationale.startswith("expected while armed away: the duty roster expects someone now. ")


@pytest.mark.parametrize("recently_real", [False, True])
def test_a_card_says_what_let_a_person_through_while_armed(policy, recently_real):
    """B-032, live 2026-10-28 21:05 (armed away, the duty roster expected
    Alex): the card said only "you recently marked this kind of event real, so
    it stays visible", not why a person was allowed."""
    inv = Investigation(id="i", review_id="r", camera="c", verdict=Verdict.BENIGN,
                        confidence=0.98, conclusion="the duty roster confirms a friend visiting.")
    decision = policy.from_investigation(make_alert(labels=["person"]), None, inv, arming="armed_away",
                                         roster_expects="the duty roster expects Alex (Wed 20:00–23:00)",
                                         recently_real=recently_real)
    assert decision.rationale.startswith(
        "expected while armed away: the duty roster expects Alex (Wed 20:00–23:00). "
        + ("You recently marked this kind of event real" if recently_real else "The duty roster confirms"))
    assert "auto-downgraded" not in decision.rationale
    home = policy.from_investigation(make_alert(labels=["person"]), None, inv, arming="armed_home",
                                     roster_expects="the duty roster expects Alex (Wed 20:00–23:00)")
    assert "expected while" not in home.rationale, "at home nothing needed letting through"


@pytest.mark.parametrize("tags, investigate", [(["owner_real"], False), (["vehicle"], True)])
def test_what_the_owner_asked_for_alerts_without_an_investigation(policy, tags, investigate):
    """D-056: live, the investigator agreed the wave matched the owner's Real
    example and still called it harmless."""
    screen = ScreenResult(verdict=Verdict.SUSPICIOUS, confidence=0.95, reason="waving both arms", tags=tags)
    decision, queued = policy.from_screening(make_alert(labels=["person"]), screen, arming="armed_home")
    assert decision.disposition is Disposition.ALERTED and queued is investigate


@pytest.mark.parametrize(
    "arming, confidence, expected",
    [
        ("armed_home", 0.98, Disposition.DOWNGRADED),  # would have been suppressed
        ("armed_home", 0.80, Disposition.DOWNGRADED),  # already only demoted
        ("armed_away", 0.98, Disposition.ESCALATED_HUMAN),  # D-049 is stricter
    ],
)
def test_after_a_recent_real_an_investigation_can_demote_but_not_hide(policy, arming, confidence, expected):
    """D-052: live, an investigation called a wave that matched the owner's
    Real example harmless (0.98); the card must stay visible."""
    inv = Investigation(id="i", review_id="r", camera="c", verdict=Verdict.BENIGN,
                        confidence=confidence, conclusion="a person already visible earlier")
    decision = policy.from_investigation(make_alert(labels=["person"]), None, inv,
                                         arming=arming, recently_real=True)
    assert decision.disposition is expected
    if confidence >= 0.95 and expected is Disposition.DOWNGRADED:
        assert decision.rationale.startswith("you recently marked this kind of event real")


# ---------------- what the owner said in /setup (D-060) ----------------


async def test_the_investigation_keeps_what_the_owner_said_in_view(tool_ctx, store):
    from watchwise.config import AlertRules, CameraSettings

    briefings: list[str] = []

    class Reading(AgentBackend):
        name = "reading"
        model = "m"

        async def run(self, ctx, screen, memory, runner):
            briefings.append(memory.briefing())
            return Conclusion(verdict=Verdict.SUSPICIOUS, confidence=0.9, conclusion="matches a rule")

    cameras = {"cam_yard": CameraSettings(notes="the back garden; the dog is out in the mornings")}
    alerts = AlertRules(always=["anyone climbing the fence"])
    investigator = Investigator(Reading(), tool_ctx, store, AgentSettings(timeout_s=5.0), MemorySettings(),
                                cameras=cameras, alerts=alerts)
    await investigator.investigate(make_alert(camera="cam_yard"), UNCERTAIN)
    assert 'camera cam_yard looks at: "the back garden; the dog is out in the mornings"' in briefings[0]
    assert 'always wants an alert for: "anyone climbing the fence"' in briefings[0]

    await investigator.investigate(make_alert(camera="cam_drive"), UNCERTAIN)
    assert "back garden" not in briefings[1] and "climbing the fence" in briefings[1]


async def test_other_angles_on_the_same_room_are_not_movement(investigator, tool_ctx, nvr_state):
    """D-062: the case above, but cam_yard and cam_door watch cam_drive's room
    from other angles: the same sighting, so nothing pushes it to suspicious."""
    import time

    from watchwise.config import CameraSettings

    t0 = time.time()
    for cam in ("cam_yard", "cam_door"):
        nvr_state.add_event(event_id=f"e-{cam}", camera=cam, start_time=t0 + 20)
    tool_ctx.config_cache["cameras"] = {
        name: CameraSettings(room="hall") for name in ("cam_drive", "cam_yard", "cam_door")
    }
    inv = await investigator.investigate(make_alert(camera="cam_drive", labels=["person"], start_time=t0), UNCERTAIN)
    assert inv.verdict is not Verdict.SUSPICIOUS
    assert "cameras recorded activity" not in inv.conclusion
