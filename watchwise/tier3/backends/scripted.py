"""A deterministic ReAct backend that uses no LLM.

This is a real investigation loop -- it plans, calls the real forensic tools
against the real NVR, observes results, updates its confidence, and decides
whether to gather more evidence or conclude. Only the *reasoning* step is a
policy function instead of a model.

It exists because:

  * the whole tier-3 pipeline (queue, tools, memory, escalation, feedback)
    must be testable and demonstrable without an API key
  * it gives an exact, reproducible baseline to compare a real model against
  * if the model backend is unreachable, investigations degrade to this
    instead of failing, which keeps the alert flow intact

It is not a substitute for a model. It cannot read a scene, and its planning
is a fixed priority order. Swap ``agent.backend: claude_sdk`` when a key is
available.
"""

from __future__ import annotations

import logging
from typing import Any

from watchwise.memory.short_term import ShortTermMemory
from watchwise.tier3.backends.base import AgentBackend, Conclusion, ToolRunner
from watchwise.types import AlertContext, ScreenResult, Verdict

logger = logging.getLogger(__name__)


class ScriptedBackend(AgentBackend):
    name = "scripted"

    def __init__(
        self,
        model: str = "scripted-react",
        *,
        max_iterations: int = 6,
        stop_confidence: float = 0.85,
    ):
        super().__init__(model)
        self.max_iterations = max_iterations
        self.stop_confidence = stop_confidence

    async def run(
        self,
        ctx: AlertContext,
        screen: ScreenResult | None,
        memory: ShortTermMemory,
        runner: ToolRunner,
    ) -> Conclusion:
        evidence: list[str] = []
        # Belief in "this is benign", starting from the screener's opinion.
        belief = self._prior(screen)
        memory.set_confidence(abs(belief - 0.5) * 2.0)

        plan = self._initial_plan(ctx, screen)
        memory.add_question("Has a human already ruled on this signature?")

        for _ in range(self.max_iterations):
            if not plan:
                break
            step = plan.pop(0)
            iteration = runner.next_iteration()
            runner.record_thought(step["thought"])

            result = await runner.call(step["tool"], step["args"])
            delta, note, followups = self._interpret(step["tool"], result, ctx)

            if note:
                evidence.append(note)
            belief = _combine(belief, delta)
            confidence = abs(belief - 0.5) * 2.0
            memory.set_confidence(confidence)
            memory.note_fact(f"{step['tool']}_belief_delta", round(delta, 3))

            for f in followups:
                if not any(p["tool"] == f["tool"] for p in plan):
                    plan.append(f)

            if confidence >= self.stop_confidence:
                runner.record_thought(
                    f"Evidence confidence {confidence:.2f} at or above the "
                    f"{self.stop_confidence:.2f} stop threshold after {iteration} "
                    "round(s); concluding."
                )
                break

            if not plan and confidence < self.stop_confidence:
                # The "proactively launch another round" behaviour:
                # rather than concluding on thin evidence, escalate to
                # actually looking at more footage.
                extra = self._escalation_step(ctx, runner)
                if extra is not None:
                    runner.record_thought(
                        f"Confidence only {confidence:.2f} and the planned tools are "
                        "exhausted; gathering another round of visual evidence."
                    )
                    plan.append(extra)

        confidence = abs(belief - 0.5) * 2.0
        verdict = self._verdict(belief, confidence)
        return Conclusion(
            verdict=verdict,
            confidence=round(confidence, 3),
            conclusion=self._narrate(verdict, confidence, evidence, runner),
            recommended_action=self._action(verdict, confidence, ctx),
            evidence=evidence,
        )

    # ---------------- planning ----------------

    @staticmethod
    def _prior(screen: ScreenResult | None) -> float:
        """Belief that the alert is benign, in [0, 1]."""
        if screen is None:
            return 0.5
        if screen.verdict is Verdict.BENIGN:
            return 0.5 + 0.5 * screen.confidence * 0.6
        if screen.verdict is Verdict.SUSPICIOUS:
            return 0.5 - 0.5 * screen.confidence * 0.6
        return 0.5

    def _initial_plan(
        self, ctx: AlertContext, screen: ScreenResult | None
    ) -> list[dict[str, Any]]:
        query = " ".join([ctx.camera, *ctx.labels, *ctx.zones]) or ctx.camera
        plan: list[dict[str, Any]] = [
            {
                "thought": "Start with prior human judgements: if a reviewer has "
                "already ruled on this exact signature, that outweighs anything "
                "I can infer from one clip.",
                "tool": "search_similar_alerts",
                "args": {
                    "query": query,
                    "camera": ctx.camera,
                    "signature": ctx.signature(),
                    "limit": 5,
                },
            },
            {
                "thought": "Check whether the alert can be trusted at all: arming "
                "state and camera/detector health change how much weight the "
                "labels deserve.",
                "tool": "get_device_status",
                "args": {"camera": ctx.camera},
            },
        ]
        if "person" in {label.lower() for label in ctx.labels}:
            plan.append(
                {
                    "thought": "A person is claimed. Whether this is expected at "
                    "this hour is decided by the roster, not by the footage.",
                    "tool": "check_duty_roster",
                    "args": {
                        "timestamp": ctx.start_time,
                        "camera": ctx.camera,
                        "zones": ctx.zones,
                        "labels": ctx.labels,
                    },
                }
            )
        plan.append(
            {
                "thought": "See whether any other camera saw the same activity, "
                "which distinguishes a path through the property from an "
                "isolated trigger.",
                "tool": "cross_reference_cameras",
                "args": {
                    "start_time": ctx.start_time,
                    "exclude_camera": ctx.camera,
                    "window_before_s": 120,
                    "window_after_s": 120,
                },
            }
        )
        return plan

    @staticmethod
    def _escalation_step(ctx: AlertContext, runner: ToolRunner) -> dict[str, Any] | None:
        if any(c.name == "describe_historical_frames" for c in runner.investigation.tool_calls):
            return None
        return {
            "thought": "Look at footage from before and after the alert window to "
            "establish approach and departure.",
            "tool": "describe_historical_frames",
            "args": {
                "camera": ctx.camera,
                "timestamp": ctx.start_time,
                "offsets_s": [-45, -15, 15],
                "question": "Is the same subject present before the alert started, "
                "and where do they go afterwards?",
            },
        }

    # ---------------- observation -> belief ----------------

    def _interpret(
        self, tool: str, result: dict[str, Any], ctx: AlertContext
    ) -> tuple[float, str, list[dict[str, Any]]]:
        """Return (belief delta toward benign, evidence note, follow-up steps)."""
        if not result.get("ok", True):
            return 0.0, f"{tool} failed: {result.get('error')}", []

        if tool == "search_similar_alerts":
            hits = result.get("pattern_matches") or []
            if hits:
                top = hits[0]
                weight = min(1.0, top["corrections"] / 5.0) * top["similarity"] * top["trust"]
                if top["kind"] == "false_alarm":
                    return (
                        0.35 * weight,
                        f"Matches a known false-alarm pattern ({top['corrections']} "
                        f"human corrections, similarity {top['similarity']}): "
                        f"{top['description']}",
                        [],
                    )
                return (
                    -0.35 * weight,
                    f"Matches a confirmed true-alarm pattern: {top['description']}",
                    [],
                )
            recurrence = result.get("signature_recurrence") or {}
            if recurrence.get("count", 0) >= 5:
                return (
                    0.10,
                    f"This signature has fired {recurrence['count']} times recently "
                    "with no adverse outcome recorded.",
                    [],
                )
            return 0.0, "No prior human judgement found for this signature.", []

        if tool == "get_device_status":
            notes = []
            delta = 0.0
            arming = (result.get("arming") or {}).get("state")
            if arming in ("disarmed", "armed_home"):
                delta += 0.20
                notes.append(f"System is {arming}; occupants are expected on site.")
            elif arming == "armed_away":
                delta -= 0.10
                notes.append("System is armed_away; nobody should be present.")
            unhealthy = result.get("unhealthy_cameras") or []
            if ctx.camera in unhealthy:
                # A broken camera does not mean benign -- it means the
                # evidence is unreliable, so pull belief back toward 0.5.
                notes.append(
                    f"Camera {ctx.camera} is degraded, so its labels are unreliable."
                )
                return -0.0, " ".join(notes), []
            return delta, " ".join(notes), []

        if tool == "check_duty_roster":
            if not result.get("configured"):
                return 0.0, "No duty roster configured; expectedness unknown.", []
            if result.get("expected"):
                best = max(result["matches"], key=lambda m: m["confidence"])
                return (
                    0.30 * best["confidence"],
                    f"Roster entry '{best['name']}' covers this time: {best['expect']}.",
                    [],
                )
            return (
                -0.20,
                "Nothing on the duty roster explains activity at this time.",
                [],
            )

        if tool == "cross_reference_cameras":
            count = result.get("event_count", 0)
            cams = result.get("cameras_with_activity") or {}
            if count == 0:
                return (
                    0.12,
                    "No other camera recorded anything in the window; activity was "
                    "confined to one view.",
                    [],
                )
            if len(cams) >= 2:
                return (
                    -0.30,
                    f"{len(cams)} other cameras recorded activity in sequence "
                    f"({result.get('interpretation', '')})",
                    [
                        {
                            "thought": "Movement across multiple cameras is the "
                            "strongest signal so far; look at the footage directly.",
                            "tool": "describe_historical_frames",
                            "args": {
                                "camera": ctx.camera,
                                "timestamp": ctx.start_time,
                                "offsets_s": [-30, 0, 30],
                                "question": "Describe the subject and what they do.",
                            },
                        }
                    ],
                )
            return -0.10, f"One other camera also recorded activity ({list(cams)}).", []

        if tool == "describe_historical_frames":
            verdict = result.get("verdict")
            conf = float(result.get("confidence") or 0.0)
            desc = result.get("description", "")
            if verdict == "benign":
                return 0.30 * conf, f"Frames before/after look ordinary: {desc}", []
            if verdict == "suspicious":
                return -0.35 * conf, f"Frames before/after look concerning: {desc}", []
            return 0.0, f"Additional frames were inconclusive: {desc}", []

        if tool == "get_historical_frames":
            return 0.0, f"{result.get('retrieved', 0)} additional frames exist.", []

        return 0.0, "", []

    # ---------------- conclusion ----------------

    @staticmethod
    def _verdict(belief: float, confidence: float) -> Verdict:
        if confidence < 0.30:
            return Verdict.UNCERTAIN
        return Verdict.BENIGN if belief > 0.5 else Verdict.SUSPICIOUS

    @staticmethod
    def _narrate(
        verdict: Verdict, confidence: float, evidence: list[str], runner: ToolRunner
    ) -> str:
        tools = ", ".join(dict.fromkeys(c.name for c in runner.investigation.tool_calls))
        body = " ".join(e for e in evidence if e) or "No decisive evidence was found."
        return (
            f"After {runner.investigation.iterations} round(s) using {tools or 'no tools'}: "
            f"{body} Concluded {verdict.value} at confidence {confidence:.2f}."
        )

    @staticmethod
    def _action(verdict: Verdict, confidence: float, ctx: AlertContext) -> str:
        if verdict is Verdict.SUSPICIOUS:
            return f"Review the {ctx.camera} clip now and verify the subject."
        if verdict is Verdict.UNCERTAIN:
            return "Human review needed; the evidence did not settle."
        return "none"


def _combine(belief: float, delta: float) -> float:
    """Additive update with saturation, so no single tool can dominate."""
    return max(0.02, min(0.98, belief + delta * (1.0 - abs(belief - 0.5) * 0.8)))
