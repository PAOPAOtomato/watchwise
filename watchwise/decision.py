"""Confidence policy: what actually happens to an alert.

Kept apart from the tiers on purpose. Screening and investigation produce
*evidence*; this module is the only place that turns evidence into an action,
so the risk posture of the whole system can be read -- and changed -- in one
file.

The governing rule: every conclusion carries a confidence, and
anything below threshold goes to a human. Over-autonomy is the failure mode
being defended against, so the policy is asymmetric:

  * suppressing an alert requires *high* confidence that it is benign
  * raising an alert requires only moderate confidence that it is suspicious
  * anything in between reaches a human

That asymmetry is deliberate. A false alarm costs a human thirty seconds; a
missed intrusion costs considerably more.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from watchwise.config import DecisionSettings
from watchwise.tier2.gate import GateResult
from watchwise.types import (
    AlertContext,
    Decision,
    Disposition,
    Investigation,
    ReviewSummary,
    ScreenResult,
    Severity,
    Stage,
    Verdict,
)

logger = logging.getLogger(__name__)

# Arming states in which nobody is expected to be moving around (D-047).
ARMED_STATES = ("armed_away", "armed_night")


def person_while_armed(ctx: AlertContext, arming: str | None) -> bool:
    """A person in view while nobody should be: tier 2 and memory may not
    dismiss it; tier 3, which reads arming and the duty roster, may."""
    return arming in ARMED_STATES and any(label.lower() == "person" for label in ctx.labels)


def _then(first: str, rest: str) -> str:
    """``first`` (a sentence or nothing), then ``rest`` as the next sentence."""
    return f"{first}{rest[:1].upper()}{rest[1:]}" if first else rest


class DecisionPolicy:
    def __init__(self, settings: DecisionSettings):
        self.settings = settings

    # ---------------- after the gate ----------------

    def from_gate(self, ctx: AlertContext, gate: GateResult) -> Decision:
        """The gate resolved it without a model call."""
        assert gate.verdict is not None

        if gate.verdict is Verdict.SUSPICIOUS:
            return Decision(
                review_id=ctx.review_id,
                disposition=Disposition.ALERTED,
                severity=Severity.ALERT,
                confidence=gate.confidence,
                rationale=f"gate/{gate.rule}: {gate.reason}",
                stage=Stage.GATED,
                matched_memory_ids=gate.matched_pattern_ids,
            )

        # Benign from the gate. Suppress only when a rule is definitive or a
        # trusted memory pattern matched; otherwise merely downgrade, so the
        # alert still exists for a human who goes looking.
        definitive = gate.rule in ("never_screen_label", "memory_pattern")
        disposition = (
            Disposition.SUPPRESSED
            if definitive and gate.confidence >= self.settings.auto_downgrade_confidence
            else Disposition.DOWNGRADED
        )
        return Decision(
            review_id=ctx.review_id,
            disposition=disposition,
            severity=Severity.SUPPRESSED
            if disposition is Disposition.SUPPRESSED
            else Severity.DETECTION,
            confidence=gate.confidence,
            rationale=f"gate/{gate.rule}: {gate.reason}",
            stage=Stage.GATED,
            matched_memory_ids=gate.matched_pattern_ids,
        )

    # ---------------- while tier 2 runs ----------------

    def provisional(self, ctx: AlertContext) -> Decision:
        """Sent when the screener is slower than the latency budget, so that
        the alert is not silent meanwhile. The screened decision replaces it.
        It alerts, because nothing has cleared the event yet."""
        return Decision(
            review_id=ctx.review_id,
            disposition=Disposition.ALERTED,
            severity=Severity.ALERT,
            confidence=0.0,
            rationale="analyzing: the vision model is still looking at this event",
            stage=Stage.INGESTED,
        )

    # ---------------- the summary of the whole event ----------------

    def with_summary(self, decision: Decision, summary: ReviewSummary) -> Decision:
        """Attach what happened over the whole event (D-045). A suspicious or
        critical rating raises an event that was downgraded or held back; it
        never lowers anything, so the summary can only make Watchwise more
        careful."""
        text = summary.text()
        if (
            self.settings.summary_can_raise
            and summary.threat_level >= 1
            and decision.disposition in (Disposition.DOWNGRADED, Disposition.SUPPRESSED)
            and decision.stage is not Stage.INGESTED
        ):
            rating = "critical" if summary.threat_level >= 2 else "suspicious"
            # A watch or "anyone" rule matched: the owner asked, nothing was
            # judged suspicious (live 2026-10-28: a parcel "rated suspicious").
            why = "" if summary.owner_asked else f"raised: the review of the whole event rated it {rating}. "
            return Decision(
                review_id=decision.review_id,
                disposition=Disposition.ALERTED,
                severity=Severity.ALERT,
                confidence=summary.confidence,
                rationale=f"{why}{summary.short_summary}",
                stage=Stage.SUMMARIZED,
                screen=decision.screen,
                investigation_id=decision.investigation_id,
                matched_memory_ids=list(decision.matched_memory_ids),
                summary=text,
            )
        return replace(decision, summary=text)

    # ---------------- faces (D-057) ----------------

    def recognized(self, decision: Decision, names: list[str], score: float = 0.0) -> Decision:
        """Someone on the household list, recognized while at home: held back,
        whatever any model thought (an alert already sent is edited)."""
        return replace(
            decision,
            disposition=Disposition.DOWNGRADED,
            severity=Severity.DETECTION,
            confidence=score,
            rationale=f"recognized as {', '.join(names)} by Frigate's face recognition",
            stage=Stage.FACE,
        )

    def unknown_face(self, decision: Decision, *, expected: str = "") -> Decision:
        """A face the NVR could not match to anyone: raises, never lowers.
        ``expected``: the duty roster expects someone here now (D-068), whose
        face the NVR can't know; then it is only noted."""
        if decision.disposition not in (Disposition.DOWNGRADED, Disposition.SUPPRESSED):
            return decision
        if expected:
            return replace(
                decision,
                rationale=_then(
                    decision.rationale.removesuffix(".") + ". " if decision.rationale else "",
                    f"Frigate saw a face it couldn't match, but {expected}, so that alone doesn't raise it.",
                ),
                stage=Stage.FACE,  # so cards and chat show this reason, not the screener's
            )
        return replace(
            decision,
            disposition=Disposition.ALERTED,
            severity=Severity.ALERT,
            confidence=0.0,
            rationale="a face Frigate's face recognition could not match to anyone it knows",
            stage=Stage.FACE,
        )

    # ---------------- after tier 2 ----------------

    def from_screening(
        self, ctx: AlertContext, screen: ScreenResult, *, arming: str | None = None
    ) -> tuple[Decision, bool]:
        """Returns (decision, needs_investigation).

        A decision is always produced, even when an investigation is queued:
        the alert has to be routed *now*, and tier 3 refines it later.
        """
        s = self.settings
        armed_person = person_while_armed(ctx, arming)

        if (
            screen.verdict is Verdict.BENIGN
            and screen.confidence >= s.screen_benign_confidence
            and not armed_person
        ):
            return (
                Decision(
                    review_id=ctx.review_id,
                    disposition=Disposition.DOWNGRADED,
                    severity=Severity.DETECTION,
                    confidence=screen.confidence,
                    rationale=f"tier-2 screening: {screen.reason}",
                    stage=Stage.SCREENED,
                    screen=screen,
                    matched_memory_ids=list(screen.memory_ids),
                ),
                False,
            )

        if (
            screen.verdict is Verdict.SUSPICIOUS
            and screen.confidence >= s.screen_suspicious_confidence
        ):
            # Surface immediately; still investigate, because the human wants
            # context with the alert, not just the alert -- unless it is what
            # the owner asked to be told about (D-056): live, an investigation
            # agreed it matched and still called it harmless.
            return (
                Decision(
                    review_id=ctx.review_id,
                    disposition=Disposition.ALERTED,
                    severity=Severity.ALERT,
                    confidence=screen.confidence,
                    rationale=f"tier-2 screening: {screen.reason}",
                    stage=Stage.SCREENED,
                    screen=screen,
                    matched_memory_ids=list(screen.memory_ids),
                ),
                not screen.owner_real,
            )

        # Everything else is the escalation case the cascade exists for.
        return (
            Decision(
                review_id=ctx.review_id,
                disposition=Disposition.ALERTED,
                severity=Severity.ALERT,
                confidence=screen.confidence,
                rationale=(
                    f"a person in view while {arming.replace('_', ' ')}: tier 2 saw "
                    f"{screen.verdict.value} @ {screen.confidence:.2f} ({screen.reason}), but "
                    "only an investigation may clear it. Alerting now; investigation in progress."
                    if armed_person and screen.verdict is Verdict.BENIGN
                    else f"tier-2 screening was not confident ({screen.verdict.value} @ "
                    f"{screen.confidence:.2f}): {screen.reason}. Alerting now; "
                    "investigation in progress."
                ),
                stage=Stage.SCREENED,
                screen=screen,
                matched_memory_ids=list(screen.memory_ids),
            ),
            True,
        )

    # ---------------- after tier 3 ----------------

    def from_investigation(
        self,
        ctx: AlertContext,
        screen: ScreenResult | None,
        inv: Investigation,
        *,
        matched_memory_ids: list[str] | None = None,
        arming: str | None = None,
        roster_expects: str | bool = False,
        recently_real: bool = False,
    ) -> Decision:
        """``roster_expects``: why a person is expected now (the duty roster or
        a recognized resident), or false."""
        decision = self._from_investigation(ctx, screen, inv, matched_memory_ids)
        allowed = ""
        if person_while_armed(ctx, arming) and decision.disposition in (
            Disposition.DOWNGRADED, Disposition.SUPPRESSED
        ):
            assert arming is not None
            if not roster_expects:
                # D-049: while nobody should be home, only the duty roster may
                # clear a person; "it looked routine before" may not.
                return replace(
                    decision,
                    disposition=Disposition.ESCALATED_HUMAN,
                    severity=Severity.ALERT,
                    rationale=(
                        f"a person in view while {arming.replace('_', ' ')}, and the duty roster "
                        f"expects nobody here now, so you decide. The investigation leaned "
                        f"benign ({inv.confidence:.2f}): {inv.conclusion}"
                    ),
                )
            # B-032: say what let a person through while armed.
            why = roster_expects if isinstance(roster_expects, str) else "the duty roster expects someone now"
            allowed = f"expected while {arming.replace('_', ' ')}: {why}. "
        if recently_real and decision.disposition is Disposition.SUPPRESSED:
            # D-052: the owner recently marked this kind of event real, so a
            # benign investigation may demote it but not hide it.
            return replace(
                decision,
                disposition=Disposition.DOWNGRADED,
                severity=Severity.DETECTION,
                rationale=_then(
                    allowed,
                    f"you recently marked this kind of event real, so it stays visible. "
                    f"The investigation found it benign ({inv.confidence:.2f}): {inv.conclusion}",
                ),
            )
        if allowed:
            return replace(decision, rationale=_then(allowed, decision.rationale.removeprefix("auto-downgraded: ")))
        return decision

    def _from_investigation(
        self,
        ctx: AlertContext,
        screen: ScreenResult | None,
        inv: Investigation,
        matched_memory_ids: list[str] | None,
    ) -> Decision:
        s = self.settings
        base = dict(
            review_id=ctx.review_id,
            stage=Stage.INVESTIGATED,
            screen=screen,
            investigation_id=inv.id,
            matched_memory_ids=matched_memory_ids or [],
        )

        # 1. Below the human-fallback floor: a human decides, full stop.
        if inv.confidence < s.human_fallback_confidence or inv.verdict is Verdict.UNCERTAIN:
            if inv.verdict is Verdict.UNCERTAIN:
                why = "the investigation could not decide"
            else:
                why = (
                    f"investigation reached {inv.verdict.value} at confidence "
                    f"{inv.confidence:.2f}, below the {s.human_fallback_confidence:.2f} "
                    "autonomy threshold"
                )
            return Decision(
                disposition=Disposition.ESCALATED_HUMAN,
                severity=Severity.ALERT,
                confidence=inv.confidence,
                rationale=f"{why}. {inv.conclusion}",
                **base,
            )

        # 2. Confidently suspicious.
        if inv.verdict is Verdict.SUSPICIOUS:
            return Decision(
                disposition=Disposition.ALERTED,
                severity=Severity.ALERT,
                confidence=inv.confidence,
                rationale=inv.conclusion,
                **base,
            )

        # 3. Confidently benign. Auto-downgrade needs the higher bar, and if
        #    policy says suspicious findings always surface, a benign finding
        #    still only downgrades -- it never deletes the record.
        if inv.confidence >= s.auto_downgrade_confidence:
            return Decision(
                disposition=Disposition.SUPPRESSED,
                severity=Severity.SUPPRESSED,
                confidence=inv.confidence,
                rationale=f"auto-downgraded: {inv.conclusion}",
                **base,
            )

        return Decision(
            disposition=Disposition.DOWNGRADED,
            severity=Severity.DETECTION,
            confidence=inv.confidence,
            rationale=(
                f"benign at confidence {inv.confidence:.2f}, below the "
                f"{s.auto_downgrade_confidence:.2f} auto-suppression bar, so the "
                f"alert is demoted rather than hidden. {inv.conclusion}"
            ),
            **base,
        )

    # ---------------- introspection ----------------

    def describe(self) -> dict[str, float | bool]:
        s = self.settings
        return {
            "screen_benign_confidence": s.screen_benign_confidence,
            "screen_suspicious_confidence": s.screen_suspicious_confidence,
            "human_fallback_confidence": s.human_fallback_confidence,
            "auto_downgrade_confidence": s.auto_downgrade_confidence,
            "always_surface_suspicious": s.always_surface_suspicious,
        }
