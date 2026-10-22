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

from watchwise.config import DecisionSettings
from watchwise.tier2.gate import GateResult
from watchwise.types import (
    AlertContext,
    Decision,
    Disposition,
    Investigation,
    ScreenResult,
    Severity,
    Stage,
    Verdict,
)

logger = logging.getLogger(__name__)


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

    # ---------------- after tier 2 ----------------

    def from_screening(self, ctx: AlertContext, screen: ScreenResult) -> tuple[Decision, bool]:
        """Returns (decision, needs_investigation).

        A decision is always produced, even when an investigation is queued:
        the alert has to be routed *now*, and tier 3 refines it later.
        """
        s = self.settings

        if screen.verdict is Verdict.BENIGN and screen.confidence >= s.screen_benign_confidence:
            return (
                Decision(
                    review_id=ctx.review_id,
                    disposition=Disposition.DOWNGRADED,
                    severity=Severity.DETECTION,
                    confidence=screen.confidence,
                    rationale=f"tier-2 screening: {screen.reason}",
                    stage=Stage.SCREENED,
                    screen=screen,
                ),
                False,
            )

        if (
            screen.verdict is Verdict.SUSPICIOUS
            and screen.confidence >= s.screen_suspicious_confidence
        ):
            # Surface immediately; still investigate, because the human wants
            # context with the alert, not just the alert.
            return (
                Decision(
                    review_id=ctx.review_id,
                    disposition=Disposition.ALERTED,
                    severity=Severity.ALERT,
                    confidence=screen.confidence,
                    rationale=f"tier-2 screening: {screen.reason}",
                    stage=Stage.SCREENED,
                    screen=screen,
                ),
                True,
            )

        # Everything else is the escalation case the cascade exists for.
        return (
            Decision(
                review_id=ctx.review_id,
                disposition=Disposition.ALERTED,
                severity=Severity.ALERT,
                confidence=screen.confidence,
                rationale=(
                    f"tier-2 screening was not confident ({screen.verdict.value} @ "
                    f"{screen.confidence:.2f}): {screen.reason}. Alerting now; "
                    "investigation in progress."
                ),
                stage=Stage.SCREENED,
                screen=screen,
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
            return Decision(
                disposition=Disposition.ESCALATED_HUMAN,
                severity=Severity.ALERT,
                confidence=inv.confidence,
                rationale=(
                    f"investigation reached {inv.verdict.value} at confidence "
                    f"{inv.confidence:.2f}, below the {s.human_fallback_confidence:.2f} "
                    f"autonomy threshold. {inv.conclusion}"
                ),
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
