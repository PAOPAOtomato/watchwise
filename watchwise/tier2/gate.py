"""The cheap, model-free gate in front of the VLM.

This is the component that actually delivers the cost reduction. A VLM call
per alert is not cheap at 10 streams; a gate that resolves the repetitive
majority for free is. It runs in microseconds and answers one question: does
this alert deserve a model call at all?

Four ways an alert can be resolved here, in priority order:

  1. ``always_alert_labels``  -- configured to always reach a human
  2. ``never_screen_labels``  -- configured to never be worth inference
  3. duplicate within the dedupe window -- the same signature we just handled
  4. a trusted long-term false-alarm pattern match

Anything the gate cannot resolve is passed through to tier 2, unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from watchwise.config import GateSettings
from watchwise.memory.long_term import PatternLibrary, PatternMatch
from watchwise.types import AlertContext, Verdict


@dataclass(slots=True)
class GateResult:
    """Either a resolution (``passed_through`` False) or a pass-through."""

    passed_through: bool
    verdict: Verdict | None = None
    confidence: float = 0.0
    reason: str = ""
    rule: str = ""
    matched_pattern: PatternMatch | None = None
    matched_pattern_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed_through": self.passed_through,
            "verdict": self.verdict.value if self.verdict else None,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "rule": self.rule,
            "matched_pattern": self.matched_pattern.to_dict() if self.matched_pattern else None,
        }


class ScreeningGate:
    def __init__(
        self,
        settings: GateSettings,
        patterns: PatternLibrary,
        *,
        store: Any = None,
        min_pattern_trust: float = 0.7,
    ):
        self.settings = settings
        self.patterns = patterns
        self.store = store
        self.min_pattern_trust = min_pattern_trust
        # signature -> (start time, review id) of the last one we resolved,
        # for in-process dedupe.
        self._recent: dict[str, tuple[float, str]] = {}

    def evaluate(self, ctx: AlertContext, *, use_memory: bool = True) -> GateResult:
        """Decide whether this alert is worth a model call.

        Time-based rules read the alert's own ``start_time`` rather than the
        wall clock, so behaviour is identical whether alerts arrive live or
        are replayed from a backlog.
        """
        if not self.settings.enabled:
            return GateResult(passed_through=True, rule="gate_disabled")

        labels = {label.lower() for label in ctx.labels}

        # 1. Hard allow-list: never spend inference deciding these.
        forced = labels & {label.lower() for label in self.settings.always_alert_labels}
        if forced:
            return GateResult(
                passed_through=False,
                verdict=Verdict.SUSPICIOUS,
                confidence=1.0,
                reason=f"label(s) {sorted(forced)} are configured to always alert",
                rule="always_alert_label",
            )

        # 2. Hard deny-list.
        if labels and labels <= {label.lower() for label in self.settings.never_screen_labels}:
            return GateResult(
                passed_through=False,
                verdict=Verdict.BENIGN,
                confidence=1.0,
                reason=f"label(s) {sorted(labels)} are configured as never-screen",
                rule="never_screen_label",
            )

        # 3. Object score floor -- a weak detection is not worth a VLM call.
        if self.settings.min_object_score > 0 and ctx.detections:
            best = max(
                (float(d.get("score") or d.get("top_score") or 0.0) for d in ctx.detections),
                default=0.0,
            )
            if best < self.settings.min_object_score:
                return GateResult(
                    passed_through=False,
                    verdict=Verdict.BENIGN,
                    confidence=0.75,
                    reason=f"best object score {best:.2f} below floor "
                    f"{self.settings.min_object_score:.2f}",
                    rule="low_score",
                )

        # 4. Recent duplicate of the same signature.
        #
        # Keyed on the *alert's own* start time, not on when we happen to be
        # processing it. Two alerts a minute apart are duplicates whether we
        # screened them live or replayed them from a backlog an hour later.
        signature = ctx.signature()
        last = self._recent.get(signature)
        # A review is never a duplicate of itself: Frigate reports one event
        # at its start and again at its end.
        if last is not None and last[1] != ctx.review_id:
            gap = abs(ctx.start_time - last[0])
            if gap <= self.settings.dedupe_window_s:
                return GateResult(
                    passed_through=False,
                    verdict=Verdict.BENIGN,
                    confidence=0.70,
                    reason=f"duplicate of a {signature} alert {gap:.0f}s away "
                    f"(window {self.settings.dedupe_window_s:.0f}s)",
                    rule="dedupe_window",
                )

        # 5. A trusted false-alarm pattern from long-term memory, unless the
        # caller says memory must not decide this one (a person while armed),
        # or a human recently marked this kind of event real: memory can't
        # tell what people did in the frames, only that it looked alike.
        if use_memory and (self.settings.memory_mode != "free" or self.recently_real(signature)):
            use_memory = False
        match = None if not use_memory else self.patterns.best_suppression_match(
            signature,
            self._describe(ctx),
            min_similarity=self.settings.memory_suppress_similarity,
            min_corrections=self.settings.memory_min_corrections,
        )
        if match is not None and match.trust >= self.min_pattern_trust:
            self._recent[signature] = (ctx.start_time, ctx.review_id)
            return GateResult(
                passed_through=False,
                verdict=Verdict.BENIGN,
                confidence=min(0.95, match.trust * match.similarity),
                reason=(
                    f"matches known false-alarm pattern '{match.description}' "
                    f"(similarity {match.similarity:.2f}, {match.corrections} corrections, "
                    f"trust {match.trust:.2f})"
                ),
                rule="memory_pattern",
                matched_pattern=match,
                matched_pattern_ids=[match.pattern_id],
            )

        self._recent[signature] = (ctx.start_time, ctx.review_id)
        self._prune(ctx.start_time)
        return GateResult(passed_through=True, rule="no_rule_matched")

    def recently_real(self, signature: str) -> bool:
        """Did a human mark this kind of event real within ``real_veto_days``?
        (D-048 for the gate; D-052 for investigations.)"""
        if self.store is None or self.settings.real_veto_days <= 0:
            return False
        since = time.time() - self.settings.real_veto_days * 86400.0
        row = self.store.query_one(
            "SELECT 1 FROM corrections WHERE signature=? AND human_verdict='suspicious' "
            "AND created_at >= ? LIMIT 1",
            (signature, since),
        )
        return row is not None

    @staticmethod
    def _describe(ctx: AlertContext) -> str:
        parts = [ctx.camera]
        parts += ctx.labels + ctx.sub_labels + ctx.zones
        return " ".join(parts)

    def _prune(self, reference: float) -> None:
        """Keep the dedupe map from growing without bound.

        ``reference`` is the current alert's start time, so pruning follows
        the stream's own clock rather than the wall clock -- correct for
        replayed and backfilled alerts too.
        """
        if len(self._recent) < 512:
            return
        cutoff = reference - max(self.settings.dedupe_window_s * 4, 300.0)
        self._recent = {k: v for k, v in self._recent.items() if v[0] >= cutoff}
