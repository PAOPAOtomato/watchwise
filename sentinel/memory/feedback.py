"""The human-correction loop: judge -> correct -> feed back into memory.

A reviewer marks an alert right or wrong. That single act has to do three
things, or the loop is not closed:

  1. record the correction (audit trail, and the source of truth for accuracy)
  2. reinforce or create a long-term pattern, so the same mistake is caught
     earlier next time
  3. score the patterns that *already* influenced this alert, so a pattern
     that starts causing misses loses its right to auto-suppress

``accuracy_report`` then measures what matters most: of the alerts we
auto-downgraded, how many did a human agree with.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sentinel.memory.long_term import PatternLibrary
from sentinel.memory.store import Store
from sentinel.types import Verdict


@dataclass(slots=True)
class CorrectionResult:
    review_id: str
    human_verdict: str
    system_verdict: str | None
    agreed: bool
    pattern_id: str | None
    patterns_scored: list[str]


class FeedbackLoop:
    def __init__(self, store: Store, patterns: PatternLibrary):
        self.store = store
        self.patterns = patterns

    def submit(
        self,
        review_id: str,
        human_verdict: str,
        *,
        note: str = "",
        reviewer: str = "unknown",
        influenced_pattern_ids: list[str] | None = None,
    ) -> CorrectionResult:
        """Record one human judgement and propagate it into memory."""
        verdict = Verdict(human_verdict)
        alert = self.store.get_alert(review_id)
        if alert is None:
            raise KeyError(f"unknown review_id {review_id!r}")

        signature = alert["signature"]
        system_verdict = self._system_verdict(alert)
        agreed = system_verdict == verdict.value if system_verdict else False

        self.store.execute(
            "INSERT INTO corrections(review_id, signature, system_verdict, system_conf, "
            "human_verdict, note, reviewer, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                review_id,
                signature,
                system_verdict,
                alert["confidence"],
                verdict.value,
                note,
                reviewer,
                time.time(),
            ),
        )

        # 2. Reinforce / create the pattern this correction teaches.
        pattern_id: str | None = None
        if verdict is Verdict.BENIGN:
            pattern_id = self.patterns.upsert(
                kind="false_alarm",
                signature=signature,
                camera=alert["camera"],
                description=note or self._default_description(alert, benign=True),
            )
        elif verdict is Verdict.SUSPICIOUS:
            pattern_id = self.patterns.upsert(
                kind="true_alarm",
                signature=signature,
                camera=alert["camera"],
                description=note or self._default_description(alert, benign=False),
            )

        # 3. Score whatever already fired on this alert.
        #
        # A pattern is judged on *its own* claim, not on whether the system's
        # final disposition happened to match. A false-alarm pattern was right
        # iff the human said benign. Conflating the two would punish a correct
        # pattern whenever some later stage overrode it, and the pattern would
        # lose its authority for someone else's mistake.
        scored: list[str] = []
        for pid in influenced_pattern_ids or []:
            row = self.patterns.get(pid)
            if row is None:
                continue
            expected = (
                Verdict.BENIGN if row["kind"] == "false_alarm" else Verdict.SUSPICIOUS
            )
            self.patterns.record_outcome(pid, agreed=verdict is expected)
            scored.append(pid)

        return CorrectionResult(
            review_id=review_id,
            human_verdict=verdict.value,
            system_verdict=system_verdict,
            agreed=agreed,
            pattern_id=pattern_id,
            patterns_scored=scored,
        )

    @staticmethod
    def _system_verdict(alert: Any) -> str | None:
        """Map the stored disposition back onto a benign/suspicious verdict."""
        disposition = alert["disposition"]
        if disposition in ("suppressed", "downgraded"):
            return Verdict.BENIGN.value
        if disposition == "alerted":
            return Verdict.SUSPICIOUS.value
        if disposition == "escalated_human":
            return Verdict.UNCERTAIN.value
        return None

    @staticmethod
    def _default_description(alert: Any, *, benign: bool) -> str:
        labels = alert["labels"]
        kind = "recurring false alarm" if benign else "confirmed genuine"
        return f"{kind} on {alert['camera']} for {labels}"

    # ---------------- measurement ----------------

    def accuracy_report(self, since: float = 0.0) -> dict[str, Any]:
        """Measure auto-downgrade accuracy: how often a human agreed with a demotion.

        Only alerts that (a) the system resolved without a human and (b) a
        human later reviewed anyway count. Everything else is unmeasurable,
        and reporting it as accuracy would be dishonest.
        """
        rows = self.store.query(
            "SELECT c.human_verdict, c.system_verdict, a.disposition "
            "FROM corrections c JOIN alerts a ON a.review_id = c.review_id "
            "WHERE c.created_at >= ?",
            (since,),
        )
        auto = [r for r in rows if r["disposition"] in ("suppressed", "downgraded")]
        correct = [r for r in auto if r["human_verdict"] == Verdict.BENIGN.value]

        total_corrections = len(rows)
        agreed = len([r for r in rows if r["human_verdict"] == r["system_verdict"]])

        return {
            "corrections_total": total_corrections,
            "system_human_agreement": round(agreed / total_corrections, 4)
            if total_corrections
            else None,
            "auto_downgraded_reviewed": len(auto),
            "auto_downgrade_accuracy": round(len(correct) / len(auto), 4) if auto else None,
            "note": (
                "auto_downgrade_accuracy is computed only over auto-resolved alerts "
                "that a human subsequently reviewed; it is null until such reviews exist."
            ),
        }

    def review_load(self, since: float = 0.0) -> dict[str, Any]:
        """How much made it to a human -- the manual-review-volume metric."""
        counts = self.store.funnel_counts(since)
        total = self.store.query_one(
            "SELECT COUNT(*) c FROM alerts WHERE start_time >= ?", (since,)
        )
        total_n = total["c"] if total else 0
        surfaced = counts.get("disposition:alerted", 0) + counts.get(
            "disposition:escalated_human", 0
        )
        return {
            "alerts_ingested": total_n,
            "surfaced_to_human": surfaced,
            "suppressed": counts.get("disposition:suppressed", 0),
            "downgraded": counts.get("disposition:downgraded", 0),
            "review_load_ratio": round(surfaced / total_n, 4) if total_n else None,
        }
