"""What the buttons on an alert card do.

Every action goes through the same core APIs a person would use over HTTP:
the feedback loop, the investigation queue and memory. The handler returns
what to show: a short toast, an optional footer and keyboard for the card,
and an optional reply message. It never sends anything itself, so the same
handler can serve any channel.
"""

from __future__ import annotations

import html
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from watchwise.config import GateSettings
from watchwise.memory.feedback import FeedbackLoop
from watchwise.memory.long_term import PatternLibrary
from watchwise.memory.store import Store
from watchwise.notify.presenter import (
    STAGE_LABELS,
    TEXT_LIMIT,
    context_from_row,
    decision_from_row,
    keyboard_for,
    plain_reason,
)
from watchwise.sinks import MemorySink
from watchwise.tier3.investigator import InvestigationQueue
from watchwise.timeutil import safe_strftime
from watchwise.types import Stage

INVESTIGATING_FOOTER = "🔎 Investigating at your request; this message will update."


@dataclass
class ActionOutcome:
    toast: str
    footer: str | None = None  # None: leave the card's footer as it is
    keyboard: str | None = None  # None: leave the card's keyboard as it is
    reply: str | None = None  # an extra message (Telegram HTML)


class ActionHandler:
    def __init__(
        self,
        *,
        store: Store,
        feedback: FeedbackLoop,
        patterns: PatternLibrary,
        investigations: InvestigationQueue | None,
        memory_sink: MemorySink,
        gate: GateSettings,
        snooze_minutes: int = 60,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.feedback = feedback
        self.patterns = patterns
        self.investigations = investigations
        self.memory_sink = memory_sink
        self.gate = gate
        self.snooze_minutes = snooze_minutes
        self.clock = clock

    async def handle(
        self, action: str, review_id: str, *, arg: str | None = None, reviewer: str = "telegram"
    ) -> ActionOutcome:
        row = self.store.get_alert(review_id)
        if row is None:
            return ActionOutcome("This alert is no longer known.")
        if action in ("ok", "real"):
            return self._feedback(row, "benign" if action == "ok" else "suspicious", reviewer)
        if action == "undo":
            return self._undo(row, arg)
        if action == "look":
            return self._look(row)
        if action == "snooze":
            return self._snooze(row)
        if action == "why":
            return ActionOutcome("Explanation below.", reply=self.explain(row))
        return ActionOutcome("Unknown action.")

    # ---------------- teaching ----------------

    def _feedback(self, row: Any, verdict: str, reviewer: str) -> ActionOutcome:
        latest = self.memory_sink.latest(row["review_id"])
        result = self.feedback.submit(
            row["review_id"],
            verdict,
            reviewer=reviewer,
            influenced_pattern_ids=list(latest.matched_memory_ids) if latest else [],
        )
        toast = "Thanks, noted as expected." if verdict == "benign" else "Thanks, noted as real."
        return ActionOutcome(
            toast,
            footer=self._learned(row, result.human_verdict, result.pattern_id),
            keyboard=f"after_feedback:{result.correction_id}",
        )

    def _learned(self, row: Any, verdict: str, pattern_id: str | None) -> str:
        labels = ", ".join(json.loads(row["labels"] or "[]")) or "activity"
        what = f"{labels} on {row['camera']}"
        if verdict != "benign":
            return f"🚨 Marked real. Watchwise will keep alerting on {what} like this."
        pattern = self.patterns.get(pattern_id) if pattern_id else None
        count = int(pattern["corrections"]) if pattern else 1
        needed = self.gate.memory_min_corrections
        if count >= needed:
            return (
                f"✅ Learned from {count} of your corrections: {what} like this is expected. "
                "Watchwise can now hold these back by itself; they are still recorded."
            )
        return (
            f"✅ Marked expected ({count} of {needed}). After {needed}, Watchwise can hold "
            f"back {what} like this by itself."
        )

    def _undo(self, row: Any, arg: str | None) -> ActionOutcome:
        try:
            self.feedback.undo(int(arg or ""))
        except (KeyError, ValueError):
            return ActionOutcome("Nothing to undo.")
        decision = decision_from_row(row)
        keyboard = keyboard_for(decision) if decision is not None else "full"
        return ActionOutcome(
            "Undone.", footer="↩️ Undone: that correction no longer counts.", keyboard=keyboard
        )

    # ---------------- other actions ----------------

    def _look(self, row: Any) -> ActionOutcome:
        if row["investigation_id"]:
            return ActionOutcome("Already investigated. Tap Why for the evidence.")
        if row["stage"] == Stage.SCREENED.value and row["disposition"] == "alerted":
            return ActionOutcome("Already being investigated; the card will update.")
        if self.investigations is None:
            return ActionOutcome("Investigations are not running.")
        if self.investigations.submit(context_from_row(row), None):
            return ActionOutcome("Investigating…", footer=INVESTIGATING_FOOTER)
        return ActionOutcome("The investigation queue is full; try again shortly.")

    def _snooze(self, row: Any) -> ActionOutcome:
        until = self.clock() + self.snooze_minutes * 60
        self.store.set_snooze(row["camera"], until)
        return ActionOutcome(
            f"{row['camera']} muted for {self.snooze_minutes} min.",
            footer=(
                f"🔇 {row['camera']} muted until {safe_strftime(until, '%H:%M')}. "
                "Its alerts are still processed and recorded."
            ),
        )

    def explain(self, row: Any) -> str:
        """The evidence behind the latest decision, as Telegram HTML."""
        esc = lambda s: html.escape(str(s), quote=False)  # noqa: E731
        lines = [
            f"<b>Why this alert</b> ({esc(row['camera'])}, "
            f"{esc(safe_strftime(row['start_time'], '%H:%M'))})"
        ]
        decision = decision_from_row(row)
        if decision is not None:
            lines.append(
                f"Decision: {decision.disposition.value}, confidence {decision.confidence:.2f}, "
                f"{esc(STAGE_LABELS.get(decision.stage, decision.stage.value))}"
            )
            lines.append(f"Reason: {esc(plain_reason(decision))}")
        latest = self.memory_sink.latest(row["review_id"])
        for pattern_id in latest.matched_memory_ids if latest else []:
            pattern = self.patterns.get(pattern_id)
            if pattern:
                lines.append(
                    f"Memory: {esc(pattern['description'])} "
                    f"({pattern['corrections']} human correction(s))"
                )
        if row["investigation_id"]:
            inv = self.store.get_investigation(row["investigation_id"])
            if inv is not None:
                trace = json.loads(inv["trace_json"] or "{}")
                tools = list(dict.fromkeys(c.get("name") for c in trace.get("tool_calls", [])))
                lines.append(
                    f"Investigation by {esc(inv['backend'])}/{esc(inv['model'])}: "
                    f"{len(trace.get('tool_calls', []))} tool call(s) "
                    f"({esc(', '.join(t for t in tools if t)) or 'none'})"
                )
                if inv["conclusion"]:
                    # Clip before escaping so the cut never lands inside an entity.
                    lines.append(f"Conclusion: {esc(str(inv['conclusion'])[:2500])}")
        return "\n".join(lines)[:TEXT_LIMIT]
