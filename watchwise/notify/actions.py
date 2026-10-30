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

from watchwise.config import CameraSettings, GateSettings, room_key
from watchwise.memory.feedback import FeedbackLoop
from watchwise.memory.long_term import PatternLibrary
from watchwise.memory.store import Store
from watchwise.notify.presenter import (
    SPOT_CHECK_KEYBOARD,
    STAGE_LABELS,
    TEXT_LIMIT,
    being_investigated,
    context_from_row,
    decision_from_row,
    keyboard_for,
    plain_reason,
)
from watchwise.sinks import MemorySink
from watchwise.tier3.investigator import InvestigationQueue
from watchwise.timeutil import safe_strftime

INVESTIGATING_FOOTER = "🔎 Investigating at your request; this message will update."
ASK_WHY = (
    "🚨 Noted: you want alerts like this. <b>What was wrong?</b> Reply in a few words, e.g. "
    "“waving both arms at the camera”, and I'll look for it next time. (Optional.)"
)
NOTE_LIMIT = 200


@dataclass
class ActionOutcome:
    toast: str
    footer: str | None = None  # None: leave the card's footer as it is
    keyboard: str | None = None  # None: leave the card's keyboard as it is
    reply: str | None = None  # an extra message (Telegram HTML)
    # Corrections the owner may describe in their own words (one per camera of a room card).
    ask_note: list[int] | None = None


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
        cameras: dict[str, CameraSettings] | None = None,
    ):
        self.store = store
        self.feedback = feedback
        self.patterns = patterns
        self.investigations = investigations
        self.memory_sink = memory_sink
        self.gate = gate
        self.snooze_minutes = snooze_minutes
        self.clock = clock
        # Which room each camera watches (D-062); Snooze mutes the whole room.
        self.cameras = cameras if cameras is not None else {}

    async def handle(
        self,
        action: str,
        review_id: str,
        *,
        arg: str | None = None,
        reviewer: str = "telegram",
        group: list[str] | None = None,
        spot_check: bool = False,
    ) -> ActionOutcome:
        """A button on the card of ``review_id``. ``group``: every event the
        card shows (a room card, D-062), the card's own first. ``spot_check``:
        the card is a held-back event the digest sent back (M2-7)."""
        row = self.store.get_alert(review_id)
        if row is None:
            return ActionOutcome("This alert is no longer known.")
        others = [r for r in (self.store.get_alert(g) for g in (group or [])[1:]) if r is not None]
        if action in ("ok", "real"):
            outcome = self._feedback(row, "benign" if action == "ok" else "suspicious", reviewer, others)
            if spot_check:
                right = action == "ok"
                outcome.toast = "Thanks: holding it back was right." if right else "Noted: that should have alerted."
                lead = "🔍 Spot-check: right to hold back." if right else "🔍 Spot-check: this should have alerted."
                outcome.footer = f"{lead} {outcome.footer}"
            return outcome
        if action == "undo":
            outcome = self._undo(row, arg)
            if spot_check and outcome.keyboard is not None:
                outcome.keyboard = SPOT_CHECK_KEYBOARD
            return outcome
        if action == "look":
            return self._look(row)
        if action == "snooze":
            return self._snooze(row, others)
        if action == "why":
            text = "\n\n".join(self.explain(r) for r in [row, *others])
            return ActionOutcome("Explanation below.", reply=text[:TEXT_LIMIT])
        return ActionOutcome("Unknown action.")

    # ---------------- teaching ----------------

    def _feedback(self, row: Any, verdict: str, reviewer: str, others: list[Any] | None = None) -> ActionOutcome:
        """One correction per event on the card, so every camera in the room
        learns from the tap (D-062)."""
        results = []
        others = others or []
        for alert in [row, *others]:
            results.append(self.feedback.submit(
                alert["review_id"],
                verdict,
                reviewer=reviewer,
                influenced_pattern_ids=self.influenced(alert["review_id"]),
            ))
        result, ids = results[0], [r.correction_id for r in results]
        toast = "Thanks, noted as expected." if verdict == "benign" else "Noted: you want alerts like this."
        footer = self._learned(row, result.human_verdict, result.pattern_id)
        if others:
            footer += f" Also recorded for {', '.join(dict.fromkeys(o['camera'] for o in others))}."
        return ActionOutcome(
            toast,
            footer=footer,
            keyboard=f"after_feedback:{','.join(map(str, ids))}",
            ask_note=ids if verdict == "suspicious" else None,
        )

    def influenced(self, review_id: str) -> list[str]:
        """The learned patterns behind the event's latest decision, which a
        correction scores. Stored with the decision, so a tap made after a
        restart (a spot-check the next morning) still scores them (B-036)."""
        stored = self.store.memory_ids(review_id)
        if stored:
            return stored
        latest = self.memory_sink.latest(review_id)  # decided before the column existed
        return list(latest.matched_memory_ids) if latest else []

    def remember_note_request(self, chat_id: int | str, message_id: int, correction_ids: list[int]) -> None:
        self.store.add_note_request(chat_id, message_id, correction_ids)

    def note_from_reply(self, chat_id: int | str, replied_to: int, text: str) -> str | None:
        """A reply to a "What was wrong?" question: keep it as the note of
        the corrections it asked about (D-053). None when the reply was to
        something else."""
        correction_ids = self.store.note_request(chat_id, replied_to)
        if not correction_ids:
            return None
        note = " ".join(text.split())[:NOTE_LIMIT]
        if not note:
            return "That reply was empty, so nothing changed."
        kept = [self.store.set_correction_note(c, note) for c in correction_ids]
        if not any(kept):
            return "That answer was undone, so there's nothing to attach this to."
        return f"✍️ Thanks. Next time I'll look for: “{html.escape(note, quote=False)}”."

    def _learned(self, row: Any, verdict: str, pattern_id: str | None) -> str:
        """What a tap teaches, as it really works with this configuration
        (B-037): a learned pattern holds events back by itself only in the
        free memory mode, and never a person while armed away or at night."""
        names = json.loads(row["labels"] or "[]")
        labels = ", ".join(names) or "activity"
        what = f"{labels} on {row['camera']}"
        if verdict != "benign":
            return f"🚨 You asked for alerts like this. Watchwise will keep alerting on {what} like this."
        pattern = self.patterns.get(pattern_id) if pattern_id else None
        count = int(pattern["corrections"]) if pattern else 1
        needed = self.gate.memory_min_corrections
        armed = " (a person still alerts while armed away or at night)" if "person" in names else ""
        if self.gate.memory_mode != "free":
            return (
                f"✅ Marked expected ({count} so far). The vision model is now shown your judgements of "
                f"{what} as examples; events it finds match them are held back{armed}, and still recorded."
            )
        if count >= needed:
            return (
                f"✅ Learned from {count} of your corrections: {what} like this is expected. "
                f"Watchwise can now hold these back by itself{armed}; they are still recorded."
            )
        return (
            f"✅ Marked expected ({count} of {needed}). After {needed}, Watchwise can hold "
            f"back {what} like this by itself{armed}."
        )

    def _undo(self, row: Any, arg: str | None) -> ActionOutcome:
        """``arg``: the corrections of one tap, comma-separated (a room card
        made one per camera)."""
        undone = 0
        for part in (arg or "").split(","):
            try:
                self.feedback.undo(int(part))
                undone += 1
            except (KeyError, ValueError):
                continue
        if not undone:
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
        decision = decision_from_row(row)
        if decision is not None and being_investigated(decision):
            return ActionOutcome("Already being investigated; the card will update.")
        if self.investigations is None:
            return ActionOutcome("Investigations are not running.")
        if self.investigations.submit(context_from_row(row), None):
            return ActionOutcome("Investigating…", footer=INVESTIGATING_FOOTER)
        return ActionOutcome("The investigation queue is full; try again shortly.")

    def _snooze(self, row: Any, others: list[Any] | None = None) -> ActionOutcome:
        """Mute the card's camera, and every other camera in its room."""
        until = self.clock() + self.snooze_minutes * 60
        key = room_key(self.cameras, row["camera"])
        cameras = [row["camera"], *(o["camera"] for o in others or [])]
        cameras += [name for name in self.cameras if room_key(self.cameras, name) == key]
        cameras = list(dict.fromkeys(cameras))
        for camera in cameras:
            self.store.set_snooze(camera, until)
        what = ", ".join(cameras)
        its = "Their" if len(cameras) > 1 else "Its"
        return ActionOutcome(
            f"{what} muted for {self.snooze_minutes} min.",
            footer=(
                f"🔇 {what} muted until {safe_strftime(until, '%H:%M')}. "
                f"{its} alerts are still processed and recorded."
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
        for pattern_id in self.influenced(row["review_id"]):
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
