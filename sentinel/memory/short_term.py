"""Short-term memory: the working context of a single investigation.

Scope is one alert. It holds what the agent has established so far -- the
original alert, the tier-2 screening result, every tool observation -- and
renders it back as a compact briefing for the next ReAct iteration.

Two jobs beyond "a list of strings":

1. **Budgeting.** Tool observations (especially cross-camera dumps) grow fast.
   The transcript is capped by item count *and* character count, oldest
   non-pinned entries evicted first, so a long investigation cannot blow the
   context window.
2. **Evidence tracking.** It keeps a running confidence and the set of open
   questions, which is what lets the loop decide "I have enough" versus
   "gather another round" rather than always running to the iteration cap.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from sentinel.types import AlertContext, ScreenResult


@dataclass(slots=True)
class MemoryItem:
    role: str  # alert | screening | thought | observation | note
    content: str
    iteration: int = 0
    pinned: bool = False
    created_at: float = field(default_factory=time.time)

    def render(self) -> str:
        prefix = f"[{self.role}"
        if self.iteration:
            prefix += f" #{self.iteration}"
        return f"{prefix}] {self.content}"


class ShortTermMemory:
    """Per-investigation scratchpad with a hard context budget."""

    def __init__(self, *, max_items: int = 40, max_chars: int = 12000):
        self.max_items = max_items
        self.max_chars = max_chars
        self.items: list[MemoryItem] = []
        self.open_questions: list[str] = []
        self.evidence_confidence: float = 0.0
        self.facts: dict[str, Any] = {}

    # ---------------- seeding ----------------

    def seed(self, ctx: AlertContext, screen: ScreenResult | None = None) -> None:
        """Pin the immutable facts of the case so they are never evicted."""
        self.add(
            "alert",
            json.dumps(
                {
                    "review_id": ctx.review_id,
                    "camera": ctx.camera,
                    "start_time": ctx.start_time,
                    "end_time": ctx.end_time,
                    "duration_s": round(ctx.duration, 1),
                    "labels": ctx.labels,
                    "sub_labels": ctx.sub_labels,
                    "zones": ctx.zones,
                    "event_ids": ctx.event_ids,
                },
                sort_keys=True,
            ),
            pinned=True,
        )
        if screen is not None:
            self.add(
                "screening",
                f"tier-2 {screen.backend}/{screen.model} said {screen.verdict.value} "
                f"(confidence {screen.confidence:.2f}): {screen.reason}",
                pinned=True,
            )
            self.evidence_confidence = min(self.evidence_confidence, screen.confidence)

    # ---------------- mutation ----------------

    def add(
        self, role: str, content: str, *, iteration: int = 0, pinned: bool = False
    ) -> MemoryItem:
        item = MemoryItem(role=role, content=content, iteration=iteration, pinned=pinned)
        self.items.append(item)
        self._evict()
        return item

    def add_thought(self, text: str, iteration: int) -> None:
        self.add("thought", text, iteration=iteration)

    def add_observation(self, tool: str, result: Any, iteration: int) -> None:
        rendered = result if isinstance(result, str) else json.dumps(result, default=str)
        self.add("observation", f"{tool} -> {rendered}", iteration=iteration)

    def note_fact(self, key: str, value: Any) -> None:
        """Structured findings the conclusion can cite verbatim."""
        self.facts[key] = value

    def set_confidence(self, value: float) -> None:
        self.evidence_confidence = max(0.0, min(1.0, value))

    def add_question(self, question: str) -> None:
        if question not in self.open_questions:
            self.open_questions.append(question)

    def resolve_question(self, question: str) -> None:
        if question in self.open_questions:
            self.open_questions.remove(question)

    def _evict(self) -> None:
        """Drop oldest unpinned items until both budgets are satisfied."""
        while len(self.items) > self.max_items:
            idx = next((i for i, it in enumerate(self.items) if not it.pinned), None)
            if idx is None:
                break
            self.items.pop(idx)

        while self._char_count() > self.max_chars:
            idx = next((i for i, it in enumerate(self.items) if not it.pinned), None)
            if idx is None:
                break
            self.items.pop(idx)

    def _char_count(self) -> int:
        return sum(len(it.render()) for it in self.items)

    # ---------------- rendering ----------------

    def transcript(self) -> str:
        return "\n".join(it.render() for it in self.items)

    def briefing(self) -> str:
        """What gets handed to the model at the top of each iteration."""
        parts = [self.transcript()]
        if self.facts:
            parts.append("[established] " + json.dumps(self.facts, sort_keys=True, default=str))
        if self.open_questions:
            parts.append("[open questions] " + "; ".join(self.open_questions))
        parts.append(f"[evidence confidence] {self.evidence_confidence:.2f}")
        return "\n".join(parts)

    def tools_used(self) -> list[str]:
        seen: list[str] = []
        for it in self.items:
            if it.role != "observation":
                continue
            name = it.content.split(" -> ", 1)[0]
            if name not in seen:
                seen.append(name)
        return seen

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [
                {
                    "role": it.role,
                    "content": it.content,
                    "iteration": it.iteration,
                    "pinned": it.pinned,
                }
                for it in self.items
            ],
            "facts": self.facts,
            "open_questions": list(self.open_questions),
            "evidence_confidence": self.evidence_confidence,
        }
