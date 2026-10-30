"""The daily digest: what Watchwise held back, and why (M2-7).

The trust contract's second promise is that every held-back alert can be
audited. Held-back events are kept; once a day the digest says how many
there were and why, and sends a few of them back as cards the owner answers
with the usual buttons. Those answers are corrections like any other, and
``FeedbackLoop.spot_checks`` counts only these random samples, so what the
owner happened to look at can't bias the figure.

Everything here is counted by code from the store. No model writes the
digest: the numbers must be exact, and a model summarizing a day of alerts
miscounts them (TD-5).

  build_digest      the digest of one window (store in, Digest out)
  render_digest     its Telegram HTML
  DigestScheduler   sends it every day at ``notify.digest.at``, and on /digest
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from watchwise.config import DigestSettings
from watchwise.memory.store import SPOT_CHECK, Store
from watchwise.timeutil import safe_strftime

logger = logging.getLogger(__name__)

SCHEDULED, ASKED = "scheduled", "asked"
FIRST_WINDOW_S = 24 * 3600.0  # the first digest looks back a day
MAX_WINDOW_S = 7 * 24 * 3600.0  # after a long outage, at most a week
CATCH_UP_S = 12 * 3600.0  # a digest missed while stopped is sent at start within this
MAX_GROUP_LINES = 8
HELD = ("downgraded", "suppressed")
ALERTS = ("alerted", "escalated_human")

_GATE_RULE = re.compile(r"^gate/([\w-]+):")
_PATTERN = re.compile(r"matches known false-alarm pattern '(.+?)' \(")
_NAMES = re.compile(r"^recognized as (.+?) by ")


@dataclass
class HoldGroup:
    """Held-back events that share one reason."""

    key: str
    icon: str
    label: str
    # A pattern learned from the owner's corrections decided these: they are
    # spot-checked first (moment 2: "I held back 9 like this. Spot-check 2?").
    learned: bool = False
    review_ids: list[str] = field(default_factory=list)


@dataclass
class Digest:
    since: float
    until: float
    events: int = 0
    sent: int = 0  # events the owner got a card for
    muted: int = 0  # alerts without a card (a snoozed camera)
    undecided: int = 0
    held: list[HoldGroup] = field(default_factory=list)
    answers: dict[str, int] = field(default_factory=dict)  # human verdict -> count
    spot_checks: list[str] = field(default_factory=list)
    # All spot-checks answered so far (FeedbackLoop.spot_checks).
    record: dict[str, Any] | None = None

    @property
    def held_count(self) -> int:
        return sum(len(g.review_ids) for g in self.held)

    @property
    def empty(self) -> bool:
        return not self.events and not self.answers


def build_digest(
    store: Store,
    since: float,
    until: float,
    *,
    spot_checks: int = 2,
    record: dict[str, Any] | None = None,
    channel: str = "telegram",
) -> Digest:
    """Count the window's events and pick the spot-checks: at most one per
    reason held back, learned patterns first, then the largest groups; the
    same window always picks the same events."""
    digest = Digest(since=since, until=until, record=record)
    cards = store.cards_between(channel, since, until)
    groups: dict[str, HoldGroup] = {}
    for row in store.alerts_between(since, until):
        review_id = str(row["review_id"])
        digest.events += 1
        card = cards.get(review_id)
        if card is not None and card != SPOT_CHECK:
            digest.sent += 1
        elif row["disposition"] in ALERTS:
            digest.muted += 1
        elif row["disposition"] in HELD:
            key, icon, label, learned = _why_held(store, row)
            groups.setdefault(key, HoldGroup(key, icon, label, learned)).review_ids.append(review_id)
        else:
            digest.undecided += 1
    digest.held = sorted(groups.values(), key=lambda g: (-len(g.review_ids), g.key))
    for row in store.corrections_between(since, until):
        verdict = str(row["human_verdict"])
        digest.answers[verdict] = digest.answers.get(verdict, 0) + 1
    seen = store.answered_between(since, until) | {r for r, kind in cards.items() if kind == SPOT_CHECK}
    digest.spot_checks = _pick(digest.held, seen, spot_checks, seed=f"{since:.0f}-{until:.0f}")
    return digest


def _pick(groups: list[HoldGroup], seen: set[str], n: int, *, seed: str) -> list[str]:
    picked: list[str] = []
    for group in sorted(groups, key=lambda g: (not g.learned, -len(g.review_ids), g.key)):
        if len(picked) >= n:
            break
        candidates = sorted(r for r in group.review_ids if r not in seen)
        if candidates:
            picked.append(random.Random(f"{seed}|{group.key}").choice(candidates))
    return picked


def _why_held(store: Store, row: Any) -> tuple[str, str, str, bool]:
    """(group key, icon, reason, learned) of one held-back event."""
    stage, rationale = str(row["stage"]), str(row["rationale"] or "")
    if stage == "face":
        match = _NAMES.match(rationale)
        names = match.group(1) if match else "a resident"
        return f"face:{names}", "👤", f"recognized {names}", False
    if stage == "gated":
        rule_match = _GATE_RULE.match(rationale)
        rule = rule_match.group(1) if rule_match else "rule"
        if rule == "memory_pattern":
            key, label = _pattern(store, row, rationale)
            return key, "🧠", f"learned from {label}", True
        if rule == "never_screen_label":
            labels = ", ".join(json.loads(row["labels"] or "[]")) or "these labels"
            return f"rule:{rule}:{labels}", "📏", f"never screened ({labels})", False
        if rule == "dedupe_window":
            return f"rule:{rule}", "📏", "a repeat of the same kind of event just before", False
        if rule == "low_score":
            return f"rule:{rule}", "📏", "the detector wasn't sure it saw anything", False
        return f"rule:{rule}", "📏", f"rule {rule}", False
    if stage == "screened":
        if _memory_ids(store, row):
            key, label = _pattern(store, row, rationale)
            return f"screened:{key}", "👁️", f"the vision model, shown {label}, found it ordinary", True
        return "screened", "👁️", "the vision model's first look found it ordinary", False
    if stage == "investigated":
        return "investigated", "🔎", "an investigation found it ordinary", False
    return f"other:{stage}", "•", "held back", False


def _memory_ids(store: Store, row: Any) -> list[str]:
    ids = store.memory_ids(str(row["review_id"]))
    if ids or not row["screen_json"]:
        return ids
    try:
        return [str(p) for p in (json.loads(row["screen_json"]) or {}).get("memory_ids") or []]
    except (ValueError, AttributeError):
        return []


def _pattern(store: Store, row: Any, rationale: str) -> tuple[str, str]:
    """The learned pattern behind an event, as (group key, "your N
    corrections for <what>"). <what> is the owner's own words when a note
    gave them, else "<labels> on <camera>"; from the rationale for events
    decided before the pattern ids were stored (B-036)."""
    for pattern_id in _memory_ids(store, row):
        pattern = store.query_one("SELECT description, corrections, signature FROM patterns WHERE id=?", (pattern_id,))
        if pattern is not None:
            n = int(pattern["corrections"])
            what = _what(str(pattern["description"]), str(pattern["signature"]))
            return f"pattern:{pattern_id}", f"your {_n(n, 'correction')} for {what}"
    match = _PATTERN.search(rationale)
    text = _what(match.group(1), str(row["signature"])) if match else "a pattern"
    return f"pattern:{text}", f"your corrections for {text}"


def _what(description: str, signature: str) -> str:
    """A pattern in words: the owner's note, or for one Watchwise named
    itself ("recurring false alarm on webcam for [...]") its labels and camera."""
    if description.startswith(("recurring false alarm on ", "confirmed genuine on ")):
        camera, labels, _ = (signature.split("|") + ["", "", ""])[:3]
        return f"{labels.replace(',', ', ')} on {camera}"
    return f"“{_short(description)}”"


def _short(text: str, limit: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _n(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def render_digest(digest: Digest) -> str:
    """The digest as Telegram HTML (well under the 4,096-character limit)."""
    esc = lambda s: html.escape(str(s), quote=False)  # noqa: E731
    head = (
        f"🗒️ <b>Watchwise digest</b> · {esc(safe_strftime(digest.since, '%a %H:%M'))} → "
        f"{esc(safe_strftime(digest.until, '%a %H:%M'))}"
    )
    record = digest.record or {}
    so_far = (
        f"Spot-checks so far: {record['right_to_hold_back']} of {record['answered']} were right to hold back."
        if record.get("answered") else ""
    )
    if digest.empty:
        return "\n".join(filter(None, [head, "Nothing happened on your cameras, and nothing waited for your answer.",
                                       so_far]))
    counts = f"{_n(digest.events, 'event')}: {digest.sent} sent to you, {digest.held_count} held back"
    if digest.muted:
        counts += f", {digest.muted} alerted while muted"
    if digest.undecided:
        counts += f", {digest.undecided} without a verdict"
    lines = [head, counts]
    if digest.held:
        lines.append("<b>Held back, and why</b>")
        for group in digest.held[:MAX_GROUP_LINES]:
            lines.append(f"{group.icon} {esc(group.label)} · {len(group.review_ids)}")
        rest = digest.held[MAX_GROUP_LINES:]
        if rest:
            lines.append(f"• {len(rest)} other reasons · {sum(len(g.review_ids) for g in rest)}")
    if digest.answers:
        # Corrections, not cards: a tap on a room card makes one per camera.
        expected, real = digest.answers.get("benign", 0), digest.answers.get("suspicious", 0)
        lines.append(f"Your answers: {expected} ✅ Expected, {real} 🚨 Alert me.")
    if so_far:
        lines.append(so_far)
    if digest.spot_checks:
        lines.append(
            f"🔍 <b>Spot-check {len(digest.spot_checks)} of the {digest.held_count}</b> (below): "
            "✅ if holding it back was right, 🚨 if you wanted an alert."
        )
    return "\n".join(lines)


class DigestScheduler:
    """Sends the digest every day at ``settings.at`` (local time), and on
    demand. Each digest covers the time since the previous one, so none
    repeats or skips anything; one missed while Watchwise was stopped goes
    out at the next start if it is less than 12 hours late."""

    def __init__(
        self,
        store: Store,
        feedback: Any,
        notifier: Any,
        settings: DigestSettings,
        *,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.feedback = feedback  # FeedbackLoop: the spot-check record
        self.notifier = notifier  # anything with send_digest(digest, chat_ids, silent=)
        self.settings = settings
        self.clock = clock
        self._started = clock()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._started = self.clock()
        self._task = asyncio.create_task(self._run(), name="digest")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        while True:
            try:
                if self.due():
                    await self.send(SCHEDULED)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("the daily digest failed; trying again in a few minutes")
            await asyncio.sleep(self.pause())

    def slot(self, now: float) -> float:
        """The latest scheduled time at or before ``now``."""
        hour, minute = (int(part) for part in self.settings.at.split(":"))
        moment = datetime.fromtimestamp(now)
        at = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if at > moment:
            at -= timedelta(days=1)
        return at.timestamp()

    def due(self, now: float | None = None) -> bool:
        if not self.settings.enabled:
            return False
        now = self.clock() if now is None else now
        slot = self.slot(now)
        last = self.store.last_digest(SCHEDULED)
        if last is None:
            return slot >= self._started  # the first one: a slot reached while running
        return float(last["until"]) < slot and now - slot <= CATCH_UP_S

    def pause(self, now: float | None = None) -> float:
        """Seconds to the next scheduled time, at most 5 minutes, so a
        changed clock or a suspended machine is noticed."""
        now = self.clock() if now is None else now
        following = datetime.fromtimestamp(self.slot(now)) + timedelta(days=1)
        return min(max(following.timestamp() - now, 1.0), 300.0)

    async def send(self, kind: str = ASKED, chat_ids: list[Any] | None = None) -> Digest:
        """Build and send the digest of the time since the previous one. A
        scheduled digest with nothing to say isn't sent (the window still
        moves on); one asked for always answers."""
        now = self.clock()
        last = self.store.last_digest()
        since = max(float(last["until"]), now - MAX_WINDOW_S) if last is not None else now - FIRST_WINDOW_S
        record = self.feedback.spot_checks() if self.feedback is not None else None
        digest = build_digest(self.store, since, now, spot_checks=self.settings.spot_checks, record=record)
        messages: dict[str, str] = {}
        if kind == ASKED or not digest.empty:
            messages = await self.notifier.send_digest(digest, chat_ids, silent=self.settings.silent)
        self.store.add_digest(kind, since, now, messages, digest.spot_checks)
        logger.info(
            "digest (%s) %s → %s: %d events, %d sent, %d held back, spot-checks %s%s",
            kind, safe_strftime(since, "%a %H:%M"), safe_strftime(now, "%a %H:%M"), digest.events,
            digest.sent, digest.held_count, digest.spot_checks or "none", "" if messages else " (not sent: nothing new)",
        )
        return digest
