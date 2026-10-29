"""Turn a decision into an alert card a person can act on in seconds.

A card is deliberately small: what was seen, where and when, one line of why,
how sure the system is and which stage decided -- plus buttons that feed the
answer straight back into memory. Channel adapters (Telegram today) only
transport it.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from watchwise.timeutil import safe_strftime
from watchwise.types import (
    AlertContext,
    Decision,
    Disposition,
    ScreenResult,
    Severity,
    Stage,
    Verdict,
)

CAPTION_LIMIT = 1024  # Telegram photo captions
TEXT_LIMIT = 4096  # Telegram text messages

ICONS = {
    Disposition.ALERTED: "🚨",
    Disposition.ESCALATED_HUMAN: "⚠️",
    Disposition.DOWNGRADED: "ℹ️",
    Disposition.SUPPRESSED: "🔕",
}
STAGE_LABELS = {
    Stage.INGESTED: "received",
    Stage.GATED: "decided by a rule or a learned pattern, no model call",
    Stage.SCREENED: "first look by the vision model",
    Stage.INVESTIGATED: "after an investigation",
    Stage.SUMMARIZED: "raised by the review of the whole event",
    Stage.FACE: "decided by Frigate's face recognition",
}
INVESTIGATING = "🔎 Investigating with the forensic tools; this message will update."
ANALYZING = "⏳ Analyzing the frames; this message will update with the verdict."


@dataclass(frozen=True)
class Button:
    label: str
    action: str


OK, REAL = Button("✅ Expected", "ok"), Button("🚨 Alert me", "real")
LOOK, SNOOZE, WHY = Button("🔍 Look closer", "look"), Button("🔇 Snooze", "snooze"), Button("❓ Why", "why")
UNDO = Button("↩️ Undo", "undo")
CLIP_LABEL = "▶️ Clip"
KEYBOARDS: dict[str, list[list[Button]]] = {
    "full": [[OK, REAL], [LOOK, SNOOZE, WHY]],  # nothing investigated yet
    "standard": [[OK, REAL], [SNOOZE, WHY]],  # investigated, or being investigated
    "after_feedback": [[UNDO, WHY]],
}


@dataclass
class AlertCard:
    review_id: str
    camera: str
    title: str
    lines: list[str]
    keyboard: str = "full"
    footer: str = ""
    unknown_face: bool = False  # offers "👤 That was <resident>" (D-071)

    def html(self, limit: int = CAPTION_LIMIT) -> str:
        """Telegram HTML within ``limit``. Text is trimmed *before* escaping,
        so a cut can never land inside an entity or a tag."""
        title = _clip(self.title, 120)
        lines = [_clip(line, 400) for line in self.lines]
        footer = _clip(self.footer, 300)
        for _ in range(100):
            text = _render(title, lines, footer)
            excess = len(text) - limit
            if excess <= 0:
                return text
            if lines:
                i = max(range(len(lines)), key=lambda k: len(lines[k]))
                if len(lines[i]) > 24:
                    lines[i] = _clip(lines[i], len(lines[i]) - max(8, excess))
                else:
                    lines.pop(i)
            elif len(footer) > 24:
                footer = _clip(footer, len(footer) - max(8, excess))
            elif len(title) > 24:
                title = _clip(title, len(title) - max(8, excess))
            else:
                break
        return _render(_clip(title, 24), [], "")


def build_card(ctx: AlertContext, decision: Decision) -> AlertCard:
    icon = ICONS.get(decision.disposition, "•")
    subject = ", ".join(label.capitalize() for label in ctx.labels) or "Motion"
    where = ctx.camera + (f" · {', '.join(ctx.zones)}" if ctx.zones else "")
    when = safe_strftime(ctx.start_time, "%H:%M")
    if _analyzing(decision):
        lines = [ANALYZING]
    else:
        lines = [f"Why: {plain_reason(decision)}", _confidence_line(decision)]
    if decision.summary:
        lines.insert(1, f"What happened: {decision.summary}")
    if being_investigated(decision):
        lines.append(INVESTIGATING)
    return AlertCard(
        review_id=ctx.review_id,
        camera=ctx.camera,
        title=f"{icon} {subject} · {where} · {when}",
        lines=lines,
        keyboard=keyboard_for(decision),
        unknown_face=unknown_face(decision),
    )


def unknown_face(decision: Decision) -> bool:
    """Raised because Frigate saw a face it matched to nobody (D-057)."""
    return decision.stage is Stage.FACE and decision.disposition is Disposition.ALERTED


# How much a verdict asks of the owner; a room card shows the most (D-062).
SERIOUSNESS = {
    Disposition.ALERTED: 2,
    Disposition.ESCALATED_HUMAN: 2,
    Disposition.DOWNGRADED: 1,
    Disposition.SUPPRESSED: 0,
}


def build_room_card(
    room: str, lead: tuple[AlertContext, Decision], others: list[tuple[AlertContext, Decision]]
) -> AlertCard:
    """One card for an incident that several cameras in one room saw. The
    most serious verdict leads (the card's own on a tie), so the card never
    shows less than any camera found; each other camera gets a line. The
    buttons stay the card's own event's, and act on all of them."""
    entries = [lead, *others]
    top_ctx, top = max(entries, key=lambda e: SERIOUSNESS.get(e[1].disposition, 0))
    card = build_card(top_ctx, top)
    card.review_id, card.camera = lead[0].review_id, lead[0].camera
    card.keyboard = keyboard_for(lead[1])  # Look closer investigates the card's own event
    card.unknown_face = any(unknown_face(d) for _, d in entries)
    subject = ", ".join(dict.fromkeys(label.capitalize() for c, _ in entries for label in c.labels)) or "Motion"
    cameras = " + ".join(dict.fromkeys(c.camera for c, _ in entries))
    when = safe_strftime(min(c.start_time for c, _ in entries), "%H:%M")
    card.title = f"{ICONS.get(top.disposition, '•')} {subject} · {room} ({cameras}) · {when}"
    for ctx, decision in entries:
        if ctx.review_id == top_ctx.review_id:
            continue
        what = "analyzing…" if _analyzing(decision) else plain_reason(decision)
        card.lines.append(f"Also {ctx.camera}: {ICONS.get(decision.disposition, '•')} {what}")
    return card


def being_investigated(decision: Decision) -> bool:
    # The policy queues an investigation for every alert raised at screening,
    # except a match to what the owner marked real (D-056).
    return (
        decision.stage is Stage.SCREENED
        and decision.disposition is Disposition.ALERTED
        and not (decision.screen is not None and decision.screen.owner_real)
    )


def _confidence_line(decision: Decision) -> str:
    """How sure, and of what. The vision model's confidence belongs to its
    own verdict, which can differ from the card's (a person while armed away
    is an alert even when the model thought it looked ordinary)."""
    stage = STAGE_LABELS.get(decision.stage, decision.stage.value)
    screen = decision.screen
    if decision.stage is Stage.SCREENED and screen is not None:
        return f"Vision model: {screen.verdict.value} @ {screen.confidence:.2f} · {stage}"
    if decision.stage is Stage.FACE:
        # Frigate's own match score, when it reported one.
        score = f" (match score {decision.confidence:.2f})" if decision.confidence else ""
        return f"{stage[0].upper()}{stage[1:]}{score}"
    return f"Confidence {decision.confidence:.2f} · {stage}"


def _analyzing(decision: Decision) -> bool:
    # The pipeline's placeholder while a slow screen runs.
    return decision.stage is Stage.INGESTED and decision.disposition is Disposition.ALERTED


def keyboard_for(decision: Decision) -> str:
    """'Look closer' is offered only when nothing is investigated or queued
    (nor about to be: an alert raised at screening is always investigated)."""
    if decision.stage is Stage.INVESTIGATED or being_investigated(decision) or _analyzing(decision):
        return "standard"
    return "full"


def _render(title: str, lines: list[str], footer: str) -> str:
    parts = [f"<b>{html.escape(title, quote=False)}</b>"]
    parts += [html.escape(line, quote=False) for line in lines]
    if footer:
        parts += ["", html.escape(footer, quote=False)]
    return "\n".join(parts)


_PREFIX = re.compile(r"^(gate/[\w-]+|tier-2 screening|auto-downgraded):\s*", re.IGNORECASE)


def plain_reason(decision: Decision) -> str:
    """The reason a person needs, without internal stage prefixes."""
    if decision.stage is Stage.SCREENED and decision.screen and decision.screen.reason:
        text = decision.screen.reason
    else:
        text = decision.rationale or ""
    return _PREFIX.sub("", text).strip() or "no reason recorded"


def clip_url(web_url: str | None, review_id: str) -> str | None:
    """Where Frigate's web page plays this alert's recording."""
    if not web_url:
        return None
    return f"{web_url.rstrip('/')}/review?id={quote(review_id, safe='')}"


def keyboard_markup(
    review_id: str, keyboard: str, *, clip: str | None = None, who: list[str] | tuple[str, ...] = ()
) -> dict[str, Any]:
    """Telegram inline keyboard. ``keyboard`` may carry an argument for Undo,
    as ``after_feedback:<correction_id>``. ``clip`` adds a link to the
    recording on its own row; ``who``, residents to offer for a face nobody
    matched ("👤 That was <name>", D-071)."""
    name, _, arg = keyboard.partition(":")
    rows = []
    for row in KEYBOARDS.get(name, KEYBOARDS["full"]):
        buttons = []
        for button in row:
            data = f"fb:{button.action}:{arg}:{review_id}" if button.action == "undo" else (
                f"fb:{button.action}:{review_id}"
            )
            if button.action == "undo" and not arg:
                continue
            if len(data.encode("utf-8")) > 64:  # Telegram's callback_data limit
                continue
            buttons.append({"text": button.label, "callback_data": data})
        if buttons:
            rows.append(buttons)
    faces = []
    for name in list(who)[:3]:
        data = f"fc:who:{name}:{review_id}"
        if len(data.encode("utf-8")) <= 64:
            faces.append({"text": f"👤 That was {name}", "callback_data": data})
    if faces:
        rows.append(faces)
    if clip:
        rows.append([{"text": CLIP_LABEL, "url": clip}])
    return {"inline_keyboard": rows}


def parse_callback(data: str) -> tuple[str, str, str | None] | None:
    """``fb:<action>:<review_id>`` or ``fb:undo:<arg>:<review_id>``."""
    if not data.startswith("fb:"):
        return None
    _, action, rest = (data.split(":", 2) + ["", ""])[:3]
    if not action or not rest:
        return None
    if action == "undo":
        arg, _, review_id = rest.partition(":")
        return (action, review_id, arg) if review_id and arg else None
    return action, rest, None


# ---------------------------------------------------------------- store rows


def context_from_row(row: Any) -> AlertContext:
    return AlertContext(
        review_id=row["review_id"],
        camera=row["camera"],
        start_time=float(row["start_time"]),
        end_time=float(row["end_time"]) if row["end_time"] else None,
        severity=Severity.ALERT,
        labels=list(json.loads(row["labels"] or "[]")),
        zones=list(json.loads(row["zones"] or "[]")),
    )


def decision_from_row(row: Any) -> Decision | None:
    if not row["disposition"]:
        return None
    screen = None
    if row["screen_json"]:
        data = json.loads(row["screen_json"]) or None
        if data:
            screen = ScreenResult(
                verdict=Verdict(data.get("verdict", "uncertain")),
                confidence=float(data.get("confidence", 0.0)),
                reason=str(data.get("reason", "")),
                tags=[str(t) for t in data.get("tags") or []],
                backend=str(data.get("backend", "")),
                model=str(data.get("model", "")),
            )
    return Decision(
        review_id=row["review_id"],
        disposition=Disposition(row["disposition"]),
        severity=Severity(row["severity"]) if row["severity"] else Severity.ALERT,
        confidence=float(row["confidence"] or 0.0),
        rationale=row["rationale"] or "",
        stage=Stage(row["stage"]),
        screen=screen,
        investigation_id=row["investigation_id"],
        summary=row["summary"] if "summary" in row.keys() else None,
    )


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: max(1, limit - 1)].rstrip() + "…"
