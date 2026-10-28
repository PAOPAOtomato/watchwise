"""What the chat model may use (D-065): four lookups and four proposals.

Kept to eight tools with short, summary-first results, because the chat
model may be a small local one. The lookups read Watchwise's own records
(every event it judged, with its verdict and the "What happened" summary)
and Frigate's event list. The ``propose_*`` tools change nothing: they store
a proposal that the owner confirms or cancels with a tap.

Text that comes from events (labels, names, summaries) is data: the system
prompt tells the model never to follow instructions found in tool results,
and nothing a tool returns can change anything without the owner's tap.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from watchwise.chat.proposals import Proposal, ProposalBook
from watchwise.config import WatchwiseConfig, room_key
from watchwise.memory.store import Store
from watchwise.memory.watches import WatchBook
from watchwise.notify.presenter import STAGE_LABELS, decision_from_row, plain_reason
from watchwise.nvr.client import NVRClient
from watchwise.setup.home import ARMING_LABELS
from watchwise.timeutil import local_zone_name, safe_strftime
from watchwise.types import Disposition

PAGE = 10
OUTCOMES = {
    Disposition.ALERTED: "alerted the owner",
    Disposition.ESCALATED_HUMAN: "asked the owner to decide",
    Disposition.DOWNGRADED: "routine, no card",
    Disposition.SUPPRESSED: "held back",
}
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
HHMM = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


@dataclass
class ChatContext:
    """Everything a tool may use for one question."""

    store: Store
    nvr: NVRClient
    config: WatchwiseConfig
    proposals: ProposalBook
    watches: WatchBook | None
    conversation: str
    cameras: list[str]  # the NVR's cameras now
    clock: Callable[[], float] = time.time
    images: list[str] = field(default_factory=list)  # review ids to send as photos
    proposed: list[Proposal] = field(default_factory=list)

    def room_of(self, camera: str) -> str:
        spec = self.config.cameras.get(camera)
        return " ".join(spec.room.split()) if spec is not None and spec.room.strip() else ""

    def resolve(self, args: dict[str, Any]) -> list[str] | str:
        """The cameras ``camera`` or ``room`` names (empty: all), or why not."""
        camera = str(args.get("camera") or "").strip()
        room = str(args.get("room") or "").strip().lower()
        if camera:
            return [camera] if camera in self.cameras else f"no camera called {camera!r}; cameras: {self.cameras}"
        if room:
            found = [c for c in self.cameras if room in (room_key(self.config.cameras, c), c.lower())]
            known = sorted({self.room_of(c) for c in self.cameras if self.room_of(c)})
            return found or f"no room called {room!r}; rooms: {known}, cameras: {self.cameras}"
        return []


@dataclass(frozen=True)
class ChatTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[ChatContext, dict[str, Any]], Awaitable[dict[str, Any]]]


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or [], "additionalProperties": False}


WHERE = {
    "camera": {"type": "string", "description": "One camera's name."},
    "room": {"type": "string", "description": "A room's name (all its cameras)."},
}


def _when(ts: float) -> str:
    return safe_strftime(ts, "%a %H:%M") or "?"


def _event(ctx: ChatContext, row: Any) -> dict[str, Any]:
    decision = decision_from_row(row)
    item = {
        "event_id": row["review_id"],
        "when": _when(float(row["start_time"])),
        "camera": row["camera"],
        "what": ", ".join(json.loads(row["labels"] or "[]")) or "motion",
        "outcome": OUTCOMES.get(decision.disposition, decision.disposition.value) if decision else "not decided yet",
    }
    room = ctx.room_of(row["camera"])
    if room:
        item["room"] = room
    happened = (row["summary"] if "summary" in row.keys() else None) or (plain_reason(decision) if decision else "")
    if happened:
        item["what_happened"] = happened[:180]
    return item


# ---------------------------------------------------------------- lookups


async def _recent_activity(ctx: ChatContext, args: dict[str, Any]) -> dict[str, Any]:
    cameras = ctx.resolve(args)
    if isinstance(cameras, str):
        return {"ok": False, "error": cameras}
    hours = min(max(float(args.get("hours") or 24), 0.1), 24 * 14)
    rows = ctx.store.recent_alerts(ctx.clock() - hours * 3600, limit=500)
    rows = [r for r in rows if not cameras or r["camera"] in cameras]
    if args.get("only_alerts"):
        rows = [r for r in rows if r["disposition"] in ("alerted", "escalated_human")]
    counts: dict[str, int] = {}
    for row in rows:
        decision = decision_from_row(row)
        key = OUTCOMES.get(decision.disposition, decision.disposition.value) if decision else "not decided yet"
        counts[key] = counts.get(key, 0) + 1
    page = max(1, int(args.get("page") or 1))
    shown = rows[(page - 1) * PAGE : page * PAGE]
    return {
        "ok": True,
        "window": f"the last {hours:g} h",
        "total": len(rows),
        "counts": counts,
        "events": [_event(ctx, r) for r in shown],
        "page": page,
        "more_pages": len(rows) > page * PAGE,
    }


async def _explain_decision(ctx: ChatContext, args: dict[str, Any]) -> dict[str, Any]:
    row = _find_event(ctx, args)
    if isinstance(row, str):
        return {"ok": False, "error": row}
    decision = decision_from_row(row)
    out = _event(ctx, row)
    if decision is not None:
        out["decided_by"] = STAGE_LABELS.get(decision.stage, decision.stage.value)
        out["reason"] = plain_reason(decision)[:400]
        out["confidence"] = round(decision.confidence, 2)
    out["card_sent"] = bool(ctx.store.query_one("SELECT 1 FROM notifications WHERE review_id=?", (row["review_id"],)))
    if row["investigation_id"]:
        inv = ctx.store.get_investigation(row["investigation_id"])
        if inv is not None and inv["conclusion"]:
            out["investigation"] = str(inv["conclusion"])[:500]
    rulings = ctx.store.query(
        "SELECT human_verdict, note FROM corrections WHERE review_id=? ORDER BY id", (row["review_id"],)
    )
    if rulings:
        out["owner_said"] = [
            ("expected" if r["human_verdict"] == "benign" else "alert me") + (f": {r['note']}" if r["note"] else "")
            for r in rulings
        ]
    return {"ok": True, "event": out}


def _find_event(ctx: ChatContext, args: dict[str, Any]) -> Any:
    event_id = str(args.get("event_id") or "").strip()
    if event_id:
        row = ctx.store.get_alert(event_id)
        return row if row is not None else f"no event {event_id!r} in Watchwise's records"
    at = str(args.get("time") or "").strip()
    match = HHMM.match(at)
    if not match:
        return "give an event_id, or a time as HH:MM"
    cameras = ctx.resolve(args)
    if isinstance(cameras, str):
        return cameras
    now = ctx.clock()
    rows = [r for r in ctx.store.recent_alerts(now - 26 * 3600, limit=1000) if not cameras or r["camera"] in cameras]
    wanted = int(match.group(1)) * 60 + int(match.group(2))

    def distance(row: Any) -> int:
        local = time.localtime(float(row["start_time"]))
        return abs(local.tm_hour * 60 + local.tm_min - wanted)

    best = min(rows, key=distance, default=None)
    if best is None or distance(best) > 10:
        return f"no event within 10 minutes of {at} in the last day"
    return best


async def _search_footage(ctx: ChatContext, args: dict[str, Any]) -> dict[str, Any]:
    cameras = ctx.resolve(args)
    if isinstance(cameras, str):
        return {"ok": False, "error": cameras}
    hours = min(max(float(args.get("hours") or 24), 0.1), 24 * 14)
    label = str(args.get("label") or "").strip().lower() or None
    try:
        events = await ctx.nvr.list_events(
            cameras=cameras or None, labels=[label] if label else None, after=ctx.clock() - hours * 3600, limit=20
        )
    except Exception as exc:
        return {"ok": False, "error": f"Frigate didn't answer ({exc})"}
    return {
        "ok": True,
        "window": f"the last {hours:g} h",
        "events": [
            {
                "when": _when(float(e.get("start_time") or 0)),
                "camera": e.get("camera"),
                "label": e.get("label"),
                "name": e.get("sub_label") if isinstance(e.get("sub_label"), str) else None,
                "seconds": round(float(e.get("end_time") or e.get("start_time") or 0) - float(e.get("start_time") or 0)),
            }
            for e in events or []
        ],
        "note": "Frigate's own detections (every tracked object), including ones Watchwise didn't need to judge.",
    }


async def _show_snapshot(ctx: ChatContext, args: dict[str, Any]) -> dict[str, Any]:
    row = _find_event(ctx, args)
    if isinstance(row, str):
        return {"ok": False, "error": row}
    if len(ctx.images) >= 3:
        return {"ok": False, "error": "at most 3 photos per answer"}
    if row["review_id"] not in ctx.images:
        ctx.images.append(row["review_id"])
    return {"ok": True, "note": f"the photo of {row['camera']} at {_when(float(row['start_time']))} is sent with your answer"}


# ---------------------------------------------------------------- proposals


def _proposed(ctx: ChatContext, proposal: Proposal) -> dict[str, Any]:
    ctx.proposed.append(proposal)
    return {
        "ok": True,
        "proposal": proposal.summary,
        "note": "Nothing has changed yet: the owner sees this with ✅ Confirm / ✖️ Cancel buttons. "
        "Tell them to tap Confirm if it is right.",
    }


async def _propose_roster_rule(ctx: ChatContext, args: dict[str, Any]) -> dict[str, Any]:
    name = re.sub(r"[^\w -]", "", str(args.get("name") or "")).strip()[:40]
    days = [str(d).lower()[:3] for d in (args.get("days") or [])]
    start, end = str(args.get("start") or ""), str(args.get("end") or "")
    if not name or not days or any(d not in DAYS for d in days):
        return {"ok": False, "error": "give a name and days as mon, tue, wed, thu, fri, sat, sun"}
    if not (HHMM.match(start) and HHMM.match(end)):
        return {"ok": False, "error": "give start and end as HH:MM (24-hour), in local time"}
    cameras = ctx.resolve(args)
    if isinstance(cameras, str):
        return {"ok": False, "error": cameras}
    labels = [str(label).lower() for label in (args.get("labels") or ["person"])][:4]
    expect = str(args.get("expect") or name)[:120]
    zone = local_zone_name()
    entry: dict[str, Any] = {
        "name": name,
        "kind": "scheduled_activity",
        "days": days,
        "start": start,
        "end": end,
        "labels": labels,
        "expect": expect,
        "confidence": 0.85,
        "timezone": zone,
    }
    if cameras:
        entry["cameras"] = cameras
    where = ", ".join(cameras) if cameras else "every camera"
    summary = (
        f"Add to the duty roster: “{name}”, {' '.join(d.capitalize() for d in days)} {start}–{end} ({zone}), "
        f"{', '.join(labels)} on {where}, expected: {expect}. Then a {labels[0]} there at those times "
        "counts as expected, also while armed."
    )
    return _proposed(ctx, ctx.proposals.propose(ctx.conversation, "roster_rule", {"entry": entry}, summary))


async def _propose_arming(ctx: ChatContext, args: dict[str, Any]) -> dict[str, Any]:
    state = str(args.get("state") or "")
    if state not in ARMING_LABELS:
        return {"ok": False, "error": f"state must be one of {list(ARMING_LABELS)}"}
    label = ARMING_LABELS[state]
    summary = f"Switch arming to {label}."
    return _proposed(ctx, ctx.proposals.propose(ctx.conversation, "arming", {"state": state, "label": label}, summary))


async def _propose_snooze(ctx: ChatContext, args: dict[str, Any]) -> dict[str, Any]:
    cameras = ctx.resolve(args)
    if isinstance(cameras, str):
        return {"ok": False, "error": cameras}
    if not cameras:
        return {"ok": False, "error": "name the camera or room to mute"}
    minutes = int(min(max(float(args.get("minutes") or 60), 5), 24 * 60))
    summary = f"Mute {', '.join(cameras)} for {minutes} min (no cards; events are still recorded and judged)."
    payload = {"cameras": cameras, "minutes": minutes}
    return _proposed(ctx, ctx.proposals.propose(ctx.conversation, "snooze", payload, summary))


async def _propose_watch(ctx: ChatContext, args: dict[str, Any]) -> dict[str, Any]:
    request = " ".join(str(args.get("request") or "").split())[:160]
    if not request:
        return {"ok": False, "error": "say what to watch for, e.g. 'the parcel by the door is picked up'"}
    cameras = ctx.resolve(args)
    if isinstance(cameras, str):
        return {"ok": False, "error": cameras}
    hours = min(max(float(args.get("hours") or 24), 1), 7 * 24)
    where = ", ".join(cameras) if cameras else "every camera"
    summary = f"Watch {where} for the next {hours:g} h and alert you when: {request}."
    payload = {"request": request, "cameras": cameras, "hours": hours, "where": where}
    return _proposed(ctx, ctx.proposals.propose(ctx.conversation, "watch", payload, summary))


TOOLS: list[ChatTool] = [
    ChatTool(
        "recent_activity",
        "What happened in a time window: every event Watchwise judged, newest first, with its outcome "
        "(alerted, routine, held back) and what happened. Use for 'anything in the kitchen today?'.",
        _schema({
            "hours": {"type": "number", "description": "How far back, in hours (default 24)."},
            **WHERE,
            "only_alerts": {"type": "boolean", "description": "Only events that alerted the owner."},
            "page": {"type": "integer", "description": "Page 2, 3... for older events."},
        }),
        _recent_activity,
    ),
    ChatTool(
        "explain_decision",
        "Why one event was alerted or not: the verdict, which stage decided, the reason, any "
        "investigation and what the owner said. Find it by event_id, or by time (HH:MM) and camera/room.",
        _schema({"event_id": {"type": "string"}, "time": {"type": "string", "description": "HH:MM, local."}, **WHERE}),
        _explain_decision,
    ),
    ChatTool(
        "search_footage",
        "Frigate's own detections in a time window (every tracked object, with names Frigate recognized), "
        "filtered by object label, camera or room.",
        _schema({
            "label": {"type": "string", "description": "e.g. person, car, cat."},
            **WHERE,
            "hours": {"type": "number", "description": "How far back, in hours (default 24)."},
        }),
        _search_footage,
    ),
    ChatTool(
        "show_snapshot",
        "Send the owner the photo of an event (by event_id, or by time HH:MM and camera/room). At most 3.",
        _schema({"event_id": {"type": "string"}, "time": {"type": "string"}, **WHERE}),
        _show_snapshot,
    ),
    ChatTool(
        "propose_roster_rule",
        "Propose a duty-roster rule for expected, recurring visits ('the cleaner comes Thursdays 1-4 pm'). "
        "Changes nothing until the owner taps Confirm.",
        _schema({
            "name": {"type": "string", "description": "Who or what, e.g. cleaner."},
            "days": {"type": "array", "items": {"type": "string"}, "description": "mon..sun"},
            "start": {"type": "string", "description": "HH:MM, local, 24-hour."},
            "end": {"type": "string", "description": "HH:MM, local, 24-hour."},
            "labels": {"type": "array", "items": {"type": "string"}, "description": "Default [person]."},
            **WHERE,
            "expect": {"type": "string", "description": "What is expected, in a few words."},
        }, ["name", "days", "start", "end"]),
        _propose_roster_rule,
    ),
    ChatTool(
        "propose_arming",
        "Propose switching arming (armed_home, armed_away, armed_night, disarmed). Changes nothing until the "
        "owner taps Confirm.",
        _schema({"state": {"type": "string", "enum": list(ARMING_LABELS)}}, ["state"]),
        _propose_arming,
    ),
    ChatTool(
        "propose_snooze",
        "Propose muting a camera or room's cards for some minutes. Changes nothing until the owner taps Confirm.",
        _schema({**WHERE, "minutes": {"type": "integer", "description": "Default 60."}}),
        _propose_snooze,
    ),
    ChatTool(
        "propose_watch",
        "Propose a watch: 'tell me when X happens' on a camera or room, checked on every event until it "
        "happens or the time runs out. Changes nothing until the owner taps Confirm.",
        _schema({
            "request": {"type": "string", "description": "What to watch for, e.g. 'the parcel is picked up'."},
            **WHERE,
            "hours": {"type": "number", "description": "How long to watch (default 24, at most 168)."},
        }, ["request"]),
        _propose_watch,
    ),
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
