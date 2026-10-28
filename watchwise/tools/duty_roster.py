"""Forensic tool category 4: the duty-roster rule base.

Encodes what is *expected*: who is on shift, which zones they are allowed in,
recurring scheduled activity (bin collection Tuesday 06:00-07:00, cleaner
Thursday afternoons), and standing exceptions. Without it the agent has no
way to tell a rostered night guard from an intruder -- both are "a person in
a restricted zone at 02:00".

The roster is plain YAML so a human can maintain it. Time handling uses the
configured timezone, and a rule may cross midnight.

Schema:

    timezone: Europe/London
    entries:
      - name: night_security
        kind: shift                       # shift | scheduled_activity | exception
        days: [mon, tue, wed, thu, fri]   # or 'all'
        start: "22:00"
        end: "06:00"                      # may wrap past midnight
        cameras: [cam_yard, cam_gate]     # omit for all
        zones: [perimeter]                # omit for all
        labels: [person]
        expect: "rostered guard patrol"
        confidence: 0.9
        timezone: Europe/Paris           # optional: this entry's own zone

An entry's own ``timezone`` wins over the file's (rules proposed in chat carry
the house's zone, so "Thursdays 1-4 pm" means local time whatever the file says).
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any

import yaml

from watchwise.timeutil import safe_datetime, zone
from watchwise.tools.base import ForensicTool, ToolContext, obj_schema

logger = logging.getLogger(__name__)

_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class DutyRoster:
    def __init__(self, path: str | Path | None = None, timezone: str = "UTC"):
        self.path = Path(path) if path else None
        self.timezone = timezone
        self._entries: list[dict[str, Any]] = []
        self._mtime: float | None = None
        self._unknown_zones: set[str] = set()
        self.load()

    def tzinfo(self) -> dt.tzinfo:
        """Resolve the configured timezone, falling back to UTC."""
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(self.timezone)
        except Exception:
            return dt.timezone.utc

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            self._entries = []
            return
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("could not read duty roster %s: %s", self.path, exc)
            self._entries = []
            return
        self.timezone = data.get("timezone", self.timezone)
        self._entries = list(data.get("entries") or [])
        self._mtime = self.path.stat().st_mtime

    def entries(self) -> list[dict[str, Any]]:
        """The rules as they are in the file now."""
        self._maybe_reload()
        return list(self._entries)

    def _maybe_reload(self) -> None:
        """Pick up edits without a restart -- rosters change often."""
        if self.path is None or not self.path.exists():
            return
        mtime = self.path.stat().st_mtime
        if self._mtime is None or mtime > self._mtime:
            self.load()

    @staticmethod
    def _parse_hhmm(value: str) -> dt.time:
        hh, _, mm = str(value).partition(":")
        return dt.time(int(hh), int(mm or 0))

    def _in_window(self, moment: dt.datetime, entry: dict[str, Any]) -> bool:
        days = entry.get("days", "all")
        if days != "all":
            allowed = {str(d).lower()[:3] for d in days}
            if _DAYS[moment.weekday()] not in allowed:
                # A window that wraps past midnight belongs to the previous day.
                start_t = self._parse_hhmm(entry.get("start", "00:00"))
                end_t = self._parse_hhmm(entry.get("end", "23:59"))
                if not (end_t <= start_t and _DAYS[(moment.weekday() - 1) % 7] in allowed):
                    return False

        start_t = self._parse_hhmm(entry.get("start", "00:00"))
        end_t = self._parse_hhmm(entry.get("end", "23:59"))
        now_t = moment.time()
        if end_t <= start_t:  # wraps midnight
            return now_t >= start_t or now_t <= end_t
        return start_t <= now_t <= end_t

    def match(
        self,
        timestamp: float,
        *,
        camera: str | None = None,
        zones: list[str] | None = None,
        labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        self._maybe_reload()
        moment = safe_datetime(timestamp, self.tzinfo())
        if moment is None:
            return []
        zones_set = {z.lower() for z in (zones or [])}
        labels_set = {label.lower() for label in (labels or [])}

        hits: list[dict[str, Any]] = []
        for entry in self._entries:
            own = zone(entry.get("timezone"))
            if own is None and entry.get("timezone") and entry["timezone"] not in self._unknown_zones:
                # Silent on Windows without tzdata, until CI caught it.
                self._unknown_zones.add(entry["timezone"])
                logger.warning("duty roster: unknown time zone %r in rule %r; using %s",
                               entry["timezone"], entry.get("name"), self.timezone)
            local = safe_datetime(timestamp, own) if own is not None else moment
            if local is None or not self._in_window(local, entry):
                continue
            cams = entry.get("cameras")
            if cams and camera and camera not in cams:
                continue
            ez = entry.get("zones")
            if ez and zones_set and not (zones_set & {z.lower() for z in ez}):
                continue
            el = entry.get("labels")
            if el and labels_set and not (labels_set & {label.lower() for label in el}):
                continue
            hits.append(
                {
                    "name": entry.get("name", "unnamed"),
                    "kind": entry.get("kind", "shift"),
                    "window": f"{entry.get('start', '00:00')}-{entry.get('end', '23:59')}",
                    "days": entry.get("days", "all"),
                    "expect": entry.get("expect", ""),
                    "confidence": float(entry.get("confidence", 0.8)),
                    "cameras": cams,
                    "zones": ez,
                    "labels": el,
                }
            )
        return hits


async def _check_duty_roster(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    roster: DutyRoster | None = ctx.roster
    timestamp = float(args["timestamp"])

    if roster is None:
        return {
            "configured": False,
            "matches": [],
            "interpretation": (
                "No duty roster is configured, so nothing can be confirmed as "
                "expected activity. Do not treat this as evidence either way."
            ),
        }

    matches = roster.match(
        timestamp,
        camera=args.get("camera"),
        zones=args.get("zones"),
        labels=args.get("labels"),
    )

    moment = safe_datetime(timestamp, roster.tzinfo())
    return {
        "configured": True,
        "local_time": moment.strftime("%Y-%m-%d %H:%M:%S %Z") if moment else None,
        "day": _DAYS[moment.weekday()] if moment else None,
        "matches": matches,
        "expected": bool(matches),
        "interpretation": _interpret(matches, moment),
    }


def _interpret(matches: list[dict[str, Any]], moment: dt.datetime | None) -> str:
    if not matches:
        when = moment.strftime("%a %H:%M") if moment else "that timestamp"
        return (
            f"Nothing on the roster covers {when} for this "
            "camera/zone/label combination. Activity here is unscheduled, which "
            "raises -- but does not by itself establish -- suspicion."
        )
    best = max(matches, key=lambda m: m["confidence"])
    others = len(matches) - 1
    tail = f" (+{others} other roster match(es))" if others else ""
    return (
        f"Roster entry '{best['name']}' ({best['kind']}, {best['window']}) covers this "
        f"time: {best['expect']}{tail}. Activity consistent with it is expected."
    )


TOOL = ForensicTool(
    name="check_duty_roster",
    description=(
        "Consult the duty-roster rule base: is there a scheduled shift, recurring "
        "activity, or standing exception that explains activity at this time, on "
        "this camera, in these zones? Use this whenever an alert involves a person "
        "outside normal hours, before concluding it is unexpected."
    ),
    category="duty_roster",
    input_schema=obj_schema(
        {
            "timestamp": {"type": "number", "description": "Unix timestamp of the alert."},
            "camera": {"type": "string", "description": "Camera the alert came from."},
            "zones": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Zones involved in the alert.",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Object labels involved, e.g. ['person'].",
            },
        },
        required=["timestamp"],
    ),
    handler=_check_duty_roster,
)
