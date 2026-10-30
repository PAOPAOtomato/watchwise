"""Trace export: a recorded run as versioned JSON, for the web demo (M2-10, M3-9).

``watchwise export`` reads Watchwise's own database, and Frigate for the
frames, and writes ``trace.json`` plus ``frames/``: the events, every
decision in order with what it was based on, each investigation step by
step, every message the phone showed and every answer the owner gave, the
chat with the tool steps behind each answer, the digests, and numbers
counted from all of it. ``docs/trace-format.md`` specifies the format; the
web demo replays it.

Privacy, by default: people in the frames are pixelated over the boxes
Frigate drew around them, and a frame with a person Frigate gave no box is
left out; names (residents, the Face Library, people on the duty roster)
become "Resident A", "Visitor A"; chat ids and usernames become "owner";
links are dropped. What the owner typed is kept as typed, names replaced,
so read the export before publishing it.
"""

from __future__ import annotations

import io
import json
import logging
import re
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Literal

from PIL import Image, ImageFilter

from watchwise import __version__
from watchwise.config import WatchwiseConfig
from watchwise.memory.feedback import spot_check_record
from watchwise.memory.store import Store
from watchwise.notify.digest import build_digest
from watchwise.notify.presenter import decision_from_row, plain_reason
from watchwise.nvr.client import NVRClient
from watchwise.timeutil import local_zone_name

logger = logging.getLogger(__name__)

FORMAT, VERSION = "watchwise-trace", "1.0"
FramesMode = Literal["blurred", "none", "full"]
NamesMode = Literal["replaced", "kept"]
PHOTO_KINDS = ("alert", "spot_check", "chat_photo")  # messages that carried a photo of an event
BLUR_MARGIN = 0.25  # each box grows by a quarter of its size on every side
MAX_EDGE = 960
_URL = re.compile(r"https?://[^\s<>()\[\]\"']*[^\s<>()\[\]\"'.,;:!?]")  # not the sentence's ")."
_TELEGRAM_ID = re.compile(r"\btelegram:[\w.-]+")
_FRIGATE_ID = re.compile(r"\b\d{10}\.\d{1,6}-[a-z0-9]{4,8}\b")  # review and object ids, e.g. 1793380473.630245-dgbw2e
_PERSON_NAME = re.compile(r"^[A-Z][a-z]+$")  # "Alex", not a role like "cleaner"


@dataclass
class ExportOptions:
    since: float
    until: float
    frames: FramesMode = "blurred"
    names: NamesMode = "replaced"
    aliases: dict[str, str] = field(default_factory=dict)  # extra name -> alias
    machine: str | None = None  # labels the measured timings
    title: str | None = None


@dataclass
class ExportResult:
    path: Path
    events: int = 0
    messages: int = 0
    frames: int = 0
    boxes_blurred: int = 0
    frames_left_out: dict[str, str] = field(default_factory=dict)  # event id -> why
    names: dict[str, str] = field(default_factory=dict)  # real -> alias (printed, never written)
    roles_kept: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- pixelation


def blur_people(jpeg: bytes, boxes: Iterable[tuple[float, float, float, float]], *,
                margin: float = BLUR_MARGIN, max_edge: int = MAX_EDGE) -> tuple[bytes, int]:
    """Pixelate each relative box (x, y, width, height in 0..1), grown by
    ``margin`` on every side, so no face can be made out. Returns the JPEG
    and how many boxes were covered."""
    with Image.open(io.BytesIO(jpeg)) as opened:
        img = opened.convert("RGB")
    img.thumbnail((max_edge, max_edge))
    width, height = img.size
    covered = 0
    for x, y, w, h in boxes:
        x0, y0 = max(0, int((x - w * margin) * width)), max(0, int((y - h * margin) * height))
        x1 = min(width, int((x + w * (1 + margin)) * width) + 1)
        y1 = min(height, int((y + h * (1 + margin)) * height) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        region = img.crop((x0, y0, x1, y1))
        block = max(8, min(region.size) // 6)
        small = region.resize((max(1, region.width // block), max(1, region.height // block)), Image.BILINEAR)
        img.paste(small.resize(region.size, Image.NEAREST).filter(ImageFilter.GaussianBlur(block / 2)), (x0, y0))
        covered += 1
    out = io.BytesIO()
    img.save(out, "JPEG", quality=80)
    return out.getvalue(), covered


def _shrink(jpeg: bytes) -> bytes:
    data, _ = blur_people(jpeg, [])
    return data


# ---------------------------------------------------------------- names


class Scrubber:
    """Replaces names with aliases, Frigate's ids with the trace's own, chat
    identities with "owner", and drops links, in every string of the trace."""

    def __init__(self, aliases: dict[str, str], ids: dict[str, str] | None = None):
        self.aliases = {k: v for k, v in aliases.items() if k and k != v}
        names = sorted(self.aliases, key=len, reverse=True)
        self._names = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE) if names else None
        self._lookup = {k.lower(): v for k, v in self.aliases.items()}
        self.ids = dict(ids or {})
        known = sorted(self.ids, key=len, reverse=True)
        self._ids = re.compile("|".join(re.escape(k) for k in known)) if known else None

    def text(self, value: str) -> str:
        if self._ids is not None:
            value = self._ids.sub(lambda m: self.ids[m.group(0)], value)
        value = _FRIGATE_ID.sub("frigate-id", value)
        value = _URL.sub("[link]", value)
        value = _TELEGRAM_ID.sub("owner", value)
        if self._names is not None:
            value = self._names.sub(lambda m: self._lookup[m.group(0).lower()], value)
        return value

    def any(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.any(v) for v in value]
        if isinstance(value, dict):  # keys too: "recognized Ray": 4
            return {self.any(k): self.any(v) for k, v in value.items()}
        return value


def name_aliases(config: WatchwiseConfig, library: Iterable[str], roster_names: Iterable[str],
                 extra: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Residents and the Face Library become "Resident A, B…"; roster names
    that look like a person's ("Alex") become "Visitor A, B…"; roles
    ("cleaner", "night_security_patrol") are kept. Returns (aliases, roles kept)."""
    aliases: dict[str, str] = {}
    for name in [*config.faces.household, *library]:
        if name and name.lower() not in {a.lower() for a in aliases}:
            aliases[name] = f"Resident {chr(ord('A') + sum(v.startswith('Resident') for v in aliases.values()))}"
    roles: list[str] = []
    for name in dict.fromkeys(roster_names):
        if not name or name.lower() in {a.lower() for a in aliases}:
            continue
        if _PERSON_NAME.match(name):
            aliases[name] = f"Visitor {chr(ord('A') + sum(v.startswith('Visitor') for v in aliases.values()))}"
        else:
            roles.append(name)
    aliases.update(extra)
    return aliases, roles


# ---------------------------------------------------------------- the export


class TraceExporter:
    def __init__(self, store: Store, config: WatchwiseConfig, nvr: NVRClient | None, options: ExportOptions):
        self.store = store
        self.config = config
        self.nvr = nvr
        self.opt = options
        self.ids: dict[str, dict[str, str]] = {}
        self.scrub = Scrubber({})

    # ---- helpers

    def _id(self, kind: str, key: Any) -> str:
        table = self.ids.setdefault(kind, {})
        if str(key) not in table:
            table[str(key)] = f"{kind}{len(table) + 1:02d}"
        return table[str(key)]

    def _known(self, kind: str, key: Any) -> str | None:
        return self.ids.get(kind, {}).get(str(key))

    def _at(self, ts: float | None) -> dict[str, Any] | None:
        if ts is None:
            return None
        return {"at": datetime.fromtimestamp(float(ts)).astimezone().isoformat(timespec="milliseconds"),
                "t": round(float(ts) - self.opt.since, 3)}

    def _when(self, ts: float | None) -> str | None:
        return None if ts is None else datetime.fromtimestamp(float(ts)).astimezone().isoformat(timespec="milliseconds")

    def _t(self, ts: float | None) -> float | None:
        return None if ts is None else round(float(ts) - self.opt.since, 3)

    @staticmethod
    def _json(text: str | None, default: Any) -> Any:
        try:
            return json.loads(text) if text else default
        except ValueError:
            return default

    # ---- run

    async def run(self, out_dir: Path) -> ExportResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        result = ExportResult(path=out_dir / "trace.json")
        since, until = self.opt.since, self.opt.until
        await self._names(result)

        alerts = self.store.alerts_between(since, until)
        review_ids = [str(a["review_id"]) for a in alerts]
        for review_id in review_ids:
            self._id("e", review_id)
        events = [self._event(a) for a in alerts]
        decisions = self._decisions(alerts)
        investigations = self._investigations(review_ids)
        messages = self._messages(review_ids)
        owner = self._owner_actions(review_ids)
        corrections = self._corrections(review_ids, owner)
        conversations, proposals = self._conversations(), self._proposals()
        frames = await self._frames(out_dir, messages, result) if self.opt.frames != "none" else []
        frame_of = {f["event"]: f["id"] for f in frames}
        for event in events:
            event["frame"] = frame_of.get(event["id"])
        for message in messages:
            if message["event"] and message.pop("_photo", False):
                message["frame"] = frame_of.get(message["event"])
            message.pop("_photo", None)

        trace = {
            "format": FORMAT,
            "version": VERSION,
            "meta": self._meta(),
            "cameras": self._cameras(alerts),
            "events": events,
            "decisions": decisions,
            "investigations": investigations,
            "frames": frames,
            "messages": messages,
            "owner_actions": owner,
            "corrections": corrections,
            "patterns": self._patterns(),
            "conversations": conversations,
            "proposals": proposals,
            "watches": self._watches(),
            "digests": self._digests(),
            "arming": self._arming(),
            "metrics": self._metrics(alerts, decisions, investigations, messages, conversations),
        }
        # Frigate's ids become the trace's own wherever they appear (tool
        # arguments and results too); names only when they are replaced.
        ids = {review_id: event for review_id, event in self.ids.get("e", {}).items()}
        trace = Scrubber(self.scrub.aliases if self.opt.names == "replaced" else {}, ids).any(trace)
        result.path.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
        result.events, result.messages, result.frames = len(events), len(messages), len(frames)
        return result

    async def _names(self, result: ExportResult) -> None:
        library: list[str] = []
        if self.nvr is not None:
            try:
                library = list(await self.nvr.faces())
            except Exception as exc:
                logger.info("Face Library not read (%s); only configured residents are replaced", exc)
        roster: list[str] = []
        path = self.config.duty_roster.path
        if path and Path(path).exists():
            try:
                import yaml

                data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
                roster = [str(e.get("name") or "") for e in data.get("entries") or [] if isinstance(e, dict)]
            except Exception as exc:
                logger.info("duty roster not read (%s)", exc)
        for row in self.store.query("SELECT payload FROM proposals"):
            entry = (self._json(row["payload"], {}) or {}).get("entry")
            if isinstance(entry, dict) and entry.get("name"):
                roster.append(str(entry["name"]))
        aliases, roles = name_aliases(self.config, library, roster, self.opt.aliases)
        if self.opt.names == "replaced":
            self.scrub = Scrubber(aliases)
            result.names, result.roles_kept = aliases, roles

    # ---- sections

    def _meta(self) -> dict[str, Any]:
        commit = None
        try:  # "-dirty" when the code had changes not yet committed
            commit = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=7"], capture_output=True,
                                    text=True, cwd=Path(__file__).resolve().parent, timeout=5).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
        models = {}
        for role in ("screener", "investigator", "chat"):
            endpoint = self.config.endpoint(role)
            if endpoint is not None:
                models[role] = f"{endpoint.provider}/{endpoint.model}"
        offset = datetime.fromtimestamp(self.opt.since).astimezone().utcoffset()
        return {
            "title": self.opt.title,
            "source": "recorded",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "generator": {"name": "watchwise", "version": __version__, "commit": commit},
            "window": {"start": self._when(self.opt.since), "end": self._when(self.opt.until),
                       "seconds": round(self.opt.until - self.opt.since, 3)},
            "timezone": local_zone_name(),
            "utc_offset_minutes": int(offset.total_seconds() // 60) if offset is not None else 0,
            "models": models,
            "machine": self.opt.machine,
            "privacy": {
                "frames": self.opt.frames,
                "names": self.opt.names,
                "people_in_frames": "pixelated over Frigate's box for each person, grown by 25% on every side"
                if self.opt.frames == "blurred" else ("not blurred: private use only" if self.opt.frames == "full" else "no frames"),
                "identities": "chat ids and usernames are 'owner'; links are '[link]'",
            },
        }

    def _cameras(self, alerts: list[Any]) -> list[dict[str, Any]]:
        names = list(dict.fromkeys([*self.config.cameras, *(str(a["camera"]) for a in alerts)]))
        cameras = []
        for name in names:
            spec = self.config.cameras.get(name)
            cameras.append({"id": name, "room": (spec.room.strip() or None) if spec else None,
                            "notes": (spec.notes.strip() or None) if spec else None})
        return cameras

    def _event(self, row: Any) -> dict[str, Any]:
        decision = decision_from_row(row)
        start, end = float(row["start_time"]), float(row["end_time"]) if row["end_time"] else None
        return {
            "id": self._id("e", row["review_id"]),
            "camera": row["camera"],
            **(self._at(start) or {}),
            "end": self._when(end),
            "duration_s": round(end - start, 1) if end else None,
            "labels": self._json(row["labels"], []),
            "zones": self._json(row["zones"], []),
            "outcome": {
                "disposition": decision.disposition.value if decision else None,
                "stage": decision.stage.value if decision else None,
                "confidence": decision.confidence if decision else None,
                "reason": plain_reason(decision) if decision else None,
                "summary": row["summary"],
            },
            "frame": None,
        }

    def _decisions(self, alerts: list[Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for alert in alerts:
            rows = self.store.query("SELECT at, decision FROM decision_log WHERE review_id=? ORDER BY id",
                                    (alert["review_id"],))
            entries = [self._json(r["decision"], {}) for r in rows]
            history = "complete"
            if not entries:  # decided before the log existed: only the latest is known
                latest = decision_from_row(alert)
                entries = [latest.to_dict()] if latest is not None else []
                history = "latest_only"
            for entry in entries:
                out.append(self._decision(alert, entry, history))
        out.sort(key=lambda d: (d["t"] if d["t"] is not None else 0.0))
        for i, d in enumerate(out, 1):
            d["id"] = f"d{i:03d}"
        return out

    def _decision(self, alert: Any, entry: dict[str, Any], history: str) -> dict[str, Any]:
        screen = entry.get("screen") or None
        decision = decision_from_row({**dict(alert), "disposition": entry.get("disposition"),
                                      "severity": entry.get("severity"), "confidence": entry.get("confidence"),
                                      "rationale": entry.get("rationale"), "stage": entry.get("stage"),
                                      "screen_json": json.dumps(screen), "summary": entry.get("summary"),
                                      "investigation_id": entry.get("investigation_id")})
        inv = entry.get("investigation_id")
        at = float(entry.get("decided_at") or alert["updated_at"])
        return {
            "id": "",
            "event": self._id("e", alert["review_id"]),
            **(self._at(at) or {}),
            "stage": entry.get("stage"),
            "disposition": entry.get("disposition"),
            "severity": entry.get("severity"),
            "confidence": entry.get("confidence"),
            "reason": plain_reason(decision) if decision else entry.get("rationale"),
            "rationale": entry.get("rationale"),
            "analyzing": entry.get("stage") == "ingested" and entry.get("disposition") == "alerted",
            "screen": None if not screen else {
                "verdict": screen.get("verdict"), "confidence": screen.get("confidence"),
                "reason": screen.get("reason"), "tags": screen.get("tags") or [],
                "model": f"{screen.get('backend')}/{screen.get('model')}",
                "latency_ms": screen.get("latency_ms"), "frames_used": screen.get("frames_used"),
                "error": screen.get("error"),
            },
            "investigation": self._id("i", inv) if inv else None,
            "summary": entry.get("summary"),
            "patterns": [self._id("p", p) for p in entry.get("matched_memory_ids") or []],
            "history": history,
        }

    def _investigations(self, review_ids: list[str]) -> list[dict[str, Any]]:
        out = []
        for review_id in review_ids:
            for row in self.store.query("SELECT * FROM investigations WHERE review_id=? ORDER BY started_at",
                                        (review_id,)):
                inv_id = self._id("i", row["id"])
                trace = self._json(row["trace_json"], {})
                steps = []
                for n, call in enumerate(trace.get("tool_calls") or [], 1):
                    steps.append({
                        "evidence_id": f"{inv_id}/{n}",
                        "iteration": call.get("iteration"),
                        "tool": call.get("name"),
                        "arguments": call.get("arguments") or {},
                        "result": call.get("result_summary"),
                        "latency_ms": call.get("latency_ms"),
                        "error": call.get("error"),
                    })
                started, finished = row["started_at"], row["finished_at"]
                out.append({
                    "id": inv_id,
                    "event": self._id("e", review_id),
                    "status": row["status"],
                    "verdict": row["verdict"],
                    "confidence": row["confidence"],
                    "conclusion": row["conclusion"],
                    "model": f"{row['backend']}/{row['model']}",
                    "started": self._at(started),
                    "finished": self._at(finished),
                    "seconds": round(float(finished) - float(started), 1) if started and finished else None,
                    "steps": steps,
                    "thoughts": trace.get("thoughts") or [],
                    # Evidence discipline (M1-6): filled once the investigator
                    # reports them; empty lists mean "not reported", not "none".
                    "observations": [],
                    "inferences": [],
                    "unknowns": [],
                    "error": row["error"],
                })
        return out

    def _in_window(self, table: str, review_ids: list[str], column: str = "review_id") -> list[Any]:
        rows = self.store.query(f"SELECT * FROM {table} WHERE at>=? AND at<=? ORDER BY id",
                                (self.opt.since, self.opt.until))
        seen = {int(r["id"]) for r in rows}
        wanted = set(review_ids)
        later = [r for r in self.store.query(f"SELECT * FROM {table} WHERE at>? ORDER BY id", (self.opt.until,))
                 if r[column] in wanted and int(r["id"]) not in seen]
        return sorted([*rows, *later], key=lambda r: int(r["id"]))

    def _messages(self, review_ids: list[str]) -> list[dict[str, Any]]:
        out = []
        for row in self._in_window("card_log", review_ids):
            review_id = str(row["review_id"] or "")
            event = self._known("e", review_id) if review_id else None
            out.append({
                "id": f"m{len(out) + 1:03d}",
                **(self._at(row["at"]) or {}),
                "message": self._id("tg", f"{row['chat_id']}:{row['message_id']}"),
                "action": row["action"],
                "kind": row["kind"],
                "event": event,
                "members": [self._known("e", m) for m in self._json(row["members"], []) if self._known("e", m)],
                "text": row["text"],
                "buttons": self._buttons(self._json(row["buttons"], {})),
                "has_photo": bool(row["has_image"]),
                "frame": None,
                "_photo": bool(row["has_image"]) and row["kind"] in PHOTO_KINDS,
            })
        return out

    def _buttons(self, markup: dict[str, Any]) -> list[list[dict[str, Any]]]:
        rows = []
        for row in markup.get("inline_keyboard") or []:
            buttons = []
            for button in row:
                if button.get("url"):
                    buttons.append({"label": button.get("text"), "link": True})  # the URL itself is private
                else:
                    buttons.append({"label": button.get("text"), **self._parse_data(str(button.get("callback_data") or ""))})
            rows.append(buttons)
        return rows

    def _parse_data(self, data: str) -> dict[str, Any]:
        """What a button does, without Frigate's ids."""
        parts = data.split(":")
        if data.startswith("fb:") and len(parts) >= 3:
            if parts[1] == "undo":
                return {"action": "undo", "event": self._known("e", parts[-1]),
                        "corrections": [self._id("c", c) for c in parts[2].split(",") if c]}
            return {"action": parts[1], "event": self._known("e", ":".join(parts[2:]))}
        if data.startswith("pr:") and len(parts) == 3:
            return {"action": "confirm" if parts[1] == "ok" else "cancel", "proposal": self._id("pr", parts[2])}
        if data.startswith("ar:"):
            return {"action": "arm", "state": data[3:]}
        if data.startswith("fc:who:") and len(parts) >= 4:
            return {"action": "that_was", "name": parts[2], "event": self._known("e", ":".join(parts[3:]))}
        if data.startswith("fc:"):
            return {"action": f"face_{parts[1]}" if len(parts) > 1 else "face"}
        return {"action": "other"}

    def _owner_actions(self, review_ids: list[str]) -> list[dict[str, Any]]:
        out = []
        for row in self._in_window("owner_log", review_ids):
            outcome = self._json(row["outcome"], {})
            entry: dict[str, Any] = {
                "id": f"a{len(out) + 1:03d}",
                **(self._at(row["at"]) or {}),
                "kind": row["kind"],
                "on_message": self._known("tg", f"{row['chat_id']}:{row['message_id']}"),
            }
            if row["kind"] == "button":
                entry.update(self._parse_data(str(row["data"])))
                keyboard = str(outcome.get("keyboard") or "")
                made = keyboard.split(":", 1)[1].split(",") if keyboard.startswith("after_feedback:") else []
                entry["outcome"] = {
                    "toast": outcome.get("toast") or outcome.get("text"),
                    "footer": outcome.get("footer"),
                    "reply": outcome.get("reply"),
                    "done": outcome.get("done"),
                    "spot_check": outcome.get("spot_check"),
                    "corrections": [self._id("c", c) for c in made if c],
                }
            elif row["kind"] == "note":
                entry.update(text=row["data"], reply=outcome.get("reply"),
                             replied_to=self._known("tg", f"{row['chat_id']}:{outcome.get('replied_to')}"))
            else:
                entry["text"] = row["data"]
            out.append(entry)
        return out

    def _corrections(self, review_ids: list[str], owner: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        spot = {str(r["review_id"]) for r in self.store.query("SELECT review_id FROM notifications WHERE kind='spot_check'")}
        for review_id in review_ids:
            for row in self.store.query("SELECT * FROM corrections WHERE review_id=? ORDER BY id", (review_id,)):
                cid = self._id("c", row["id"])
                effects = self.store.query("SELECT pattern_id, effect FROM correction_effects WHERE correction_id=?",
                                           (row["id"],))
                out[cid] = {
                    "id": cid,
                    "event": self._id("e", review_id),
                    **(self._at(row["created_at"]) or {}),
                    "verdict": row["human_verdict"],
                    "system_verdict": row["system_verdict"],
                    "note": row["note"] or None,
                    "via": "spot_check" if review_id in spot else ("card" if str(row["reviewer"] or "").startswith("telegram") else "other"),
                    "undone": None,
                    "effects": [{"pattern": self._id("p", e["pattern_id"]), "effect": e["effect"]} for e in effects],
                }
        # Undo deletes a correction; the owner's taps still say what it was.
        made: dict[str, dict[str, Any]] = {}
        for action in owner:
            if action.get("kind") != "button":
                continue
            for cid in (action.get("outcome") or {}).get("corrections") or []:
                made.setdefault(cid, action)
            if action.get("action") == "undo":
                for cid in action.get("corrections") or []:
                    tap = made.get(cid)
                    if cid not in out and tap is not None:
                        out[cid] = {
                            "id": cid, "event": tap.get("event"), "at": tap.get("at"), "t": tap.get("t"),
                            "verdict": "benign" if tap.get("action") == "ok" else "suspicious",
                            "system_verdict": None, "note": None,
                            "via": "spot_check" if (tap.get("outcome") or {}).get("spot_check") else "card",
                            "undone": {"at": action.get("at"), "t": action.get("t")}, "effects": [],
                        }
        return sorted(out.values(), key=lambda c: c["t"] if c["t"] is not None else 0.0)

    def _patterns(self) -> list[dict[str, Any]]:
        out = []
        for key, pid in sorted(self.ids.get("p", {}).items(), key=lambda kv: kv[1]):
            row = self.store.query_one("SELECT * FROM patterns WHERE id=?", (key,))
            if row is None:
                out.append({"id": pid, "gone": True})
                continue
            agreements, disagreements = int(row["agreements"]), int(row["disagreements"])
            out.append({
                "id": pid, "kind": row["kind"], "camera": row["camera"], "signature": row["signature"],
                "description": row["description"], "corrections": int(row["corrections"]),
                "agreements": agreements, "disagreements": disagreements,
                "trust": round((agreements + 1.0) / (agreements + disagreements + 2.0), 3),
                "first_seen": self._at(row["created_at"]), "last_seen": self._at(row["last_seen"]),
                "state": "at export",
            })
        return out

    def _conversations(self) -> list[dict[str, Any]]:
        rows = self.store.query("SELECT * FROM chat_messages WHERE created_at>=? AND created_at<=? ORDER BY id",
                                (self.opt.since, self.opt.until))
        turns: list[dict[str, Any]] = []
        question: Any = None
        for row in rows:
            if row["role"] == "user":
                question = row
                continue
            meta = self._json(row["meta"], {})
            turns.append({
                "id": f"q{len(turns) + 1:02d}",
                **(self._at(row["created_at"]) or {}),
                "question": question["content"] if question is not None else None,
                "answer": row["content"],
                "about": self._known("e", meta.get("about")) if meta.get("about") else None,
                "tools": meta.get("tools") or [],
                "photos": [self._known("e", r) for r in meta.get("images") or [] if self._known("e", r)],
                "proposals": [self._id("pr", p) for p in meta.get("proposals") or []],
                "seconds": meta.get("seconds"),
                "prompt_tokens": meta.get("prompt_tokens"),
                "guard": meta.get("guard") or meta.get("fixed"),
                "steps_recorded": bool(meta),
            })
            question = None
        return [{"id": "owner", "turns": turns}] if turns else []

    def _proposals(self) -> list[dict[str, Any]]:
        out = []
        for row in self.store.query("SELECT * FROM proposals WHERE created_at>=? AND created_at<=? ORDER BY created_at",
                                    (self.opt.since, self.opt.until)):
            out.append({
                "id": self._id("pr", row["id"]),
                **(self._at(row["created_at"]) or {}),
                "kind": row["kind"], "summary": row["summary"], "payload": self._json(row["payload"], {}),
                "status": row["status"], "result": row["result"] or None,
                "decided": self._at(row["decided_at"]),
            })
        return out

    def _watches(self) -> list[dict[str, Any]]:
        out = []
        for row in self.store.query("SELECT * FROM watches WHERE created_at<=? AND expires_at>=? ORDER BY id",
                                    (self.opt.until, self.opt.since)):
            out.append({
                "id": f"w{len(out) + 1:02d}", "request": row["request"], "cameras": self._json(row["cameras"], []),
                "created": self._at(row["created_at"]), "expires": self._at(row["expires_at"]),
                "status": row["status"], "matched": self._known("e", row["review_id"]) if row["review_id"] else None,
            })
        return out

    def _digests(self) -> list[dict[str, Any]]:
        out = []
        for row in self.store.query("SELECT * FROM digests WHERE until>=? AND until<=? ORDER BY id",
                                    (self.opt.since, self.opt.until)):
            messages = self._json(row["messages"], {})
            out.append({
                "id": f"g{len(out) + 1:02d}", "kind": row["kind"],
                "covers": {"start": self._when(row["since"]), "end": self._when(row["until"])},
                **(self._at(row["until"]) or {}),
                "messages": [self._known("tg", f"{chat}:{mid}") for chat, mid in messages.items()],
                "spot_checks": [self._known("e", r) for r in self._json(row["spot_checks"], [])],
                "sent": bool(messages),
            })
        return out

    def _arming(self) -> list[dict[str, Any]]:
        before = self.store.query_one("SELECT * FROM arming_log WHERE at<? ORDER BY at DESC LIMIT 1", (self.opt.since,))
        rows = self.store.query("SELECT * FROM arming_log WHERE at>=? AND at<=? ORDER BY at",
                                (self.opt.since, self.opt.until))
        out = []
        for row in [*( [before] if before is not None else []), *rows]:
            source = str(row["source"]).split(":", 1)[0]
            out.append({**(self._at(row["at"]) or {}), "state": row["state"], "source": source})
        return out

    async def _frames(self, out_dir: Path, messages: list[dict[str, Any]], result: ExportResult) -> list[dict[str, Any]]:
        wanted = list(dict.fromkeys(m["event"] for m in messages if m.get("_photo") and m["event"]))
        reverse = {v: k for k, v in self.ids.get("e", {}).items()}
        frames: list[dict[str, Any]] = []
        (out_dir / "frames").mkdir(exist_ok=True)
        for event in wanted:
            review_id = reverse.get(event)
            if review_id is None or self.nvr is None:
                continue
            try:
                frame = await self._frame(review_id, event, out_dir, len(frames) + 1, result)
            except Exception as exc:
                result.frames_left_out[event] = f"Frigate gave no frame ({type(exc).__name__})"
                continue
            if frame is not None:
                frames.append(frame)
        return frames

    async def _frame(self, review_id: str, event: str, out_dir: Path, n: int, result: ExportResult) -> dict[str, Any] | None:
        """The best frame of the event's person (Frigate's ``snapshot_frame_time``),
        from the recording, with every person's box pixelated."""
        assert self.nvr is not None
        review = await self.nvr.get_review(review_id)
        objects = []
        for event_id in (review.get("data") or {}).get("detections") or []:
            try:
                objects.append(await self.nvr.get_event(event_id))
            except Exception:
                continue
        people = [o for o in objects if str(o.get("label", "")).split("-")[0] == "person"]
        if not people:
            result.frames_left_out[event] = "no person or object box from Frigate"
            return None
        lead = people[0]
        data = lead.get("data") or {}
        at = data.get("snapshot_frame_time")
        if not at or not data.get("box"):
            # Frigate's box belongs to that one frame; without it nothing can be blurred safely.
            result.frames_left_out[event] = "no best frame with a box from Frigate"
            return None
        others = [p for p in people[1:] if float(p.get("start_time") or 0) <= float(at) <= float(p.get("end_time") or 1e12)]
        if others and self.opt.frames == "blurred":
            # Their boxes come from other frames: they could be anywhere in this one.
            result.frames_left_out[event] = "more than one person in view: can't be sure every one is covered"
            return None
        boxes = [tuple(float(v) for v in data["box"])]
        boxes += [tuple(float(v) for v in a["box"]) for a in data.get("attributes") or [] if a.get("box")]
        jpeg = (await self.nvr.recording_frame(str(review.get("camera") or lead.get("camera")), float(at))).jpeg
        if self.opt.frames == "blurred":
            jpeg, covered = blur_people(jpeg, boxes)
            result.boxes_blurred += covered
        else:
            jpeg = _shrink(jpeg)
        name = f"frames/f{n:02d}.jpg"
        (out_dir / name).write_bytes(jpeg)
        with Image.open(io.BytesIO(jpeg)) as img:
            width, height = img.size
        return {"id": f"f{n:02d}", "event": event, "file": name, **(self._at(float(at)) or {}),
                "source": "recording at the person's best frame", "width": width, "height": height,
                "privacy": self.opt.frames, "boxes": len(boxes) if self.opt.frames == "blurred" else 0}

    def _metrics(self, alerts: list[Any], decisions: list[dict[str, Any]], investigations: list[dict[str, Any]],
                 messages: list[dict[str, Any]], conversations: list[dict[str, Any]]) -> dict[str, Any]:
        digest = build_digest(self.store, self.opt.since, self.opt.until, spot_checks=0)
        final = [d.stage.value for d in (decision_from_row(a) for a in alerts) if d is not None]
        stages = {s: final.count(s) for s in ("gated", "face", "screened", "investigated", "summarized", "ingested")}
        first_card: list[float] = []
        starts = {self._known("e", a["review_id"]): float(a["start_time"]) for a in alerts}
        for message in messages:
            event = message.get("event")
            if message["kind"] == "alert" and message["action"] == "sent" and event in starts:
                first_card.append(round(message["t"] - (starts[event] - self.opt.since), 1))
        # Later decisions carry the first look along: count each result once.
        looks = {(d["event"], d["screen"].get("reason"), d["screen"].get("latency_ms"))
                 for d in decisions if d.get("screen")}
        screens = [latency for _, _, latency in looks if latency]
        inv_s = [i["seconds"] for i in investigations if i.get("seconds")]
        chat_s = [t["seconds"] for c in conversations for t in c["turns"] if t.get("seconds")]
        spot = spot_check_record(self.store, self.opt.since)
        label = f"measured ({self.opt.machine or 'machine not given'}, {datetime.fromtimestamp(self.opt.since).date()})"

        def stats(values: list[float]) -> dict[str, Any] | None:
            return {"n": len(values), "median": round(median(values), 2), "max": round(max(values), 2)} if values else None

        return {
            "counts": {
                "label": "counted from this recording",
                "events": digest.events, "sent_to_owner": digest.sent, "held_back": digest.held_count,
                "alerted_while_muted": digest.muted,
                "held_back_by_reason": {g.label: len(g.review_ids) for g in digest.held},
                "final_stage": stages,
                "model_calls": {"screenings": len(looks),
                                "investigations": len(investigations),
                                "chat_answers": sum(len(c["turns"]) for c in conversations)},
                "owner_answers": digest.answers,
                "spot_checks": spot,
            },
            "timings": {
                "label": label,
                "first_card_after_event_start_s": stats(first_card),
                "screen_ms": stats([float(s) for s in screens]),
                "investigation_s": stats([float(s) for s in inv_s]),
                "chat_answer_s": stats([float(s) for s in chat_s]),
            },
        }


def parse_when(text: str, now: float | None = None) -> float:
    """A time for --since/--until: "-6h", "-90m", "-2d", "2026-10-30 12:00"
    (local time) or an ISO time with its offset."""
    now = time.time() if now is None else now
    match = re.fullmatch(r"-(\d+(?:\.\d+)?)([mhd])", text.strip())
    if match:
        return now - float(match.group(1)) * {"m": 60, "h": 3600, "d": 86400}[match.group(2)]
    moment = datetime.fromisoformat(text.strip().replace(" ", "T", 1))
    return moment.timestamp()
