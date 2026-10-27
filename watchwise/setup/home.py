"""What the owner tells ``/setup`` about their home, written into the config
file (D-060).

Plan -> confirm -> apply, like ``watchwise init``:

  * the plan merges the answers into the file as it is now and touches only
    the ``cameras``, ``faces``, ``alerts`` and ``arming`` blocks, and only
    those whose values change; every other line, comments included, stays
    exactly as it was
  * nothing is written before the owner confirms; the same answers again
    plan no change and write nothing
  * apply writes only if the file is still the one the plan was made
    against, validates the whole result first, and backs up the file it
    replaces; ``restore`` puts that backup back the same way
  * ``update_live`` copies the new values into the running service's config
    objects, which the screener, pipeline, investigator and arming state
    hold, so the next event uses them without a restart
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from watchwise.config import WatchwiseConfig, interpolate_env
from watchwise.setup.plan import FileChange, apply, plan_file
from watchwise.setup.render import replace_top_level_block

HOME_BLOCKS = ("cameras", "faces", "alerts", "arming")
# What a regenerated config (`watchwise init`) carries over from the old one:
# everything the owner set by command or chat.
OWNER_BLOCKS = ("notify", *HOME_BLOCKS)

# A name as Frigate's Face Library and `/face` accept it.
NAME = re.compile(r"^[A-Za-z0-9_]{1,32}$")
NOTE_LIMIT = 300
ROOM_LIMIT = 40
RULE_LIMIT = 120
MAX_RULES = 8
DEFAULT_ARMING = "armed_away"
ARMING_LABELS = {
    "armed_home": "Home",
    "armed_away": "Away",
    "armed_night": "Night",
    "disarmed": "Off (disarmed)",
}

COMMENTS = {
    "cameras": (
        "# What each camera looks at, in the owner's words, and its room (from /setup in\n"
        "# Telegram; cameras in one room share one card per incident). The notes go into\n"
        "# the model prompts: with a cloud model, to its provider."
    ),
    "faces": "# Residents, as Frigate's face recognition names them.",
    "alerts": (
        "# What should always alert, in the owner's words (from /setup). The vision\n"
        "# model checks every event for these; with a cloud model, they go to its provider."
    ),
    "arming": "# Assumed while no arming state is set (/arm in Telegram, or POST /arming).",
}


class SetupError(RuntimeError):
    """The config file can't be planned against or written."""


class FileChanged(SetupError):
    """The file changed after the plan was made; nothing was written."""


@dataclass
class HomeAnswers:
    notes: dict[str, str] = field(default_factory=dict)  # camera -> what it looks at
    rooms: dict[str, str] = field(default_factory=dict)  # camera -> the room it watches
    household: list[str] = field(default_factory=list)
    always_alert: list[str] = field(default_factory=list)
    arming_default: str = DEFAULT_ARMING

    @classmethod
    def from_config(cls, config: WatchwiseConfig) -> "HomeAnswers":
        return cls(
            notes={name: cam.notes for name, cam in config.cameras.items() if cam.notes},
            rooms={name: cam.room for name, cam in config.cameras.items() if cam.room},
            household=list(config.faces.household),
            always_alert=list(config.alerts.always),
            arming_default=config.arming.default,
        )


@dataclass
class HomePlan:
    path: Path
    base_text: str  # the file as it was when the plan was made
    change: FileChange
    lines: list[str]  # every answer, in plain words, marked where it changes
    blocks: list[str] = field(default_factory=list)  # the top-level blocks that change

    @property
    def changes_anything(self) -> bool:
        return self.change.writes


# ---------------------------------------------------------------- answers


def clean_note(text: str) -> str:
    return " ".join(str(text).split())[:NOTE_LIMIT]


def clean_room(text: str) -> str:
    return " ".join(str(text).split())[:ROOM_LIMIT]


def clean_rules(lines: list[str]) -> list[str]:
    rules: list[str] = []
    for line in lines:
        rule = " ".join(str(line).split()).lstrip("-•*· ").strip()[:RULE_LIMIT]
        if rule and rule.lower() not in {r.lower() for r in rules}:
            rules.append(rule)
    return rules[:MAX_RULES]


def split_names(text: str) -> tuple[list[str], list[str]]:
    """Names typed by the owner: (usable, rejected)."""
    usable: list[str] = []
    rejected: list[str] = []
    for part in re.split(r"[,\n;]+", text):
        name = part.strip()
        if not name:
            continue
        if NAME.match(name):
            if name.lower() not in {n.lower() for n in usable}:
                usable.append(name)
        else:
            rejected.append(name)
    return usable, rejected


# ---------------------------------------------------------------- plan / apply


def validate(text: str) -> WatchwiseConfig:
    """The config the service would load from ``text`` (raises if it wouldn't)."""
    return WatchwiseConfig.model_validate(interpolate_env(yaml.safe_load(text) or {}))


def read_config(path: Path | None) -> tuple[str, WatchwiseConfig]:
    if path is None or not path.is_file():
        raise SetupError(
            "Watchwise is running without a config file, so there is nowhere to save "
            "answers. Run `watchwise init`, then start `watchwise serve --config <file>`."
        )
    text = path.read_text(encoding="utf-8")
    try:
        return text, validate(text)
    except (yaml.YAMLError, ValueError) as exc:
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise SetupError(f"{path.name} isn't a valid config right now ({first}); fix it by hand first.") from None


def plan_home(path: Path | None, answers: HomeAnswers) -> HomePlan:
    """What saving ``answers`` would change in the file. Writes nothing."""
    text, current = read_config(path)
    assert path is not None
    data = yaml.safe_load(text) or {}
    new_text = text
    blocks = _blocks(data, answers)
    for key, value in blocks.items():
        new_text = replace_top_level_block(new_text, key, value, comment=COMMENTS.get(key))
    validate(new_text)  # never propose a file the service would refuse
    change = plan_file(path, new_text, reason="what /setup was told")
    return HomePlan(path, text, change, describe(HomeAnswers.from_config(current), answers), list(blocks))


def apply_home(plan: HomePlan, *, now: float | None = None) -> WatchwiseConfig:
    """Write a confirmed plan (backing up the file) and return the new config.
    Raises FileChanged, writing nothing, if the file is no longer the one
    the plan was made against."""
    current = plan.path.read_text(encoding="utf-8") if plan.path.is_file() else None
    if current != plan.base_text:
        raise FileChanged(f"{plan.path.name} changed after the plan was made")
    if not plan.change.writes or plan.change.content is None:
        return validate(current)
    config = validate(plan.change.content)
    apply([plan.change], now=now)
    return config


def restore(path: Path, backup: Path, written: str, *, now: float | None = None) -> tuple[FileChange, WatchwiseConfig]:
    """Put back the file /setup replaced, if nobody changed it since."""
    current = path.read_text(encoding="utf-8") if path.is_file() else None
    if current != written:
        raise FileChanged(
            f"{path.name} changed after /setup saved it, so I left it alone. "
            f"The version from before /setup is {backup.name}."
        )
    if not backup.is_file():
        raise SetupError(f"The backup {backup.name} is gone, so there is nothing to restore.")
    text = backup.read_text(encoding="utf-8")
    config = validate(text)
    change = plan_file(path, text, reason="undo /setup")
    apply([change], now=now)
    return change, config


def update_live(config: WatchwiseConfig, new: WatchwiseConfig) -> None:
    """Copy the /setup blocks into the running service's config, in place:
    the parts of the service that use them hold these very objects."""
    for name in type(config.faces).model_fields:
        setattr(config.faces, name, getattr(new.faces, name))
    for name in type(config.alerts).model_fields:
        setattr(config.alerts, name, getattr(new.alerts, name))
    for name in type(config.arming).model_fields:
        setattr(config.arming, name, getattr(new.arming, name))
    config.cameras.clear()
    config.cameras.update(new.cameras)


def keep_owner_blocks(text: str, existing: str) -> str:
    """A regenerated config keeps what the owner set by command or chat
    (`watchwise telegram link`, /setup): those blocks are carried over."""
    try:
        old = yaml.safe_load(existing) or {}
    except yaml.YAMLError:
        return text
    if not isinstance(old, dict):
        return text
    for key in OWNER_BLOCKS:
        if old.get(key) is not None:
            text = replace_top_level_block(text, key, old[key], comment=COMMENTS.get(key))
    return text


def _blocks(data: dict[str, Any], answers: HomeAnswers) -> dict[str, Any]:
    """The blocks whose values change, merged with what else they hold."""
    cameras = {str(k): dict(v or {}) for k, v in (data.get("cameras") or {}).items()}
    for name in dict.fromkeys([*cameras, *answers.notes, *answers.rooms]):
        spec = cameras.setdefault(name, {})
        for key, value in (("notes", clean_note(answers.notes.get(name, ""))),
                           ("room", clean_room(answers.rooms.get(name, "")))):
            if value:
                spec[key] = value
            else:
                spec.pop(key, None)
        if not spec:
            del cameras[name]
    faces = dict(data.get("faces") or {})
    faces["household"] = list(answers.household)
    alerts = dict(data.get("alerts") or {})
    alerts["always"] = clean_rules(answers.always_alert)
    arming = dict(data.get("arming") or {})
    arming["default"] = answers.arming_default
    wanted = {
        "cameras": (cameras, bool(cameras)),
        "faces": (faces, bool(answers.household)),
        "alerts": (alerts, bool(alerts["always"])),
        "arming": (arming, answers.arming_default != DEFAULT_ARMING),
    }
    blocks: dict[str, Any] = {}
    for key, (value, worth_adding) in wanted.items():
        if (key in data and data[key] != value) or (key not in data and worth_adding):
            blocks[key] = value
    return blocks


def describe(old: HomeAnswers, new: HomeAnswers) -> list[str]:
    """Every answer in plain words (not HTML), saying what changes."""
    lines: list[str] = []
    for camera in dict.fromkeys([*new.notes, *new.rooms, *old.notes, *old.rooms]):
        before, after = clean_note(old.notes.get(camera, "")), clean_note(new.notes.get(camera, ""))
        old_room, room = clean_room(old.rooms.get(camera, "")), clean_room(new.rooms.get(camera, ""))
        if not (before or after or old_room or room):
            continue
        line = f"📷 {camera}" + (f" · room {room}" if room else "") + (f": “{after}”" if after else ": no notes")
        marks = []
        if before and not after:
            marks.append(f"− “{before}”")
        if old_room and not room:
            marks.append(f"− room {old_room}")
        if (after and after != before) or (room and room.lower() != old_room.lower()):
            marks.append("new")
        lines.append(line + (f" ({', '.join(marks)})" if marks else ""))
    added = [n for n in new.household if n.lower() not in {o.lower() for o in old.household}]
    removed = [n for n in old.household if n.lower() not in {h.lower() for h in new.household}]
    change = ", ".join([*(f"+ {n}" for n in added), *(f"− {n}" for n in removed)])
    lines.append(f"👪 Residents: {', '.join(new.household) or 'none'}" + (f" ({change})" if change else ""))
    rules = clean_rules(new.always_alert)
    old_rules = {r.lower() for r in clean_rules(old.always_alert)}
    if rules:
        lines.append("🚨 Always alert for:")
        lines += [f"   • {r}" + ("" if r.lower() in old_rules else " (new)") for r in rules]
    else:
        lines.append("🚨 Always alert for: nothing extra")
    for gone in clean_rules(old.always_alert):
        if gone.lower() not in {r.lower() for r in rules}:
            lines.append(f"   − {gone}")
    arming = f"🔒 When no arming state is set: {ARMING_LABELS.get(new.arming_default, new.arming_default)}"
    if new.arming_default != old.arming_default:
        arming += f" (was {ARMING_LABELS.get(old.arming_default, old.arming_default)})"
    lines.append(arming)
    return lines
