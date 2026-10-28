"""Add one rule to the duty roster file, plan -> confirm -> apply (D-065).

A rule the owner confirmed in chat ("the cleaner comes Thursdays 1-4 pm") is
appended to the ``entries:`` list as text, so every other line and comment
stays as it was; the file is backed up first (``setup.plan.apply``). The
roster reloads itself when the file changes, so the next event uses it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from watchwise.setup.plan import FileChange, plan_file

HEADER = (
    "# The duty-roster rule base: what is EXPECTED, so an investigation can tell\n"
    "# a rostered visit from an intruder. Edited live; reloaded on change.\n"
)


def plan_roster_entry(path: Path, entry: dict[str, Any], *, timezone: str = "UTC") -> FileChange:
    """What adding ``entry`` would change. Writes nothing."""
    if not path.exists():
        text = HEADER + yaml.safe_dump({"timezone": timezone, "entries": [entry]}, sort_keys=False, width=100)
        return plan_file(path, text, reason="a duty-roster rule confirmed in chat")
    current = path.read_text(encoding="utf-8")
    data = yaml.safe_load(current) or {}
    entries = data.get("entries") if isinstance(data, dict) else None
    text = _appended(current, entry) if isinstance(entries, list) else None
    if text is None or (yaml.safe_load(text) or {}).get("entries") != [*(entries or []), entry]:
        # No plain `entries:` list to append to: write the whole file again
        # (the diff in the plan shows exactly what changes).
        data = dict(data) if isinstance(data, dict) else {}
        data.setdefault("timezone", timezone)
        data["entries"] = [*(entries or []), entry]
        text = HEADER + yaml.safe_dump(data, sort_keys=False, width=100)
    return plan_file(path, text, reason="a duty-roster rule confirmed in chat")


def _appended(text: str, entry: dict[str, Any]) -> str | None:
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if re.match(r"^entries:\s*(#.*)?$", line)), None)
    if start is None:
        return None
    end, indent = start + 1, None
    while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t-"):
        item = re.match(r"^(\s*)- ", lines[end])
        if indent is None and item:
            indent = item.group(1)
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    block = yaml.safe_dump([entry], sort_keys=False, default_flow_style=False, width=100)
    pad = indent if indent is not None else "  "
    block = "".join(pad + line if line.strip() else line for line in block.splitlines(keepends=True))
    before = "".join(lines[:end])
    if before and not before.endswith("\n"):
        before += "\n"
    return before + block + "".join(lines[end:])
