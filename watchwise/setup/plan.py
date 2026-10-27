"""Plan -> confirm -> apply, for every file setup touches.

Rules (borrowed from REA's setup discipline):
  * show the full plan, including diffs, before writing anything
  * additive: never delete, and never touch a secrets file that exists
  * idempotent: identical content is reported as "unchanged" and not written
  * reversible: an overwritten file is first copied to <name>.bak-<timestamp>
"""

from __future__ import annotations

import difflib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Action = Literal["create", "update", "unchanged", "keep"]


@dataclass
class FileChange:
    path: Path
    action: Action
    content: str | None
    reason: str = ""
    diff: str = ""
    backup: Path | None = None

    @property
    def writes(self) -> bool:
        return self.action in ("create", "update")


def plan_file(path: Path, content: str, *, reason: str = "", overwrite: bool = True) -> FileChange:
    """What writing ``content`` to ``path`` would do. Writes nothing."""
    if not path.exists():
        return FileChange(path, "create", content, reason)
    current = path.read_text(encoding="utf-8")
    if current == content:
        return FileChange(path, "unchanged", None, reason)
    if not overwrite:
        return FileChange(path, "keep", None, reason or "exists; left as it is")
    diff = "".join(
        difflib.unified_diff(
            current.splitlines(keepends=True),
            content.splitlines(keepends=True),
            fromfile=f"{path} (current)",
            tofile=f"{path} (proposed)",
        )
    )
    return FileChange(path, "update", content, reason, diff)


def apply(changes: list[FileChange], *, now: float | None = None) -> list[FileChange]:
    """Write the planned changes, backing up anything overwritten."""
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now if now is not None else time.time()))
    for change in changes:
        if not change.writes or change.content is None:
            continue
        change.path.parent.mkdir(parents=True, exist_ok=True)
        if change.action == "update" and change.path.exists():
            backup = change.path.with_name(f"{change.path.name}.bak-{stamp}")
            n = 1
            while backup.exists():  # two writes in one second keep both backups
                backup = change.path.with_name(f"{change.path.name}.bak-{stamp}-{n}")
                n += 1
            shutil.copy2(change.path, backup)
            change.backup = backup
        tmp = change.path.with_name(f".{change.path.name}.tmp")
        tmp.write_text(change.content, encoding="utf-8", newline="\n")
        os.replace(tmp, change.path)
    return changes
