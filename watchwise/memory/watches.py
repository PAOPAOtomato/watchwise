""""Tell me when..." watches (D-065).

The owner asks in chat ("tell me when the parcel is picked up") and confirms;
from then on the vision model checks the request on every event from the
cameras it names, until one matches or the watch expires. A match alerts (it
can only raise an event, like an always-alert rule) and ends the watch.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass

from watchwise.memory.store import Store

MAX_HOURS = 7 * 24


@dataclass(frozen=True)
class Watch:
    id: int
    request: str
    cameras: list[str]  # empty: every camera
    expires_at: float


class WatchBook:
    def __init__(self, store: Store, *, clock: Callable[[], float] = time.time):
        self.store = store
        self.clock = clock

    def add(self, request: str, cameras: list[str], hours: float) -> int:
        now = self.clock()
        hours = max(0.1, min(float(hours), MAX_HOURS))
        cur = self.store.execute(
            "INSERT INTO watches(request, cameras, created_at, expires_at) VALUES (?,?,?,?)",
            (request, json.dumps(list(cameras)), now, now + hours * 3600),
        )
        return int(cur.lastrowid)

    def active(self, camera: str | None = None) -> list[Watch]:
        """Watches still running (all, or those covering ``camera``)."""
        rows = self.store.query(
            "SELECT * FROM watches WHERE status='active' AND expires_at>? ORDER BY id", (self.clock(),)
        )
        watches = [
            Watch(int(r["id"]), str(r["request"]), list(json.loads(r["cameras"] or "[]")), float(r["expires_at"]))
            for r in rows
        ]
        return [w for w in watches if camera is None or not w.cameras or camera in w.cameras]

    def fulfil(self, watch_ids: list[int], review_id: str) -> None:
        for watch_id in watch_ids:
            self.store.execute(
                "UPDATE watches SET status='done', review_id=? WHERE id=? AND status='active'",
                (review_id, int(watch_id)),
            )

    def cancel(self, watch_id: int) -> bool:
        cur = self.store.execute("UPDATE watches SET status='cancelled' WHERE id=? AND status='active'", (int(watch_id),))
        return cur.rowcount > 0
