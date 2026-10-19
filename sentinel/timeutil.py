"""Timestamp helpers that tolerate bad input from upstream.

Tier 1 payloads are not always well formed -- a truncated MQTT message or a
manual event can carry a zero, negative or absurd ``start_time``. Python's
``datetime.fromtimestamp`` raises on those (OSError on Windows, ValueError or
OverflowError elsewhere), and a screening or forensic component must never
die because of it.
"""

from __future__ import annotations

import datetime as dt


def safe_datetime(ts: float, tz: dt.tzinfo | None = None) -> dt.datetime | None:
    try:
        return dt.datetime.fromtimestamp(ts, tz)
    except (OSError, ValueError, OverflowError):
        return None


def safe_strftime(
    ts: float, fmt: str = "%H:%M:%S", tz: dt.tzinfo | None = None
) -> str | None:
    moment = safe_datetime(ts, tz)
    return moment.strftime(fmt) if moment else None


def safe_hour(ts: float, default: int = 12) -> int:
    moment = safe_datetime(ts)
    return moment.hour if moment else default
