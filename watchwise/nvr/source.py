"""Alert sources: where candidate alerts come from.

Tier 1 is NVR. It already decided that something happened and wrote a
ReviewSegment; we subscribe to that decision rather than re-doing motion or
object detection. Two implementations:

  MQTTAlertSource  -- production: subscribes to <prefix>/reviews
  ReplayAlertSource -- offline: replays recorded review payloads from JSON

Both normalize into ``AlertContext`` and push onto the same asyncio.Queue, so
nothing downstream knows or cares which one is running.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from watchwise.config import MQTTSettings
from watchwise.types import AlertContext, ReviewSummary, Severity

logger = logging.getLogger(__name__)


def review_payload_to_context(payload: dict[str, Any]) -> AlertContext | None:
    """Normalize one NVR review-update payload into an AlertContext.

    NVR publishes ``{"type": "new"|"update"|"end", "before": {...},
    "after": {...}}``. We key off ``after`` because it carries the most
    complete label/zone set at that point in time.
    """
    after = payload.get("after") or payload.get("before")
    if not after:
        return None

    data = after.get("data") or {}
    severity_raw = (after.get("severity") or "alert").lower()
    try:
        severity = Severity(severity_raw)
    except ValueError:
        severity = Severity.ALERT

    return AlertContext(
        summary=_summary(data.get("metadata")),
        review_id=str(after.get("id")),
        camera=str(after.get("camera") or "unknown"),
        start_time=float(after.get("start_time") or 0.0),
        end_time=float(after["end_time"]) if after.get("end_time") else None,
        severity=severity,
        # The NVR calls a person it has named "person-verified" (B-025); it is
        # still a person, for memory and for the arming rules.
        labels=list(dict.fromkeys(str(o).removesuffix("-verified") for o in data.get("objects") or [])),
        sub_labels=list(data.get("sub_labels") or []),
        zones=list(data.get("zones") or []),
        event_ids=list(data.get("detections") or []),
        thumb_path=after.get("thumb_path"),
        raw=payload,
    )


def _summary(metadata: Any) -> ReviewSummary | None:
    """Frigate's GenAI review summary (``data.metadata``), if it has one."""
    if not isinstance(metadata, dict) or not (metadata.get("shortSummary") or metadata.get("title")):
        return None
    try:
        level = max(0, min(2, int(metadata.get("potential_threat_level") or 0)))
        confidence = max(0.0, min(1.0, float(metadata.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        level, confidence = 0, 0.0
    return ReviewSummary(
        title=str(metadata.get("title") or ""),
        short_summary=str(metadata.get("shortSummary") or ""),
        threat_level=level,
        confidence=confidence,
        concerns=[str(c) for c in metadata.get("other_concerns") or []],
    )


class AlertSource(ABC):
    """Pushes AlertContext objects onto a queue until stopped."""

    def __init__(self, queue: "asyncio.Queue[AlertContext]"):
        self.queue = queue
        self._task: asyncio.Task[None] | None = None

    @abstractmethod
    async def run(self) -> None: ...

    async def start(self) -> None:
        self._task = asyncio.create_task(self.run(), name=type(self).__name__)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _emit(self, ctx: AlertContext) -> None:
        try:
            self.queue.put_nowait(ctx)
        except asyncio.QueueFull:
            # Never block tier 1. Dropping the oldest is the right trade:
            # a stale unscreened alert is worth less than a fresh one.
            with contextlib.suppress(asyncio.QueueEmpty):
                dropped = self.queue.get_nowait()
                logger.warning("screening queue full, dropped %s", dropped.review_id)
            with contextlib.suppress(asyncio.QueueFull):
                self.queue.put_nowait(ctx)


class MQTTAlertSource(AlertSource):
    """Subscribes to NVR's review topic over MQTT.

    By default an event is screened when it starts (``new``), so the verdict
    does not wait for it to end, and its ``end`` report is passed on too: the
    pipeline calls the model again only if the finished segment shows a new
    label or zone. ``update`` reports are skipped, since screening a segment
    on every change would burn a model call per update, except when the NVR
    has just named someone in it (a new sub-label, D-057). ``genai`` reports
    (the NVR's own summary of a finished segment) always pass through.

    With ``on_face``, the NVR's face-recognition attempts
    (``<prefix>/tracked_object_update``, type ``face``) are handed to it.
    """

    def __init__(
        self,
        queue: "asyncio.Queue[AlertContext]",
        settings: MQTTSettings,
        *,
        screen_on: Iterable[str] = ("new", "end"),
        severities: Iterable[str] = ("alert",),
        on_face: Callable[[dict[str, Any]], None] | None = None,
    ):
        super().__init__(queue)
        self.settings = settings
        self.screen_on = set(screen_on)
        self.severities = set(severities)
        self.on_face = on_face
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None

    @property
    def topic(self) -> str:
        return f"{self.settings.topic_prefix}/reviews"

    @property
    def face_topic(self) -> str:
        return f"{self.settings.topic_prefix}/tracked_object_update"

    async def run(self) -> None:
        import paho.mqtt.client as mqtt  # imported lazily; optional dependency

        self._loop = asyncio.get_running_loop()
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=self.settings.client_id
        )
        if self.settings.username:
            client.username_pw_set(self.settings.username, self.settings.password)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        self._client = client

        client.connect_async(self.settings.host, self.settings.port, self.settings.keepalive)
        client.loop_start()
        logger.info("MQTT alert source connecting to %s:%s", self.settings.host, self.settings.port)
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            client.loop_stop()
            client.disconnect()

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: Any, props: Any = None) -> None:
        client.subscribe(self.topic)
        logger.info("subscribed to %s", self.topic)
        if self.on_face is not None:
            client.subscribe(self.face_topic)
            logger.info("subscribed to %s", self.face_topic)

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        # Runs on paho's thread; hop back to the event loop.
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            logger.warning("undecodable review payload on %s", msg.topic)
            return
        if msg.topic == self.face_topic:
            if self.on_face is not None and payload.get("type") == "face" and self._loop:
                self._loop.call_soon_threadsafe(self.on_face, payload)
            return
        if not self.wanted(payload):
            return
        ctx = review_payload_to_context(payload)
        if ctx is None or ctx.severity.value not in self.severities:
            return
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._emit(ctx), self._loop)

    def wanted(self, payload: dict[str, Any]) -> bool:
        kind = payload.get("type")
        if kind in self.screen_on or kind == "genai":
            return True
        return kind == "update" and _sub_labels(payload, "after") != _sub_labels(payload, "before")


def _sub_labels(payload: dict[str, Any], side: str) -> list[str]:
    return list(((payload.get(side) or {}).get("data") or {}).get("sub_labels") or [])


class ReplayAlertSource(AlertSource):
    """Replays a JSON file of review payloads. Used for offline testing.

    The file is either a JSON array of review-update payloads, or JSON lines.
    ``speed`` scales the original inter-arrival gaps (0 = as fast as possible).
    """

    def __init__(
        self,
        queue: "asyncio.Queue[AlertContext]",
        path: str | Path,
        *,
        speed: float = 0.0,
        loop_forever: bool = False,
    ):
        super().__init__(queue)
        self.path = Path(path)
        self.speed = speed
        self.loop_forever = loop_forever

    def _load(self) -> list[dict[str, Any]]:
        text = self.path.read_text(encoding="utf-8").strip()
        if text.startswith("["):
            return json.loads(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    async def run(self) -> None:
        payloads = self._load()
        while True:
            prev_ts: float | None = None
            for payload in payloads:
                ctx = review_payload_to_context(payload)
                if ctx is None:
                    continue
                if self.speed > 0 and prev_ts is not None:
                    gap = max(0.0, ctx.start_time - prev_ts) / self.speed
                    await asyncio.sleep(min(gap, 10.0))
                prev_ts = ctx.start_time
                await self._emit(ctx)
            if not self.loop_forever:
                return
            await asyncio.sleep(1.0)
