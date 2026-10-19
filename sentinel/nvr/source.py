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
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from sentinel.config import MQTTSettings
from sentinel.types import AlertContext, Severity

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
        review_id=str(after.get("id")),
        camera=str(after.get("camera") or "unknown"),
        start_time=float(after.get("start_time") or 0.0),
        end_time=float(after["end_time"]) if after.get("end_time") else None,
        severity=severity,
        labels=list(data.get("objects") or []),
        sub_labels=list(data.get("sub_labels") or []),
        zones=list(data.get("zones") or []),
        event_ids=list(data.get("detections") or []),
        thumb_path=after.get("thumb_path"),
        raw=payload,
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

    Only ``type == "end"`` segments are screened by default: a finished
    segment has its final label and zone set, and screening a segment that is
    still growing would burn a model call per update.
    """

    def __init__(
        self,
        queue: "asyncio.Queue[AlertContext]",
        settings: MQTTSettings,
        *,
        screen_on: Iterable[str] = ("end",),
        severities: Iterable[str] = ("alert",),
    ):
        super().__init__(queue)
        self.settings = settings
        self.screen_on = set(screen_on)
        self.severities = set(severities)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client: Any = None

    @property
    def topic(self) -> str:
        return f"{self.settings.topic_prefix}/reviews"

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

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        # Runs on paho's thread; hop back to the event loop.
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            logger.warning("undecodable review payload on %s", msg.topic)
            return
        if payload.get("type") not in self.screen_on:
            return
        ctx = review_payload_to_context(payload)
        if ctx is None or ctx.severity.value not in self.severities:
            return
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._emit(ctx), self._loop)


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
