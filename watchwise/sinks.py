"""Where decisions go.

Sinks are how watchwise gives its answer back without touching NVR's internals:

  MemorySink   -- in-process ring buffer, backs the HTTP API and the tests
  MQTTSink     -- publishes to watchwise/... topics alongside NVR's own, so
                  Home Assistant or any existing automation can consume the
                  refined verdict instead of the raw one
  WebhookSink  -- POSTs to an arbitrary URL
  LogSink      -- structured log line, always on

Note what is deliberately *not* here: writing back into NVR's database or
overriding its review severity. NVR owns that record. Watchwise publishes a
parallel, additive verdict and lets the consumer decide which to believe --
which also means removing watchwise leaves NVR working exactly as before.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any

from watchwise.types import AlertContext, Decision

logger = logging.getLogger(__name__)


class MemorySink:
    """Keeps the last N decisions for the API and for assertions in tests."""

    def __init__(self, maxlen: int = 500):
        self.decisions: deque[tuple[AlertContext, Decision]] = deque(maxlen=maxlen)

    async def __call__(self, ctx: AlertContext, decision: Decision) -> None:
        self.decisions.append((ctx, decision))

    def latest(self, review_id: str) -> Decision | None:
        for _, d in reversed(self.decisions):
            if d.review_id == review_id:
                return d
        return None

    def all_for(self, review_id: str) -> list[Decision]:
        return [d for _, d in self.decisions if d.review_id == review_id]

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        items = list(self.decisions)[-limit:]
        return [
            {**d.to_dict(), "camera": ctx.camera, "labels": ctx.labels}
            for ctx, d in reversed(items)
        ]


class LogSink:
    async def __call__(self, ctx: AlertContext, decision: Decision) -> None:
        logger.info(
            "%s %s/%s -> %s (%.2f) [%s] %s",
            decision.stage.value,
            ctx.camera,
            ctx.review_id,
            decision.disposition.value,
            decision.confidence,
            ",".join(ctx.labels) or "-",
            decision.rationale[:160],
        )


class MQTTSink:
    """Publishes verdicts on a topic tree parallel to NVR's own."""

    def __init__(self, client: Any, topic_prefix: str = "watchwise"):
        self.client = client
        self.topic_prefix = topic_prefix.rstrip("/")

    async def __call__(self, ctx: AlertContext, decision: Decision) -> None:
        payload = {
            **decision.to_dict(),
            "camera": ctx.camera,
            "labels": ctx.labels,
            "zones": ctx.zones,
            "start_time": ctx.start_time,
        }
        body = json.dumps(payload, default=str)
        for topic in (
            f"{self.topic_prefix}/verdicts",
            f"{self.topic_prefix}/{ctx.camera}/verdict",
        ):
            self.client.publish(topic, body, qos=0, retain=False)


class MQTTPublisher:
    """A paho client that stays connected for publishing (reconnects itself).

    ``MQTTSink`` needs only ``publish``; tests pass any object that has one.
    """

    def __init__(self, settings: Any):
        import paho.mqtt.client as mqtt  # imported lazily; optional at runtime

        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=f"{settings.client_id}-publisher"
        )
        if settings.username:
            self.client.username_pw_set(settings.username, settings.password)
        self.client.connect_async(settings.host, settings.port, settings.keepalive)
        self.client.loop_start()

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False) -> Any:
        return self.client.publish(topic, payload, qos=qos, retain=retain)

    async def aclose(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()


class WebhookSink:
    def __init__(self, url: str, timeout_s: float = 5.0):
        import httpx

        self.url = url
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def __call__(self, ctx: AlertContext, decision: Decision) -> None:
        await self._client.post(
            self.url,
            json={**decision.to_dict(), "camera": ctx.camera, "labels": ctx.labels},
        )

    async def aclose(self) -> None:
        await self._client.aclose()
