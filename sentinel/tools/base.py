"""Transport-agnostic forensic tools.

A tool is defined once here, then exposed three ways without being rewritten:

  * to the Claude Agent SDK, as an in-process MCP server
  * to any MCP client, as a stdio MCP server
  * to the scripted backend and the tests, as a plain awaitable

That separation is deliberate. The tool logic is the valuable part and it
should not be coupled to a particular agent framework.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from sentinel.memory.long_term import PatternLibrary
from sentinel.memory.store import Store
from sentinel.nvr.client import NVRClient

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Everything the forensic tools are allowed to touch.

    Passing this explicitly (rather than reaching for globals) is what makes
    the tools testable against a fake NVR and an in-memory store.
    """

    nvr: NVRClient
    store: Store
    patterns: PatternLibrary
    roster: Any | None = None
    cameras: list[str] = field(default_factory=list)
    config_cache: dict[str, Any] = field(default_factory=dict)

    async def camera_names(self) -> list[str]:
        if self.cameras:
            return self.cameras
        try:
            self.cameras = await self.nvr.cameras()
        except Exception as exc:
            logger.warning("could not list cameras from NVR: %s", exc)
            self.cameras = []
        return self.cameras


@dataclass(slots=True)
class ForensicTool:
    """One callable capability offered to the agent."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[ToolContext, dict[str, Any]], Awaitable[dict[str, Any]]]
    category: str = "general"

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            result = await self.handler(ctx, args)
        except Exception as exc:
            logger.exception("tool %s failed", self.name)
            return {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
            }
        result.setdefault("ok", True)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        return result


def obj_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def summarize(result: dict[str, Any], limit: int = 400) -> str:
    """A one-line rendering of a tool result, for the ReAct trace."""
    import json

    if not result.get("ok", True):
        return f"ERROR: {result.get('error')}"
    payload = {k: v for k, v in result.items() if k not in ("ok", "latency_ms")}
    text = json.dumps(payload, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."
