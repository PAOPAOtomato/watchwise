"""Load the local models before the first alert needs them.

Ollama loads a model on its first request and unloads it after ``keep_alive``
without one. Loading means reading gigabytes from disk, which can take longer
than the screening timeout, so a cold model turns the first alert, and the
first alert after a quiet night, into "uncertain" for no reason but timing.

At start-up the warmer loads the screener, then the other local models, then
the screener again: if they don't all fit in memory, the model on the alert
path is the one left loaded. While the service runs it touches the screener
every half ``keep_alive`` so that it stays loaded. All of this runs in the
background and never delays start-up or an alert.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from watchwise.llm.base import ChatModel
from watchwise.tier2.vlm.base import VLMBackend

logger = logging.getLogger(__name__)

# What Ollama uses when a request sets no keep_alive.
OLLAMA_DEFAULT_KEEP_ALIVE_S = 300.0
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
_UNIT_S = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def keep_alive_seconds(value: str | None) -> float | None:
    """Ollama's ``keep_alive`` ("30m", "1h30m", "300", "-1") in seconds.

    None means the model is never unloaded. A value that can't be read counts
    as Ollama's default.
    """
    text = (value or "").strip()
    if not text:
        return OLLAMA_DEFAULT_KEEP_ALIVE_S
    try:
        seconds = float(text)
    except ValueError:
        body = text.lstrip("-")
        parts = _DURATION.findall(body)
        if not parts or "".join(n + u for n, u in parts) != body:
            return OLLAMA_DEFAULT_KEEP_ALIVE_S
        seconds = sum(float(n) * _UNIT_S[u] for n, u in parts)
        if text.startswith("-"):
            seconds = -seconds
    return None if seconds < 0 else seconds


def refresh_interval(keep_alive: str | None) -> float | None:
    """How often to touch the screener so that Ollama keeps it loaded."""
    seconds = keep_alive_seconds(keep_alive)
    if not seconds:
        return None  # never unloaded, or unloaded after every request anyway
    return max(30.0, seconds / 2)


class ModelWarmer:
    """A background worker with ``start()`` / ``stop()``."""

    def __init__(
        self,
        screener: VLMBackend,
        others: list[tuple[str, ChatModel]] | None = None,
        *,
        refresh_s: float | None = None,
    ):
        self.screener = screener
        self.others = list(others or [])
        self.refresh_s = refresh_s
        # role -> {"model", "ready", "load_s", "error"}, for /health. A role
        # whose model needs no loading (cloud, stand-in) has no entry.
        self.status: dict[str, dict[str, Any]] = {}
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="model-warmup")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def warm(self) -> None:
        """Load each local model once: the screener first, and last again."""
        await self._load_screener()
        loaded_other = False
        for role, chat in self.others:
            loaded_other |= await self._load(role, chat.describe(), chat.load)
        if loaded_other and self.status.get("screener", {}).get("ready"):
            await self._load_screener(quiet=True)

    async def _run(self) -> None:
        await self.warm()
        if "screener" not in self.status:
            return
        while self.refresh_s:
            await asyncio.sleep(self.refresh_s)
            await self._load_screener(quiet=True)

    async def _load_screener(self, *, quiet: bool = False) -> bool:
        vlm = self.screener
        vlm.warming = True
        try:
            return await self._load("screener", f"{vlm.name}/{vlm.model}", vlm.load, quiet=quiet)
        finally:
            vlm.warming = False

    async def _load(
        self, role: str, what: str, load: Callable[[], Awaitable[bool]], *, quiet: bool = False
    ) -> bool:
        before = self.status.get(role, {})
        started = time.perf_counter()
        try:
            needed = await load()
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            if before.get("error") != error:  # once, not on every refresh
                logger.warning(
                    "warm-up: could not load the %s model %s: %s. The first alert may be "
                    "slow or come back uncertain",
                    role, what, error,
                )
            self.status[role] = {"model": what, "ready": False, "load_s": None, "error": error}
            return False
        if not needed:
            return False
        seconds = round(time.perf_counter() - started, 1)
        if quiet and before.get("ready") and seconds < 1.0:
            return True  # still loaded
        logger.info(
            "warm-up: %s model %s %s after %.1fs",
            role, what, "reloaded" if before.get("ready") else "ready", seconds,
        )
        self.status[role] = {"model": what, "ready": True, "load_s": seconds, "error": None}
        return True
