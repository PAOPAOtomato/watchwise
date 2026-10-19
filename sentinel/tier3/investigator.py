"""Runs one investigation end to end, and the async worker pool behind it.

The critical property here is that tier 3 is *off the critical path*. An
alert is already dispatched by the time an investigation starts; the
investigation refines the record afterwards. So:

  * the queue is bounded and drops the oldest item when full
  * every investigation has a wall-clock timeout
  * a backend failure downgrades to "uncertain", which routes to a human,
    rather than raising into the alert path
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable

from sentinel.config import AgentSettings, MemorySettings
from sentinel.memory.short_term import ShortTermMemory
from sentinel.memory.store import Store
from sentinel.tier3.backends.base import AgentBackend, ToolRunner
from sentinel.tools.base import ToolContext
from sentinel.types import (
    AlertContext,
    Investigation,
    ScreenResult,
    Verdict,
    new_id,
)

logger = logging.getLogger(__name__)

InvestigationDone = Callable[[AlertContext, Investigation], Awaitable[None]]


class Investigator:
    """Executes a single investigation."""

    def __init__(
        self,
        backend: AgentBackend,
        tool_ctx: ToolContext,
        store: Store,
        agent_settings: AgentSettings,
        memory_settings: MemorySettings,
    ):
        self.backend = backend
        self.tool_ctx = tool_ctx
        self.store = store
        self.settings = agent_settings
        self.memory_settings = memory_settings

    async def investigate(
        self, ctx: AlertContext, screen: ScreenResult | None
    ) -> Investigation:
        inv = Investigation(
            id=new_id("inv"),
            review_id=ctx.review_id,
            camera=ctx.camera,
            status="running",
            backend=self.backend.name,
            model=self.backend.model,
            started_at=time.time(),
        )
        self.store.upsert_investigation(inv.to_dict())

        memory = ShortTermMemory(
            max_items=self.memory_settings.short_term_max_items,
            max_chars=self.memory_settings.short_term_max_chars,
        )
        memory.seed(ctx, screen)
        runner = ToolRunner(
            self.tool_ctx,
            memory,
            inv,
            max_calls=max(4, self.settings.max_iterations * 2),
        )

        try:
            conclusion = await asyncio.wait_for(
                self.backend.run(ctx, screen, memory, runner),
                timeout=self.settings.timeout_s,
            )
            inv.verdict = conclusion.verdict
            inv.confidence = conclusion.confidence
            inv.conclusion = conclusion.conclusion
            if conclusion.recommended_action and conclusion.recommended_action != "none":
                inv.conclusion += f" Recommended action: {conclusion.recommended_action}"
            inv.status = "done"
        except TimeoutError:
            inv.status = "failed"
            inv.verdict = Verdict.UNCERTAIN
            inv.confidence = 0.0
            inv.error = f"timed out after {self.settings.timeout_s:.0f}s"
            inv.conclusion = (
                "Investigation timed out before reaching a conclusion; escalating "
                "to human review."
            )
            logger.warning("investigation %s timed out", inv.id)
        except Exception as exc:
            inv.status = "failed"
            inv.verdict = Verdict.UNCERTAIN
            inv.confidence = 0.0
            inv.error = f"{type(exc).__name__}: {exc}"
            inv.conclusion = f"Investigation failed ({exc}); escalating to human review."
            logger.exception("investigation %s failed", inv.id)
        finally:
            inv.finished_at = time.time()
            inv.iterations = max(inv.iterations, runner.iteration)
            self.store.upsert_investigation(inv.to_dict())

        return inv


class InvestigationQueue:
    """Bounded queue plus a worker pool, so tier 3 never blocks tier 1/2."""

    def __init__(
        self,
        investigator: Investigator,
        settings: AgentSettings,
        on_done: InvestigationDone | None = None,
    ):
        self.investigator = investigator
        self.settings = settings
        self.on_done = on_done
        self.queue: asyncio.Queue[tuple[AlertContext, ScreenResult | None]] = asyncio.Queue(
            maxsize=settings.queue_size
        )
        self._workers: list[asyncio.Task[None]] = []
        self.stats = {"submitted": 0, "completed": 0, "dropped": 0, "failed": 0}

    def submit(self, ctx: AlertContext, screen: ScreenResult | None) -> bool:
        """Non-blocking. Returns False if the item had to be dropped."""
        try:
            self.queue.put_nowait((ctx, screen))
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                dropped, _ = self.queue.get_nowait()
                self.queue.task_done()
                self.stats["dropped"] += 1
                logger.warning(
                    "investigation queue full; dropped oldest (%s)", dropped.review_id
                )
            try:
                self.queue.put_nowait((ctx, screen))
            except asyncio.QueueFull:
                self.stats["dropped"] += 1
                return False
        self.stats["submitted"] += 1
        return True

    async def start(self) -> None:
        for i in range(max(1, self.settings.max_concurrency)):
            self._workers.append(
                asyncio.create_task(self._worker(), name=f"investigation-worker-{i}")
            )

    async def stop(self) -> None:
        for task in self._workers:
            task.cancel()
        for task in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._workers.clear()

    async def drain(self, timeout: float = 60.0) -> None:
        """Wait for the queue to empty. Used by tests and graceful shutdown."""
        await asyncio.wait_for(self.queue.join(), timeout=timeout)

    async def _worker(self) -> None:
        while True:
            ctx, screen = await self.queue.get()
            try:
                inv = await self.investigator.investigate(ctx, screen)
                if inv.status == "failed":
                    self.stats["failed"] += 1
                else:
                    self.stats["completed"] += 1
                if self.on_done is not None:
                    await self.on_done(ctx, inv)
            except asyncio.CancelledError:
                self.queue.task_done()
                raise
            except Exception:
                # A worker must outlive any single bad alert.
                self.stats["failed"] += 1
                logger.exception("investigation worker error on %s", ctx.review_id)
            self.queue.task_done()
