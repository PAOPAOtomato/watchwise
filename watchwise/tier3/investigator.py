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

from watchwise.config import AgentSettings, AlertRules, CameraSettings, MemorySettings
from watchwise.memory.short_term import ShortTermMemory
from watchwise.memory.store import Store
from watchwise.tier3.backends.base import AgentBackend, ToolRunner
from watchwise.tools.base import ToolContext
from watchwise.types import (
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
        *,
        cameras: dict[str, CameraSettings] | None = None,
        alerts: AlertRules | None = None,
    ):
        self.backend = backend
        self.tool_ctx = tool_ctx
        self.store = store
        self.settings = agent_settings
        self.memory_settings = memory_settings
        # What the owner said in /setup (D-060), updated in place by it.
        self.cameras = cameras if cameras is not None else {}
        self.alerts = alerts if alerts is not None else AlertRules()

    def owner_context(self, ctx: AlertContext) -> str:
        """What the owner said about this camera, and what they always want
        an alert for, pinned for the whole investigation."""
        parts: list[str] = []
        spec = self.cameras.get(ctx.camera)
        if spec is not None and spec.notes:
            parts.append(f'The owner says camera {ctx.camera} looks at: "{spec.notes}".')
        if self.alerts.always:
            rules = "; ".join(f'"{rule}"' for rule in self.alerts.always)
            parts.append(
                f"The owner always wants an alert for: {rules}. If this event shows one of "
                "these, conclude suspicious, even if it looks harmless."
            )
        return " ".join(parts)

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
        owner = self.owner_context(ctx)
        if owner:
            memory.add("owner", owner, pinned=True)
        runner = ToolRunner(
            self.tool_ctx,
            memory,
            inv,
            max_calls=max(4, self.settings.max_iterations * 2),
        )

        try:
            # asyncio.timeout, not wait_for: on Python 3.11, wait_for can
            # swallow a cancellation that arrives as the work finishes, which
            # left a worker running after stop() and hung shutdown.
            async with asyncio.timeout(self.settings.timeout_s):
                conclusion = await self.backend.run(ctx, screen, memory, runner)
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

    async def stop(self, *, grace_s: float = 5.0) -> None:
        """Cancel the workers and wait until every one of them has exited.

        A cancellation can be absorbed by code running inside a worker, so a
        worker still alive after ``grace_s`` is cancelled again rather than
        awaited forever.
        """
        pending = {t for t in self._workers if not t.done()}
        while pending:
            for task in pending:
                task.cancel()
            _, pending = await asyncio.wait(pending, timeout=grace_s)
            if pending:
                logger.warning(
                    "%d investigation worker(s) survived cancellation; cancelling again",
                    len(pending),
                )
        for task in self._workers:
            if not task.cancelled() and task.exception() is not None:
                logger.error("investigation worker ended with an error: %r", task.exception())
        self._workers.clear()

    async def drain(self, timeout: float = 60.0) -> None:
        """Wait for the queue to empty. Used by tests and graceful shutdown."""
        async with asyncio.timeout(timeout):
            await self.queue.join()

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
