"""The cascade: source -> gate -> tier 2 -> decide -> dispatch -> tier 3.

One alert takes at most two passes through this module:

  pass 1 (on the alert path)
      gate, then optionally screen, then decide and dispatch. This is what a
      human sees. When the screen takes longer than the latency budget, an
      "analyzing" decision goes out first and the verdict replaces it.

  pass 2 (asynchronous, off the alert path)
      an investigation refines the record and may downgrade or escalate it.
      A revised decision is dispatched as an update.

The split is what makes the latency budget achievable at all: nothing that
calls a large model is allowed to sit between tier 1 and the alert.

Frigate reports an event when it starts and again when it ends. The second
report of a review costs a model call only if it shows something the first
did not: a new object label or zone. Alerts are handled concurrently (the
screener caps the model calls), but the reports of one review in order.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from watchwise.decision import DecisionPolicy
from watchwise.memory.store import Store
from watchwise.tier2.screener import Screener
from watchwise.tier3.investigator import InvestigationQueue
from watchwise.types import (
    AlertContext,
    Decision,
    Investigation,
    ScreenResult,
    Stage,
)

logger = logging.getLogger(__name__)

DecisionSink = Callable[[AlertContext, Decision], Awaitable[None]]

# Alerts in pass 1 at once; the rest wait in the queue.
MAX_IN_FLIGHT = 16
# A screen this quick needs no "analyzing" decision in front of it.
MIN_PROVISIONAL_WAIT_S = 0.5
# How many reviews are remembered, to pair an event's end with its start.
REMEMBERED_REVIEWS = 2048
ARMED = ("armed_away", "armed_night")


@dataclass
class _Handled:
    # Labels and "zone:<name>" entries the review was last decided on.
    scope: frozenset[str]
    decision: Decision


class Pipeline:
    def __init__(
        self,
        alert_queue: "asyncio.Queue[AlertContext]",
        screener: Screener,
        policy: DecisionPolicy,
        store: Store,
        investigations: InvestigationQueue | None = None,
        sinks: list[DecisionSink] | None = None,
        *,
        arming: Any = None,
    ):
        self.alert_queue = alert_queue
        self.screener = screener
        self.policy = policy
        self.store = store
        self.investigations = investigations
        self.sinks = sinks or []
        # Anything with get() -> {"state": ...}; picks the latency budget.
        self.arming = arming
        self._task: asyncio.Task[None] | None = None
        self._slots = asyncio.Semaphore(MAX_IN_FLIGHT)
        self._in_flight: set[asyncio.Task[None]] = set()
        # review_id -> [lock, users]: one review's reports in arrival order.
        self._reviews: dict[str, list[Any]] = {}
        self._handled: OrderedDict[str, _Handled] = OrderedDict()
        # review_id -> what tier 2 concluded, so the tier-3 callback can
        # produce a revised decision with the full picture.
        self._pending: dict[str, tuple[AlertContext, ScreenResult | None, list[str]]] = {}
        self.metrics = {
            "ingested": 0,
            "gate_resolved": 0,
            "vlm_calls": 0,
            "investigations_queued": 0,
            "screen_latency_ms_total": 0.0,
            "rescreens": 0,
            "reports_without_model": 0,
            "provisional_dispatches": 0,
            "first_dispatches": 0,
            "first_dispatch_ms_total": 0.0,
        }

    # ---------------- lifecycle ----------------

    async def start(self) -> None:
        if self.investigations is not None:
            self.investigations.on_done = self._on_investigation_done
            await self.investigations.start()
        self._task = asyncio.create_task(self._run(), name="pipeline")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for task in list(self._in_flight):
            task.cancel()
        if self._in_flight:
            await asyncio.gather(*self._in_flight, return_exceptions=True)
        if self.investigations is not None:
            await self.investigations.stop()

    async def _run(self) -> None:
        while True:
            await self._slots.acquire()
            try:
                ctx = await self.alert_queue.get()
            except BaseException:
                self._slots.release()
                raise
            task = asyncio.create_task(self._handle_queued(ctx), name=f"alert:{ctx.review_id}")
            self._in_flight.add(task)
            task.add_done_callback(self._in_flight.discard)

    async def _handle_queued(self, ctx: AlertContext) -> None:
        try:
            await self.handle(ctx)
        except Exception:
            logger.exception("pipeline failed on %s", ctx.review_id)
        finally:
            self.alert_queue.task_done()
            self._slots.release()

    # ---------------- the cascade ----------------

    async def handle(self, ctx: AlertContext) -> Decision:
        """Process one report of an alert through pass 1, queueing pass 2 if needed."""
        entry = self._reviews.setdefault(ctx.review_id, [asyncio.Lock(), 0])
        entry[1] += 1
        try:
            async with entry[0]:
                return await self._handle(ctx)
        finally:
            entry[1] -= 1
            if entry[1] == 0:
                self._reviews.pop(ctx.review_id, None)

    async def _handle(self, ctx: AlertContext) -> Decision:
        started = time.perf_counter()
        record = ctx.to_dict()
        record["signature"] = ctx.signature()
        scope = frozenset(ctx.labels) | frozenset(f"zone:{z}" for z in ctx.zones)
        seen = self._handled.get(ctx.review_id)

        if seen is not None and scope <= seen.scope:
            # A later report of an event already decided, showing nothing
            # new: keep the record current, without a model call.
            row = self.store.get_alert(ctx.review_id)
            self.store.upsert_alert(record, row["stage"] if row is not None else seen.decision.stage.value)
            self.metrics["reports_without_model"] += 1
            return seen.decision
        first = seen is None
        if first:
            self.metrics["ingested"] += 1
        else:
            scope |= seen.scope
            self.metrics["rescreens"] += 1
            logger.info(
                "re-screening %s: it now also shows %s",
                ctx.review_id, ", ".join(sorted(scope - seen.scope)),
            )
        self.store.upsert_alert(record, Stage.INGESTED.value)

        gate_result = self.screener.check_gate(ctx)
        if not gate_result.passed_through:
            # Resolved by the gate: no model was called at all.
            self.metrics["gate_resolved"] += 1
            decision = self.policy.from_gate(ctx, gate_result)
            self.store.upsert_alert(record, Stage.GATED.value)
            self.store.record_decision(decision.to_dict())
            await self._dispatch(ctx, decision, first=first)
            self._remember(ctx.review_id, scope, decision)
            return decision

        screen, provisional_sent = await self._screen(ctx, allow_provisional=first)
        self.metrics["vlm_calls"] += 1
        self.metrics["screen_latency_ms_total"] += screen.latency_ms
        self.store.upsert_alert(record, Stage.SCREENED.value)

        decision, needs_investigation = self.policy.from_screening(ctx, screen)
        self.store.record_decision(decision.to_dict())
        await self._dispatch(ctx, decision, first=first and not provisional_sent)
        self._remember(ctx.review_id, scope, decision)

        if needs_investigation and self.investigations is not None:
            memory_ids = list(dict.fromkeys([*gate_result.matched_pattern_ids, *screen.memory_ids]))
            pending = (ctx, screen, memory_ids)
            if ctx.review_id in self._pending:
                # Already under investigation (this is a re-screen): give
                # its revision the newer picture rather than start another.
                self._pending[ctx.review_id] = pending
            else:
                self._pending[ctx.review_id] = pending
                if self.investigations.submit(ctx, screen):
                    self.metrics["investigations_queued"] += 1
                else:
                    self._pending.pop(ctx.review_id, None)
                    logger.warning(
                        "could not queue investigation for %s; alert stands as dispatched",
                        ctx.review_id,
                    )

        logger.debug(
            "pass 1 for %s took %.1fms -> %s",
            ctx.review_id,
            (time.perf_counter() - started) * 1000.0,
            decision.disposition.value,
        )
        return decision

    async def _screen(self, ctx: AlertContext, *, allow_provisional: bool) -> tuple[ScreenResult, bool]:
        """Run tier 2. If it outlasts the latency budget, dispatch an
        "analyzing" decision first. Returns (result, whether one was sent)."""
        work = asyncio.create_task(self.screener.screen_frames(ctx), name=f"screen:{ctx.review_id}")
        sent = False
        try:
            budget = self._provisional_after() if allow_provisional else None
            if budget is not None:
                done, _ = await asyncio.wait({work}, timeout=max(budget, MIN_PROVISIONAL_WAIT_S))
                if not done:
                    provisional = self.policy.provisional(ctx)
                    self.store.record_decision(provisional.to_dict())
                    self.metrics["provisional_dispatches"] += 1
                    await self._dispatch(ctx, provisional, first=True)
                    sent = True
            return await work, sent
        finally:
            if not work.done():
                work.cancel()

    def _provisional_after(self) -> float | None:
        settings = self.policy.settings
        state = (self.arming.get() or {}).get("state") if self.arming is not None else None
        return settings.provisional_after_armed_s if state in ARMED else settings.provisional_after_s

    def _remember(self, review_id: str, scope: frozenset[str], decision: Decision) -> None:
        self._handled[review_id] = _Handled(scope, decision)
        self._handled.move_to_end(review_id)
        while len(self._handled) > REMEMBERED_REVIEWS:
            self._handled.popitem(last=False)

    async def _on_investigation_done(
        self, ctx: AlertContext, inv: Investigation
    ) -> None:
        """Pass 2: revise the record once the agent has finished."""
        _, screen, memory_ids = self._pending.pop(
            ctx.review_id, (ctx, None, [])
        )
        decision = self.policy.from_investigation(
            ctx, screen, inv, matched_memory_ids=memory_ids
        )
        self.store.record_decision(decision.to_dict())
        await self._dispatch(ctx, decision)

    async def _dispatch(self, ctx: AlertContext, decision: Decision, *, first: bool = False) -> None:
        if first:
            # Event -> phone latency: Frigate's report to our first dispatch.
            waited = time.time() - ctx.received_at
            self.metrics["first_dispatches"] += 1
            self.metrics["first_dispatch_ms_total"] += waited * 1000.0
            logger.info(
                "first dispatch for %s (%s) %.1fs after Frigate reported it",
                ctx.review_id,
                "analyzing" if decision.stage is Stage.INGESTED else decision.disposition.value,
                waited,
            )
        for sink in self.sinks:
            try:
                await sink(ctx, decision)
            except Exception:
                logger.exception("decision sink %r failed", getattr(sink, "__name__", sink))

    # ---------------- reporting ----------------

    def cost_funnel(self) -> dict[str, float | int | None]:
        """The measured cost funnel: how many alerts reached each model tier.

        ``vlm_call_rate`` is the fraction of alerts that reached a model at
        all; ``agent_call_rate`` the fraction that reached the expensive one.
        These are counted, not asserted.
        """
        ingested = self.metrics["ingested"]
        vlm = self.metrics["vlm_calls"]
        agent = self.metrics["investigations_queued"]
        return {
            "alerts_ingested": ingested,
            "gate_resolved_without_model": self.metrics["gate_resolved"],
            "vlm_calls": vlm,
            "investigations_queued": agent,
            "vlm_call_rate": round(vlm / ingested, 4) if ingested else None,
            "agent_call_rate": round(agent / ingested, 4) if ingested else None,
            "mean_screen_latency_ms": round(
                self.metrics["screen_latency_ms_total"] / vlm, 1
            )
            if vlm
            else None,
        }

    def latency(self) -> dict[str, float | int | None]:
        """How quickly alerts reached the sinks (counted, not asserted)."""
        firsts = self.metrics["first_dispatches"]
        return {
            "first_dispatches": firsts,
            "mean_first_dispatch_s": round(self.metrics["first_dispatch_ms_total"] / firsts / 1000.0, 2)
            if firsts
            else None,
            "provisional_dispatches": self.metrics["provisional_dispatches"],
            "rescreens": self.metrics["rescreens"],
            "reports_without_model": self.metrics["reports_without_model"],
        }
