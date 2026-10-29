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

When an event ends, Watchwise asks the screener model what happened over the
whole of it (or Frigate's GenAI summary arrives). The summary is attached to
every decision on that review from then on: the card says what happened, and
a suspicious or critical rating raises an event that was downgraded or held
back. It never lowers one.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from watchwise.config import FaceSettings
from watchwise.decision import ARMED_STATES, DecisionPolicy, person_while_armed
from watchwise.memory.store import Store
from watchwise.tier2.screener import Screener
from watchwise.tier3.investigator import InvestigationQueue
from watchwise.types import (
    AlertContext,
    Decision,
    Disposition,
    Investigation,
    ReviewSummary,
    ScreenResult,
    Severity,
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
# How many tracked objects' face-recognition results are remembered.
REMEMBERED_FACES = 4096


@dataclass(frozen=True)
class FaceSeen:
    """One face result from the NVR: an attempt (the best match so far over
    that person's attempts, or None) or a confirmed recognition of a
    household member (the review's sub-label, which holds an event back)."""

    camera: str
    name: str | None
    score: float = 0.0
    confirmed: bool = False


class FaceFeed:
    """Hands the NVR's face results to whoever listens (the /setup
    recognition check). Listeners run on the event loop and must not block."""

    def __init__(self) -> None:
        self._listeners: list[Callable[[FaceSeen], None]] = []

    def subscribe(self, listener: Callable[[FaceSeen], None]) -> Callable[[], None]:
        """Start listening; call the returned function to stop."""
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def publish(self, seen: FaceSeen) -> None:
        for listener in list(self._listeners):
            try:
                listener(seen)
            except Exception:
                logger.exception("a face listener failed")


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
        roster: Any = None,
        faces: FaceSettings | None = None,
        face_feed: FaceFeed | None = None,
    ):
        self.alert_queue = alert_queue
        self.screener = screener
        self.policy = policy
        self.store = store
        self.investigations = investigations
        self.sinks = sinks or []
        # Anything with get() -> {"state": ...}; picks the latency budget.
        self.arming = arming
        # The duty roster: while armed, only it may clear a person (D-049).
        self.roster = roster
        # Who lives here, as the NVR's face recognition names them (D-057).
        self.faces = faces if faces is not None else FaceSettings()
        self.face_feed = face_feed if face_feed is not None else FaceFeed()
        # event id -> (attempts, best match so far or None, its score).
        self._faces: OrderedDict[str, tuple[int, str | None, float]] = OrderedDict()
        # review_id -> household names the NVR recognized in it.
        self._recognized: OrderedDict[str, list[str]] = OrderedDict()
        self._task: asyncio.Task[None] | None = None
        self._slots = asyncio.Semaphore(MAX_IN_FLIGHT)
        self._in_flight: set[asyncio.Task[None]] = set()
        # review_id -> [lock, users]: one review's reports in arrival order.
        self._reviews: dict[str, list[Any]] = {}
        self._handled: OrderedDict[str, _Handled] = OrderedDict()
        # review_id -> the summary of the whole event (Frigate's or ours), once known.
        self._summaries: OrderedDict[str, ReviewSummary] = OrderedDict()
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
            "summaries": 0,
            "own_summaries": 0,
            "raised_by_summary": 0,
            "held_by_face": 0,
            "raised_by_unknown_face": 0,
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
        self._note_recognized(ctx)
        if ctx.summary is not None and ctx.raw.get("type") == "genai":
            return await self._summarized(ctx)
        return await self._handle_report(ctx)

    async def _summarized(self, ctx: AlertContext) -> Decision:
        """Frigate's summary of the event, after it ended (D-045)."""
        assert ctx.summary is not None
        if ctx.review_id not in self._handled:
            # Not seen before (Watchwise started mid-event): judge it as a
            # first report; the summary joins that decision.
            self._keep_summary(ctx.review_id, ctx.summary)
            return await self._handle_report(ctx)
        return await self._apply_summary(ctx, ctx.summary)

    async def _handle_report(self, ctx: AlertContext) -> Decision:
        if self._held_by_face(ctx.review_id):
            decision = await self._recognized_report(ctx)
            if ctx.raw.get("type") == "end":
                # A "tell me when..." watch, or an always-alert rule for
                # "anyone", outranks a recognized face (B-029, D-067): a name
                # says who it is, not what they did. It costs only their own
                # question, once, when the event ends.
                watched = await self.screener.watch_event(ctx, residents=True)
                if watched is not None:
                    decision = await self._apply_summary(ctx, watched)
            return decision
        decision = await self._judge_report(ctx)
        if ctx.raw.get("type") == "end" and ctx.summary is None:
            # The event is over: say what happened across all of it (D-046).
            summary = await self.screener.summarize(ctx)
            if summary is not None:
                self.metrics["own_summaries"] += 1
                decision = await self._apply_summary(ctx, summary)
        if ctx.raw.get("type") == "end" and self._unknown_face(ctx):
            # D-068: a rostered visitor's face isn't in the NVR's library.
            expected = self._roster_reason(ctx)
            raised = self.policy.unknown_face(decision, expected=expected)
            if raised is not decision:
                if expected:
                    logger.info("a face nobody matched in %s, but %s: not raising it", ctx.review_id, expected)
                else:
                    self.metrics["raised_by_unknown_face"] += 1
                    logger.info("a face nobody matched in %s: raising it to an alert", ctx.review_id)
                decision = await self._decide(ctx, raised)
        return decision

    # ---------------- faces (D-057) ----------------

    def note_face(self, update: dict[str, Any]) -> None:
        """One face-recognition attempt reported by the NVR. Its ``name`` is
        the best match so far over all of that object's attempts, or None."""
        event_id = str(update.get("id") or "")
        if not event_id:
            return
        attempts = self._faces.get(event_id, (0, None, 0.0))[0] + 1
        name = update.get("name") or None
        score = float(update.get("score") or 0.0)
        self._faces[event_id] = (attempts, str(name) if name else None, score)
        self._faces.move_to_end(event_id)
        while len(self._faces) > REMEMBERED_FACES:
            self._faces.popitem(last=False)
        self.face_feed.publish(FaceSeen(str(update.get("camera") or ""), str(name) if name else None, score))

    def _note_recognized(self, ctx: AlertContext) -> None:
        """The NVR names a person once it is sure (a sub-label on the review)."""
        household = {name.lower(): name for name in self.faces.household}
        names = [household[s.lower()] for s in ctx.sub_labels if s.lower() in household]
        if not names:
            return
        known = self._recognized.get(ctx.review_id, [])
        for name in dict.fromkeys(names):
            if name not in known:
                self.face_feed.publish(FaceSeen(ctx.camera, name, confirmed=True))
        self._recognized[ctx.review_id] = list(dict.fromkeys(names))
        self._recognized.move_to_end(ctx.review_id)
        while len(self._recognized) > REMEMBERED_REVIEWS:
            self._recognized.popitem(last=False)

    def _held_by_face(self, review_id: str) -> list[str]:
        """Household names recognized in this review while at home: such an
        event is held back. While armed they count like the roster instead."""
        if self._arming_state() in ARMED_STATES:
            return []
        return self._recognized.get(review_id, [])

    def _face_score(self, ctx: AlertContext, names: list[str]) -> float:
        scores = [self._faces[e][2] for e in ctx.event_ids if e in self._faces and self._faces[e][1] in names]
        return max(scores, default=0.0)

    def _unknown_face(self, ctx: AlertContext) -> bool:
        """Did the NVR see a face in this event that matched nobody it knows?"""
        if not (self.faces.household and self.faces.alert_on_unknown) or self._recognized.get(ctx.review_id):
            return False
        return any(
            self._faces[e][1] is None and self._faces[e][0] >= self.faces.min_unknown_attempts
            for e in ctx.event_ids
            if e in self._faces
        )

    async def _recognized_report(self, ctx: AlertContext) -> Decision:
        """A report of an event in which a household member was recognized
        while at home: recorded, no model call, held back."""
        record = ctx.to_dict()
        record["signature"] = ctx.signature()
        seen = self._handled.get(ctx.review_id)
        if seen is None:
            self.metrics["ingested"] += 1
        self.store.upsert_alert(record, Stage.FACE.value)
        if seen is not None and seen.decision.stage is Stage.FACE:
            return seen.decision
        base = seen.decision if seen is not None else Decision(
            review_id=ctx.review_id, disposition=Disposition.DOWNGRADED, severity=Severity.DETECTION,
            confidence=0.0, rationale="", stage=Stage.INGESTED,
        )
        decision = await self._decide(ctx, base, first=seen is None)
        scope = frozenset(ctx.labels) | frozenset(f"zone:{z}" for z in ctx.zones)
        self._remember(ctx.review_id, scope | (seen.scope if seen is not None else frozenset()), decision)
        return decision

    async def _apply_summary(self, ctx: AlertContext, summary: ReviewSummary) -> Decision:
        logger.info("what happened in %s: %s (threat level %d)", ctx.review_id, summary.text()[:200],
                    summary.threat_level)
        self._keep_summary(ctx.review_id, summary)
        seen = self._handled[ctx.review_id]
        return await self._decide(ctx, replace(seen.decision, decided_at=time.time()))

    def _keep_summary(self, review_id: str, summary: ReviewSummary) -> None:
        self._summaries[review_id] = summary
        self._summaries.move_to_end(review_id)
        while len(self._summaries) > REMEMBERED_REVIEWS:
            self._summaries.popitem(last=False)
        self.metrics["summaries"] += 1

    async def _judge_report(self, ctx: AlertContext) -> Decision:
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

        arming = self._arming_state()
        gate_result = self.screener.check_gate(ctx, use_memory=not person_while_armed(ctx, arming))
        if not gate_result.passed_through:
            # Resolved by the gate: no model was called at all.
            self.metrics["gate_resolved"] += 1
            self.store.upsert_alert(record, Stage.GATED.value)
            decision = await self._decide(ctx, self.policy.from_gate(ctx, gate_result), first=first)
            self._remember(ctx.review_id, scope, decision)
            return decision

        screen, provisional_sent = await self._screen(ctx, allow_provisional=first)
        self.metrics["vlm_calls"] += 1
        self.metrics["screen_latency_ms_total"] += screen.latency_ms
        self.store.upsert_alert(record, Stage.SCREENED.value)

        decision, needs_investigation = self.policy.from_screening(ctx, screen, arming=arming)
        decision = await self._decide(ctx, decision, first=first and not provisional_sent)
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
                    self.metrics["provisional_dispatches"] += 1
                    await self._decide(ctx, self.policy.provisional(ctx), first=True)
                    sent = True
            return await work, sent
        finally:
            if not work.done():
                work.cancel()

    def _arming_state(self) -> str | None:
        return (self.arming.get() or {}).get("state") if self.arming is not None else None

    def _provisional_after(self) -> float | None:
        settings = self.policy.settings
        armed = self._arming_state() in ARMED_STATES
        return settings.provisional_after_armed_s if armed else settings.provisional_after_s

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
            ctx,
            screen,
            inv,
            matched_memory_ids=memory_ids,
            arming=self._arming_state(),
            roster_expects=self._roster_expects(ctx),
            recently_real=self._recently_real(ctx),
        )
        await self._decide(ctx, decision)

    def _recently_real(self, ctx: AlertContext) -> bool:
        """Did the owner mark this kind of event real recently (D-052)? If we
        can't tell, assume so: the alert stays visible."""
        try:
            return self.screener.gate.recently_real(ctx.signature())
        except Exception:
            logger.exception("recent-Real lookup failed for %s", ctx.review_id)
            return True

    def _roster_expects(self, ctx: AlertContext) -> str:
        """Why a person on this camera is expected right then, or "": the duty
        roster, or a household member the NVR recognized (D-057). The reason
        goes on the card (B-032)."""
        names = self._recognized.get(ctx.review_id)
        if names:
            return f"recognized as {', '.join(names)} by Frigate's face recognition"
        return self._roster_reason(ctx)

    def _roster_reason(self, ctx: AlertContext) -> str:
        """The duty roster's reason to expect a person here then, or ""."""
        if self.roster is None:
            return ""
        try:
            hits = self.roster.match(ctx.start_time, camera=ctx.camera, zones=ctx.zones, labels=["person"])
        except Exception:
            logger.exception("duty roster lookup failed for %s", ctx.review_id)
            return ""
        if not hits:
            return ""
        best = max(hits, key=lambda hit: hit["confidence"])
        days = best["days"]
        when = "every day" if days == "all" else " ".join(str(d).capitalize() for d in days)
        return f"the duty roster expects {best['name']} ({when} {best['window'].replace('-', '–')})"

    async def _decide(self, ctx: AlertContext, decision: Decision, *, first: bool = False) -> Decision:
        """Record and dispatch a decision, with the summary of the whole event
        attached once it is known (which may raise it, never lower it). A
        household member recognized at home holds it back, whatever came
        before or after (D-057)."""
        names = self._held_by_face(ctx.review_id)
        summary = self._summaries.get(ctx.review_id)
        # What the owner asked to be told about outranks a recognized face (B-029).
        watched = summary is not None and summary.owner_asked
        if names and not watched:
            held = self._handled.get(ctx.review_id)
            if held is None or held.decision.stage is not Stage.FACE:
                self.metrics["held_by_face"] += 1
                logger.info("%s recognized in %s: holding it back", ", ".join(names), ctx.review_id)
            decision = self.policy.recognized(decision, names, self._face_score(ctx, names))
        elif summary is not None:
            raised = self.policy.with_summary(decision, summary)
            if raised.stage is Stage.SUMMARIZED and decision.stage is not Stage.SUMMARIZED:
                self.metrics["raised_by_summary"] += 1
                logger.info(
                    "the whole-event review of %s rated it threat level %d: raising it to an alert",
                    ctx.review_id, summary.threat_level,
                )
            decision = raised
        self.store.record_decision(decision.to_dict())
        entry = self._handled.get(ctx.review_id)
        if entry is not None:
            entry.decision = decision
        await self._dispatch(ctx, decision, first=first)
        return decision

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
            "summaries": self.metrics["summaries"],
            "own_summaries": self.metrics["own_summaries"],
            "raised_by_summary": self.metrics["raised_by_summary"],
        }
