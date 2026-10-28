"""Tier 2: gate -> frame sampling -> VLM -> ScreenResult.

The screener owns the concurrency cap. With N cameras all alerting at once,
an uncapped screener will queue every request onto one GPU and turn a 400ms
call into a 6s one. A semaphore keeps per-alert latency predictable and lets
the queue absorb the burst instead.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from watchwise.config import AlertRules, CameraSettings, VLMSettings
from watchwise.memory.long_term import PatternLibrary
from watchwise.memory.watches import Watch, WatchBook
from watchwise.tier2 import prompts
from watchwise.tier2.gate import GateResult, ScreeningGate
from watchwise.tier2.sampling import FrameSampler
from watchwise.tier2.vlm.base import VLMBackend, parse_summary_reply, parse_watch_reply
from watchwise.types import OWNER_REAL_TAG, AlertContext, ReviewSummary, ScreenResult, Verdict

logger = logging.getLogger(__name__)

# Frames across a whole event for the watches' question.
WATCH_FRAMES = 4


@dataclass(frozen=True)
class _Rule:
    """An always-alert rule asked like a watch (for a recognized resident)."""

    request: str


class Screener:
    def __init__(
        self,
        gate: ScreeningGate,
        sampler: FrameSampler,
        vlm: VLMBackend,
        settings: VLMSettings,
        patterns: PatternLibrary | None = None,
        *,
        arming: Any = None,
        cameras: dict[str, CameraSettings] | None = None,
        alerts: AlertRules | None = None,
        watches: WatchBook | None = None,
    ):
        self.gate = gate
        self.sampler = sampler
        self.vlm = vlm
        self.settings = settings
        self.patterns = patterns
        # Anything with get() -> {"state": ...}; the prompt says who should be home.
        self.arming = arming
        # What the owner said in /setup (D-060). The service's own config
        # objects, which /setup updates in place.
        self.cameras = cameras if cameras is not None else {}
        self.alerts = alerts if alerts is not None else AlertRules()
        # "Tell me when..." requests from the chat, checked on every event.
        self.watches = watches
        self._sem = asyncio.Semaphore(max(1, settings.max_concurrency))
        self.stats = {"gated": 0, "screened": 0, "errors": 0, "summaries": 0}

    async def screen(self, ctx: AlertContext) -> tuple[GateResult, ScreenResult | None]:
        """Screen one alert.

        Returns ``(gate_result, screen_result)``. ``screen_result`` is None
        exactly when the gate resolved the alert without any model call --
        that pairing is what the cost funnel counts.
        """
        gate_result = self.check_gate(ctx)
        if not gate_result.passed_through:
            return gate_result, None
        return gate_result, await self.screen_frames(ctx)

    def check_gate(self, ctx: AlertContext, *, use_memory: bool = True) -> GateResult:
        """The free, model-free half of screening."""
        gate_result = self.gate.evaluate(ctx, use_memory=use_memory)
        if not gate_result.passed_through:
            self.stats["gated"] += 1
            logger.debug(
                "gate resolved %s via %s: %s",
                ctx.review_id,
                gate_result.rule,
                gate_result.reason,
            )
        return gate_result

    async def screen_frames(self, ctx: AlertContext) -> ScreenResult:
        """The model half, for an alert the gate let through."""
        matches = self._memory_matches(ctx)
        extra = self._memory_context(ctx, matches)
        async with self._sem:
            frames = await self.sampler.for_alert(ctx, self.settings.frames)
            if not frames:
                self.stats["errors"] += 1
                return ScreenResult(
                    verdict=Verdict.UNCERTAIN,
                    confidence=0.0,
                    reason="no frames could be retrieved from NVR for this segment",
                    backend=self.vlm.name,
                    model=self.vlm.model,
                    error="no_frames",
                )

            prompt = prompts.build_prompt(
                ctx,
                len(frames),
                extra,
                arming=self._arming_state(),
                camera_notes=self.camera_notes(ctx.camera),
                always_alert=self.alerts.always,
            )
            result = await self.vlm.screen(frames, prompt, prompts.SYSTEM)

        self.stats["screened"] += 1
        if result.error:
            self.stats["errors"] += 1
        # The patterns close enough that the gate could one day act on them
        # alone. They shaped this verdict, so a human correction of it scores
        # them; that is how a new pattern earns the trust the gate requires.
        threshold = self.gate.settings.memory_suppress_similarity
        result.memory_ids = [m.pattern_id for m in matches if m.similarity >= threshold]
        if result.verdict is Verdict.BENIGN and result.owner_real:
            # D-055: live, the model saw "waving both arms, which matches a
            # previously marked real event" and still said benign, because a
            # resident at home is ordinary. The owner asked for these.
            result.verdict = Verdict.SUSPICIOUS
        return result

    def _watches(self, ctx: AlertContext) -> list:
        if self.watches is None:
            return []
        try:
            return self.watches.active(ctx.camera)
        except Exception:
            logger.exception("could not read the watches for %s", ctx.review_id)
            return []

    async def _check_watches(self, ctx: AlertContext, frames: list, watches: list) -> list[tuple[Any, str]]:
        """Ask the strict watch question for each watch on these frames: the
        watches seen, with the model's evidence (B-029)."""
        seen = []
        for watch in watches:
            raw = await self.vlm.describe(frames, prompts.build_watch_prompt(ctx, len(frames), watch.request),
                                          prompts.WATCH_SYSTEM)
            answer = parse_watch_reply(raw)
            label = f"watch {watch.id}" if isinstance(watch, Watch) else f"rule “{watch.request}”"
            logger.info("%s on %s: %s", label, ctx.review_id,
                        f"seen={answer[0]} ({answer[1]})" if answer else f"no readable answer ({(raw or '')[:80]!r})")
            if answer is not None and answer[0]:
                seen.append((watch, answer[1]))
        return seen


    async def summarize(self, ctx: AlertContext) -> ReviewSummary | None:
        """What happened over the whole event, from frames spread across it
        (D-046). None when switched off or when anything fails: the card then
        simply goes without it."""
        count = self.settings.summary_frames
        watches = self._watches(ctx)
        if count <= 0:
            return await self.watch_event(ctx, watches)
        async with self._sem:
            frames = await self.sampler.for_alert(ctx, count, front_bias=False)
            if not frames:
                return None
            prompt = prompts.build_summary_prompt(
                ctx,
                len(frames),
                arming=self._arming_state(),
                concerns=self.settings.summary_concerns,
                owner_real=self._owner_real(ctx),
                camera_notes=self.camera_notes(ctx.camera),
                always_alert=self.alerts.always,
            )
            raw = await self.vlm.describe(frames, prompt, prompts.SUMMARY_SYSTEM)
            seen = await self._check_watches(ctx, frames, watches)
        self.stats["summaries"] += 1
        summary = parse_summary_reply(raw)
        if summary is None and raw:
            logger.info("no usable summary for %s: %s", ctx.review_id, raw[:200])
        if seen:
            if summary is None:
                summary = ReviewSummary(title="", short_summary=seen[0][1])
            self._summary_watch(ctx, summary, seen)
        return summary

    async def watch_event(
        self, ctx: AlertContext, watches: list | None = None, *, residents: bool = False
    ) -> ReviewSummary | None:
        """Only the owner's own questions, on frames across the whole event:
        the watches, and with ``residents`` (a resident recognized at home,
        held back before any other model call) the always-alert rules worded
        for "anyone" (D-067). One call per question. A summary carrying the
        match, or None."""
        watches = self._watches(ctx) if watches is None else watches
        if residents:
            watches = [*watches, *(_Rule(rule) for rule in self.alerts.for_everyone())]
        if not watches:
            return None
        async with self._sem:
            frames = await self.sampler.for_alert(ctx, WATCH_FRAMES, front_bias=False)
            if not frames:
                return None
            seen = await self._check_watches(ctx, frames, watches)
        if not seen:
            return None
        summary = ReviewSummary(title="", short_summary="")
        self._summary_watch(ctx, summary, seen)
        return summary

    def _summary_watch(self, ctx: AlertContext, summary: ReviewSummary, seen: list[tuple[Any, str]]) -> None:
        """Frames across the whole event show what the owner asked about: it
        alerts (never lowers anything); a watch ends, a rule stays. Checked
        once, when the event ends (D-066: one call per event per question)."""
        summary.owner_asked = True
        summary.threat_level = max(summary.threat_level, 1)
        watched = [(w, e) for w, e in seen if isinstance(w, Watch)]
        ruled = [(r, e) for r, e in seen if not isinstance(r, Watch)]
        parts = []
        if watched:
            asked = "; ".join(f"“{w.request}”" for w, _ in watched)
            parts.append(f"👀 You asked to be told when {asked}: {watched[0][1]}")
        if ruled:
            asked = "; ".join(f"“{r.request}”" for r, _ in ruled)
            parts.append(f"🚨 You asked to always be alerted for {asked}: {ruled[0][1]}")
        summary.short_summary = " ".join([*parts, summary.short_summary]).strip()
        if watched:
            assert self.watches is not None
            self.watches.fulfil([w.id for w, _ in watched], ctx.review_id)
        logger.info("asked-for %s matched in %s", [w.request for w, _ in seen], ctx.review_id)

    def _arming_state(self) -> str | None:
        return (self.arming.get() or {}).get("state") if self.arming is not None else None

    def camera_notes(self, camera: str) -> str:
        spec = self.cameras.get(camera)
        return spec.notes if spec is not None else ""

    def _memory_matches(self, ctx: AlertContext) -> list[Any]:
        if self.patterns is None:
            return []
        return self.patterns.search(
            " ".join([ctx.camera, *ctx.labels, *ctx.zones]),
            signature=ctx.signature(),
            top_k=2,
            min_similarity=0.5,
        )

    def _memory_context(self, ctx: AlertContext, matches: list[Any] | None = None) -> str:
        """Give the VLM the recurring-mistake context a human reviewer has.

        These are patterns that were *not* strong enough for the gate to
        suppress on. Surfacing them still helps: the model can confirm or
        rule out the known failure mode instead of rediscovering it.
        """
        if matches is None:
            matches = self._memory_matches(ctx)
        lines: list[str] = []
        if matches:
            lines.append("Known recurring judgements for this camera and label set:")
            for m in matches:
                lines.append(
                    f"- ({m.kind}, {m.corrections} human corrections) {m.description}"
                )
            lines.append(
                "Treat these as priors from past human review, not as the answer."
            )
        lines += self._owner_examples(ctx)
        return "\n".join(lines)

    def _owner_real(self, ctx: AlertContext) -> list[str]:
        """What the owner marked real on this kind of event, for the summary,
        which can only raise (D-053)."""
        if self.patterns is None:
            return []
        texts: list[str] = []
        for verdict, text in self.patterns.store.correction_examples(ctx.signature()):
            if verdict == "suspicious" and text[:200] not in texts and len(texts) < 3:
                texts.append(text[:200])
        return texts

    def _owner_examples(self, ctx: AlertContext) -> list[str]:
        """What the owner said about earlier events of this kind, as what was
        seen in them (D-051): the camera, labels and zones alone can't tell
        someone sitting at a desk from someone gesturing at the camera."""
        if self.patterns is None:
            return []
        seen: dict[str, list[str]] = {"benign": [], "suspicious": []}
        for verdict, text in self.patterns.store.correction_examples(ctx.signature()):
            texts = seen.get(verdict)
            if texts is not None and text not in texts and len(texts) < 4:
                texts.append(text[:200])
        if not (seen["benign"] or seen["suspicious"]):
            return []
        lines = ["", "How the owner judged earlier events like this one:"]
        lines += [f'- marked expected: "{t}"' for t in seen["benign"]]
        lines += [f'- marked real: "{t}"' for t in seen["suspicious"]]
        rule = "If this event clearly matches what they marked expected, it is routine (benign). "
        if seen["suspicious"]:
            rule += (
                "If it resembles what they marked real, even if it also looks like something "
                "they marked expected, they want an alert even if it looks harmless: answer "
                f'"suspicious" and add the tag "{OWNER_REAL_TAG}". '
            )
        lines.append(rule + "If it shows something different, it is not routine.")
        return lines

    async def aclose(self) -> None:
        await self.vlm.aclose()
