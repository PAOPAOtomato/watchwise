"""Tier 2: gate -> frame sampling -> VLM -> ScreenResult.

The screener owns the concurrency cap. With N cameras all alerting at once,
an uncapped screener will queue every request onto one GPU and turn a 400ms
call into a 6s one. A semaphore keeps per-alert latency predictable and lets
the queue absorb the burst instead.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from watchwise.config import AlertRules, CameraSettings, VLMSettings
from watchwise.memory.long_term import PatternLibrary
from watchwise.memory.watches import WatchBook
from watchwise.tier2 import prompts
from watchwise.tier2.gate import GateResult, ScreeningGate
from watchwise.tier2.sampling import FrameSampler
from watchwise.tier2.vlm.base import VLMBackend, parse_summary_reply
from watchwise.types import OWNER_REAL_TAG, OWNER_WATCH_TAG, AlertContext, ReviewSummary, ScreenResult, Verdict

logger = logging.getLogger(__name__)


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
        watches = self._watches(ctx)
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
                watches=[w.request for w in watches],
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
        if result.owner_watch and watches:
            self._watch_matched(ctx, result, watches)
        return result

    def _watches(self, ctx: AlertContext) -> list:
        if self.watches is None:
            return []
        try:
            return self.watches.active(ctx.camera)
        except Exception:
            logger.exception("could not read the watches for %s", ctx.review_id)
            return []

    def _watch_matched(self, ctx: AlertContext, result: ScreenResult, watches: list) -> None:
        """What the owner asked to be told about happened (D-065): it alerts
        (never lowers anything) and the watch ends."""
        numbers = set()
        for tag in result.tags:
            suffix = tag.lower().removeprefix(OWNER_WATCH_TAG).lstrip("_")
            if suffix.isdigit():
                numbers.add(int(suffix))
        matched = [w for n, w in enumerate(watches, 1) if n in numbers] or watches
        if result.verdict is Verdict.BENIGN:
            result.verdict = Verdict.SUSPICIOUS
        asked = "; ".join(f"“{w.request}”" for w in matched)
        result.reason = f"👀 You asked to be told when {asked}: {result.reason}"
        assert self.watches is not None
        self.watches.fulfil([w.id for w in matched], ctx.review_id)
        logger.info("watch %s matched in %s", [w.id for w in matched], ctx.review_id)

    async def summarize(self, ctx: AlertContext) -> ReviewSummary | None:
        """What happened over the whole event, from frames spread across it
        (D-046). None when switched off or when anything fails: the card then
        simply goes without it."""
        count = self.settings.summary_frames
        if count <= 0:
            return None
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
        self.stats["summaries"] += 1
        summary = parse_summary_reply(raw)
        if summary is None and raw:
            logger.info("no usable summary for %s: %s", ctx.review_id, raw[:200])
        return summary

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
