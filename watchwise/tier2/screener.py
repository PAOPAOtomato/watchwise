"""Tier 2: gate -> frame sampling -> VLM -> ScreenResult.

The screener owns the concurrency cap. With N cameras all alerting at once,
an uncapped screener will queue every request onto one GPU and turn a 400ms
call into a 6s one. A semaphore keeps per-alert latency predictable and lets
the queue absorb the burst instead.
"""

from __future__ import annotations

import asyncio
import logging

from watchwise.config import VLMSettings
from watchwise.memory.long_term import PatternLibrary
from watchwise.tier2 import prompts
from watchwise.tier2.gate import GateResult, ScreeningGate
from watchwise.tier2.sampling import FrameSampler
from watchwise.tier2.vlm.base import VLMBackend
from watchwise.types import AlertContext, ScreenResult, Verdict

logger = logging.getLogger(__name__)


class Screener:
    def __init__(
        self,
        gate: ScreeningGate,
        sampler: FrameSampler,
        vlm: VLMBackend,
        settings: VLMSettings,
        patterns: PatternLibrary | None = None,
    ):
        self.gate = gate
        self.sampler = sampler
        self.vlm = vlm
        self.settings = settings
        self.patterns = patterns
        self._sem = asyncio.Semaphore(max(1, settings.max_concurrency))
        self.stats = {"gated": 0, "screened": 0, "errors": 0}

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

    def check_gate(self, ctx: AlertContext) -> GateResult:
        """The free, model-free half of screening."""
        gate_result = self.gate.evaluate(ctx)
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
        extra = self._memory_context(ctx)
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

            prompt = prompts.build_prompt(ctx, len(frames), extra)
            result = await self.vlm.screen(frames, prompt, prompts.SYSTEM)

        self.stats["screened"] += 1
        if result.error:
            self.stats["errors"] += 1
        return result

    def _memory_context(self, ctx: AlertContext) -> str:
        """Give the VLM the recurring-mistake context a human reviewer has.

        These are patterns that were *not* strong enough for the gate to
        suppress on. Surfacing them still helps: the model can confirm or
        rule out the known failure mode instead of rediscovering it.
        """
        if self.patterns is None:
            return ""
        matches = self.patterns.search(
            " ".join([ctx.camera, *ctx.labels, *ctx.zones]),
            signature=ctx.signature(),
            top_k=2,
            min_similarity=0.5,
        )
        if not matches:
            return ""
        lines = ["Known recurring judgements for this camera and label set:"]
        for m in matches:
            lines.append(
                f"- ({m.kind}, {m.corrections} human corrections) {m.description}"
            )
        lines.append(
            "Treat these as priors from past human review, not as the answer."
        )
        return "\n".join(lines)

    async def aclose(self) -> None:
        await self.vlm.aclose()
