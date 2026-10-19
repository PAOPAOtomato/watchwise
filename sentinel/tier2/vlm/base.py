"""The pluggable VLM interface used by the tier-2 screener.

Every backend takes the same thing -- a handful of JPEG frames plus a text
prompt -- and returns the same thing: a parsed ``ScreenResult``. Backends are
responsible only for transport and for turning the model's reply into JSON;
verdict policy lives in the screener, not here.

Parsing is deliberately forgiving. Small local VL models routinely wrap JSON
in prose or a code fence, and a screening tier that hard-fails on that would
escalate everything to the expensive tier for no reason.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from sentinel.types import Frame, ScreenResult, Verdict

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class VLMBackend(ABC):
    """A vision-language model that can screen a handful of frames."""

    name: str = "base"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def infer(self, frames: list[Frame], prompt: str, system: str) -> str:
        """Return the model's raw text reply."""

    async def aclose(self) -> None:
        """Release transport resources. Default: nothing to do."""

    async def screen(self, frames: list[Frame], prompt: str, system: str) -> ScreenResult:
        import time

        started = time.perf_counter()
        try:
            raw = await self.infer(frames, prompt, system)
        except Exception as exc:  # transport, timeout, model unavailable
            logger.warning("VLM %s/%s failed: %s", self.name, self.model, exc)
            return ScreenResult(
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                reason=f"screening backend error: {exc}",
                backend=self.name,
                model=self.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                frames_used=len(frames),
                error=str(exc),
            )

        result = parse_screen_reply(raw)
        result.backend = self.name
        result.model = self.model
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        result.frames_used = len(frames)
        return result


def parse_screen_reply(raw: str) -> ScreenResult:
    """Coerce a model reply into a ScreenResult.

    On anything unparseable we return UNCERTAIN with zero confidence, which
    routes the alert to tier 3 rather than silently dropping it. Failing
    *open* is the only safe default for a surveillance filter.
    """
    text = (raw or "").strip()
    if not text:
        return ScreenResult(
            verdict=Verdict.UNCERTAIN, confidence=0.0, reason="empty model reply"
        )

    payload: dict[str, Any] | None = None
    fence = _FENCE.search(text)
    candidate = fence.group(1).strip() if fence else text
    for attempt in (candidate, _JSON_BLOCK.search(candidate)):
        if attempt is None:
            continue
        blob = attempt if isinstance(attempt, str) else attempt.group(0)
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break

    if payload is None:
        return ScreenResult(
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            reason=f"unparseable model reply: {text[:200]}",
            error="parse_error",
        )

    verdict_raw = str(payload.get("verdict", "uncertain")).strip().lower()
    aliases = {
        "benign": Verdict.BENIGN,
        "normal": Verdict.BENIGN,
        "nothing": Verdict.BENIGN,
        "false_alarm": Verdict.BENIGN,
        "suspicious": Verdict.SUSPICIOUS,
        "threat": Verdict.SUSPICIOUS,
        "alert": Verdict.SUSPICIOUS,
        "uncertain": Verdict.UNCERTAIN,
        "unsure": Verdict.UNCERTAIN,
        "unknown": Verdict.UNCERTAIN,
    }
    verdict = aliases.get(verdict_raw, Verdict.UNCERTAIN)

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    tags = payload.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]

    return ScreenResult(
        verdict=verdict,
        confidence=confidence,
        reason=str(payload.get("reason", "")).strip()[:1000],
        tags=[str(t) for t in tags][:10],
    )
