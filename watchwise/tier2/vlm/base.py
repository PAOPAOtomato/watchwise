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
from dataclasses import dataclass
from typing import Any

from watchwise.llm.base import split_reasoning
from watchwise.types import Frame, ScreenResult, Verdict

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

# ScreenResult.error codes. The first three mean the model answered without a
# usable verdict; TIMEOUT means it did not answer in time. Any other backend
# failure carries its own description.
EMPTY_REPLY = "empty_reply"
PARSE_ERROR = "parse_error"
TRUNCATED = "truncated"
TIMEOUT = "timeout"
REPLY_PROBLEMS = frozenset({EMPTY_REPLY, PARSE_ERROR, TRUNCATED})


@dataclass
class VLMReply:
    """A backend's reply, plus what the server said about it."""

    text: str
    # The model stopped at its output cap (Ollama done_reason / OpenAI
    # finish_reason "length"). Thinking models spend the cap on hidden
    # reasoning first, so a cap sized for the JSON alone returns nothing.
    truncated: bool = False
    max_tokens: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    # The model reasoned before (or instead of) answering.
    thinking: bool = False


class VLMBackend(ABC):
    """A vision-language model that can screen a handful of frames."""

    name: str = "base"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def infer(self, frames: list[Frame], prompt: str, system: str) -> str | VLMReply:
        """Return the model's raw text reply, or a ``VLMReply`` when the
        backend knows more (truncation, token counts)."""

    async def aclose(self) -> None:
        """Release transport resources. Default: nothing to do."""

    async def screen(self, frames: list[Frame], prompt: str, system: str) -> ScreenResult:
        import time

        started = time.perf_counter()
        try:
            raw = await self.infer(frames, prompt, system)
        except Exception as exc:  # transport, timeout, model unavailable
            # httpx timeouts have an empty message; never report an empty error.
            detail = str(exc).strip() or type(exc).__name__
            timed_out = _is_timeout(exc)
            logger.warning(
                "VLM %s/%s failed: %s%s", self.name, self.model,
                "timed out: " if timed_out else "", detail,
            )
            return ScreenResult(
                verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                reason=(
                    f"the screening model did not answer in time ({detail})"
                    if timed_out
                    else f"screening backend error: {detail}"
                ),
                backend=self.name,
                model=self.model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                frames_used=len(frames),
                error=TIMEOUT if timed_out else detail,
            )

        reply = raw if isinstance(raw, VLMReply) else VLMReply(str(raw or ""))
        result = parse_screen_reply(reply.text)
        if reply.truncated and result.error:
            # A cut-off answer is a sizing problem, not a model that cannot
            # follow the format; say which.
            cap = f"the {reply.max_tokens}-token" if reply.max_tokens else "its"
            if result.error == PARSE_ERROR:
                when = "mid-answer"
            else:
                when = "while it was still thinking" if reply.thinking else "before it answered"
            result.reason = f"the reply was cut off at {cap} output cap {when}"
            result.error = TRUNCATED
        if reply.input_tokens or reply.output_tokens:
            result.tokens = {"input": reply.input_tokens, "output": reply.output_tokens}
        result.backend = self.name
        result.model = self.model
        result.latency_ms = (time.perf_counter() - started) * 1000.0
        result.frames_used = len(frames)
        return result


def _is_timeout(exc: BaseException | None) -> bool:
    """True for a timeout from any transport, including one that a provider
    re-raised as its own error type (the original is kept as ``__cause__``)."""
    while exc is not None:
        if isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__:
            return True
        exc = exc.__cause__
    return False


def parse_screen_reply(raw: str) -> ScreenResult:
    """Coerce a model reply into a ScreenResult.

    On anything unparseable we return UNCERTAIN with zero confidence, which
    routes the alert to tier 3 rather than silently dropping it. Failing
    *open* is the only safe default for a surveillance filter.
    """
    # Inline <think> blocks can hold braces of their own; only the answer counts.
    text, _ = split_reasoning((raw or "").strip())
    if not text:
        return ScreenResult(
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            reason="empty model reply",
            error=EMPTY_REPLY,
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
            error=PARSE_ERROR,
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
