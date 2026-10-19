"""The agent-backend interface, plus the tool-invocation recorder.

A backend's only job is to drive a think -> act -> observe loop and return a
conclusion. Everything shared -- executing tools, recording the trace,
enforcing the iteration cap, updating short-term memory -- lives in
``ToolRunner`` so that a new backend cannot quietly skip it.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from sentinel.memory.short_term import ShortTermMemory
from sentinel.tools.base import ToolContext, summarize
from sentinel.tools.registry import TOOLS_BY_NAME
from sentinel.types import AlertContext, Investigation, ScreenResult, ToolCall, Verdict

logger = logging.getLogger(__name__)


class ToolRunner:
    """Executes forensic tools while recording the ReAct trace.

    Also enforces two things the model is not trusted to enforce itself: the
    tool-call budget, and not repeating an identical call.
    """

    def __init__(
        self,
        ctx: ToolContext,
        memory: ShortTermMemory,
        investigation: Investigation,
        *,
        max_calls: int = 12,
    ):
        self.ctx = ctx
        self.memory = memory
        self.investigation = investigation
        self.max_calls = max_calls
        self.iteration = 0
        self._seen: dict[str, dict[str, Any]] = {}

    @property
    def calls_made(self) -> int:
        return len(self.investigation.tool_calls)

    @property
    def budget_exhausted(self) -> bool:
        return self.calls_made >= self.max_calls

    def next_iteration(self) -> int:
        self.iteration += 1
        self.investigation.iterations = self.iteration
        return self.iteration

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        import json

        if self.budget_exhausted:
            return {"ok": False, "error": f"tool budget of {self.max_calls} calls exhausted"}

        key = f"{name}:{json.dumps(args, sort_keys=True, default=str)}"
        if key in self._seen:
            cached = dict(self._seen[key])
            cached["note"] = "identical call already made this investigation; cached result"
            return cached

        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}

        started = time.perf_counter()
        result = await tool.run(self.ctx, args)
        latency = (time.perf_counter() - started) * 1000.0

        self._seen[key] = result
        self.investigation.tool_calls.append(
            ToolCall(
                iteration=max(1, self.iteration),
                name=name,
                arguments=args,
                result_summary=summarize(result),
                latency_ms=round(latency, 1),
                error=None if result.get("ok", True) else str(result.get("error")),
            )
        )
        self.memory.add_observation(name, result, self.iteration)
        return result

    def record_thought(self, text: str) -> None:
        self.investigation.thoughts.append(text)
        self.memory.add_thought(text, self.iteration)


@dataclass(slots=True)
class Conclusion:
    verdict: Verdict
    confidence: float
    conclusion: str
    recommended_action: str = "none"
    evidence: list[str] | None = None


class AgentBackend(ABC):
    """Drives one investigation to a conclusion."""

    name: str = "base"

    def __init__(self, model: str = "unknown"):
        self.model = model

    @abstractmethod
    async def run(
        self,
        ctx: AlertContext,
        screen: ScreenResult | None,
        memory: ShortTermMemory,
        runner: ToolRunner,
    ) -> Conclusion: ...

    async def aclose(self) -> None:
        """Release backend resources. Default: nothing to do."""


def parse_conclusion(raw: str) -> Conclusion:
    """Parse a backend's final JSON object, failing safe to uncertain."""
    import json
    import re

    text = (raw or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return Conclusion(
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            conclusion=f"agent produced no parseable conclusion: {raw[:300]}",
        )
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Conclusion(
            verdict=Verdict.UNCERTAIN,
            confidence=0.0,
            conclusion=f"agent conclusion was not valid JSON: {raw[:300]}",
        )

    try:
        verdict = Verdict(str(payload.get("verdict", "uncertain")).lower())
    except ValueError:
        verdict = Verdict.UNCERTAIN
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    evidence = payload.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]

    return Conclusion(
        verdict=verdict,
        confidence=confidence,
        conclusion=str(payload.get("conclusion", "")).strip(),
        recommended_action=str(payload.get("recommended_action", "none")).strip(),
        evidence=[str(e) for e in evidence][:12],
    )
