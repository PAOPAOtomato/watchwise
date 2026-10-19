"""Investigation backend: a ReAct loop over native tool calling.

Works with any ``ChatModel`` -- a local model on Ollama, an OpenAI-compatible
server, or Claude -- so tier 3 runs on whatever the user's hardware profile
chose. The loop is deliberately plain:

    think (model turn) -> call tools -> observe (tool results) -> repeat

and everything that must not depend on the model's good behaviour stays in
``ToolRunner``: the tool-call budget, duplicate-call caching, and the audit
trace. The model never touches the pipeline directly; whatever it concludes
goes through the decision policy like every other tier.

Small models misbehave in predictable ways, each handled here:

* answering in prose instead of JSON  -> one nudge, then fail safe
* invalid tool arguments              -> an error result it can correct
* calling a tool that does not exist  -> an error result listing real ones
* never concluding                    -> a final "conclude now" turn, then
                                         ``uncertain`` (a human decides)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from watchwise.llm.base import (
    INVALID_ARGUMENTS,
    ChatModel,
    Message,
    ToolSpec,
    Usage,
)
from watchwise.memory.short_term import ShortTermMemory
from watchwise.tier3 import prompts
from watchwise.tier3.backends.base import AgentBackend, Conclusion, ToolRunner, parse_conclusion
from watchwise.tools.base import ForensicTool
from watchwise.tools.registry import ALL_TOOLS
from watchwise.types import AlertContext, ScreenResult, Verdict

logger = logging.getLogger(__name__)

NUDGE = (
    "Give your final answer now: exactly one JSON object in the format from your "
    "instructions, and nothing else."
)
CONCLUDE = (
    "You have reached the investigation budget. Do not call more tools. Give your "
    "final answer now as exactly one JSON object; if the evidence is insufficient, "
    'say "uncertain" with an honest confidence.'
)


class ToolLoopBackend(AgentBackend):
    name = "tools"

    def __init__(
        self,
        chat: ChatModel,
        *,
        max_iterations: int = 6,
        tools: list[ForensicTool] | None = None,
        result_char_limit: int = 2500,
    ):
        super().__init__(chat.model)
        self.chat = chat
        self.max_iterations = max(1, max_iterations)
        self.tools = list(tools or ALL_TOOLS)
        self.result_char_limit = result_char_limit
        self.specs = [ToolSpec(t.name, t.description, t.input_schema) for t in self.tools]
        self._names = {t.name for t in self.tools}
        self.usage = Usage()

    async def aclose(self) -> None:
        await self.chat.aclose()

    async def run(
        self,
        ctx: AlertContext,
        screen: ScreenResult | None,
        memory: ShortTermMemory,
        runner: ToolRunner,
    ) -> Conclusion:
        messages = [
            Message.system(prompts.SYSTEM),
            Message.user(prompts.build_task_prompt(ctx, screen, memory.briefing())),
        ]
        usage = Usage()
        nudged = False
        try:
            for _ in range(self.max_iterations):
                runner.next_iteration()
                result = await self.chat.complete(messages, tools=self.specs)
                usage.add(result.usage)
                if result.reasoning:
                    runner.record_thought(result.reasoning[:1500])
                if result.stop_reason == "refusal":
                    return _declined()

                if result.tool_calls:
                    if result.content:
                        runner.record_thought(result.content[:1500])
                    messages.append(result.as_message())
                    for call in result.tool_calls:
                        outcome = await self._execute(call.name, call.arguments, runner)
                        messages.append(
                            Message.tool_result(
                                call,
                                self._render(outcome),
                                is_error=not outcome.get("ok", True),
                            )
                        )
                    if runner.budget_exhausted:
                        break
                    continue

                # No tool calls: this should be the conclusion.
                runner.record_thought(result.content[:1500] or "(empty reply)")
                if _has_json_object(result.content) or nudged:
                    return parse_conclusion(result.content)
                nudged = True
                messages.append(result.as_message())
                messages.append(Message.user(NUDGE))

            # Out of rounds (or tool budget): one last turn to conclude.
            messages.append(Message.user(CONCLUDE))
            result = await self.chat.complete(messages, tools=self.specs)
            usage.add(result.usage)
            if result.stop_reason == "refusal":
                return _declined()
            if result.tool_calls or not _has_json_object(result.content):
                return Conclusion(
                    verdict=Verdict.UNCERTAIN,
                    confidence=0.0,
                    conclusion=(
                        "The investigation reached its budget without a usable "
                        "conclusion from the model."
                    ),
                    recommended_action="Human review needed.",
                    evidence=[],
                )
            runner.record_thought(result.content[:1500])
            return parse_conclusion(result.content)
        finally:
            self.usage.add(usage)
            runner.record_thought(
                f"model usage: {usage.input_tokens} input / {usage.output_tokens} output "
                f"tokens ({self.chat.describe()})"
            )

    async def _execute(
        self, name: str, arguments: dict[str, Any], runner: ToolRunner
    ) -> dict[str, Any]:
        if INVALID_ARGUMENTS in arguments:
            return {
                "ok": False,
                "error": "the arguments were not valid JSON; send a JSON object "
                f"matching the tool's schema. Received: {arguments[INVALID_ARGUMENTS]!r}",
            }
        if name not in self._names:
            return {
                "ok": False,
                "error": f"unknown tool {name!r}; available tools: {sorted(self._names)}",
            }
        return await runner.call(name, arguments)

    def _render(self, outcome: dict[str, Any]) -> str:
        text = json.dumps(outcome, default=str, sort_keys=True)
        if len(text) <= self.result_char_limit:
            return text
        cut = len(text) - self.result_char_limit
        return text[: self.result_char_limit] + f" ...[truncated {cut} chars]"


def _has_json_object(text: str) -> bool:
    return "{" in (text or "") and "}" in (text or "")


def _declined() -> Conclusion:
    return Conclusion(
        verdict=Verdict.UNCERTAIN,
        confidence=0.0,
        conclusion="The model declined to continue this investigation.",
        recommended_action="Human review needed.",
        evidence=[],
    )
