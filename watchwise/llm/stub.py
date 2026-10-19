"""A deterministic chat model for tests, demos and wiring checks.

With a script, it replays the given results in order (each entry may also be
a callable that sees the conversation). Without one, it behaves like the
dullest possible competent model: call the first offered tool once, then
answer with an honest "uncertain" conclusion. It never pretends to judge.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Union

from watchwise.llm.base import ChatModel, ChatResult, Message, ToolCallRequest, ToolSpec

Step = Union[ChatResult, Callable[[list[Message], list[ToolSpec] | None], ChatResult]]

STUB_CONCLUSION = {
    "verdict": "uncertain",
    "confidence": 0.5,
    "conclusion": "stub model: no real judgement was made",
    "recommended_action": "none",
    "evidence": [],
}


class StubChat(ChatModel):
    provider = "stub"

    def __init__(self, model: str = "stub", script: list[Step] | None = None):
        super().__init__(model)
        self.script: list[Step] = list(script or [])
        self.calls: list[tuple[list[Message], list[ToolSpec] | None]] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        self.calls.append((list(messages), tools))
        if self.script:
            step = self.script.pop(0)
            return step(messages, tools) if callable(step) else step
        if tools and not any(m.role == "tool" for m in messages):
            call = ToolCallRequest(id="stub_call_1", name=tools[0].name, arguments={})
            return ChatResult(content="", tool_calls=[call], stop_reason="tool_calls", model=self.model)
        return ChatResult(content=json.dumps(STUB_CONCLUSION), model=self.model)
