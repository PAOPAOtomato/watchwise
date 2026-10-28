"""The "Ask Watchwise" loop (D-065).

One question is one short ReAct loop over the ``chat`` role's model, like an
investigation: the model looks things up with the tools in ``tools.py``, then
answers in plain text. What must not depend on the model's good behaviour
stays here: the tool budget, the deadline, and failing open with an honest
"I couldn't answer that" instead of a guess (standing rule 5).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from watchwise.chat import prompts
from watchwise.chat.proposals import Proposal, ProposalBook
from watchwise.chat.tools import TOOLS, TOOLS_BY_NAME, ChatContext, _event
from watchwise.config import WatchwiseConfig
from watchwise.llm.base import INVALID_ARGUMENTS, ChatModel, Message, ToolSpec
from watchwise.memory.store import Store
from watchwise.memory.watches import WatchBook
from watchwise.nvr.client import NVRClient

logger = logging.getLogger(__name__)

HISTORY = 12  # messages of earlier turns kept in context
MAX_ROUNDS = 5
MAX_TOOL_CALLS = 8
RESULT_LIMIT = 2500
DEADLINE_S = 150.0


@dataclass
class ChatAnswer:
    text: str
    images: list[str] = field(default_factory=list)  # review ids to send as photos
    proposals: list[Proposal] = field(default_factory=list)


class ChatService:
    def __init__(
        self,
        model: ChatModel,
        *,
        store: Store,
        nvr: NVRClient,
        config: WatchwiseConfig,
        proposals: ProposalBook,
        watches: WatchBook | None = None,
        arming: Any = None,
        roster: Any = None,
        clock: Callable[[], float] = time.time,
        deadline_s: float = DEADLINE_S,
    ):
        self.model = model
        self.store = store
        self.nvr = nvr
        self.config = config
        self.proposals = proposals
        self.watches = watches
        self.arming = arming
        self.roster = roster
        self.clock = clock
        self.deadline_s = deadline_s
        self.specs = [ToolSpec(t.name, t.description, t.parameters) for t in TOOLS]

    def reset(self, conversation: str) -> None:
        self.store.clear_chat(conversation)

    async def ask(self, conversation: str, question: str, *, about: str | None = None) -> ChatAnswer:
        """Answer one question; ``about``: the event whose card the owner replied to."""
        try:
            cameras = await self.nvr.cameras()
        except Exception:
            cameras = sorted(self.config.cameras)
        ctx = ChatContext(
            store=self.store, nvr=self.nvr, config=self.config, proposals=self.proposals,
            watches=self.watches, conversation=conversation, cameras=cameras, clock=self.clock,
        )
        history = [Message(role=r["role"], content=r["content"]) for r in self.store.chat_history(conversation, HISTORY)]
        state = prompts.house_state(
            now=self.clock(), cameras=cameras, config=self.config, arming=self._arming(),
            roster=self._roster(), watches=self.watches.active() if self.watches else [],
            about=self._about(ctx, about),
        )
        messages = [Message.system(prompts.SYSTEM), *history, Message.user(f"{state}\n\nThe owner asks: {question}")]
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.deadline_s):
                text = await self._loop(ctx, messages)
        except TimeoutError:
            text = self._unanswered(ctx, "it took too long")
        except Exception as exc:
            logger.exception("chat answer failed")
            text = self._unanswered(ctx, f"the model failed: {type(exc).__name__}")
        self.store.add_chat_message(conversation, "user", question)
        self.store.add_chat_message(conversation, "assistant", text)
        used = [m.name for m in messages if m.role == "tool"]
        logger.info(
            "chat: answered in %.1fs with %d tool call(s) %s, %d photo(s), %d proposal(s)",
            time.monotonic() - started, len(used), used, len(ctx.images), len(ctx.proposed),
        )
        return ChatAnswer(text, ctx.images, ctx.proposed)

    async def _loop(self, ctx: ChatContext, messages: list[Message]) -> str:
        calls = 0
        for _ in range(MAX_ROUNDS):
            result = await self.model.complete(messages, tools=self.specs)
            if result.stop_reason == "refusal":
                return "I can't help with that one."
            if not result.tool_calls:
                if result.content.strip():
                    return result.content.strip()
                messages.append(Message.user(prompts.NUDGE))
                continue
            messages.append(result.as_message())
            for call in result.tool_calls:
                calls += 1
                outcome = (
                    await self._run(ctx, call.name, call.arguments)
                    if calls <= MAX_TOOL_CALLS
                    else {"ok": False, "error": "tool budget used up; answer with what you have"}
                )
                messages.append(Message.tool_result(call, _render(outcome), is_error=not outcome.get("ok", True)))
        messages.append(Message.user(prompts.NUDGE))
        result = await self.model.complete(messages, tools=None)
        return result.content.strip() or self._unanswered(ctx, "the model gave no answer")

    async def _run(self, ctx: ChatContext, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool {name!r}; tools: {sorted(TOOLS_BY_NAME)}"}
        if INVALID_ARGUMENTS in arguments:
            return {"ok": False, "error": "the arguments were not valid JSON; send an object matching the schema"}
        try:
            return await tool.handler(ctx, arguments)
        except Exception as exc:
            logger.exception("chat tool %s failed", name)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _unanswered(self, ctx: ChatContext, why: str) -> str:
        """Fail open: say so, and keep anything already proposed or found."""
        text = f"Sorry, I couldn't answer that ({why}). Try again, or ask more simply."
        if ctx.proposed:
            text += " The change I proposed is below; it does nothing unless you confirm it."
        return text

    def _arming(self) -> dict[str, Any] | None:
        try:
            return self.arming.get() if self.arming is not None else None
        except Exception:
            return None

    def _roster(self) -> list[dict[str, Any]]:
        try:
            return self.roster.entries() if self.roster is not None else []
        except Exception:
            return []

    def _about(self, ctx: ChatContext, review_id: str | None) -> dict[str, Any] | None:
        row = self.store.get_alert(review_id) if review_id else None
        return _event(ctx, row) if row is not None else None


def _render(outcome: dict[str, Any]) -> str:
    text = json.dumps(outcome, default=str, ensure_ascii=False)
    return text if len(text) <= RESULT_LIMIT else text[:RESULT_LIMIT] + " ...[truncated]"
