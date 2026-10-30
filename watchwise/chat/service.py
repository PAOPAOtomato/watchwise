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
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from watchwise.chat import prompts
from watchwise.chat.proposals import TTL_S, Proposal, ProposalBook
from watchwise.chat.tools import TOOLS, TOOLS_BY_NAME, ChatContext, _event
from watchwise.config import WatchwiseConfig
from watchwise.llm.base import INVALID_ARGUMENTS, ChatModel, Message, ToolCallRequest, ToolSpec
from watchwise.memory.store import Store
from watchwise.memory.watches import WatchBook
from watchwise.nvr.client import NVRClient

logger = logging.getLogger(__name__)

HISTORY = 12  # messages of earlier turns kept in context
MAX_ROUNDS = 5
MAX_TOOL_CALLS = 8
RESULT_LIMIT = 2500
DEADLINE_S = 150.0

# The context budget (D-069). Past the model's context window Ollama silently
# drops the oldest messages, which in a long tool loop can be the question.
IDLE_S = 6 * 3600  # a question after this much quiet starts a new conversation
ANSWER_CHARS = 600  # an earlier answer, as sent back with later questions
CHARS_PER_TOKEN = 3.0  # live, 3.5 came out 10-14% under Ollama's own counts
DEFAULT_WINDOW = 32768  # hosted models: their context isn't configured here

# An answer that says a change was proposed or made (B-031). Checked only
# when no propose_ tool ran in that answer.
CLAIMED_CHANGE = re.compile(
    r"\bI(?:'ve|’ve| have)?\s+(?:just\s+|now\s+)?(?:proposed|added|set up|set|switched|armed|disarmed|muted|"
    r"created|scheduled|started watching|updated|changed|saved|confirmed|activated)\b"
    r"|\b(?:is|are)\s+now\s+(?:active|armed|disarmed|muted|on|set|in place|confirmed|watching)\b"
    r"|\barming\s+(?:is\s+now|was\s+switched|has\s+been\s+(?:set|switched))\b"
    r"|\b(?:has|have)\s+been\s+(?:added|applied|confirmed|set up|activated|created|armed|disarmed|muted|"
    r"switched|changed|updated|proposed)\b"
    r"|\b(?:rules?|changes?|watch(?:es)?)\s+(?:is|are)\s+(?:confirmed|active|applied)\b"
    r"|\bI(?:'ll|’ll| will)\s+(?:alert|notify|tell|message)\s+you\s+when\b"
    r"|\bI(?:'ll|’ll| will)\s+let\s+you\s+know\s+when\b"
    r"|^\W*(?:✅\s*)?(?:confirmed|done|applied)\b"
    r"|\b(?:is|are)\s+now\s+(?:in|on)\s+the\s+(?:duty\s+)?roster\b",
    re.IGNORECASE,
)
# A message that only confirms: live, a typed "Confirm" got "✅ Confirmed: ...
# is now in the duty roster" from the model. Only the button applies a change.
BARE_CONFIRM = re.compile(
    r"^\W*(?:confirm(?:ed)?|yes|yep|yeah|y|ok(?:ay)?|sure|do it|go ahead|apply( it)?|✅|👍)\W*$", re.IGNORECASE
)
# Pointing to the button is honest while an earlier proposal still waits.
TAP_CONFIRM = re.compile(r"\btap\s+(?:the\s+)?(?:✅\s*)?confirm\b", re.IGNORECASE)


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
        budget_tokens: int | None = None,
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
        self.budget = budget_tokens or chat_budget(config.endpoint("chat"))
        self._schema_tokens = _tokens(json.dumps([[s.name, s.description, s.parameters] for s in self.specs]))

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
        history = self._history(conversation)
        waiting = self.proposals.waiting(conversation)
        if BARE_CONFIRM.match(question.strip()):
            text = prompts.typed_confirmation(waiting)
            now = self.clock()
            self.store.add_chat_message(conversation, "user", question, at=now)
            self.store.add_chat_message(conversation, "assistant", text, at=now, meta={
                "about": about, "fixed": "typed_confirmation", "waiting": [p.id for p in waiting],
            })
            logger.info("chat: a typed confirmation, answered without the model (%d waiting)", len(waiting))
            return ChatAnswer(text)
        state = prompts.house_state(
            now=self.clock(), cameras=cameras, config=self.config, arming=self._arming(),
            roster=self._roster(), watches=self.watches.active() if self.watches else [],
            waiting=waiting, about=self._about(ctx, about),
        )
        asked = Message.user(f"{state}\n\nThe owner asks: {question}")
        messages = [Message.system(prompts.SYSTEM), *history, asked]
        fit = _Fit(asked)
        started = time.monotonic()
        meta: dict[str, Any] = {"about": about}
        try:
            async with asyncio.timeout(self.deadline_s):
                text = await self._loop(ctx, messages, fit)
                if self._unbacked(ctx, text, waiting):
                    logger.warning("chat: the answer claimed a change that no tool proposed; asking again")
                    meta["guard"] = "claimed_change"
                    messages += [Message(role="assistant", content=text), Message.user(prompts.UNBACKED)]
                    text = await self._loop(ctx, messages, fit)
                    if self._unbacked(ctx, text, waiting):
                        meta["guard"] = "nothing_changed"
                        text = prompts.NOTHING_CHANGED
        except TimeoutError:
            meta["error"] = "timeout"
            text = self._unanswered(ctx, "it took too long")
        except Exception as exc:
            logger.exception("chat answer failed")
            meta["error"] = type(exc).__name__
            text = self._unanswered(ctx, f"the model failed: {type(exc).__name__}")
        start = next((i for i, m in enumerate(messages) if m is asked), 0)
        meta.update(
            tools=_steps(messages[start + 1:]), images=list(ctx.images), proposals=[p.id for p in ctx.proposed],
            seconds=round(time.monotonic() - started, 1), prompt_tokens=fit.reported,
        )
        now = self.clock()
        self.store.add_chat_message(conversation, "user", question, at=now)
        self.store.add_chat_message(conversation, "assistant", text, at=now, meta=meta)
        used = [m.name for m in messages[start:] if m.role == "tool"]  # not the history's
        logger.info(
            "chat: answered in %.1fs with %d tool call(s) %s, %d photo(s), %d proposal(s); "
            "prompt %d tokens (estimated %d of %d)%s",
            time.monotonic() - started, len(used), used, len(ctx.images), len(ctx.proposed),
            fit.reported, fit.estimated, self.budget, fit.describe(),
        )
        return ChatAnswer(text, ctx.images, ctx.proposed)

    async def _loop(self, ctx: ChatContext, messages: list[Message], fit: "_Fit") -> str:
        calls = 0
        for _ in range(MAX_ROUNDS):
            self._fit(messages, fit)
            result = await self.model.complete(messages, tools=self.specs)
            fit.reported = max(fit.reported, result.usage.input_tokens)
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
        self._fit(messages, fit)
        result = await self.model.complete(messages, tools=self.specs, tool_choice="none")
        fit.reported = max(fit.reported, result.usage.input_tokens)
        return result.content.strip() or self._unanswered(ctx, "the model gave no answer")

    def _history(self, conversation: str) -> list[Message]:
        """Earlier turns of this conversation, back to the last quiet gap of
        IDLE_S: after one, a question starts afresh (D-069). Stored answers
        stay whole; the copy sent back is shortened."""
        rows = self.store.chat_history(conversation, HISTORY)
        kept: list[Any] = []
        later = self.clock()
        for row in reversed(rows):
            if later - float(row["created_at"]) > IDLE_S:
                break
            kept.append(row)
            later = float(row["created_at"])
        if len(kept) < len(rows):
            logger.info("chat: %d earlier message(s) left out after a quiet gap", len(rows) - len(kept))
        while kept and kept[-1]["role"] != "user":
            kept.pop()  # start at a question
        history: list[Message] = []
        since = self.clock() - IDLE_S
        for row in reversed(kept):
            if row["role"] == "assistant":
                history += self._proposal_steps(conversation, since, float(row["created_at"]))
                history.append(Message(role="assistant", content=_shorten(row["content"])))
                since = float(row["created_at"])  # a question shares its answer's time
            else:
                history.append(Message(role="user", content=row["content"]))
        return history

    def _proposal_steps(self, conversation: str, after: float, until: float) -> list[Message]:
        """The tool calls behind an earlier answer's proposals, and how each
        ended. Live, with only the answers' text in the history ("✅ Proposal:
        ... Tap Confirm"), the small model wrote that text again instead of
        calling the tool (B-031); this shows it how a proposal is made."""
        rows = self.store.query(
            "SELECT id, kind, payload, summary, status, created_at FROM proposals "
            "WHERE conversation=? AND created_at>? AND created_at<=? ORDER BY created_at",
            (conversation, after, until),
        )
        steps: list[Message] = []
        for row in rows:
            call = ToolCallRequest(f"h-{row['id']}", f"propose_{row['kind']}", _arguments(row["kind"], json.loads(row["payload"])))
            status = row["status"]
            if status == "pending" and self.clock() - float(row["created_at"]) > TTL_S:
                status = "expired"
            then = {"applied": "the owner tapped Confirm; it was applied", "cancelled": "the owner cancelled it",
                    "pending": "still waiting for the owner's tap"}.get(status, status)
            steps += [Message(role="assistant", tool_calls=[call]),
                      Message.tool_result(call, json.dumps({"ok": True, "proposal": row["summary"], "then": then},
                                                           ensure_ascii=False))]
        return steps

    def _fit(self, messages: list[Message], fit: "_Fit") -> None:
        """Keep the prompt within the budget (D-069): whole earlier turns go
        first, oldest first; then this answer's older tool results shrink to a
        stub. The system prompt, the question and the newest tool result stay."""
        size = self._schema_tokens + sum(_tokens(_text(m)) for m in messages)
        while size > self.budget and next(i for i, m in enumerate(messages) if m is fit.asked) > 1:
            end = 2
            while messages[end] is not fit.asked and messages[end].role != "user":
                end += 1
            size -= sum(_tokens(_text(m)) for m in messages[1:end])
            fit.dropped += end - 1
            del messages[1:end]
        results = [i for i, m in enumerate(messages) if m.role == "tool"][:-1]
        for i in results:
            if size <= self.budget:
                break
            old = messages[i]
            stub = f"[{old.name} result, {len(old.content)} characters, shown earlier; left out to save room]"
            if len(stub) < len(old.content):
                size -= _tokens(old.content) - _tokens(stub)
                messages[i] = replace(old, content=stub)
                fit.shortened += 1
        fit.estimated = max(fit.estimated, int(size))
        if size > self.budget and not fit.warned:
            fit.warned = True
            logger.warning("chat: the prompt is still ~%d tokens, over the %d budget", size, self.budget)

    @staticmethod
    def _unbacked(ctx: ChatContext, text: str, waiting: list[Proposal]) -> bool:
        """Does the answer claim a change, with no proposal made in it (or,
        for "tap Confirm", none waiting either)?"""
        if ctx.proposed:
            return False
        return bool(CLAIMED_CHANGE.search(text)) or (not waiting and bool(TAP_CONFIRM.search(text)))

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


@dataclass
class _Fit:
    """One answer's context bookkeeping (D-069)."""

    asked: Message  # the question, with the house state
    dropped: int = 0  # earlier messages left out
    shortened: int = 0  # tool results replaced by a stub
    estimated: int = 0
    reported: int = 0  # the provider's own count, when it gives one
    warned: bool = False

    def describe(self) -> str:
        parts = []
        if self.dropped:
            parts.append(f"{self.dropped} earlier message(s) left out")
        if self.shortened:
            parts.append(f"{self.shortened} tool result(s) shortened")
        return f"; {', '.join(parts)}" if parts else ""


def _arguments(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    """A proposal's tool arguments, rebuilt from what it stored."""
    entry = payload.get("entry") or {}
    cameras = payload.get("cameras") or entry.get("cameras") or []
    one = {"camera": cameras[0]} if len(cameras) == 1 else {}
    if kind == "roster_rule":
        return {k: entry[k] for k in ("name", "days", "start", "end") if k in entry} | one
    if kind == "roster_removal":
        return {"name": entry.get("name")}
    if kind == "arming":
        return {"state": payload.get("state")}
    if kind == "snooze":
        return one | {"minutes": payload.get("minutes")}
    if kind == "watch":
        return {"request": payload.get("request"), "hours": payload.get("hours")} | one
    return {}


def chat_budget(endpoint: Any) -> int:
    """Tokens a question may fill: the model's context window less room for
    its answer and a 10% margin."""
    window = (endpoint.context_window if endpoint is not None else None) or DEFAULT_WINDOW
    answer = endpoint.max_tokens if endpoint is not None else 2048
    return int((window - answer) * 0.9)


def _tokens(text: str) -> float:
    return len(text) / CHARS_PER_TOKEN


def _text(message: Message) -> str:
    calls = "".join(json.dumps(c.arguments, default=str) + c.name for c in message.tool_calls)
    return message.content + calls


def _shorten(text: str) -> str:
    return text if len(text) <= ANSWER_CHARS else text[: ANSWER_CHARS - 1].rstrip() + "…"


def _render(outcome: dict[str, Any]) -> str:
    text = json.dumps(outcome, default=str, ensure_ascii=False)
    return text if len(text) <= RESULT_LIMIT else text[:RESULT_LIMIT] + " ...[truncated]"


def _steps(messages: list[Message]) -> list[dict[str, Any]]:
    """The tool calls behind one answer, each with what it returned, as the
    model saw it (kept with the answer for the trace export, M3-9)."""
    steps: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.role == "assistant":
            for call in message.tool_calls:
                by_id[call.id] = {"tool": call.name, "arguments": call.arguments, "ok": None, "result": ""}
                steps.append(by_id[call.id])
        elif message.role == "tool" and message.tool_call_id in by_id:
            by_id[message.tool_call_id].update(ok=not message.is_error, result=message.content)
    return steps
