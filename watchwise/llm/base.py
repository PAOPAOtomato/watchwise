"""The provider-neutral chat-model interface.

Every model Watchwise talks to -- a 2B vision model on a laptop CPU, a 14B
model on a GPU, or a cloud API -- sits behind one small interface: send a list
of messages (optionally with images and tool definitions), get back text
and/or tool calls. The tier-3 investigation loop and the chat agent are
written against this interface only, which is what lets a user swap hardware
and models by editing config instead of code.

Providers are responsible for transport and for normalizing their replies.
Small local models are untidy: they wrap JSON in prose, write tool calls into
their text instead of the structured field, and leak ``<think>`` blocks. The
helpers at the bottom of this module tidy that up once, for every provider.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

Role = Literal["system", "user", "assistant", "tool"]
StopReason = Literal["stop", "tool_calls", "length", "refusal"]

# Marker placed in ``ToolCallRequest.arguments`` when a model's arguments could
# not be parsed as JSON. The caller reports it back instead of running the tool.
INVALID_ARGUMENTS = "__invalid_arguments__"


class ChatModelError(RuntimeError):
    """Transport or provider failure. Callers treat it as "no answer"."""


@dataclass(slots=True)
class ImagePart:
    data: bytes
    media_type: str = "image/jpeg"

    def __repr__(self) -> str:  # keep bytes out of logs
        return f"ImagePart({self.media_type}, {len(self.data)} bytes)"


@dataclass(slots=True)
class ToolCallRequest:
    """A model asking for one tool to be run."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolSpec:
    """A tool offered to the model: name, description, JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    images: list[ImagePart] = field(default_factory=list)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    # For role="tool": which call this answers, and the tool's name.
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    # Provider-native content to replay verbatim on the next request to the
    # same provider (e.g. Anthropic thinking blocks, which must be passed back
    # unchanged). Other providers ignore it.
    native: Any = None
    native_provider: str | None = None

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls(role="system", content=text)

    @classmethod
    def user(cls, text: str, images: list[ImagePart] | None = None) -> "Message":
        return cls(role="user", content=text, images=list(images or []))

    @classmethod
    def tool_result(
        cls, call: ToolCallRequest, content: str, *, is_error: bool = False
    ) -> "Message":
        return cls(
            role="tool",
            content=content,
            tool_call_id=call.id,
            name=call.name,
            is_error=is_error,
        )


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


@dataclass(slots=True)
class ChatResult:
    content: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: StopReason = "stop"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    latency_ms: float = 0.0
    # Visible reasoning, when a provider returns it (never required).
    reasoning: str = ""
    native: Any = None
    native_provider: str | None = None

    def as_message(self) -> Message:
        """The assistant turn to append to the conversation."""
        return Message(
            role="assistant",
            content=self.content,
            tool_calls=list(self.tool_calls),
            native=self.native,
            native_provider=self.native_provider,
        )


class ChatModel(ABC):
    """One model behind one provider."""

    provider: str = "base"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        """Run one model turn."""

    async def list_models(self) -> list[str]:
        """Models the server offers, where the provider can say. Default: unknown."""
        return []

    async def load(self) -> bool:
        """Load the model into memory before it is needed, on servers that
        load on demand. False means there is nothing to load (the default)."""
        return False

    async def aclose(self) -> None:
        """Release transport resources. Default: nothing to do."""

    def describe(self) -> str:
        return f"{self.provider}/{self.model}"


# ---------------------------------------------------------------- helpers


def openai_url(base_url: str, path: str) -> str:
    """Join an OpenAI-style base URL and an endpoint path.

    Accepts both conventions people paste in:
      http://localhost:11434          -> http://localhost:11434/v1/chat/completions
      https://api.openai.com/v1       -> https://api.openai.com/v1/chat/completions
      https://api.groq.com/openai/v1  -> https://api.groq.com/openai/v1/chat/completions
    """
    base = base_url.rstrip("/")
    if urlsplit(base).path in ("", "/"):
        base += "/v1"
    return f"{base}/{path.lstrip('/')}"


_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def split_reasoning(text: str) -> tuple[str, str]:
    """Separate ``<think>...</think>`` blocks from the visible reply.

    Returns ``(visible, reasoning)``. An unclosed ``<think>`` (a reply cut off
    mid-thought) is treated as reasoning with no visible answer.
    """
    if not text:
        return "", ""
    reasoning = [m.group(1).strip() for m in _THINK_BLOCK.finditer(text)]
    visible = _THINK_BLOCK.sub("", text)
    lowered = visible.lower()
    if "<think>" in lowered:
        cut = lowered.index("<think>")
        reasoning.append(visible[cut + len("<think>"):].strip())
        visible = visible[:cut]
    return visible.strip(), "\n".join(r for r in reasoning if r)


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Coerce a tool call's arguments into a dict, as forgivingly as is safe.

    Unparseable arguments come back as ``{INVALID_ARGUMENTS: raw}`` so the
    caller can tell the model what went wrong instead of running the tool
    with garbage.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {INVALID_ARGUMENTS: str(raw)[:500]}
    for candidate in (raw, *(m.group(0) for m in [_OBJECT.search(raw)] if m)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {INVALID_ARGUMENTS: raw[:500]}


def _loads_any(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _OBJECT.search(text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _call_from_object(obj: Any, tool_names: set[str], index: int) -> ToolCallRequest | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):  # {"type": "function", "function": {...}}
        obj = obj["function"]
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if name not in tool_names:
        return None
    for key in ("arguments", "args", "parameters", "input"):
        if key in obj:
            args = obj[key]
            break
    else:
        args = {}
    return ToolCallRequest(id=f"call_text_{index}", name=name, arguments=parse_arguments(args))


def extract_text_tool_calls(
    text: str, tool_names: set[str]
) -> tuple[list[ToolCallRequest], str]:
    """Recover tool calls a model wrote into its reply text.

    Handles ``<tool_call>{...}</tool_call>`` tags (Qwen/Hermes style) and a
    reply that is nothing but a JSON object (optionally fenced) naming a known
    tool. Only names in ``tool_names`` are accepted, so a final-answer JSON
    object is never mistaken for a call. Returns ``(calls, remaining_text)``.
    """
    if not text or not tool_names:
        return [], text
    calls: list[ToolCallRequest] = []
    for match in _TOOL_TAG.finditer(text):
        call = _call_from_object(_loads_any(match.group(1)), tool_names, len(calls))
        if call is not None:
            calls.append(call)
    if calls:
        return calls, _TOOL_TAG.sub("", text).strip()

    fence = _FENCE.search(text)
    candidate = fence.group(1) if fence else text
    parsed = _loads_any(candidate)
    objects = parsed if isinstance(parsed, list) else [parsed]
    for obj in objects:
        call = _call_from_object(obj, tool_names, len(calls))
        if call is not None:
            calls.append(call)
    if calls:
        return calls, ""
    return [], text
