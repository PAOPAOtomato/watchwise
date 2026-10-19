"""Claude via the official Anthropic SDK (Messages API).

Design notes, all of them requirements of the current API rather than taste:

* **Thinking** is left at the model default (adaptive on current models), and
  the assistant's content blocks are replayed *verbatim* on the next turn --
  thinking blocks must be passed back unchanged, and the conversation is kept
  append-only so they stay valid.
* **Parallel tool results** go back in a single user message.
* **Refusals** come back as HTTP 200 with ``stop_reason == "refusal"``. For
  models that support it, server-side fallback (``fallbacks="default"``) is on
  by default so a declined request is retried on Anthropic's recommended
  fallback model; whatever still comes back refused is reported as
  ``stop_reason="refusal"``, which Sentinel treats as "ask a human".
* **Prompt caching** uses top-level automatic caching; the system prompt and
  tool list are kept byte-stable so the prefix actually caches.
* ``temperature`` is not sent: current models reject non-default sampling.

Requires the optional dependency: ``pip install "sentinel[anthropic]"``.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

from sentinel.llm.base import (
    ChatModel,
    ChatModelError,
    ChatResult,
    Message,
    ToolCallRequest,
    ToolSpec,
    Usage,
)

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept the server-side ``fallbacks="default"`` parameter.
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)


class AnthropicChat(ChatModel):
    provider = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key_env: str | None = "ANTHROPIC_API_KEY",
        base_url: str | None = None,
        timeout_s: float = 180.0,
        max_tokens: int = 16000,
        effort: str | None = None,
        refusal_fallback: bool = True,
        cache: bool = True,
        max_retries: int = 2,
        http_client: Any = None,
    ):
        super().__init__(model)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # optional dependency
            raise ChatModelError(
                'the anthropic provider needs the SDK: pip install "sentinel[anthropic]"'
            ) from exc

        self.max_tokens = max_tokens
        self.effort = effort
        self.refusal_fallback = refusal_fallback and model in FALLBACK_MODELS
        self.cache = cache
        kwargs: dict[str, Any] = {"timeout": timeout_s, "max_retries": max_retries}
        key = os.environ.get(api_key_env) if api_key_env else None
        if key:
            kwargs["api_key"] = key  # otherwise the SDK resolves env/profile itself
        if base_url:
            kwargs["base_url"] = base_url
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = AsyncAnthropic(**kwargs)

    async def aclose(self) -> None:
        await self._client.close()

    async def list_models(self) -> list[str]:
        try:
            return sorted([m.id async for m in self._client.models.list()])
        except Exception:
            return []

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        import anthropic

        system, wire = self.to_wire(messages)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": wire,
        }
        if system:
            params["system"] = system
        if tools:
            params["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]
        if self.cache:
            params["cache_control"] = {"type": "ephemeral"}
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.refusal_fallback:
            params["betas"] = [FALLBACK_BETA]
            params["fallbacks"] = "default"

        started = time.perf_counter()
        try:
            response = await self._client.beta.messages.create(**params)
        except anthropic.AuthenticationError as exc:
            raise ChatModelError(
                f"{self.describe()}: authentication failed; set ANTHROPIC_API_KEY in .env"
            ) from exc
        except anthropic.NotFoundError as exc:
            raise ChatModelError(f"{self.describe()}: unknown model or endpoint: {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise ChatModelError(f"{self.describe()}: rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ChatModelError(
                f"{self.describe()}: HTTP {exc.status_code}: {exc.message}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise ChatModelError(f"{self.describe()}: connection failed: {exc}") from exc

        blocks = list(response.content or [])
        text = "\n".join(b.text for b in blocks if b.type == "text").strip()
        thinking = "\n".join((b.thinking or "") for b in blocks if b.type == "thinking").strip()
        calls = [
            ToolCallRequest(id=b.id, name=b.name, arguments=dict(b.input or {}))
            for b in blocks
            if b.type == "tool_use"
        ]
        if response.stop_reason == "refusal":
            stop = "refusal"
            category = getattr(response.stop_details, "category", None)
            thinking = f"declined by the model (category: {category or 'unspecified'})"
        elif calls:
            stop = "tool_calls"
        elif response.stop_reason == "max_tokens":
            stop = "length"
        else:
            stop = "stop"

        usage = response.usage
        return ChatResult(
            content="" if stop == "refusal" else text,
            tool_calls=[] if stop == "refusal" else calls,
            stop_reason=stop,
            usage=Usage(
                input_tokens=int(usage.input_tokens or 0),
                output_tokens=int(usage.output_tokens or 0),
                cache_read_tokens=int(usage.cache_read_input_tokens or 0),
                cache_write_tokens=int(usage.cache_creation_input_tokens or 0),
            ),
            model=str(response.model or self.model),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            reasoning=thinking,
            native=[b.to_dict() for b in blocks],
            native_provider=self.provider,
        )

    def to_wire(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """Convert neutral messages to (system, Messages-API messages)."""
        system = "\n\n".join(m.content for m in messages if m.role == "system" and m.content)
        wire: list[dict[str, Any]] = []

        def push(role: str, blocks: list[dict[str, Any]]) -> None:
            # The API needs alternating roles; consecutive tool results and a
            # following user note become one user message (results first).
            if wire and wire[-1]["role"] == role:
                wire[-1]["content"].extend(blocks)
            else:
                wire.append({"role": role, "content": blocks})

        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                push(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                            "is_error": m.is_error,
                        }
                    ],
                )
            elif m.role == "assistant":
                if m.native is not None and m.native_provider == self.provider:
                    blocks = [dict(b) for b in m.native]
                else:
                    blocks = [{"type": "text", "text": m.content}] if m.content else []
                    blocks += [
                        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                        for c in m.tool_calls
                    ]
                if blocks:
                    push("assistant", blocks)
            else:
                blocks = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img.media_type,
                            "data": base64.standard_b64encode(img.data).decode("ascii"),
                        },
                    }
                    for img in m.images
                ]
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                push("user", blocks)
        return system, wire
