"""Any OpenAI-compatible chat endpoint: OpenAI, Groq, Gemini, OpenRouter,
LM Studio, llama.cpp server, vLLM, OpenVINO Model Server, and Ollama's /v1.

This is the widest door in the model layer: most hosted providers and most
local inference servers speak this dialect, so one class covers them all.
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any

import httpx

from sentinel.llm.base import (
    ChatModel,
    ChatModelError,
    ChatResult,
    Message,
    ToolCallRequest,
    ToolSpec,
    Usage,
    extract_text_tool_calls,
    openai_url,
    parse_arguments,
    split_reasoning,
)


class OpenAICompatChat(ChatModel):
    provider = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        api_key_env: str | None = None,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(model)
        if not base_url:
            raise ChatModelError("openai_compat needs a base_url")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        key = api_key or (os.environ.get(api_key_env, "") if api_key_env else "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s, headers=headers)
        if client is not None and headers:
            self._client.headers.update(headers)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def list_models(self) -> list[str]:
        try:
            r = await self._client.get(openai_url(self.base_url, "models"))
            r.raise_for_status()
            return sorted(str(m.get("id")) for m in r.json().get("data", []))
        except (httpx.HTTPError, ValueError):
            return []

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._wire(m) for m in messages],
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            body["tool_choice"] = "auto"
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            r = await self._client.post(openai_url(self.base_url, "chat/completions"), json=body)
        except httpx.HTTPError as exc:
            raise ChatModelError(f"{self.describe()}: {type(exc).__name__}: {exc}") from exc
        if r.status_code >= 400:
            raise ChatModelError(f"{self.describe()}: HTTP {r.status_code}: {r.text[:300]}")
        try:
            data = r.json()
            choice = data["choices"][0]
        except (ValueError, KeyError, IndexError) as exc:
            raise ChatModelError(f"{self.describe()}: malformed response: {r.text[:300]}") from exc

        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):  # some servers return content parts
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        visible, reasoning = split_reasoning(str(content))
        reasoning = "\n".join(
            r for r in (message.get("reasoning_content") or message.get("reasoning") or "", reasoning) if r
        )

        calls = [
            ToolCallRequest(
                id=str(tc.get("id") or f"call_{i}"),
                name=str((tc.get("function") or {}).get("name", "")),
                arguments=parse_arguments((tc.get("function") or {}).get("arguments")),
            )
            for i, tc in enumerate(message.get("tool_calls") or [])
        ]
        if not calls and tools:
            calls, visible = extract_text_tool_calls(visible, {t.name for t in tools})

        finish = choice.get("finish_reason")
        stop = (
            "tool_calls"
            if calls
            else "length"
            if finish == "length"
            else "refusal"
            if finish == "content_filter"
            else "stop"
        )
        usage = data.get("usage") or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        return ChatResult(
            content=visible,
            tool_calls=calls,
            stop_reason=stop,
            usage=Usage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                cache_read_tokens=int(cached),
            ),
            model=str(data.get("model") or self.model),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            reasoning=reasoning,
        )

    @staticmethod
    def _wire(m: Message) -> dict[str, Any]:
        if m.role == "tool":
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
        if m.role == "assistant":
            out: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                out["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in m.tool_calls
                ]
            return out
        if m.images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": m.content}]
            for img in m.images:
                b64 = base64.b64encode(img.data).decode("ascii")
                parts.append(
                    {"type": "image_url", "image_url": {"url": f"data:{img.media_type};base64,{b64}"}}
                )
            return {"role": m.role, "content": parts}
        return {"role": m.role, "content": m.content}
