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
from urllib.parse import urlsplit

import httpx

from watchwise.llm.base import (
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


# Our effort levels, as OpenAI's reasoning_effort values.
_REASONING_EFFORT = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}


def error_message(response: httpx.Response) -> str:
    """The human-readable part of an error response, whatever its shape."""
    try:
        data = response.json()
    except ValueError:
        return response.text[:300]
    err = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(err, dict):
        return str(err.get("message") or err.get("detail") or err)[:300]
    return str(err)[:300]


class RequestTuning:
    """Which optional request parameters a server accepts, learned once.

    "OpenAI-compatible" servers disagree on details. OpenAI's current models
    require ``max_completion_tokens`` and accept only the default temperature;
    most local servers expect ``max_tokens``; not all accept
    ``reasoning_effort`` or ``response_format``. Instead of a per-server table,
    a 400 that names one of these parameters switches or drops it (once each)
    and the request is retried. The result is remembered for the client's
    lifetime, so the cost is at most one rejected, unbilled request each.
    """

    def __init__(self, base_url: str, *, effort: str | None = None):
        host = (urlsplit(base_url).hostname or "").lower()
        self.token_param = "max_completion_tokens" if host.endswith("openai.com") else "max_tokens"
        self.send_temperature = True
        self.reasoning_effort = _REASONING_EFFORT.get(effort) if effort else None
        self.send_response_format = True
        self._token_switched = False

    def apply(
        self,
        body: dict[str, Any],
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        out = dict(body)
        out[self.token_param] = max_tokens
        if self.send_temperature:
            out["temperature"] = temperature
        if self.reasoning_effort:
            out["reasoning_effort"] = self.reasoning_effort
        if json_mode and self.send_response_format:
            out["response_format"] = {"type": "json_object"}
        return out

    def adapt(self, response: httpx.Response) -> bool:
        """Adjust to a parameter the server rejected. True means retry."""
        if response.status_code != 400:
            return False
        try:
            data = response.json()
        except ValueError:
            return False
        err = data.get("error") if isinstance(data, dict) else None
        if not isinstance(err, dict):
            return False
        param = str(err.get("param") or "")
        message = str(err.get("message") or "")
        if param in ("max_tokens", "max_completion_tokens") or "max_completion_tokens" in message:
            if self._token_switched:
                return False
            self.token_param = (
                "max_tokens" if self.token_param == "max_completion_tokens" else "max_completion_tokens"
            )
            self._token_switched = True
            return True
        if param == "temperature" and self.send_temperature:
            self.send_temperature = False
            return True
        if param == "reasoning_effort" and self.reasoning_effort:
            self.reasoning_effort = None
            return True
        if param == "response_format" and self.send_response_format:
            self.send_response_format = False
            return True
        return False


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
        effort: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(model)
        if not base_url:
            raise ChatModelError("openai_compat needs a base_url")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tuning = RequestTuning(base_url, effort=effort)
        key = api_key or (os.environ.get(api_key_env, "") if api_key_env else "")
        if api_key_env and not key:
            # A named key that is missing is a setup error, not "no auth":
            # say so now instead of failing later with an opaque HTTP 401.
            raise ChatModelError(f"{model}: {api_key_env} is not set; add it to .env")
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
        tool_choice: str = "auto",
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._wire(m) for m in messages],
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
            body["tool_choice"] = tool_choice

        url = openai_url(self.base_url, "chat/completions")
        for _ in range(5):  # at most one adaptation per tunable parameter
            request = self.tuning.apply(
                body,
                max_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature,
                json_mode=json_mode,
            )
            started = time.perf_counter()
            try:
                r = await self._client.post(url, json=request)
            except httpx.HTTPError as exc:
                raise ChatModelError(f"{self.describe()}: {type(exc).__name__}: {exc}") from exc
            if not self.tuning.adapt(r):
                break
        if r.status_code >= 400:
            raise ChatModelError(f"{self.describe()}: HTTP {r.status_code}: {error_message(r)}")
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
