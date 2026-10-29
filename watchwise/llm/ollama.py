"""Ollama's native chat API (/api/chat).

Ollama also serves an OpenAI-compatible /v1 endpoint, but that dialect has no
way to set the context length, and Ollama's default is small enough to cut a
tool-calling prompt (system prompt + tool schemas + evidence) off silently.
The native API takes ``options.num_ctx`` per request, so it is the one used.
"""

from __future__ import annotations

import base64
import time
from typing import Any

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
    parse_arguments,
    split_reasoning,
)

DEFAULT_URL = "http://localhost:11434"
# Reading a model from disk into memory can take minutes on a slow disk, far
# longer than any per-request timeout.
LOAD_TIMEOUT_S = 600.0


class OllamaChat(ChatModel):
    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        *,
        timeout_s: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        context_window: int | None = 8192,
        keep_alive: str | None = None,
        think: bool | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(model)
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.keep_alive = keep_alive
        self.think = think
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def list_models(self) -> list[str]:
        try:
            r = await self._client.get(f"{self.base_url}/api/tags")
            r.raise_for_status()
            return sorted(str(m.get("name")) for m in r.json().get("models", []))
        except (httpx.HTTPError, ValueError):
            return []

    async def load(self) -> bool:
        # With no messages, Ollama only loads the model. It must get the same
        # num_ctx as real requests, or the first real request reloads it.
        body: dict[str, Any] = {"model": self.model, "messages": []}
        if self.context_window:
            body["options"] = {"num_ctx": self.context_window}
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        try:
            r = await self._client.post(f"{self.base_url}/api/chat", json=body, timeout=LOAD_TIMEOUT_S)
        except httpx.HTTPError as exc:
            raise ChatModelError(
                f"{self.describe()}: cannot reach Ollama at {self.base_url} ({type(exc).__name__})"
            ) from exc
        if r.status_code == 404:
            raise ChatModelError(f"{self.describe()}: model not found. Run: ollama pull {self.model}")
        if r.status_code >= 400:
            raise ChatModelError(f"{self.describe()}: HTTP {r.status_code}: {r.text[:300]}")
        return True

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        tool_choice: str = "auto",
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "num_predict": max_tokens or self.max_tokens,
        }
        if self.context_window:
            options["num_ctx"] = self.context_window
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [self._wire(m) for m in messages],
            "stream": False,
            "options": options,
        }
        if tools and tool_choice != "none":  # Ollama has no tool_choice
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
        if json_mode:
            body["format"] = "json"
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        if self.think is not None:
            body["think"] = self.think

        started = time.perf_counter()
        try:
            r = await self._client.post(f"{self.base_url}/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise ChatModelError(
                f"{self.describe()}: cannot reach Ollama at {self.base_url} "
                f"({type(exc).__name__}). Is it running?"
            ) from exc
        if r.status_code == 404:
            raise ChatModelError(
                f"{self.describe()}: model not found on the Ollama server. "
                f"Run: ollama pull {self.model}"
            )
        if r.status_code >= 400:
            raise ChatModelError(f"{self.describe()}: HTTP {r.status_code}: {r.text[:300]}")
        try:
            data = r.json()
        except ValueError as exc:
            raise ChatModelError(f"{self.describe()}: malformed response: {r.text[:300]}") from exc

        message = data.get("message") or {}
        visible, inline_reasoning = split_reasoning(str(message.get("content") or ""))
        reasoning = "\n".join(r for r in (message.get("thinking") or "", inline_reasoning) if r)

        calls = []
        for i, tc in enumerate(message.get("tool_calls") or []):
            fn = tc.get("function") or {}
            calls.append(
                ToolCallRequest(
                    id=str(tc.get("id") or f"call_{i}"),
                    name=str(fn.get("name", "")),
                    arguments=parse_arguments(fn.get("arguments")),
                )
            )
        if not calls and tools:
            calls, visible = extract_text_tool_calls(visible, {t.name for t in tools})

        stop = (
            "tool_calls" if calls else "length" if data.get("done_reason") == "length" else "stop"
        )
        return ChatResult(
            content=visible,
            tool_calls=calls,
            stop_reason=stop,
            usage=Usage(
                input_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
            ),
            model=str(data.get("model") or self.model),
            latency_ms=(time.perf_counter() - started) * 1000.0,
            reasoning=reasoning,
        )

    @staticmethod
    def _wire(m: Message) -> dict[str, Any]:
        out: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.images:
            out["images"] = [base64.b64encode(i.data).decode("ascii") for i in m.images]
        if m.role == "assistant" and m.tool_calls:
            out["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
            ]
        if m.role == "tool" and m.name:
            out["tool_name"] = m.name
        return out
