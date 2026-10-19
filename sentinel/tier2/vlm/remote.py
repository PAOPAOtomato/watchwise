"""Real VLM backends: Ollama and any OpenAI-compatible vision endpoint.

Both are thin -- the whole point of tier 2 is that it stays cheap. Note that
NVR already ships an ``ollama`` genai provider; we do not reuse that class
because it lives inside the NVR process and is wired to NVR's own prompts.
Talking to the same Ollama server over HTTP keeps the two independent.
"""

from __future__ import annotations

import base64
import os
from typing import Any

import httpx

from sentinel.llm.base import openai_url
from sentinel.tier2.vlm.base import VLMBackend
from sentinel.types import Frame


class OllamaVLM(VLMBackend):
    """Local Ollama server. Free per call; latency is whatever the GPU gives."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        *,
        timeout_s: float = 20.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
        context_window: int | None = None,
        keep_alive: str | None = None,
    ):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.keep_alive = keep_alive
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def infer(self, frames: list[Frame], prompt: str, system: str) -> str:
        images = [base64.b64encode(f.jpeg).decode("ascii") for f in frames]
        options: dict[str, Any] = {"temperature": self.temperature, "num_predict": self.max_tokens}
        if self.context_window:
            options["num_ctx"] = self.context_window
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "images": images,
            "stream": False,
            "format": "json",
            "options": options,
        }
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        r = await self._client.post(f"{self.base_url}/api/generate", json=body)
        r.raise_for_status()
        return str(r.json().get("response", ""))


class OpenAICompatVLM(VLMBackend):
    """Any /v1/chat/completions vision endpoint: vLLM, LM Studio, llama.cpp.

    Kept separate from the Anthropic path on purpose -- this backend is for a
    self-hosted OpenAI-shaped server, not for Claude.
    """

    name = "openai_compat"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key_env: str | None = None,
        timeout_s: float = 20.0,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        key = os.environ.get(api_key_env, "") if api_key_env else ""
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.AsyncClient(timeout=timeout_s, headers=headers)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def infer(self, frames: list[Frame], prompt: str, system: str) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for f in frames:
            b64 = base64.b64encode(f.jpeg).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        r = await self._client.post(openai_url(self.base_url, "chat/completions"), json=body)
        r.raise_for_status()
        data = r.json()
        return str(data["choices"][0]["message"]["content"])
