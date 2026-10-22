"""Real VLM backends: Ollama and any OpenAI-compatible vision endpoint.

Both are thin -- the whole point of tier 2 is that it stays cheap. Note that
NVR already ships an ``ollama`` genai provider; we do not reuse that class
because it lives inside the NVR process and is wired to NVR's own prompts.
Talking to the same Ollama server over HTTP keeps the two independent.
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any

import httpx

from watchwise.llm.base import openai_url
from watchwise.llm.ollama import LOAD_TIMEOUT_S
from watchwise.llm.openai_compat import RequestTuning, error_message
from watchwise.tier2.vlm.base import VLMBackend, VLMReply
from watchwise.types import Frame

logger = logging.getLogger(__name__)


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
        max_tokens: int = 1024,
        context_window: int | None = None,
        keep_alive: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.keep_alive = keep_alive
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def load(self) -> bool:
        # With no prompt, Ollama only loads the model. It must get the same
        # num_ctx as real screens, or the first real screen reloads it.
        body: dict[str, Any] = {"model": self.model}
        if self.context_window:
            body["options"] = {"num_ctx": self.context_window}
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        r = await self._client.post(f"{self.base_url}/api/generate", json=body, timeout=LOAD_TIMEOUT_S)
        if r.status_code == 404:
            raise RuntimeError(
                f"model {self.model} is not on the Ollama server at {self.base_url}; "
                f"run: ollama pull {self.model}"
            )
        r.raise_for_status()
        return True

    async def infer(self, frames: list[Frame], prompt: str, system: str) -> VLMReply:
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
        if r.status_code == 404:
            raise RuntimeError(
                f"model {self.model} is not on the Ollama server at {self.base_url}; "
                f"run: ollama pull {self.model}"
            )
        r.raise_for_status()
        data = r.json()
        text = str(data.get("response") or "")
        return VLMReply(
            text,
            truncated=data.get("done_reason") == "length",
            max_tokens=self.max_tokens,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            thinking=bool(data.get("thinking")) or "<think>" in text.lower(),
        )


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
        max_tokens: int = 1024,
        effort: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tuning = RequestTuning(self.base_url, effort=effort)
        key = os.environ.get(api_key_env, "") if api_key_env else ""
        if api_key_env and not key:
            # Screening must fail open, so this is a warning, not an error:
            # calls will fail and screen as "uncertain" until the key is set.
            logger.warning("%s is not set; screening with %s will fail", api_key_env, model)
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s, headers=headers)
        if client is not None and headers:
            self._client.headers.update(headers)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def infer(self, frames: list[Frame], prompt: str, system: str) -> VLMReply:
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
        }
        url = openai_url(self.base_url, "chat/completions")
        for _ in range(5):  # at most one adaptation per tunable parameter
            request = self.tuning.apply(
                body, max_tokens=self.max_tokens, temperature=self.temperature, json_mode=True
            )
            r = await self._client.post(url, json=request)
            if not self.tuning.adapt(r):
                break
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {error_message(r)}")
        data = r.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        # content is null when a reasoning model spent the whole cap thinking.
        text = message.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        text = str(text)
        usage = data.get("usage") or {}
        reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        return VLMReply(
            text,
            truncated=choice.get("finish_reason") == "length",
            max_tokens=self.max_tokens,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            thinking=bool(
                message.get("reasoning_content") or message.get("reasoning") or reasoning_tokens
            )
            or "<think>" in text.lower(),
        )
