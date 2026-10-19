"""Backend selection. One place that maps config -> concrete VLM."""

from __future__ import annotations

from watchwise.config import ModelEndpoint, VLMSettings
from watchwise.tier2.vlm.base import VLMBackend
from watchwise.tier2.vlm.stub import StubVLM


def build_vlm(settings: VLMSettings) -> VLMBackend:
    if settings.backend == "stub":
        return StubVLM(settings.model)

    if settings.backend == "anthropic":
        from watchwise.llm.registry import build_chat_model
        from watchwise.tier2.vlm.chat_adapter import ChatModelVLM

        chat = build_chat_model(
            ModelEndpoint(
                provider="anthropic",
                model=settings.model,
                base_url=settings.base_url,
                api_key_env=settings.api_key_env,
                timeout_s=settings.timeout_s,
                effort=settings.effort,
                refusal_fallback=settings.refusal_fallback,
            )
        )
        return ChatModelVLM(chat)

    # Imported lazily so a stub-only deployment never needs httpx clients open.
    from watchwise.tier2.vlm.remote import OllamaVLM, OpenAICompatVLM

    if settings.backend == "ollama":
        return OllamaVLM(
            settings.model,
            settings.base_url or "http://localhost:11434",
            timeout_s=settings.timeout_s,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            context_window=settings.context_window,
            keep_alive=settings.keep_alive,
        )
    if settings.backend == "openai_compat":
        if not settings.base_url:
            raise ValueError("vlm.base_url is required for the openai_compat backend")
        return OpenAICompatVLM(
            settings.model,
            settings.base_url,
            api_key_env=settings.api_key_env,
            timeout_s=settings.timeout_s,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    raise ValueError(f"unknown vlm backend: {settings.backend}")
