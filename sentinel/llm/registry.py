"""One place that maps a configured endpoint to a concrete chat model."""

from __future__ import annotations

from sentinel.config import ModelEndpoint
from sentinel.llm.base import ChatModel, ChatModelError


def build_chat_model(endpoint: ModelEndpoint) -> ChatModel:
    provider = endpoint.provider
    if provider == "stub":
        from sentinel.llm.stub import StubChat

        return StubChat(endpoint.model)
    if provider == "ollama":
        from sentinel.llm.ollama import OllamaChat

        return OllamaChat(
            endpoint.model,
            endpoint.base_url,
            timeout_s=endpoint.timeout_s,
            temperature=endpoint.temperature,
            max_tokens=endpoint.max_tokens,
            context_window=endpoint.context_window or 8192,
            keep_alive=endpoint.keep_alive,
        )
    if provider == "openai_compat":
        from sentinel.llm.openai_compat import OpenAICompatChat

        if not endpoint.base_url:
            raise ChatModelError(
                f"model {endpoint.model!r}: provider openai_compat needs base_url"
            )
        return OpenAICompatChat(
            endpoint.model,
            endpoint.base_url,
            api_key_env=endpoint.api_key_env,
            timeout_s=endpoint.timeout_s,
            temperature=endpoint.temperature,
            max_tokens=endpoint.max_tokens,
        )
    if provider == "anthropic":
        from sentinel.llm.anthropic import AnthropicChat

        return AnthropicChat(
            endpoint.model,
            api_key_env=endpoint.api_key_env or "ANTHROPIC_API_KEY",
            base_url=endpoint.base_url,
            timeout_s=endpoint.timeout_s,
            # Thinking shares this budget on current Claude models, so the
            # small local-model default would starve it.
            max_tokens=max(endpoint.max_tokens, 16000),
            effort=endpoint.effort,
            refusal_fallback=endpoint.refusal_fallback,
        )
    raise ChatModelError(f"unknown model provider {provider!r}")
