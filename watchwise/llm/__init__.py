"""Provider-neutral model layer: one interface over local and cloud models."""

from watchwise.llm.base import (
    ChatModel,
    ChatModelError,
    ChatResult,
    ImagePart,
    Message,
    ToolCallRequest,
    ToolSpec,
    Usage,
)
from watchwise.llm.registry import build_chat_model

__all__ = [
    "ChatModel",
    "ChatModelError",
    "ChatResult",
    "ImagePart",
    "Message",
    "ToolCallRequest",
    "ToolSpec",
    "Usage",
    "build_chat_model",
]
