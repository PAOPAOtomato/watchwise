"""Use any vision-capable ``ChatModel`` as a tier-2 screener.

The dedicated Ollama and OpenAI-compatible VLM backends stay the default for
local screening. This adapter is how a model that only exists behind the chat
layer (for example Claude, in the ``cloud`` profile) screens frames.
"""

from __future__ import annotations

from watchwise.llm.base import ChatModel, ImagePart, Message
from watchwise.tier2.vlm.base import VLMBackend, VLMReply
from watchwise.types import Frame


class ChatModelVLM(VLMBackend):
    def __init__(self, chat: ChatModel, *, max_tokens: int | None = None):
        super().__init__(chat.model)
        self.chat = chat
        self.name = chat.provider
        self.max_tokens = max_tokens

    async def infer(self, frames: list[Frame], prompt: str, system: str) -> VLMReply:
        images = [ImagePart(f.jpeg, "image/jpeg") for f in frames]
        result = await self.chat.complete(
            [Message.system(system), Message.user(prompt, images=images)],
            max_tokens=self.max_tokens,
        )
        if result.stop_reason == "refusal":
            # Surfaces as a backend error, which screens as "uncertain".
            raise RuntimeError("the model declined to screen these frames")
        return VLMReply(
            result.content,
            truncated=result.stop_reason == "length",
            max_tokens=self.max_tokens or getattr(self.chat, "max_tokens", None),
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            thinking=bool(result.reasoning),
        )

    async def aclose(self) -> None:
        await self.chat.aclose()
