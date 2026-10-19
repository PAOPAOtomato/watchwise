"""Use any vision-capable ``ChatModel`` as a tier-2 screener.

The dedicated Ollama and OpenAI-compatible VLM backends stay the default for
local screening. This adapter is how a model that only exists behind the chat
layer (for example Claude, in the ``cloud`` profile) screens frames.
"""

from __future__ import annotations

from sentinel.llm.base import ChatModel, ImagePart, Message
from sentinel.tier2.vlm.base import VLMBackend
from sentinel.types import Frame


class ChatModelVLM(VLMBackend):
    def __init__(self, chat: ChatModel, *, max_tokens: int | None = None):
        super().__init__(chat.model)
        self.chat = chat
        self.name = chat.provider
        self.max_tokens = max_tokens

    async def infer(self, frames: list[Frame], prompt: str, system: str) -> str:
        images = [ImagePart(f.jpeg, "image/jpeg") for f in frames]
        result = await self.chat.complete(
            [Message.system(system), Message.user(prompt, images=images)],
            max_tokens=self.max_tokens,
        )
        if result.stop_reason == "refusal":
            # Surfaces as a backend error, which screens as "uncertain".
            raise RuntimeError("the model declined to screen these frames")
        return result.content

    async def aclose(self) -> None:
        await self.chat.aclose()
