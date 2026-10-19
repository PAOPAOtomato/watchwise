"""A fake model server speaking two dialects: OpenAI-compatible and Ollama.

Like the fake NVR, this is a real ASGI app rather than a mock, so providers
are exercised through their real HTTP code: URL shapes, request bodies,
status codes and response parsing.

Tests queue replies (``state.replies``) and inspect what was sent
(``state.requests``). A reply is a dict with any of ``content``,
``tool_calls``, ``thinking``, ``finish_reason``; the server wraps it in the
dialect the client used.

    POST /v1/chat/completions   GET /v1/models      (OpenAI-compatible)
    POST /api/chat              GET /api/tags       (Ollama native)
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

FINAL = {"content": json.dumps({"verdict": "benign", "confidence": 0.9, "conclusion": "fake final"})}


class FakeLLMState:
    def __init__(self) -> None:
        self.replies: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.status: int = 200
        self.models: list[str] = ["fake-model"]
        # Request parameters this server refuses, the way OpenAI's newer models
        # refuse max_tokens or a non-default temperature: param -> message.
        self.reject_params: dict[str, str] = {}

    def next_reply(self) -> dict[str, Any]:
        return self.replies.pop(0) if self.replies else dict(FINAL)


def build_fake_llm(state: FakeLLMState | None = None) -> tuple[FastAPI, FakeLLMState]:
    state = state or FakeLLMState()
    app = FastAPI(title="fake-llm")

    async def record(request: Request) -> dict[str, Any]:
        body = await request.json()
        state.requests.append(body)
        state.headers.append(dict(request.headers))
        return body

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> JSONResponse:
        body = await record(request)
        if state.status != 200:
            return JSONResponse({"error": {"message": "fake failure"}}, status_code=state.status)
        for param, text in state.reject_params.items():
            if param in body:
                return JSONResponse(
                    {
                        "error": {
                            "message": text,
                            "type": "invalid_request_error",
                            "param": param,
                            "code": "unsupported_parameter",
                        }
                    },
                    status_code=400,
                )
        reply = state.next_reply()
        message: dict[str, Any] = {"role": "assistant", "content": reply.get("content", "")}
        if reply.get("tool_calls"):
            message["tool_calls"] = [
                {
                    "id": call.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"]
                        if isinstance(call["arguments"], str)
                        else json.dumps(call["arguments"]),
                    },
                }
                for i, call in enumerate(reply["tool_calls"])
            ]
        if reply.get("thinking"):
            message["reasoning_content"] = reply["thinking"]
        finish = reply.get("finish_reason") or ("tool_calls" if reply.get("tool_calls") else "stop")
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": body.get("model", "fake-model"),
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            }
        )

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": m, "object": "model"} for m in state.models]}

    @app.post("/api/chat")
    async def ollama_chat(request: Request) -> JSONResponse:
        body = await record(request)
        if body.get("model") not in state.models:
            return JSONResponse({"error": f"model '{body.get('model')}' not found"}, status_code=404)
        if state.status != 200:
            return JSONResponse({"error": "fake failure"}, status_code=state.status)
        reply = state.next_reply()
        message: dict[str, Any] = {"role": "assistant", "content": reply.get("content", "")}
        if reply.get("tool_calls"):
            message["tool_calls"] = [
                {"function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in reply["tool_calls"]
            ]
        if reply.get("thinking"):
            message["thinking"] = reply["thinking"]
        return JSONResponse(
            {
                "model": body["model"],
                "message": message,
                "done": True,
                "done_reason": reply.get("finish_reason", "stop"),
                "prompt_eval_count": 13,
                "eval_count": 5,
            }
        )

    @app.get("/api/tags")
    async def tags() -> dict[str, Any]:
        return {"models": [{"name": m} for m in state.models]}

    return app, state
