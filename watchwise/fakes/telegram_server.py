"""A fake Telegram Bot API, for exercising the bot without Telegram.

Like the other fakes it is a real ASGI app, so ``TelegramAPI`` runs its real
HTTP code: JSON bodies, multipart photo uploads, error envelopes, long-poll
``getUpdates`` offsets. Tests queue updates (``push_message``,
``push_button``) and inspect what was sent (``sent``, ``edits``, ``answers``).

    POST /bot<token>/<method>   getMe, sendMessage, sendPhoto,
                                editMessageCaption, editMessageText,
                                answerCallbackQuery, getUpdates
"""

from __future__ import annotations

import json
from email.parser import BytesParser
from email.policy import default as email_policy
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class FakeTelegramState:
    def __init__(self, token: str = "test-token") -> None:
        self.token = token
        self.username = "watchwise_test_bot"
        self.sent: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.answers: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.calls: list[str] = []
        # method -> list of (status, description, retry_after) to fail with, in order
        self.failures: dict[str, list[tuple[int, str, int | None]]] = {}
        self._next_update = 1
        self._next_message = 100

    def push_message(self, chat_id: int, text: str, *, first_name: str = "Test") -> None:
        self.updates.append(
            {
                "update_id": self._next_update,
                "message": {
                    "message_id": self._next_update,
                    "chat": {"id": chat_id, "type": "private", "first_name": first_name},
                    "from": {"id": chat_id, "first_name": first_name},
                    "text": text,
                },
            }
        )
        self._next_update += 1

    def push_button(self, chat_id: int, data: str, *, message_id: int = 100, username: str = "tester") -> None:
        self.updates.append(
            {
                "update_id": self._next_update,
                "callback_query": {
                    "id": f"cbq-{self._next_update}",
                    "from": {"id": chat_id, "username": username},
                    "message": {"message_id": message_id, "chat": {"id": chat_id}},
                    "data": data,
                },
            }
        )
        self._next_update += 1

    def buttons(self, record: dict[str, Any]) -> list[str]:
        """The callback_data of every button on a sent or edited message."""
        markup = record.get("reply_markup") or {}
        rows = markup.get("inline_keyboard", [])
        return [b["callback_data"] for row in rows for b in row if "callback_data" in b]

    def links(self, record: dict[str, Any]) -> list[tuple[str, str]]:
        """(label, url) of every link button on a sent or edited message."""
        markup = record.get("reply_markup") or {}
        return [(b["text"], b["url"]) for row in markup.get("inline_keyboard", []) for b in row if "url" in b]


def _parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, bytes]]:
    message = BytesParser(policy=email_policy).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )
    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if part.get_filename():
            files[name] = part.get_payload(decode=True) or b""
        else:
            fields[name] = (part.get_payload(decode=True) or b"").decode("utf-8")
    return fields, files


def build_fake_telegram(state: FakeTelegramState | None = None) -> tuple[FastAPI, FakeTelegramState]:
    state = state or FakeTelegramState()
    app = FastAPI(title="fake-telegram")

    def ok(result: Any) -> JSONResponse:
        return JSONResponse({"ok": True, "result": result})

    @app.post("/bot{token}/{method}")
    async def call(token: str, method: str, request: Request) -> JSONResponse:
        if token != state.token:
            return JSONResponse({"ok": False, "error_code": 401, "description": "Unauthorized"}, 401)
        content_type = request.headers.get("content-type", "")
        files: dict[str, bytes] = {}
        if content_type.startswith("multipart/form-data"):
            params, files = _parse_multipart(await request.body(), content_type)
        else:
            raw = await request.body()
            params = json.loads(raw) if raw else {}
        state.calls.append(method)

        queued = state.failures.get(method)
        if queued:
            status, description, retry_after = queued.pop(0)
            payload: dict[str, Any] = {"ok": False, "error_code": status, "description": description}
            if retry_after is not None:
                payload["parameters"] = {"retry_after": retry_after}
            return JSONResponse(payload, status_code=status)

        markup = params.get("reply_markup")
        if isinstance(markup, str):
            markup = json.loads(markup)

        if method == "getMe":
            return ok({"id": 42, "is_bot": True, "username": state.username})
        if method in ("sendMessage", "sendPhoto"):
            message_id = state._next_message
            state._next_message += 1
            state.sent.append(
                {
                    "method": method,
                    "chat_id": int(params["chat_id"]),
                    "message_id": message_id,
                    "text": params.get("text") or params.get("caption") or "",
                    "parse_mode": params.get("parse_mode"),
                    "reply_markup": markup,
                    "reply_to": params.get("reply_to_message_id"),
                    "photo": files.get("photo"),
                }
            )
            return ok({"message_id": message_id, "chat": {"id": int(params["chat_id"])}})
        if method in ("editMessageCaption", "editMessageText"):
            state.edits.append(
                {
                    "method": method,
                    "chat_id": int(params["chat_id"]),
                    "message_id": int(params["message_id"]),
                    "text": params.get("caption") or params.get("text") or "",
                    "reply_markup": markup,
                }
            )
            return ok(True)
        if method == "answerCallbackQuery":
            state.answers.append(params)
            return ok(True)
        if method == "getUpdates":
            offset = int(params.get("offset") or 0)
            pending = [u for u in state.updates if u["update_id"] >= offset]
            state.updates = pending  # confirmed updates are forgotten, like Telegram
            return ok(pending)
        return JSONResponse({"ok": False, "error_code": 404, "description": "Not Found"}, 404)

    return app, state
