"""Telegram: alert cards with buttons, delivered by a bot you own.

  TelegramAPI       a thin Bot API client. The token is part of every URL, so
                    errors are re-raised without the URL and httpx's request
                    logging is silenced (it would print the token).
  TelegramNotifier  a decision sink. It queues work on its own task, so a
                    slow network never delays the alert pipeline; sends a card
                    for each alert, then edits that card in place as the
                    verdict is refined.
  TelegramBot       long-polls for button presses and commands (no inbound
                    port or public URL needed) and hands them to the
                    ActionHandler.

Telegram is a cloud relay: card text, and the snapshot unless
``send_images`` is false, passes through Telegram's servers.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from watchwise.config import TelegramSettings
from watchwise.memory.store import Store
from watchwise.notify.actions import ActionHandler
from watchwise.notify.presenter import (
    CAPTION_LIMIT,
    TEXT_LIMIT,
    AlertCard,
    build_card,
    clip_url,
    context_from_row,
    decision_from_row,
    keyboard_markup,
    parse_callback,
)
from watchwise.nvr.client import NVRClient
from watchwise.tier2.sampling import downscale_jpeg
from watchwise.types import AlertContext, Decision

logger = logging.getLogger(__name__)

# httpx logs every request URL at INFO, and Bot API URLs contain the token.
for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

CHANNEL = "telegram"


class TelegramError(RuntimeError):
    """A Bot API failure. Never contains the token."""


class TelegramAPI:
    def __init__(
        self,
        token: str,
        *,
        base_url: str = "https://api.telegram.org",
        client: httpx.AsyncClient | None = None,
        timeout_s: float = 60.0,
    ):
        if not token:
            raise TelegramError("no bot token")
        self._base = f"{base_url.rstrip('/')}/bot{token}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def call(self, method: str, *, files: dict[str, Any] | None = None, **params: Any) -> Any:
        params = {k: v for k, v in params.items() if v is not None}
        url = f"{self._base}/{method}"
        for attempt in range(2):
            try:
                if files:
                    data = {
                        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                        for k, v in params.items()
                    }
                    r = await self._client.post(url, data=data, files=files)
                else:
                    r = await self._client.post(url, json=params)
            except httpx.HTTPError as exc:
                raise TelegramError(f"{method}: {type(exc).__name__}") from None
            try:
                body = r.json()
            except ValueError:
                raise TelegramError(f"{method}: HTTP {r.status_code}") from None
            if body.get("ok"):
                return body.get("result")
            retry_after = (body.get("parameters") or {}).get("retry_after")
            if r.status_code == 429 and retry_after is not None and attempt == 0:
                await asyncio.sleep(min(float(retry_after), 30.0))
                continue
            raise TelegramError(f"{method}: {body.get('description') or r.status_code}")
        raise TelegramError(f"{method}: rate limited")

    async def get_me(self) -> dict[str, Any]:
        return await self.call("getMe")

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
        reply_to: int | None = None,
    ) -> dict[str, Any]:
        return await self.call(
            "sendMessage",
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
            reply_markup=reply_markup,
            reply_to_message_id=reply_to,
            disable_web_page_preview=True,
        )

    async def send_photo(
        self,
        chat_id: int | str,
        photo: bytes,
        caption: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.call(
            "sendPhoto",
            files={"photo": ("snapshot.jpg", photo, "image/jpeg")},
            chat_id=chat_id,
            caption=caption,
            parse_mode="HTML",
            reply_markup=reply_markup,
        )

    async def edit(
        self,
        chat_id: int | str,
        message_id: int | str,
        text: str,
        *,
        has_image: bool,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        method, key = ("editMessageCaption", "caption") if has_image else ("editMessageText", "text")
        try:
            await self.call(
                method,
                chat_id=chat_id,
                message_id=message_id,
                parse_mode="HTML",
                reply_markup=reply_markup,
                **{key: text},
            )
        except TelegramError as exc:
            if "message is not modified" not in str(exc):
                raise

    async def answer_callback(self, callback_id: str, text: str | None = None) -> None:
        await self.call("answerCallbackQuery", callback_query_id=callback_id, text=text)

    async def get_updates(self, offset: int | None = None, timeout: int = 0) -> list[dict[str, Any]]:
        result = await self.call(
            "getUpdates",
            offset=offset,
            timeout=timeout,
            allowed_updates=["message", "callback_query"],
        )
        return list(result or [])


class TelegramNotifier:
    """A decision sink that keeps one card per alert per chat up to date."""

    def __init__(
        self,
        api: TelegramAPI,
        settings: TelegramSettings,
        store: Store,
        nvr: NVRClient,
        *,
        web_url: str | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.api = api
        self.settings = settings
        self.store = store
        self.nvr = nvr
        # Frigate's web page, for each card's link to its recording.
        self.web_url = web_url
        self.clock = clock
        self._queue: asyncio.Queue[tuple[AlertContext, Decision]] = asyncio.Queue(maxsize=256)
        self._task: asyncio.Task[None] | None = None
        self._last_sent: dict[str, float] = {}

    # ---------------- lifecycle ----------------

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="telegram-notifier")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def drain(self, timeout: float = 10.0) -> None:
        async with asyncio.timeout(timeout):
            await self._queue.join()

    async def __call__(self, ctx: AlertContext, decision: Decision) -> None:
        """The sink. Never blocks the pipeline: work is queued."""
        if not self.settings.chat_ids:
            return
        try:
            self._queue.put_nowait((ctx, decision))
        except asyncio.QueueFull:
            logger.warning("telegram queue full; dropping the card update for %s", ctx.review_id)

    async def _run(self) -> None:
        while True:
            ctx, decision = await self._queue.get()
            try:
                await self.process(ctx, decision)
            except Exception:
                logger.exception("telegram card for %s failed", ctx.review_id)
            finally:
                self._queue.task_done()

    # ---------------- cards ----------------

    async def process(self, ctx: AlertContext, decision: Decision) -> None:
        rows = self.store.notifications_for(ctx.review_id, CHANNEL)
        if rows:
            await self._revise(ctx, decision, rows)
        elif self._wanted(ctx, decision):
            await self._send(ctx, decision)

    def _wanted(self, ctx: AlertContext, decision: Decision) -> bool:
        if decision.disposition.value not in self.settings.notify_on:
            return False
        until = self.store.snoozed_until(ctx.camera)
        return not (until and until > self.clock())

    async def _send(self, ctx: AlertContext, decision: Decision) -> None:
        card = build_card(ctx, decision)
        image = await self._snapshot(ctx) if self.settings.send_images else None
        markup = keyboard_markup(ctx.review_id, card.keyboard, clip=clip_url(self.web_url, ctx.review_id))
        for chat_id in self.settings.chat_ids:
            await self._pace(str(chat_id))
            if image:
                msg = await self.api.send_photo(chat_id, image, card.html(CAPTION_LIMIT), reply_markup=markup)
            else:
                msg = await self.api.send_message(chat_id, card.html(TEXT_LIMIT), reply_markup=markup)
            self.store.add_notification(
                ctx.review_id, CHANNEL, str(chat_id), str(msg["message_id"]), has_image=bool(image)
            )
        self.store.set_notification_state(ctx.review_id, CHANNEL, footer="", keyboard=card.keyboard)

    async def _revise(self, ctx: AlertContext, decision: Decision, rows: list[Any]) -> None:
        """A refined verdict for a card already sent: edit it in place."""
        card = build_card(ctx, decision)
        footer, keyboard = rows[0]["footer"], rows[0]["keyboard"]
        if footer.startswith("🔎"):
            footer = ""  # the investigation the person asked for has finished
        if not keyboard.startswith("after_feedback"):
            keyboard = card.keyboard
        self.store.set_notification_state(ctx.review_id, CHANNEL, footer=footer, keyboard=keyboard)
        await self._edit_all(rows, card, footer, keyboard)

    async def apply(self, review_id: str, *, footer: str | None, keyboard: str | None) -> None:
        """Show a button's outcome on the card (re-rendered from the store)."""
        rows = self.store.notifications_for(review_id, CHANNEL)
        alert = self.store.get_alert(review_id)
        decision = decision_from_row(alert) if alert is not None else None
        if not rows or decision is None:
            return
        footer = rows[0]["footer"] if footer is None else footer
        keyboard = rows[0]["keyboard"] if keyboard is None else keyboard
        self.store.set_notification_state(review_id, CHANNEL, footer=footer, keyboard=keyboard)
        await self._edit_all(rows, build_card(context_from_row(alert), decision), footer, keyboard)

    async def _edit_all(self, rows: list[Any], card: AlertCard, footer: str, keyboard: str) -> None:
        card.footer = footer
        markup = keyboard_markup(card.review_id, keyboard, clip=clip_url(self.web_url, card.review_id))
        for row in rows:
            limit = CAPTION_LIMIT if row["has_image"] else TEXT_LIMIT
            try:
                await self.api.edit(
                    row["chat_id"],
                    row["message_id"],
                    card.html(limit),
                    has_image=bool(row["has_image"]),
                    reply_markup=markup,
                )
            except TelegramError as exc:
                logger.warning("could not update the card for %s: %s", card.review_id, exc)

    async def _snapshot(self, ctx: AlertContext) -> bytes | None:
        try:
            if ctx.event_ids:
                data = await self.nvr.event_snapshot(ctx.event_ids[0])
            else:
                middle = ctx.start_time + min(ctx.duration / 2, 5.0)
                data = (await self.nvr.recording_frame(ctx.camera, middle)).jpeg
            return downscale_jpeg(data, max_edge=1280, quality=80)
        except Exception as exc:
            logger.info("no snapshot for %s (%s); sending text only", ctx.review_id, exc)
            return None

    async def _pace(self, chat_id: str) -> None:
        wait = self._last_sent.get(chat_id, 0.0) + self.settings.min_interval_s - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_sent[chat_id] = time.monotonic()


HELP = (
    "<b>Watchwise</b> sends a card for each alert that needs you.\n"
    "✅ Expected: teach it this is normal (after a few, it stops alerting on it)\n"
    "🚨 Real: confirm it was worth an alert\n"
    "🔍 Look closer: run a full investigation now\n"
    "🔇 Snooze: mute this camera for a while (alerts are still recorded)\n"
    "❓ Why: the evidence behind the decision\n"
    "↩️ Undo: take back your last answer on that card"
)


class TelegramBot:
    """Long-polls Telegram and turns button presses into actions."""

    def __init__(
        self,
        api: TelegramAPI,
        settings: TelegramSettings,
        handler: ActionHandler,
        notifier: TelegramNotifier,
    ):
        self.api = api
        self.settings = settings
        self.handler = handler
        self.notifier = notifier
        self._offset: int | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="telegram-bot")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _run(self) -> None:
        backoff = 1.0
        while True:
            try:
                await self.poll_once(timeout=self.settings.poll_timeout_s)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("telegram polling failed (%s); retrying in %.0fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def poll_once(self, timeout: int = 0) -> int:
        updates = await self.api.get_updates(offset=self._offset, timeout=timeout)
        for update in updates:
            self._offset = int(update["update_id"]) + 1
            try:
                await self.handle(update)
            except Exception:
                logger.exception("telegram update %s failed", update.get("update_id"))
        return len(updates)

    def allowed(self, chat_id: Any) -> bool:
        try:
            return int(chat_id) in {int(c) for c in self.settings.chat_ids}
        except (TypeError, ValueError):
            return False

    async def handle(self, update: dict[str, Any]) -> None:
        if "callback_query" in update:
            await self._on_button(update["callback_query"])
        elif "message" in update:
            await self._on_message(update["message"])

    async def _on_message(self, message: dict[str, Any]) -> None:
        chat_id = (message.get("chat") or {}).get("id")
        text = (message.get("text") or "").strip()
        if chat_id is None:
            return
        if text.startswith(("/start", "/id")):
            if self.allowed(chat_id):
                await self.api.send_message(chat_id, "✅ Watchwise is connected. Alerts will appear here.")
            else:
                await self.api.send_message(
                    chat_id,
                    f"This chat's id is <code>{int(chat_id)}</code>. To get Watchwise alerts "
                    "here, run <code>watchwise telegram link</code> on the machine running "
                    "Watchwise.",
                )
            return
        if not self.allowed(chat_id):
            return  # strangers get no further answers
        if text.startswith("/help"):
            await self.api.send_message(chat_id, HELP)
        else:
            await self.api.send_message(
                chat_id,
                "Conversation with Watchwise arrives in a later version. For now, use the "
                "buttons on alert cards (/help explains them).",
            )

    async def _on_button(self, query: dict[str, Any]) -> None:
        message = query.get("message") or {}
        chat_id = (message.get("chat") or {}).get("id")
        if chat_id is None or not self.allowed(chat_id):
            await self._answer(query, "Not authorized for this chat.")
            return
        parsed = parse_callback(str(query.get("data") or ""))
        if parsed is None:
            await self._answer(query, "Unknown button.")
            return
        action, review_id, arg = parsed
        user = query.get("from") or {}
        outcome = await self.handler.handle(
            action, review_id, arg=arg, reviewer=f"telegram:{user.get('username') or user.get('id')}"
        )
        await self._answer(query, outcome.toast)
        if outcome.footer is not None or outcome.keyboard is not None:
            await self.notifier.apply(review_id, footer=outcome.footer, keyboard=outcome.keyboard)
        if outcome.reply:
            await self.api.send_message(chat_id, outcome.reply, reply_to=message.get("message_id"))

    async def _answer(self, query: dict[str, Any], text: str) -> None:
        """Answer a button press. Telegram refuses answers to presses that
        are too old (made while the service was stopped); the press itself
        has still been handled, so its card must still be updated."""
        try:
            await self.api.answer_callback(query["id"], text)
        except TelegramError as exc:
            logger.info("could not answer a button press (%s); handling it anyway", exc)
