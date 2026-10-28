"""Telegram: alert cards with buttons, delivered by a bot you own.

  TelegramAPI       a thin Bot API client. The token is part of every URL, so
                    errors are re-raised without the URL and httpx's request
                    logging is silenced (it would print the token).
  TelegramNotifier  a decision sink. It queues work on its own task, so a
                    slow network never delays the alert pipeline; sends a card
                    for each alert, then edits that card in place as the
                    verdict is refined. Cameras in one room share one card
                    per incident (D-062).
  TelegramBot       long-polls for button presses and commands (no inbound
                    port or public URL needed) and hands them to the
                    ActionHandler, /setup, /arm and /face.

Telegram is a cloud relay: card text, and the snapshot unless
``send_images`` is false, passes through Telegram's servers.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from watchwise.config import CameraSettings, RoomSettings, TelegramSettings, room_key, same_room
from watchwise.memory.store import Store
from watchwise.notify.actions import ASK_WHY, ActionHandler
from watchwise.notify.faces import FaceEnroller
from watchwise.notify.presenter import (
    CAPTION_LIMIT,
    TEXT_LIMIT,
    AlertCard,
    build_card,
    SERIOUSNESS,
    build_room_card,
    clip_url,
    context_from_row,
    decision_from_row,
    keyboard_markup,
    parse_callback,
)
from watchwise.chat.proposals import ProposalBook
from watchwise.chat.service import ChatService
from watchwise.notify.setup_chat import Reply, SetupChat
from watchwise.nvr.client import NVRClient
from watchwise.setup.home import ARMING_LABELS
from watchwise.tier2.sampling import downscale_jpeg
from watchwise.timeutil import safe_strftime
from watchwise.types import AlertContext, Decision

logger = logging.getLogger(__name__)

# httpx logs every request URL at INFO, and Bot API URLs contain the token.
for _name in ("httpx", "httpcore"):
    logging.getLogger(_name).setLevel(logging.WARNING)

CHANNEL = "telegram"
# How far back a room card can still be joined, or an event pulled onto one.
RECENT_CARDS_S = 3600.0


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

    async def send_chat_action(self, chat_id: int | str, action: str = "typing") -> None:
        await self.call("sendChatAction", chat_id=chat_id, action=action)

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
        cameras: dict[str, CameraSettings] | None = None,
        rooms: RoomSettings | None = None,
    ):
        self.api = api
        self.settings = settings
        self.store = store
        self.nvr = nvr
        # Frigate's web page, for each card's link to its recording.
        self.web_url = web_url
        self.clock = clock
        # Which room each camera watches (the service's config, which /setup
        # updates in place), and when two cameras' events are one incident.
        self.cameras = cameras if cameras is not None else {}
        self.rooms = rooms if rooms is not None else RoomSettings()
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
        lead = self.store.card_lead(ctx.review_id, CHANNEL)
        if lead is not None:
            # Another camera's event on a room card has a new verdict.
            if self._stands_out(ctx, decision, lead):
                self.store.remove_card_member(ctx.review_id, CHANNEL)
                logger.info("%s on %s leaves the room card of %s for its own card", ctx.review_id, ctx.camera, lead)
                await self.apply(lead, footer=None, keyboard=None)
                await self._send(ctx, decision)
            else:
                await self.apply(lead, footer=None, keyboard=None)
            return
        rows = self.store.notifications_for(ctx.review_id, CHANNEL)
        if rows:
            await self._revise(ctx, decision, rows)
            return
        joined = self._open_card_in_room(ctx)
        if joined is not None and not self._stands_out(ctx, decision, joined):
            self.store.add_card_member(joined, ctx.review_id, CHANNEL)
            logger.info("%s on %s joins the room card of %s", ctx.review_id, ctx.camera, joined)
            await self.apply(joined, footer=None, keyboard=None)
        elif self._wanted(ctx, decision):
            await self._send(ctx, decision)

    def card_group(self, review_id: str) -> list[str]:
        """Every event a card shows: its own, then the room's other cameras'."""
        return [review_id, *self.store.card_members(review_id, CHANNEL)]

    # ---------------- rooms (D-062) ----------------

    def _open_card_in_room(self, ctx: AlertContext) -> str | None:
        """The card of an incident this event is part of: another camera in
        its room saw something overlapping it in time, and the card's first
        event began at most ``rooms.max_incident_s`` before this one."""
        key = room_key(self.cameras, ctx.camera)
        if key.startswith("camera:"):
            return None  # a camera in a room of its own
        for card in self.store.recent_cards(CHANNEL, since=self.clock() - RECENT_CARDS_S):
            if room_key(self.cameras, card["camera"]) != key or not self._in_incident(ctx, card):
                continue
            members = (self.store.get_alert(m) for m in self.store.card_members(card["review_id"], CHANNEL))
            entries = [card, *(m for m in members if m is not None)]
            if any(e["camera"] != ctx.camera and self._overlaps(ctx, e) for e in entries):
                return str(card["review_id"])
        return None

    def _room_companions(self, ctx: AlertContext) -> list[str]:
        """Events on the room's other cameras that overlap this one and have
        no card of their own: a new card shows them too."""
        if room_key(self.cameras, ctx.camera).startswith("camera:"):
            return []
        companions: list[str] = []
        for row in self.store.alerts_on(same_room(self.cameras, ctx.camera), ctx.start_time - RECENT_CARDS_S):
            review_id = str(row["review_id"])
            if (
                self.store.notifications_for(review_id, CHANNEL)
                or self.store.card_lead(review_id, CHANNEL)
                or not self._overlaps(ctx, row)
                or not self._in_incident(row, ctx)
            ):
                continue
            companions.append(review_id)
        if companions:
            logger.info("%s on %s: the room card also shows %s", ctx.review_id, ctx.camera, ", ".join(companions))
        return companions

    def _in_incident(self, later: Any, first: Any) -> bool:
        """Does ``later`` start soon enough after ``first`` to share its card
        (B-028)? Without a limit, a room with someone in it would keep one
        card open, and a new incident there would only edit it, silently."""
        start = later.start_time if isinstance(later, AlertContext) else float(later["start_time"])
        begun = first.start_time if isinstance(first, AlertContext) else float(first["start_time"])
        return start - begun <= self.rooms.max_incident_s

    def _stands_out(self, ctx: AlertContext, decision: Decision, lead: str) -> bool:
        """Would this event go unnoticed on the room's card (B-028)? Telegram
        doesn't notify edits, so an event that would alert by itself gets its
        own card when it is more serious than everything else on the card,
        or when the owner has already answered the card."""
        if not self._wanted(ctx, decision):
            return False
        rows = self.store.notifications_for(lead, CHANNEL)
        if rows and str(rows[0]["keyboard"]).startswith("after_feedback"):
            return True
        levels = []
        for review_id in [lead, *self.store.card_members(lead, CHANNEL)]:
            row = self.store.get_alert(review_id) if review_id != ctx.review_id else None
            other = decision_from_row(row) if row is not None else None
            if other is not None:
                levels.append(SERIOUSNESS.get(other.disposition, 0))
        return SERIOUSNESS.get(decision.disposition, 0) > max(levels, default=-1)

    def _overlaps(self, ctx: AlertContext, row: Any) -> bool:
        """Do the two events overlap, give or take the merge window? An event
        that hasn't ended runs until now."""
        now = self.clock()
        window = self.rooms.merge_window_s
        start, end = ctx.start_time, ctx.end_time or now
        other_start = float(row["start_time"])
        other_end = float(row["end_time"]) if row["end_time"] else now
        return start <= other_end + window and other_start <= end + window

    def _room_name(self, camera: str) -> str:
        spec = self.cameras.get(camera)
        return " ".join(spec.room.split()) if spec is not None and spec.room.strip() else camera

    def _card(
        self,
        lead: str,
        ctx: AlertContext | None = None,
        decision: Decision | None = None,
        members: list[str] | None = None,
    ) -> AlertCard | None:
        """The card for ``lead`` (its live verdict when given, else the
        stored one), with the room's other cameras' latest verdicts."""
        if ctx is None or decision is None:
            alert = self.store.get_alert(lead)
            decision = decision_from_row(alert) if alert is not None else None
            if alert is None or decision is None:
                return None
            ctx = context_from_row(alert)
        others: list[tuple[AlertContext, Decision]] = []
        for review_id in self.store.card_members(lead, CHANNEL) if members is None else members:
            row = self.store.get_alert(review_id)
            other = decision_from_row(row) if row is not None else None
            if row is not None and other is not None:
                others.append((context_from_row(row), other))
        if not others:
            return build_card(ctx, decision)
        return build_room_card(self._room_name(ctx.camera), (ctx, decision), others)

    def _wanted(self, ctx: AlertContext, decision: Decision) -> bool:
        if decision.disposition.value not in self.settings.notify_on:
            return False
        until = self.store.snoozed_until(ctx.camera)
        return not (until and until > self.clock())

    async def _send(self, ctx: AlertContext, decision: Decision) -> None:
        companions = self._room_companions(ctx)
        card = self._card(ctx.review_id, ctx, decision, members=companions) or build_card(ctx, decision)
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
        for review_id in companions:
            self.store.add_card_member(ctx.review_id, review_id, CHANNEL)

    async def _revise(self, ctx: AlertContext, decision: Decision, rows: list[Any]) -> None:
        """A refined verdict for a card already sent: edit it in place."""
        card = self._card(ctx.review_id, ctx, decision) or build_card(ctx, decision)
        footer, keyboard = rows[0]["footer"], rows[0]["keyboard"]
        if footer.startswith("🔎"):
            footer = ""  # the investigation the person asked for has finished
        if not keyboard.startswith("after_feedback"):
            keyboard = card.keyboard
        self.store.set_notification_state(ctx.review_id, CHANNEL, footer=footer, keyboard=keyboard)
        await self._edit_all(rows, card, footer, keyboard)

    async def apply(self, review_id: str, *, footer: str | None, keyboard: str | None) -> None:
        """Show a button's outcome, or a room camera's new verdict, on the
        card (re-rendered from the store). None keeps what the card has."""
        rows = self.store.notifications_for(review_id, CHANNEL)
        card = self._card(review_id)
        if not rows or card is None:
            return
        footer = rows[0]["footer"] if footer is None else footer
        keyboard = rows[0]["keyboard"] if keyboard is None else keyboard
        self.store.set_notification_state(review_id, CHANNEL, footer=footer, keyboard=keyboard)
        await self._edit_all(rows, card, footer, keyboard)

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
    "🚨 Alert me: you want alerts like this (reply to my question to say what was wrong)\n"
    "🔍 Look closer: run a full investigation now\n"
    "🔇 Snooze: mute this camera for a while (alerts are still recorded)\n"
    "❓ Why: the evidence behind the decision\n"
    "↩️ Undo: take back your last answer on that card\n"
    "Ask in your own words, e.g. “anything in the kitchen today?”, or reply to a card to ask "
    "about it. Changes (roster, arming, muting, “tell me when…”) only happen when you tap ✅ Confirm. "
    "/new starts a new conversation\n"
    "/setup: tell Watchwise about your home (cameras, residents, what always alerts)\n"
    "/arm: see or switch arming (Home, Away, Night, Off)\n"
    "/face &lt;name&gt;: add your face to Frigate's Face Library from what the camera sees now"
)
ARM_BUTTONS = [
    [("🏠 Home", "ar:armed_home"), ("🚪 Away", "ar:armed_away")],
    [("🌙 Night", "ar:armed_night"), ("🔓 Off", "ar:disarmed")],
]
ARM_WORDS = {
    "home": "armed_home",
    "away": "armed_away",
    "night": "armed_night",
    "off": "disarmed",
    "disarm": "disarmed",
    "disarmed": "disarmed",
    **{state: state for state in ARMING_LABELS},
}


def _markup(buttons: list[list[tuple[str, str]]]) -> dict[str, Any] | None:
    if not buttons:
        return None
    return {"inline_keyboard": [[{"text": label, "callback_data": data} for label, data in row] for row in buttons]}


class TelegramBot:
    """Long-polls Telegram and turns button presses into actions."""

    def __init__(
        self,
        api: TelegramAPI,
        settings: TelegramSettings,
        handler: ActionHandler,
        notifier: TelegramNotifier,
        enroller: FaceEnroller | None = None,
        *,
        setup: SetupChat | None = None,
        arming: Any = None,
        chat: ChatService | None = None,
        proposals: ProposalBook | None = None,
    ):
        self.api = api
        self.settings = settings
        self.handler = handler
        self.notifier = notifier
        self.enroller = enroller
        self.setup = setup
        # Anything with get() and set(state, source); /arm reads and switches it.
        self.arming = arming
        # "Ask Watchwise" (D-065): questions, and the changes it proposes.
        self.chat = chat
        self.proposals = proposals
        self._chat_locks: dict[str, asyncio.Lock] = {}
        self._offset: int | None = None
        self._task: asyncio.Task[None] | None = None
        # Work that outlives one update (the recognition check).
        self.tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="telegram-bot")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for task in list(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

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
        replied_to = (message.get("reply_to_message") or {}).get("message_id")
        if replied_to is not None and text and not text.startswith("/"):
            answer = self.handler.note_from_reply(chat_id, int(replied_to), text)
            if answer is not None:
                await self.api.send_message(chat_id, answer, reply_to=message.get("message_id"))
                return
        if text.startswith("/setup"):
            await self._send(chat_id, await self.setup.start(int(chat_id)) if self.setup else [_NO_SETUP])
        elif text.startswith("/cancel"):
            await self._send(chat_id, self.setup.cancel(int(chat_id)) if self.setup else [_NO_SETUP])
        elif text.startswith("/arm"):
            await self._arm(chat_id, text, message.get("from") or {})
        elif self.setup is not None and text and not text.startswith("/") and self.setup.waiting(int(chat_id)):
            await self._send(chat_id, await self.setup.answer(int(chat_id), text))
        elif text.startswith("/face"):
            await self._face(chat_id, text)
        elif text.startswith("/help"):
            await self.api.send_message(chat_id, HELP)
        elif text.startswith("/new") and self.chat is not None:
            self.chat.reset(str(chat_id))
            await self.api.send_message(chat_id, "🆕 New conversation: earlier questions are forgotten.")
        elif self.chat is not None and text and not text.startswith("/"):
            about = self.notifier.store.review_for_message(CHANNEL, chat_id, replied_to) if replied_to else None
            self._spawn(self._ask(chat_id, text, message.get("message_id"), about))
        else:
            await self.api.send_message(
                chat_id,
                "Questions need a chat model (roles.chat in the config). For now, use the "
                "buttons on alert cards (/help explains them).",
            )

    # ---------------- Ask Watchwise (D-065) ----------------

    async def _ask(self, chat_id: Any, text: str, message_id: Any, about: str | None) -> None:
        """Answer one question (in the background: it can take a while on a
        local model), in order per chat."""
        assert self.chat is not None
        lock = self._chat_locks.setdefault(str(chat_id), asyncio.Lock())
        async with lock:
            try:
                with contextlib.suppress(TelegramError):
                    await self.api.send_chat_action(chat_id, "typing")
                answer = await self.chat.ask(str(chat_id), text, about=about)
                await self.api.send_message(chat_id, html.escape(answer.text, quote=False)[:4000], reply_to=message_id)
                for review_id in answer.images:
                    await self._photo(chat_id, review_id)
                for proposal in answer.proposals:
                    await self.api.send_message(
                        chat_id,
                        f"📝 {html.escape(proposal.summary, quote=False)}\nNothing changes unless you confirm.",
                        reply_markup=_markup([[("✅ Confirm", f"pr:ok:{proposal.id}"), ("✖️ Cancel", f"pr:no:{proposal.id}")]]),
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("answering a question failed")

    async def _photo(self, chat_id: Any, review_id: str) -> None:
        row = self.notifier.store.get_alert(review_id)
        if row is None or not self.settings.send_images:
            return
        ctx = context_from_row(row)
        image = await self.notifier._snapshot(ctx)
        if image:
            when = safe_strftime(ctx.start_time, "%a %H:%M")
            await self.api.send_photo(chat_id, image, f"📷 {html.escape(ctx.camera, quote=False)} · {when}")

    async def _proposal_button(self, query: dict[str, Any], chat_id: Any, data: str) -> None:
        _, action, proposal_id = (data.split(":", 2) + ["", ""])[:3]
        if self.proposals is None or not proposal_id:
            await self._answer(query, "Unknown button.")
            return
        user = query.get("from") or {}
        if action == "ok":
            outcome = self.proposals.confirm(proposal_id)
            logger.info("proposal %s confirmed by telegram:%s", proposal_id, user.get("username") or user.get("id"))
        else:
            outcome = self.proposals.cancel(proposal_id)
        await self._answer(query, outcome.text[:190])
        message = query.get("message") or {}
        if message.get("message_id") is not None:
            mark = "✅" if action == "ok" and outcome.done else "✖️"
            await self.api.edit(chat_id, message["message_id"], f"{mark} {html.escape(outcome.text, quote=False)}", has_image=False)

    async def _on_button(self, query: dict[str, Any]) -> None:
        message = query.get("message") or {}
        chat_id = (message.get("chat") or {}).get("id")
        if chat_id is None or not self.allowed(chat_id):
            await self._answer(query, "Not authorized for this chat.")
            return
        data = str(query.get("data") or "")
        if data.startswith("fc:"):
            await self._face_button(query, chat_id, data)
            return
        if data.startswith("su:") and self.setup is not None:
            await self._answer(query, "")
            await self._send(chat_id, await self.setup.press(int(chat_id), data), pressed=message.get("message_id"))
            return
        if data.startswith("ar:"):
            await self._arm_button(query, chat_id, data[3:])
            return
        if data.startswith("pr:"):
            await self._proposal_button(query, chat_id, data)
            return
        parsed = parse_callback(data)
        if parsed is None:
            await self._answer(query, "Unknown button.")
            return
        action, review_id, arg = parsed
        user = query.get("from") or {}
        outcome = await self.handler.handle(
            action,
            review_id,
            arg=arg,
            reviewer=f"telegram:{user.get('username') or user.get('id')}",
            group=self.notifier.card_group(review_id),
        )
        await self._answer(query, outcome.toast)
        if outcome.footer is not None or outcome.keyboard is not None:
            await self.notifier.apply(review_id, footer=outcome.footer, keyboard=outcome.keyboard)
        if outcome.reply:
            await self.api.send_message(chat_id, outcome.reply, reply_to=message.get("message_id"))
        if outcome.ask_note is not None and self.settings.ask_why_on_real:
            sent = await self.api.send_message(
                chat_id,
                ASK_WHY,
                reply_to=message.get("message_id"),
                reply_markup={"force_reply": True, "input_field_placeholder": "What was wrong?"},
            )
            self.handler.remember_note_request(chat_id, int(sent["message_id"]), outcome.ask_note)

    # ---------------- /setup (D-060) ----------------

    async def _send(self, chat_id: Any, replies: list[Reply], *, pressed: int | None = None) -> None:
        """Send what /setup returned: messages, an in-place edit of the
        pressed message, a /face capture, a recognition check."""
        for reply in replies:
            markup = _markup(reply.buttons)
            if reply.edit and pressed is not None:
                await self.api.edit(chat_id, pressed, reply.text, has_image=False, reply_markup=markup)
            elif reply.photo is not None:
                await self.api.send_photo(chat_id, reply.photo, reply.text, reply_markup=markup)
            elif reply.text:
                await self.api.send_message(chat_id, reply.text, reply_markup=markup)
            if reply.face:
                await self._face_capture(chat_id, reply.face, reply.face_camera)
            if reply.check:
                self._spawn(self._check(chat_id))

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _check(self, chat_id: Any) -> None:
        assert self.setup is not None
        try:
            await self._send(chat_id, [await self.setup.check(int(chat_id))])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("the recognition check failed")

    # ---------------- /arm ----------------

    async def _arm(self, chat_id: Any, text: str, user: dict[str, Any]) -> None:
        if self.arming is None:
            await self.api.send_message(chat_id, "Arming isn't available here.")
            return
        parts = text.split()
        if len(parts) > 1:
            state = ARM_WORDS.get(parts[1].lower())
            if state is None:
                await self.api.send_message(chat_id, "Send /arm home, away, night or off (or just /arm).")
                return
            self.arming.set(state, source=f"telegram:{user.get('username') or user.get('id')}")
        await self.api.send_message(chat_id, self._arming_text(), reply_markup=_markup(ARM_BUTTONS))

    async def _arm_button(self, query: dict[str, Any], chat_id: Any, state: str) -> None:
        if self.arming is None or state not in ARMING_LABELS:
            await self._answer(query, "Unknown button.")
            return
        user = query.get("from") or {}
        self.arming.set(state, source=f"telegram:{user.get('username') or user.get('id')}")
        await self._answer(query, f"Arming: {ARMING_LABELS[state]}")
        message_id = (query.get("message") or {}).get("message_id")
        if message_id is not None:
            await self.api.edit(chat_id, message_id, self._arming_text(), has_image=False, reply_markup=_markup(ARM_BUTTONS))

    def _arming_text(self) -> str:
        now = self.arming.get() if self.arming is not None else {}
        state = str(now.get("state") or "")
        label = html.escape(ARMING_LABELS.get(state, state), quote=False)
        if now.get("source") == "default":
            head = f"🔒 Arming: <b>{label}</b> (nothing set yet; that's the default from /setup)"
        elif now.get("since"):
            head = f"🔒 Arming: <b>{label}</b> (since {safe_strftime(float(now['since']), '%a %H:%M')})"
        else:
            head = f"🔒 Arming: <b>{label}</b>"
        return (
            f"{head}\nHome or Off: ordinary household activity is expected. Away or Night: any "
            "person in view alerts unless the duty roster or a recognized resident explains them."
        )

    # ---------------- /face ----------------

    async def _face(self, chat_id: Any, text: str) -> None:
        """/face <name> [camera]: show the camera's current frame, ask first (D-058)."""
        parts = text.split()
        if self.enroller is not None and len(parts) < 2:
            await self.api.send_message(chat_id, "Send /face &lt;name&gt; [camera] while you face the camera, e.g. /face alex.")
            return
        await self._face_capture(chat_id, parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else None)

    async def _face_capture(self, chat_id: Any, name: str, camera: str | None) -> None:
        if self.enroller is None:
            await self.api.send_message(chat_id, "Adding faces isn't available here.")
            return
        token, capture, message = await self.enroller.capture(name, camera)
        text = html.escape(message, quote=False)
        if capture is None:
            await self.api.send_message(chat_id, text)
            return
        markup = {"inline_keyboard": [[
            {"text": "✅ Add", "callback_data": f"fc:add:{token}"},
            {"text": "✖️ Cancel", "callback_data": f"fc:cancel:{token}"},
        ]]}
        if self.settings.send_images:
            await self.api.send_photo(chat_id, capture.jpeg, text, reply_markup=markup)
        else:
            await self.api.send_message(chat_id, text, reply_markup=markup)

    async def _face_button(self, query: dict[str, Any], chat_id: Any, data: str) -> None:
        _, action, token = (data.split(":", 2) + ["", ""])[:3]
        if self.enroller is None or not token:
            await self._answer(query, "Unknown button.")
            return
        if action == "add":
            await self._answer(query, "Adding…")
            result = await self.enroller.confirm(token)
        else:
            await self._answer(query, "Cancelled.")
            result = self.enroller.cancel(token)
        message_id = (query.get("message") or {}).get("message_id")
        await self.api.send_message(chat_id, html.escape(result, quote=False), reply_to=message_id)

    async def _answer(self, query: dict[str, Any], text: str | None) -> None:
        """Answer a button press. Telegram refuses answers to presses that
        are too old (made while the service was stopped); the press itself
        has still been handled, so its card must still be updated."""
        try:
            await self.api.answer_callback(query["id"], text or None)
        except TelegramError as exc:
            logger.info("could not answer a button press (%s); handling it anyway", exc)


_NO_SETUP = Reply("/setup isn't available here.")
