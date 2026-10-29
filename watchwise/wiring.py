"""Composition root: build the whole system from a config object.

Every dependency is constructed here and injected downward. Nothing else in
the package reads config at import time, which is what makes each layer
independently testable -- a test builds exactly the slice it needs.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from watchwise.chat.proposals import ProposalBook
from watchwise.chat.service import ChatService
from watchwise.config import WatchwiseConfig
from watchwise.decision import DecisionPolicy
from watchwise.llm.base import ChatModel
from watchwise.memory.feedback import FeedbackLoop
from watchwise.memory.long_term import PatternLibrary
from watchwise.memory.store import Store
from watchwise.memory.vectors import build_embedder
from watchwise.memory.watches import WatchBook
from watchwise.nvr.client import NVRClient
from watchwise.nvr.source import AlertSource, MQTTAlertSource
from watchwise.pipeline import FaceFeed, Pipeline
from watchwise.sinks import LogSink, MemorySink, MQTTPublisher, MQTTSink, WebhookSink
from watchwise.tier2.gate import ScreeningGate
from watchwise.tier2.sampling import FrameSampler
from watchwise.tier2.screener import Screener
from watchwise.tier2.vlm.registry import build_vlm
from watchwise.tier3.backends.base import AgentBackend
from watchwise.tier3.backends.scripted import ScriptedBackend
from watchwise.tier3.investigator import InvestigationQueue, Investigator
from watchwise.tools.base import ToolContext
from watchwise.tools.device_status import ArmingState
from watchwise.tools.duty_roster import DutyRoster
from watchwise.types import AlertContext
from watchwise.warmup import ModelWarmer, refresh_interval

logger = logging.getLogger(__name__)


@dataclass
class System:
    """Everything the service needs, already wired together."""

    config: WatchwiseConfig
    store: Store
    patterns: PatternLibrary
    feedback: FeedbackLoop
    nvr: NVRClient
    tool_ctx: ToolContext
    screener: Screener
    policy: DecisionPolicy
    investigations: InvestigationQueue
    pipeline: Pipeline
    alert_queue: "asyncio.Queue[AlertContext]"
    memory_sink: MemorySink
    arming: ArmingState
    sources: list[AlertSource] = field(default_factory=list)
    # Long-running workers with start()/stop(): the model warmer and the
    # Telegram notifier and bot.
    background: list[Any] = field(default_factory=list)
    warmer: ModelWarmer | None = None
    notifier: Any = None
    bot: Any = None
    _closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)

    async def start(self) -> None:
        await self.pipeline.start()
        for source in self.sources:
            await source.start()
        for worker in self.background:
            await worker.start()

    async def stop(self) -> None:
        for worker in reversed(self.background):
            await worker.stop()
        for source in self.sources:
            await source.stop()
        await self.pipeline.stop()
        for closer in self._closers:
            try:
                await closer()
            except Exception:
                logger.exception("cleanup failed")
        self.store.close()


def build_agent_backend(
    config: WatchwiseConfig, tool_ctx: ToolContext
) -> tuple[AgentBackend, str]:
    """Pick the investigation backend, degrading loudly rather than silently."""
    backend = config.investigator_backend()

    if backend == "tools":
        endpoint = config.endpoint("investigator")
        if endpoint is None:
            logger.warning(
                "agent.backend=tools needs roles.investigator; falling back to scripted"
            )
            return _scripted(config), "fell back to scripted: no roles.investigator"
        from watchwise.llm.base import ChatModelError
        from watchwise.llm.registry import build_chat_model
        from watchwise.tier3.backends.tool_loop import ToolLoopBackend

        try:
            chat = build_chat_model(endpoint)
        except ChatModelError as exc:
            logger.warning("investigator model unavailable (%s); falling back to scripted", exc)
            return _scripted(config), f"fell back to scripted: {exc}"
        return ToolLoopBackend(chat, max_iterations=config.agent.max_iterations), "ready"

    if backend == "claude_sdk":
        from watchwise.tier3.backends.claude_sdk import ClaudeSDKBackend

        ok, why = ClaudeSDKBackend.ensure_available()
        if not ok:
            logger.warning(
                "agent.backend=claude_sdk requested but unavailable (%s); "
                "falling back to the scripted backend",
                why,
            )
            return _scripted(config), f"fell back to scripted: {why}"
        return (
            ClaudeSDKBackend(
                tool_ctx,
                model=config.agent.model,
                max_iterations=config.agent.max_iterations,
                effort=config.agent.effort,
                max_budget_usd=config.agent.max_budget_usd,
            ),
            "ready",
        )

    return _scripted(config), "configured"


def _scripted(config: WatchwiseConfig) -> ScriptedBackend:
    return ScriptedBackend(
        max_iterations=config.agent.max_iterations,
        stop_confidence=config.agent.stop_confidence,
    )


def build_system(
    config: WatchwiseConfig,
    *,
    nvr_client: NVRClient | None = None,
    start_sources: bool = True,
    telegram_api: Any = None,
    mqtt_publisher: Any = None,
    config_path: str | Path | None = None,
    chat_model: ChatModel | None = None,
) -> System:
    """Compose the service. ``config_path`` is the file ``config`` came from;
    /setup in Telegram writes the owner's answers there. ``chat_model``
    replaces the ``chat`` role's model (tests)."""
    store = Store(config.memory.db_path)
    watches = WatchBook(store)
    embedder = build_embedder(config.memory.embedding_backend, config.memory.embedding_dim)
    patterns = PatternLibrary(store, embedder, config.memory.pattern_half_life_days)
    feedback = FeedbackLoop(store, patterns)

    nvr = nvr_client or NVRClient(config.nvr)
    roster = DutyRoster(config.duty_roster.path, config.duty_roster.timezone)
    arming = ArmingState(_arming_path(config), config.arming)

    screening = config.screener_settings()
    vlm = build_vlm(screening)
    tool_ctx = ToolContext(
        nvr=nvr,
        store=store,
        patterns=patterns,
        roster=roster,
        config_cache={
            "vlm": vlm,
            "arming_path": str(_arming_path(config)),
            "arming": arming,
            "cameras": config.cameras,  # rooms, for cross-camera evidence (D-062)
        },
    )

    gate = ScreeningGate(config.gate, patterns, store=store)
    sampler = FrameSampler(
        nvr, max_edge=screening.max_image_edge, quality=screening.jpeg_quality
    )
    # The owner's /setup answers: these very objects, updated in place by it.
    screener = Screener(
        gate, sampler, vlm, screening, patterns, arming=arming, cameras=config.cameras, alerts=config.alerts,
        watches=watches,
    )
    policy = DecisionPolicy(config.decision)

    backend, backend_status = build_agent_backend(config, tool_ctx)
    logger.info("investigation backend: %s (%s)", backend.name, backend_status)

    investigator = Investigator(
        backend, tool_ctx, store, config.agent, config.memory, cameras=config.cameras, alerts=config.alerts
    )
    investigations = InvestigationQueue(investigator, config.agent)

    warmer = None
    if config.service.warm_up_models:
        chat = getattr(backend, "chat", None)
        others = [("investigator", chat)] if isinstance(chat, ChatModel) else []
        warmer = ModelWarmer(vlm, others, refresh_s=refresh_interval(screening.keep_alive))

    memory_sink = MemorySink()
    sinks: list[Any] = [memory_sink, LogSink()]
    closers: list[Callable[[], Awaitable[None]]] = [screener.aclose, backend.aclose, nvr.aclose]

    notify = config.notify
    if notify.mqtt.enabled:
        publisher = mqtt_publisher or MQTTPublisher(config.mqtt)
        sinks.append(MQTTSink(publisher, topic_prefix=notify.mqtt.topic_prefix))
        if hasattr(publisher, "aclose"):
            closers.append(publisher.aclose)
    if notify.webhook_url:
        webhook = WebhookSink(notify.webhook_url)
        sinks.append(webhook)
        closers.append(webhook.aclose)

    face_feed = FaceFeed()
    proposals = ProposalBook(
        store,
        roster_path=Path(config.duty_roster.path) if config.duty_roster.path else None,
        arming=arming,
        watches=watches,
    )
    chat = _build_chat(config, chat_model, store=store, nvr=nvr, proposals=proposals, watches=watches,
                       arming=arming, roster=roster)
    if chat is not None:
        closers.append(chat.model.aclose)
    notifier = bot = None
    if notify.telegram.enabled:
        notifier, bot, api = _build_telegram(
            config, store, nvr, feedback, patterns, investigations, memory_sink, telegram_api,
            arming=arming, face_feed=face_feed, config_path=config_path, chat=chat, proposals=proposals,
        )
        if notifier is not None:
            sinks.append(notifier)
            if telegram_api is None:
                closers.append(api.aclose)

    alert_queue: asyncio.Queue[AlertContext] = asyncio.Queue(maxsize=256)
    pipeline = Pipeline(
        alert_queue,
        screener,
        policy,
        store,
        investigations=investigations,
        sinks=sinks,
        arming=arming,
        roster=roster,
        faces=config.faces,
        face_feed=face_feed,
    )

    sources: list[AlertSource] = []
    if start_sources and config.mqtt.enabled:
        sources.append(
            MQTTAlertSource(
                alert_queue,
                config.mqtt,
                screen_on=config.mqtt.screen_on,
                # Always: /setup can add the first resident, and its
                # recognition check listens, while the service runs.
                on_face=pipeline.note_face,
            )
        )

    system = System(
        config=config,
        store=store,
        patterns=patterns,
        feedback=feedback,
        nvr=nvr,
        tool_ctx=tool_ctx,
        screener=screener,
        policy=policy,
        investigations=investigations,
        pipeline=pipeline,
        alert_queue=alert_queue,
        memory_sink=memory_sink,
        arming=arming,
        sources=sources,
        background=[w for w in (warmer, notifier, bot) if w is not None],
        warmer=warmer,
        notifier=notifier,
        bot=bot,
    )
    system._closers = closers
    return system


def _build_telegram(
    config: WatchwiseConfig,
    store: Store,
    nvr: NVRClient,
    feedback: FeedbackLoop,
    patterns: PatternLibrary,
    investigations: InvestigationQueue,
    memory_sink: MemorySink,
    api: Any = None,
    *,
    arming: ArmingState | None = None,
    face_feed: FaceFeed | None = None,
    config_path: str | Path | None = None,
    chat: ChatService | None = None,
    proposals: ProposalBook | None = None,
) -> tuple[Any, Any, Any]:
    """Notifier + bot, or (None, None, None) if no token is configured.

    A missing token is a warning, not a startup failure: alerts are still
    processed and available over the API.
    """
    from watchwise.notify.actions import ActionHandler
    from watchwise.notify.faces import FaceEnroller
    from watchwise.notify.setup_chat import SetupChat
    from watchwise.notify.telegram import TelegramAPI, TelegramBot, TelegramNotifier

    settings = config.notify.telegram
    if api is None:
        token = os.environ.get(settings.bot_token_env, "")
        if not token:
            logger.warning(
                "notify.telegram is enabled but %s is not set; no phone alerts will be sent",
                settings.bot_token_env,
            )
            return None, None, None
        api = TelegramAPI(token, base_url=settings.api_base_url)
    if not settings.chat_ids:
        logger.warning(
            "notify.telegram has no chat_ids; run `watchwise telegram link` to connect a chat"
        )
    notifier = TelegramNotifier(
        api, settings, store, nvr, web_url=config.nvr.web_url, cameras=config.cameras, rooms=config.rooms,
        faces=config.faces,
    )
    handler = ActionHandler(
        store=store,
        feedback=feedback,
        patterns=patterns,
        investigations=investigations,
        memory_sink=memory_sink,
        gate=config.gate,
        snooze_minutes=settings.snooze_minutes,
        cameras=config.cameras,
    )
    enroller = FaceEnroller(nvr, config.faces)
    setup = SetupChat(
        config,
        Path(config_path) if config_path else None,
        nvr,
        arming=arming,
        face_feed=face_feed,
        send_images=settings.send_images,
    )
    bot = TelegramBot(
        api, settings, handler, notifier, enroller=enroller, setup=setup, arming=arming, chat=chat, proposals=proposals
    )
    return notifier, bot, api


def _build_chat(config: WatchwiseConfig, model: ChatModel | None, **parts: Any) -> ChatService | None:
    """"Ask Watchwise" on the ``chat`` role's model (D-065), or None when no
    real model is configured: questions then get a short explanation."""
    if model is None:
        endpoint = config.endpoint("chat")
        if endpoint is None or endpoint.provider == "stub":
            return None
        from watchwise.llm.registry import build_chat_model

        try:
            model = build_chat_model(endpoint)
        except Exception as exc:
            logger.warning("chat model unavailable (%s); questions in Telegram won't be answered", exc)
            return None
    logger.info("Ask Watchwise: questions in Telegram go to %s", model.describe())
    return ChatService(model, config=config, **parts)


def _arming_path(config: WatchwiseConfig) -> Path:
    return Path(config.memory.db_path).parent / "arming.json"


async def build_tool_context(
    config_path: str | None = None,
) -> tuple[ToolContext, Callable[[], Awaitable[None]]]:
    """Minimal wiring for the standalone MCP server: tools only, no pipeline."""
    config = WatchwiseConfig.load(config_path)
    store = Store(config.memory.db_path)
    embedder = build_embedder(config.memory.embedding_backend, config.memory.embedding_dim)
    patterns = PatternLibrary(store, embedder, config.memory.pattern_half_life_days)
    nvr = NVRClient(config.nvr)
    vlm = build_vlm(config.screener_settings())
    ctx = ToolContext(
        nvr=nvr,
        store=store,
        patterns=patterns,
        roster=DutyRoster(config.duty_roster.path, config.duty_roster.timezone),
        config_cache={
            "vlm": vlm,
            "arming_path": str(_arming_path(config)),
            "arming": ArmingState(_arming_path(config), config.arming),
            "cameras": config.cameras,
        },
    )

    async def cleanup() -> None:
        await vlm.aclose()
        await nvr.aclose()
        store.close()

    return ctx, cleanup
