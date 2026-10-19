"""Composition root: build the whole system from a config object.

Every dependency is constructed here and injected downward. Nothing else in
the package reads config at import time, which is what makes each layer
independently testable -- a test builds exactly the slice it needs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from watchwise.config import WatchwiseConfig
from watchwise.decision import DecisionPolicy
from watchwise.memory.feedback import FeedbackLoop
from watchwise.memory.long_term import PatternLibrary
from watchwise.memory.store import Store
from watchwise.memory.vectors import build_embedder
from watchwise.nvr.client import NVRClient
from watchwise.nvr.source import AlertSource, MQTTAlertSource
from watchwise.pipeline import Pipeline
from watchwise.sinks import LogSink, MemorySink
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
    _closers: list[Callable[[], Awaitable[None]]] = field(default_factory=list)

    async def start(self) -> None:
        await self.pipeline.start()
        for source in self.sources:
            await source.start()

    async def stop(self) -> None:
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
) -> System:
    store = Store(config.memory.db_path)
    embedder = build_embedder(config.memory.embedding_backend, config.memory.embedding_dim)
    patterns = PatternLibrary(store, embedder, config.memory.pattern_half_life_days)
    feedback = FeedbackLoop(store, patterns)

    nvr = nvr_client or NVRClient(config.nvr)
    roster = DutyRoster(config.duty_roster.path, config.duty_roster.timezone)
    arming = ArmingState(_arming_path(config))

    screening = config.screener_settings()
    vlm = build_vlm(screening)
    tool_ctx = ToolContext(
        nvr=nvr,
        store=store,
        patterns=patterns,
        roster=roster,
        config_cache={"vlm": vlm, "arming_path": str(_arming_path(config))},
    )

    gate = ScreeningGate(config.gate, patterns, store=store)
    sampler = FrameSampler(
        nvr, max_edge=screening.max_image_edge, quality=screening.jpeg_quality
    )
    screener = Screener(gate, sampler, vlm, screening, patterns)
    policy = DecisionPolicy(config.decision)

    backend, backend_status = build_agent_backend(config, tool_ctx)
    logger.info("investigation backend: %s (%s)", backend.name, backend_status)

    investigator = Investigator(backend, tool_ctx, store, config.agent, config.memory)
    investigations = InvestigationQueue(investigator, config.agent)

    memory_sink = MemorySink()
    alert_queue: asyncio.Queue[AlertContext] = asyncio.Queue(maxsize=256)
    pipeline = Pipeline(
        alert_queue,
        screener,
        policy,
        store,
        investigations=investigations,
        sinks=[memory_sink, LogSink()],
    )

    sources: list[AlertSource] = []
    if start_sources and config.mqtt.enabled:
        sources.append(MQTTAlertSource(alert_queue, config.mqtt))

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
    )
    system._closers = [screener.aclose, backend.aclose, nvr.aclose]
    return system


def _arming_path(config: WatchwiseConfig) -> Any:
    from pathlib import Path

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
        config_cache={"vlm": vlm, "arming_path": str(_arming_path(config))},
    )

    async def cleanup() -> None:
        await vlm.aclose()
        await nvr.aclose()
        store.close()

    return ctx, cleanup
