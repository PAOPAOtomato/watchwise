"""Configuration for the watchwise sidecar.

Everything that could differ between machines is here: base URLs, the broker
address, which models run where, and thresholds. Loaded from YAML with
``${VAR}`` / ``${VAR:-default}`` interpolation and ``WATCHWISE_*`` env-var
overrides, so one file works both on the host and inside docker compose.

Models are configured in two layers:

  models:  named endpoints -- a provider (stub, ollama, openai_compat,
           anthropic), a model name, and where to reach it
  roles:   which endpoint does which job -- ``screener`` (tier-2 vision),
           ``investigator`` (tier-3 tool-calling agent), ``chat``

``watchwise init`` writes both from a hardware profile; anyone can edit them by
hand. Older configs with only ``vlm:`` and ``agent:`` sections keep working.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Provider = Literal["stub", "ollama", "openai_compat", "anthropic"]
Effort = Literal["low", "medium", "high", "xhigh", "max"]
ROLE_NAMES = ("screener", "investigator", "chat", "chat_escalation")


class NVRSettings(BaseModel):
    """How to reach the NVR (Frigate) base layer."""

    base_url: str = "http://localhost:5000"
    # Frigate's web page as your phone reaches it (e.g. over Tailscale), not
    # the internal API above. Alert cards then link each alert's recording
    # there; the link (host name and review id) passes through Telegram.
    web_url: str | None = None
    api_prefix: str = "/api"
    # NVR supports header auth when behind its proxy; empty means none.
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 10.0
    verify_tls: bool = True


class MQTTSettings(BaseModel):
    host: str = "localhost"
    port: int = 1883
    username: str | None = None
    password: str | None = None
    # Must match the NVR's own mqtt.topic_prefix; upstream Frigate uses "frigate".
    topic_prefix: str = "frigate"
    client_id: str = "watchwise"
    keepalive: int = 60
    enabled: bool = True
    # Which review reports to screen: "new" (the event started) and "end".
    # A later report of the same review costs a model call only if it shows
    # a new object label or zone.
    screen_on: list[Literal["new", "update", "end"]] = Field(default_factory=lambda: ["new", "end"])


class GateSettings(BaseModel):
    """The cheap, model-free pre-screening filter.

    This is the single biggest lever on AI cost: anything the gate resolves
    never reaches a VLM or the agent.
    """

    enabled: bool = True
    # Drop a repeat of the same signature seen within this window.
    dedupe_window_s: float = 60.0
    # Minimum tracked-object score before we bother screening at all.
    min_object_score: float = 0.0
    # Labels that always go straight through to a human alert.
    always_alert_labels: list[str] = Field(default_factory=list)
    # Labels that are never worth a model call.
    never_screen_labels: list[str] = Field(default_factory=list)
    # Suppress when a long-term misjudgment pattern matches at/above this.
    memory_suppress_similarity: float = 0.86
    # A pattern must have this many confirmed human corrections before it is
    # trusted to auto-suppress on its own.
    memory_min_corrections: int = 3
    # Memory never holds back a kind of event (camera, labels, zones) that a
    # human marked "real" within this many days.
    real_veto_days: float = 30.0
    # free: a trusted routine is held back here, before any model looks (it
    # can only compare camera, labels and zones). model_checked: never here;
    # the screener sees descriptions of the events the owner marked expected
    # or real and decides whether this one matches (one model call per event).
    memory_mode: Literal["free", "model_checked"] = "free"


class ModelEndpoint(BaseModel):
    """One model on one server, referenced by name from ``roles``."""

    provider: Provider = "stub"
    model: str = "stub"
    # Ollama: http://localhost:11434. OpenAI-compatible: the provider's base
    # URL, with or without the trailing /v1. Anthropic: leave unset.
    base_url: str | None = None
    # Name of the environment variable holding the key -- never the key itself.
    api_key_env: str | None = None
    timeout_s: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 2048
    # Ollama only: context length (num_ctx). Ollama's own default is small
    # enough to silently truncate a tool-calling prompt.
    context_window: int | None = None
    # Ollama only: how long to keep the model loaded between calls.
    keep_alive: str | None = None
    # Ollama only: false turns off a thinking model's hidden reasoning (a
    # large share of its time per call); unset leaves the model's default.
    think: bool | None = None
    # Reasoning depth / spend; unset means the model default. Anthropic: the
    # effort setting. OpenAI-compatible: sent as reasoning_effort (dropped
    # automatically if the server does not accept it).
    effort: Effort | None = None
    # Anthropic only: re-run a declined request on Anthropic's recommended
    # fallback model (server-side) instead of returning the refusal.
    refusal_fallback: bool = True


class RolesSettings(BaseModel):
    """Which named model endpoint performs each job."""

    screener: str | None = None
    investigator: str | None = None
    chat: str | None = None
    chat_escalation: str | None = None


class VLMSettings(BaseModel):
    """Tier-2 screening. ``roles.screener`` overrides the model fields here,
    including the output cap; the tuning fields (frames, image size, timeout,
    concurrency) always apply."""

    backend: Provider = "stub"
    model: str = "stub-vlm"
    base_url: str | None = None
    api_key_env: str | None = None
    # Frames sampled across the review segment and sent to the VLM.
    frames: int = 3
    max_image_edge: int = 640
    jpeg_quality: int = 70
    timeout_s: float = 20.0
    temperature: float = 0.0
    # Output cap. With roles.screener, the endpoint's max_tokens is used and
    # this applies only if set explicitly. Thinking models (the default
    # qwen3-vl tags) spend 400-700 tokens reasoning before the JSON verdict.
    max_tokens: int = 1024
    # Concurrency cap so N cameras cannot stampede one GPU.
    max_concurrency: int = 2
    # When an event ends, this many frames spread across all of it go to the
    # screener model for a "What happened" summary on the card (one extra call
    # per event). 0 turns it off.
    summary_frames: int = 4
    # Things the summary should watch for and name.
    summary_concerns: list[str] = Field(
        default_factory=lambda: ["Unusual or erratic behaviour or gestures by a person"]
    )
    context_window: int | None = None
    keep_alive: str | None = None
    effort: Effort | None = None
    refusal_fallback: bool = True


class AgentSettings(BaseModel):
    # auto: the ``tools`` loop when roles.investigator names a real model,
    # otherwise the deterministic ``scripted`` loop.
    backend: Literal["auto", "scripted", "tools", "claude_sdk"] = "auto"
    # Used by the claude_sdk backend only; the tools backend takes its model
    # from roles.investigator.
    model: str = "claude-opus-5-5"
    # ReAct budget: typical investigations take 2-3 rounds; this is the hard cap.
    max_iterations: int = 6
    # Stop early once evidence reaches this confidence.
    stop_confidence: float = 0.85
    timeout_s: float = 180.0
    max_concurrency: int = 2
    # Bounded queue: tier 3 must never apply backpressure to tier 1/2.
    queue_size: int = 128
    effort: Effort = "medium"
    max_budget_usd: float | None = 0.50


class MemorySettings(BaseModel):
    db_path: str = "./data/watchwise.db"
    embedding_backend: Literal["hash", "nvr_clip"] = "hash"
    embedding_dim: int = 256
    # Short-term memory: how much of one investigation stays in context.
    short_term_max_items: int = 40
    short_term_max_chars: int = 12000
    # Long-term: decay so a pattern that stops recurring loses influence.
    pattern_half_life_days: float = 30.0


class DecisionSettings(BaseModel):
    """Confidence policy. Anything below threshold goes to a human."""

    # Tier 2 may resolve on its own at or above these.
    screen_benign_confidence: float = 0.80
    screen_suspicious_confidence: float = 0.80
    # Tier 3 conclusions below this are escalated to a human regardless.
    human_fallback_confidence: float = 0.60
    # Auto-downgrade requires both agreement and this confidence.
    auto_downgrade_confidence: float = 0.85
    # If true, a suspicious verdict always reaches a human even at high conf.
    always_surface_suspicious: bool = True
    # How long the first card may wait for the screener. A slower screen gets
    # an "Analyzing..." card at once, which the verdict then edits in place.
    # None: always wait for the verdict.
    provisional_after_s: float | None = 10.0
    # The same while armed away or at night, when a quick card matters more
    # than an extra buzz for something harmless.
    provisional_after_armed_s: float | None = 0.0
    # Frigate's GenAI summary of a finished event may raise one that was
    # downgraded or held back to an alert (suspicious or critical rating).
    # It never lowers anything.
    summary_can_raise: bool = True


NotifyOn = Literal["alerted", "escalated_human", "downgraded", "suppressed"]


class TelegramSettings(BaseModel):
    """Alert cards on your phone, with buttons that teach the memory.

    Telegram is a cloud relay: card text (and the snapshot, unless
    ``send_images`` is false) passes through Telegram's servers.
    """

    enabled: bool = False
    # Name of the variable in .env holding the bot token -- never the token.
    bot_token_env: str = "WATCHWISE_TELEGRAM_BOT_TOKEN"
    # Allow-list: only these chats receive alerts or may press buttons.
    chat_ids: list[int] = Field(default_factory=list)
    send_images: bool = True
    # Which dispositions produce a card.
    notify_on: list[NotifyOn] = Field(default_factory=lambda: ["alerted", "escalated_human"])
    snooze_minutes: int = 60
    # After 🚨 Real, ask "What was wrong?"; the reply teaches what to look for.
    ask_why_on_real: bool = True
    # Pause between messages to one chat (Telegram asks for <= 1/s per chat).
    min_interval_s: float = 1.0
    poll_timeout_s: int = 30
    api_base_url: str = "https://api.telegram.org"


class MQTTPublishSettings(BaseModel):
    """Publish every verdict on MQTT (uses the broker in ``mqtt:``)."""

    enabled: bool = False
    topic_prefix: str = "watchwise"


class NotifySettings(BaseModel):
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    mqtt: MQTTPublishSettings = Field(default_factory=MQTTPublishSettings)
    # POST every verdict as JSON to this URL.
    webhook_url: str | None = None


class FaceSettings(BaseModel):
    """Who lives here, as the NVR's face recognition knows them (D-057).

    Recognition runs in the NVR (Frigate's Face Library), on this machine;
    Watchwise only reads the names it reports. A recognized name appears in
    card text, which Telegram relays.
    """

    # Names exactly as in the NVR's Face Library. Empty: faces change nothing.
    household: list[str] = Field(default_factory=list)
    # A face the NVR saw but could not match to anyone raises the event when it ends.
    alert_on_unknown: bool = True
    # Attempts without any match before a face counts as unknown.
    min_unknown_attempts: int = Field(default=2, ge=1)


class DutyRosterSettings(BaseModel):
    """Rule base: who/what is expected where and when."""

    path: str | None = None
    timezone: str = "UTC"


class ServiceSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 5055
    log_level: str = "INFO"
    # Load the local models at start-up, and keep the screener loaded while
    # the service runs, so that no alert waits for a model to load.
    warm_up_models: bool = True


class WatchwiseConfig(BaseModel):
    # Informational: the hardware profile `watchwise init` generated this from.
    profile: str | None = None
    nvr: NVRSettings = Field(default_factory=NVRSettings)
    mqtt: MQTTSettings = Field(default_factory=MQTTSettings)
    gate: GateSettings = Field(default_factory=GateSettings)
    models: dict[str, ModelEndpoint] = Field(default_factory=dict)
    roles: RolesSettings = Field(default_factory=RolesSettings)
    vlm: VLMSettings = Field(default_factory=VLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    decision: DecisionSettings = Field(default_factory=DecisionSettings)
    notify: NotifySettings = Field(default_factory=NotifySettings)
    duty_roster: DutyRosterSettings = Field(default_factory=DutyRosterSettings)
    faces: FaceSettings = Field(default_factory=FaceSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)

    @model_validator(mode="after")
    def _roles_name_known_models(self) -> "WatchwiseConfig":
        for role in ROLE_NAMES:
            name = getattr(self.roles, role)
            if name and name not in self.models:
                raise ValueError(
                    f"roles.{role} refers to model {name!r}, but models: only "
                    f"defines {sorted(self.models) or 'nothing'}"
                )
        return self

    # ---------------- role resolution ----------------

    def endpoint(self, role: str) -> ModelEndpoint | None:
        """The model endpoint configured for ``role``, if any."""
        if role not in ROLE_NAMES:
            raise ValueError(f"unknown role {role!r}; roles are {ROLE_NAMES}")
        name = getattr(self.roles, role)
        return self.models.get(name) if name else None

    def screener_settings(self) -> VLMSettings:
        """Tier-2 settings with ``roles.screener`` applied on top of ``vlm``.

        The output cap comes from the endpoint, sized for its model, unless
        ``vlm.max_tokens`` is set explicitly. (It used to be a fixed 512, or
        256 on the CPU profiles, which a thinking model fills before it
        answers.)
        """
        ep = self.endpoint("screener")
        if ep is None:
            return self.vlm
        update: dict[str, Any] = {
            "backend": ep.provider,
            "model": ep.model,
            "base_url": ep.base_url,
            "api_key_env": ep.api_key_env,
            "temperature": ep.temperature,
            "context_window": ep.context_window,
            "keep_alive": ep.keep_alive,
            "effort": ep.effort,
            "refusal_fallback": ep.refusal_fallback,
        }
        if "max_tokens" not in self.vlm.model_fields_set:
            update["max_tokens"] = ep.max_tokens
        return self.vlm.model_copy(update=update)

    def screener_max_tokens_key(self) -> str:
        """The config key that sets the screener's output cap."""
        if self.endpoint("screener") is None or "max_tokens" in self.vlm.model_fields_set:
            return "vlm.max_tokens"
        return f"models.{self.roles.screener}.max_tokens"

    def investigator_backend(self) -> str:
        """Resolve ``agent.backend: auto`` to a concrete backend name."""
        if self.agent.backend != "auto":
            return self.agent.backend
        ep = self.endpoint("investigator")
        return "tools" if ep is not None and ep.provider != "stub" else "scripted"

    # ---------------- loading ----------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "WatchwiseConfig":
        data: dict[str, Any] = {}
        path = path or os.environ.get("WATCHWISE_CONFIG")
        if path:
            p = Path(path)
            if p.exists():
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        cfg = cls.model_validate(interpolate_env(data))
        return _apply_env_overrides(cfg)


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def interpolate_env(value: Any, env: dict[str, str] | None = None) -> Any:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in every string value.

    An unset variable with no default expands to an empty string, which the
    settings treat as "not configured" rather than as a literal URL.
    """
    source = os.environ if env is None else env
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: source.get(m.group(1)) or (m.group(2) or ""), value)
    if isinstance(value, dict):
        return {k: interpolate_env(v, env) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v, env) for v in value]
    return value


_ENV_MAP = {
    "WATCHWISE_NVR_URL": ("nvr", "base_url"),
    "WATCHWISE_MQTT_HOST": ("mqtt", "host"),
    "WATCHWISE_MQTT_PORT": ("mqtt", "port"),
    "WATCHWISE_MQTT_PREFIX": ("mqtt", "topic_prefix"),
    "WATCHWISE_MQTT_ENABLED": ("mqtt", "enabled"),
    "WATCHWISE_VLM_BACKEND": ("vlm", "backend"),
    "WATCHWISE_VLM_MODEL": ("vlm", "model"),
    "WATCHWISE_VLM_BASE_URL": ("vlm", "base_url"),
    "WATCHWISE_AGENT_BACKEND": ("agent", "backend"),
    "WATCHWISE_AGENT_MODEL": ("agent", "model"),
    "WATCHWISE_DB_PATH": ("memory", "db_path"),
    "WATCHWISE_PORT": ("service", "port"),
    "WATCHWISE_LOG_LEVEL": ("service", "log_level"),
}


def _coerce(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw


def _apply_env_overrides(cfg: WatchwiseConfig) -> WatchwiseConfig:
    for env, (section, key) in _ENV_MAP.items():
        raw = os.environ.get(env)
        if raw is None:
            continue
        sub = getattr(cfg, section)
        setattr(sub, key, _coerce(getattr(sub, key), raw))
    return cfg
