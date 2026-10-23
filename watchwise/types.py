"""Core data types shared across the three tiers.

These are deliberately transport-agnostic: the same objects flow from the
MQTT/replay source, through screening, into the investigation queue, and out
to the HTTP API and the memory store.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Severity(str, Enum):
    """Mirrors NVR's review severities plus our own downgrade level."""

    ALERT = "alert"
    DETECTION = "detection"
    SUPPRESSED = "suppressed"


class Verdict(str, Enum):
    """What a tier concluded about an alert."""

    BENIGN = "benign"  # confidently nothing; downgrade
    SUSPICIOUS = "suspicious"  # confidently something; keep/raise
    UNCERTAIN = "uncertain"  # not confident either way; escalate


class Disposition(str, Enum):
    """Final routing decision for an alert."""

    SUPPRESSED = "suppressed"  # never shown to a human
    DOWNGRADED = "downgraded"  # demoted to detection
    ALERTED = "alerted"  # shown as an alert
    ESCALATED_HUMAN = "escalated_human"  # explicitly queued for human review


class Stage(str, Enum):
    """Which tier produced the current state. Used for the cost funnel."""

    INGESTED = "ingested"
    GATED = "gated"  # stopped by the cheap pre-VLM gate
    SCREENED = "screened"  # tier 2 VLM ran
    INVESTIGATED = "investigated"  # tier 3 agent ran


@dataclass(slots=True)
class AlertContext:
    """One candidate alert, as handed to us by tier 1 (NVR).

    ``review_id`` is NVR's ReviewSegment id; ``event_ids`` are the tracked
    object ids inside it. Everything else is denormalized so downstream tiers
    do not need a live NVR connection to reason about the alert.
    """

    review_id: str
    camera: str
    start_time: float
    end_time: float | None = None
    severity: Severity = Severity.ALERT
    labels: list[str] = field(default_factory=list)
    sub_labels: list[str] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)
    thumb_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    received_at: float = field(default_factory=time.time)

    @property
    def duration(self) -> float:
        return (self.end_time or time.time()) - self.start_time

    def signature(self) -> str:
        """Stable, human-readable identity used for repeat-pattern matching."""
        labels = ",".join(sorted(set(self.labels))) or "unknown"
        zones = ",".join(sorted(set(self.zones))) or "nozone"
        return f"{self.camera}|{labels}|{zones}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass(slots=True)
class ScreenResult:
    """Tier-2 output."""

    verdict: Verdict
    confidence: float
    reason: str
    tags: list[str] = field(default_factory=list)
    backend: str = "unknown"
    model: str = "unknown"
    latency_ms: float = 0.0
    frames_used: int = 0
    tokens: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    # Learned patterns the model was shown that a correction should score.
    memory_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


@dataclass(slots=True)
class ToolCall:
    """One observed step of the ReAct loop."""

    iteration: int
    name: str
    arguments: dict[str, Any]
    result_summary: str
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class Investigation:
    """Tier-3 output: the full ReAct trace plus the conclusion."""

    id: str
    review_id: str
    camera: str
    status: str = "pending"  # pending | running | done | failed
    verdict: Verdict = Verdict.UNCERTAIN
    confidence: float = 0.0
    conclusion: str = ""
    iterations: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    backend: str = "unknown"
    model: str = "unknown"
    started_at: float = 0.0
    finished_at: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        if not self.started_at or not self.finished_at:
            return 0.0
        return (self.finished_at - self.started_at) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        d["latency_ms"] = self.latency_ms
        return d


@dataclass(slots=True)
class Decision:
    """The final, policy-applied outcome for one alert."""

    review_id: str
    disposition: Disposition
    severity: Severity
    confidence: float
    rationale: str
    stage: Stage
    screen: ScreenResult | None = None
    investigation_id: str | None = None
    matched_memory_ids: list[str] = field(default_factory=list)
    decided_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "disposition": self.disposition.value,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "stage": self.stage.value,
            "screen": self.screen.to_dict() if self.screen else None,
            "investigation_id": self.investigation_id,
            "matched_memory_ids": list(self.matched_memory_ids),
            "decided_at": self.decided_at,
        }


@dataclass(slots=True)
class Frame:
    """A JPEG pulled from NVR, tagged with where it came from."""

    camera: str
    timestamp: float
    jpeg: bytes
    source: str = "recording"  # recording | snapshot | thumbnail

    def __repr__(self) -> str:  # keep bytes out of logs
        return (
            f"Frame(camera={self.camera!r}, ts={self.timestamp:.2f}, "
            f"source={self.source!r}, bytes={len(self.jpeg)})"
        )
