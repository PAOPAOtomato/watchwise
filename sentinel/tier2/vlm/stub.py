"""A deterministic, model-free screening backend.

This is the default until a real VLM endpoint is configured. It exists for
three reasons:

  * every other layer becomes testable without a GPU or an API key
  * CI gets stable, exact assertions instead of model-dependent ones
  * the pipeline can be run end to end on the live NVR to validate wiring
    before spending anything on inference

It is a *heuristic*, not a model: it reads the labels, zones and time of day
off the alert context that the prompt already contains, and applies the same
rules a human would state as a first pass. It will be wrong in the ways a
rule engine is wrong -- that is the point of replacing it with a real VLM.
"""

from __future__ import annotations

import json
import re

from sentinel.timeutil import safe_hour
from sentinel.tier2.vlm.base import VLMBackend
from sentinel.types import Frame

# Labels that essentially never justify waking a human on their own.
_LOW_INTEREST = {"cat", "dog", "bird", "squirrel", "rabbit", "raccoon"}
# Labels that are inherently worth a look.
_HIGH_INTEREST = {"person", "face"}
_VEHICLE = {"car", "truck", "motorcycle", "bicycle", "bus"}

_LABELS_RE = re.compile(r'"labels"\s*:\s*\[(.*?)\]', re.DOTALL)
_ZONES_RE = re.compile(r'"zones"\s*:\s*\[(.*?)\]', re.DOTALL)
_START_RE = re.compile(r'"start_time"\s*:\s*([0-9.]+)')
_DURATION_RE = re.compile(r'"duration_s"\s*:\s*([0-9.]+)')


def _extract_list(pattern: re.Pattern[str], text: str) -> list[str]:
    m = pattern.search(text)
    if not m:
        return []
    return [s.strip().strip('"').strip("'") for s in m.group(1).split(",") if s.strip()]


class StubVLM(VLMBackend):
    """Rule-based stand-in with the same interface as a real VLM."""

    name = "stub"

    def __init__(self, model: str = "stub-vlm"):
        super().__init__(model)

    async def infer(self, frames: list[Frame], prompt: str, system: str) -> str:
        labels = {label.lower() for label in _extract_list(_LABELS_RE, prompt)}
        zones = {z.lower() for z in _extract_list(_ZONES_RE, prompt)}

        start_match = _START_RE.search(prompt)
        hour = 12
        if start_match:
            hour = safe_hour(float(start_match.group(1)))
        night = hour >= 22 or hour < 6

        duration_match = _DURATION_RE.search(prompt)
        duration = float(duration_match.group(1)) if duration_match else 0.0

        verdict, confidence, reason, tags = self._decide(
            labels, zones, night=night, duration=duration, frame_count=len(frames)
        )
        return json.dumps(
            {"verdict": verdict, "confidence": confidence, "reason": reason, "tags": tags}
        )

    @staticmethod
    def _decide(
        labels: set[str], zones: set[str], *, night: bool, duration: float, frame_count: int
    ) -> tuple[str, float, str, list[str]]:
        if not labels:
            return (
                "uncertain",
                0.30,
                "no object labels on the segment; cannot classify from motion alone",
                ["no-labels"],
            )

        if labels and labels <= _LOW_INTEREST:
            return (
                "benign",
                0.90,
                f"only low-interest animal labels present ({sorted(labels)})",
                ["animal"],
            )

        if labels & _HIGH_INTEREST:
            if night and zones:
                return (
                    "suspicious",
                    0.82,
                    f"person detected at night inside monitored zone(s) {sorted(zones)}",
                    ["person", "night", "in-zone"],
                )
            if duration < 2.0:
                return (
                    "uncertain",
                    0.45,
                    f"person label but the segment is only {duration:.1f}s; "
                    "too brief to judge intent",
                    ["person", "short"],
                )
            return (
                "suspicious",
                0.72,
                f"person present for {duration:.1f}s in {sorted(zones) or 'unzoned area'}",
                ["person"],
            )

        if labels & _VEHICLE:
            if night:
                return (
                    "uncertain",
                    0.50,
                    "vehicle at night; headlights commonly cause spurious labels",
                    ["vehicle", "night"],
                )
            return ("benign", 0.78, f"daytime vehicle activity ({sorted(labels)})", ["vehicle"])

        return (
            "uncertain",
            0.40,
            f"labels {sorted(labels)} not covered by screening heuristics",
            ["unhandled"],
        )
