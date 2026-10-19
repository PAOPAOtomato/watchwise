"""Forensic tool category 2: device and arming status.

Answers "can I even trust this alert?" before the agent reasons about what it
shows. A camera whose feed dropped two minutes ago, a detector running at 2
FPS, or a system that is disarmed because the owner is home all change the
correct conclusion far more than anything in the frames.

Everything here is read from NVR's /api/stats and /api/config plus a local
arming state that the operator sets through our own API -- NVR has no concept
of "armed", so that piece is ours.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from watchwise.tools.base import ForensicTool, ToolContext, obj_schema


class ArmingState:
    """Trivial persisted arming state: armed / disarmed / home, per system.

    Deliberately a file rather than a table -- an operator should be able to
    flip it with an editor when the service is down.
    """

    VALID = ("armed_away", "armed_night", "armed_home", "disarmed")

    def __init__(self, path: str | Path = "./data/arming.json"):
        self.path = Path(path)

    def get(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"state": "armed_away", "since": None, "source": "default"}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {"state": "armed_away", "since": None, "source": "unreadable"}

    def set(self, state: str, source: str = "api") -> dict[str, Any]:
        if state not in self.VALID:
            raise ValueError(f"state must be one of {self.VALID}")
        payload = {"state": state, "since": time.time(), "source": source}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        return payload


async def _get_device_status(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    camera = args.get("camera")

    try:
        stats = await ctx.nvr.stats()
    except Exception as exc:
        return {
            "ok": False,
            "error": f"NVR stats unavailable: {exc}",
            "hint": "If NVR itself is unreachable, treat detector metadata as stale.",
        }

    cameras_stats: dict[str, Any] = stats.get("cameras", {}) or {}
    detectors: dict[str, Any] = stats.get("detectors", {}) or {}
    service = stats.get("service", {}) or {}

    if camera:
        selected = {camera: cameras_stats.get(camera, {})}
    else:
        selected = cameras_stats

    camera_health = {}
    for name, cs in selected.items():
        camera_fps = float(cs.get("camera_fps") or 0.0)
        detection_fps = float(cs.get("detection_fps") or 0.0)
        process_fps = float(cs.get("process_fps") or 0.0)
        issues = []
        if camera_fps <= 0.0:
            issues.append("no frames arriving from the camera")
        elif camera_fps < 3.0:
            issues.append(f"camera fps very low ({camera_fps:.1f})")
        if process_fps <= 0.0 and camera_fps > 0.0:
            issues.append("frames arriving but not being processed")
        camera_health[name] = {
            "camera_fps": camera_fps,
            "detection_fps": detection_fps,
            "process_fps": process_fps,
            "skipped_fps": cs.get("skipped_fps"),
            "pid": cs.get("pid"),
            "healthy": not issues,
            "issues": issues,
        }

    detector_health = {}
    for name, ds in detectors.items():
        speed = float(ds.get("inference_speed") or 0.0)
        detector_health[name] = {
            "inference_speed_ms": speed,
            "detection_start": ds.get("detection_start"),
            # A detector that has effectively stopped inferring makes every
            # label on this alert suspect.
            "healthy": speed > 0.0,
        }

    arming = ctx.config_cache.get("arming_state")
    if arming is None:
        arming = ArmingState(ctx.config_cache.get("arming_path", "./data/arming.json")).get()

    unhealthy = [n for n, h in camera_health.items() if not h["healthy"]]

    return {
        "arming": arming,
        "cameras": camera_health,
        "detectors": detector_health,
        "uptime_s": service.get("uptime"),
        "nvr_version": service.get("version"),
        "storage": service.get("storage"),
        "unhealthy_cameras": unhealthy,
        "interpretation": _interpret(arming, camera_health, detector_health, camera),
    }


def _interpret(
    arming: dict[str, Any],
    camera_health: dict[str, Any],
    detector_health: dict[str, Any],
    camera: str | None,
) -> str:
    notes = [f"System is {arming.get('state', 'unknown')}."]
    if arming.get("state") in ("disarmed", "armed_home"):
        notes.append(
            "Occupant presence is expected, so routine indoor/near-door activity "
            "is less likely to be an intrusion."
        )
    target = camera_health.get(camera) if camera else None
    if target and not target["healthy"]:
        notes.append(f"Camera {camera} is degraded: {'; '.join(target['issues'])}.")
    elif target:
        notes.append(f"Camera {camera} is healthy at {target['camera_fps']:.1f} fps.")
    dead = [n for n, d in detector_health.items() if not d["healthy"]]
    if dead:
        notes.append(f"Detector(s) {dead} report no inference; labels may be stale.")
    return " ".join(notes)


TOOL = ForensicTool(
    name="get_device_status",
    description=(
        "Check whether the system and the camera can be trusted right now: arming "
        "state (armed_away / armed_night / armed_home / disarmed), per-camera frame "
        "rates and health, detector inference health, NVR uptime and storage. Call "
        "this before concluding that an alert is genuine -- a degraded camera or a "
        "disarmed system changes the interpretation of the same footage."
    ),
    category="device_status",
    input_schema=obj_schema(
        {
            "camera": {
                "type": "string",
                "description": "Camera to focus on. Omit for a whole-system view.",
            }
        }
    ),
    handler=_get_device_status,
)
