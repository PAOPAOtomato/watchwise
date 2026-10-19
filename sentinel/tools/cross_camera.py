"""Forensic tool category 3: multi-camera cross-referencing.

A single camera cannot distinguish "someone walked up the drive, rang the
bell and left" from "someone walked up the drive and then appeared at the
back of the house". Correlating what the neighbouring cameras saw in the same
window is usually the single most decisive piece of evidence available, and
it costs one HTTP call.

The tool reports a *track*: which cameras saw what, in time order, with the
gaps between them, so the agent can reason about a path rather than a set.
"""

from __future__ import annotations

from typing import Any

from sentinel.timeutil import safe_strftime
from sentinel.tools.base import ForensicTool, ToolContext, obj_schema


async def _cross_reference_cameras(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    start = float(args["start_time"])
    window_before = float(args.get("window_before_s", 120))
    window_after = float(args.get("window_after_s", 120))
    exclude = args.get("exclude_camera")
    labels = args.get("labels") or None
    requested = args.get("cameras") or None

    cameras = requested or await ctx.camera_names()
    if exclude:
        cameras = [c for c in cameras if c != exclude]

    after = start - window_before
    before = start + window_after

    try:
        events = await ctx.nvr.list_events(
            cameras=cameras or None,
            labels=labels,
            after=after,
            before=before,
            limit=100,
        )
    except Exception as exc:
        return {"ok": False, "error": f"NVR event query failed: {exc}"}

    timeline: list[dict[str, Any]] = []
    for e in events or []:
        e_start = float(e.get("start_time") or 0.0)
        timeline.append(
            {
                "event_id": e.get("id"),
                "camera": e.get("camera"),
                "label": e.get("label"),
                "sub_label": e.get("sub_label"),
                "zones": e.get("zones") or [],
                "start_time": e_start,
                "end_time": e.get("end_time"),
                "score": e.get("score") or e.get("top_score"),
                "offset_s": round(e_start - start, 1),
                "local_time": safe_strftime(e_start) if e_start else None,
            }
        )
    timeline.sort(key=lambda x: x["start_time"])

    by_camera: dict[str, int] = {}
    for item in timeline:
        cam = str(item["camera"])
        by_camera[cam] = by_camera.get(cam, 0) + 1

    return {
        "window": {"after": after, "before": before, "anchor": start},
        "cameras_searched": cameras,
        "event_count": len(timeline),
        "cameras_with_activity": by_camera,
        "timeline": timeline[:50],
        "interpretation": _interpret(timeline, by_camera, exclude),
    }


def _interpret(
    timeline: list[dict[str, Any]], by_camera: dict[str, int], exclude: str | None
) -> str:
    if not timeline:
        return (
            "No other camera saw anything in this window. Activity confined to a "
            "single camera is more consistent with a localized or spurious trigger "
            "than with someone moving through the property."
        )
    if len(by_camera) == 1:
        cam = next(iter(by_camera))
        return f"Only {cam} recorded activity ({by_camera[cam]} event(s)) in this window."

    ordered = " -> ".join(
        dict.fromkeys(str(item["camera"]) for item in timeline)
    )
    span = timeline[-1]["start_time"] - timeline[0]["start_time"]
    return (
        f"{len(by_camera)} cameras recorded activity across {span:.0f}s, in the order "
        f"{ordered}. A consistent progression across adjacent cameras indicates "
        f"someone moving through the property rather than an isolated trigger."
    )


TOOL = ForensicTool(
    name="cross_reference_cameras",
    description=(
        "Find what every OTHER camera recorded around the same time, as an ordered "
        "timeline with offsets relative to the alert. Use this to distinguish a "
        "localized or spurious trigger from a person actually moving across the "
        "property, and to establish direction of travel."
    ),
    category="cross_camera",
    input_schema=obj_schema(
        {
            "start_time": {
                "type": "number",
                "description": "Unix timestamp to anchor the window on (the alert start).",
            },
            "window_before_s": {
                "type": "number",
                "description": "Seconds to look back (default 120).",
            },
            "window_after_s": {
                "type": "number",
                "description": "Seconds to look forward (default 120).",
            },
            "exclude_camera": {
                "type": "string",
                "description": "Camera to leave out, normally the alerting one.",
            },
            "cameras": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Restrict to these cameras. Omit to search all.",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Restrict to these object labels, e.g. ['person'].",
            },
        },
        required=["start_time"],
    ),
    handler=_cross_reference_cameras,
)
