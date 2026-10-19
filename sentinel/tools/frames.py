"""Retrieving additional historical frames on demand.

This is an agent action rather than a tool category, and
it is the one that most often resolves an "uncertain" screening: the screener
saw three frames from inside the segment, but the answer is in what happened
thirty seconds *before* it started.

Two tools:

  get_historical_frames -- descriptive metadata plus retrievable URLs, which
      is what a text-only agent loop can use directly.
  describe_historical_frames -- actually pulls the frames and runs the tier-2
      VLM on them, returning a description. This is how the agent "looks
      again" without needing image support in its own loop.
"""

from __future__ import annotations

from typing import Any

from sentinel.timeutil import safe_strftime
from sentinel.tools.base import ForensicTool, ToolContext, obj_schema


def _plan(start: float, offsets: list[float]) -> list[dict[str, Any]]:
    out = []
    for off in offsets:
        ts = start + off
        out.append({"timestamp": ts, "offset_s": off, "local_time": safe_strftime(ts)})
    return out


async def _get_historical_frames(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    camera = str(args["camera"])
    anchor = float(args["timestamp"])
    offsets = [float(o) for o in (args.get("offsets_s") or [-60, -30, -10, 10, 30])]
    limit = int(args.get("limit", 6))
    offsets = offsets[:limit]

    planned = _plan(anchor, offsets)
    available: list[dict[str, Any]] = []
    for item in planned:
        try:
            frame = await ctx.nvr.recording_frame(camera, item["timestamp"], height=360)
            item["bytes"] = len(frame.jpeg)
            item["available"] = True
        except Exception as exc:
            item["available"] = False
            item["error"] = str(exc)
        available.append(item)

    got = [f for f in available if f.get("available")]
    return {
        "camera": camera,
        "anchor": anchor,
        "frames": available,
        "retrieved": len(got),
        "clip_url": ctx.nvr.clip_url(camera, anchor - 30, anchor + 30),
        "interpretation": (
            f"{len(got)} of {len(available)} requested frames exist in the recordings. "
            "Use describe_historical_frames to have them actually looked at."
            if got
            else "No recordings exist for those timestamps; the window may predate "
            "retention or recording may be disabled for this camera."
        ),
    }


async def _describe_historical_frames(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    """Pull frames and run the screening VLM over them with a custom question."""
    vlm = ctx.config_cache.get("vlm")
    if vlm is None:
        return {
            "ok": False,
            "error": "no VLM backend wired into the tool context",
        }

    camera = str(args["camera"])
    anchor = float(args["timestamp"])
    offsets = [float(o) for o in (args.get("offsets_s") or [-30, -10, 10])][:4]
    question = str(args.get("question") or "Describe what is visible and what changes.")

    from sentinel.tier2.sampling import downscale_jpeg

    frames = []
    for item in _plan(anchor, offsets):
        try:
            frame = await ctx.nvr.recording_frame(camera, item["timestamp"], height=480)
        except Exception:
            continue
        frame.jpeg = downscale_jpeg(frame.jpeg, max_edge=640, quality=70)
        frames.append(frame)

    if not frames:
        return {"ok": False, "error": "no frames retrievable for those offsets"}

    system = (
        "You are examining surveillance frames on request from an investigation "
        "agent. Answer the question directly and factually. Reply with one JSON "
        'object: {"verdict": "benign"|"suspicious"|"uncertain", "confidence": '
        '0.0-1.0, "reason": "<what you see>", "tags": []}'
    )
    prompt = (
        f"{len(frames)} frame(s) from camera {camera}, offsets "
        f"{offsets} seconds relative to the alert.\n\nQuestion: {question}\n\n"
        "Respond with the JSON object only."
    )
    result = await vlm.screen(frames, prompt, system)
    return {
        "camera": camera,
        "frames_examined": len(frames),
        "offsets_s": offsets,
        "question": question,
        "verdict": result.verdict.value,
        "confidence": result.confidence,
        "description": result.reason,
        "model": f"{result.backend}/{result.model}",
    }


GET_FRAMES_TOOL = ForensicTool(
    name="get_historical_frames",
    description=(
        "Check which historical frames exist around a timestamp on a camera, at "
        "offsets you choose (negative = before the alert). Returns availability and "
        "a clip URL. Use it to establish what footage exists before asking for it "
        "to be described."
    ),
    category="frames",
    input_schema=obj_schema(
        {
            "camera": {"type": "string"},
            "timestamp": {"type": "number", "description": "Unix timestamp to anchor on."},
            "offsets_s": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Second offsets from the anchor; negative is earlier.",
            },
            "limit": {"type": "integer", "description": "Max frames (default 6)."},
        },
        required=["camera", "timestamp"],
    ),
    handler=_get_historical_frames,
)

DESCRIBE_FRAMES_TOOL = ForensicTool(
    name="describe_historical_frames",
    description=(
        "Actually look at historical frames and answer a specific question about "
        "them, using the vision model. This is how you gather NEW visual evidence "
        "when the initial screening was uncertain -- for example 'was this person "
        "already present 60 seconds earlier?' or 'is the gate open in these frames?'"
    ),
    category="frames",
    input_schema=obj_schema(
        {
            "camera": {"type": "string"},
            "timestamp": {"type": "number"},
            "offsets_s": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Up to 4 second-offsets from the anchor.",
            },
            "question": {
                "type": "string",
                "description": "The specific visual question to answer.",
            },
        },
        required=["camera", "timestamp", "question"],
    ),
    handler=_describe_historical_frames,
)
