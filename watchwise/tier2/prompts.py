"""Prompts for the tier-2 screener.

Design constraints, in order of importance:

  * **Cheap.** The system prompt is short and byte-stable so it caches well
    and costs little on hosted backends.
  * **Structured.** The reply must be one JSON object. Small local VL models
    follow a schema far more reliably than free prose.
  * **Honest about uncertainty.** The screener's job is to be *confidently*
    right about the easy majority and to say "uncertain" otherwise, because
    uncertain is what escalates to the agent. A screener that guesses defeats
    the whole cascade.
"""

from __future__ import annotations

import json

from watchwise.timeutil import safe_hour, safe_strftime
from watchwise.types import AlertContext

SYSTEM = """You are the fast screening stage of a video surveillance pipeline.
You look at a few frames from one motion event and decide whether it needs a \
human's attention.

Answer with exactly one JSON object and nothing else:
{"verdict": "benign"|"suspicious"|"uncertain", "confidence": 0.0-1.0,
 "reason": "<one sentence citing what you actually see>", "tags": ["<short>", ...]}

Rules:
- "benign": you can see it is ordinary activity. Routine deliveries, residents, \
pets, weather, vegetation, vehicles passing on a public road.
- "suspicious": you can see something a person should look at. Loitering, \
forced entry, concealment, an unexpected person in a restricted area, someone \
handling doors, windows or vehicles that are not theirs.
- "uncertain": you cannot tell from these frames. Use this freely. An \
"uncertain" answer costs a deeper investigation; a wrong confident answer \
costs a missed intrusion or a false alarm.
- Set confidence to how sure you are of the verdict itself, not how sure you \
are that something moved.
- Judge only what is visible. Do not infer intent you cannot see."""


def build_prompt(ctx: AlertContext, frame_count: int, extra_context: str = "") -> str:
    """Volatile, per-alert half of the prompt.

    Kept after the stable system text so hosted backends can cache the prefix.
    """
    metadata = {
        "camera": ctx.camera,
        "local_time": safe_strftime(ctx.start_time, "%Y-%m-%d %H:%M:%S"),
        "hour_of_day": safe_hour(ctx.start_time),
        "start_time": ctx.start_time,
        "duration_s": round(ctx.duration, 1),
        "labels": ctx.labels,
        "sub_labels": ctx.sub_labels,
        "zones": ctx.zones,
    }
    lines = [
        f"{frame_count} frame(s) from one motion event, in chronological order.",
        "",
        "Detector metadata (the object detector's own labels; it can be wrong):",
        json.dumps(metadata, indent=2, sort_keys=True),
    ]
    if extra_context:
        lines += ["", "Additional context:", extra_context]
    lines += ["", "Respond with the JSON object only."]
    return "\n".join(lines)
