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


# What each arming state tells the model about who should be around.
ARMING_NOTES = {
    "armed_away": "armed away: the residents are out, so nobody is expected on site "
    "unless the additional context says so.",
    "armed_night": "armed for the night: the residents are home and asleep, so movement "
    "outdoors or in shared areas is unexpected.",
    "armed_home": "armed while home: the residents are home and awake.",
    "disarmed": "disarmed: the residents are around; ordinary household activity is expected.",
}


def build_prompt(
    ctx: AlertContext, frame_count: int, extra_context: str = "", *, arming: str | None = None
) -> str:
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
    if arming in ARMING_NOTES:
        lines += ["", f"Household: {ARMING_NOTES[arming]}"]
    if extra_context:
        lines += ["", "Additional context:", extra_context]
    lines += ["", "Respond with the JSON object only."]
    return "\n".join(lines)


SUMMARY_SYSTEM = """You describe what happened during one event seen by a security camera.
The frames are spread across the whole event, in chronological order. Describe \
what the people, animals or vehicles did over time (arriving, passing, \
lingering, leaving), not a single moment.

Answer with exactly one JSON object and nothing else:
{"title": "<at most 8 words>", "summary": "<one or two sentences, start to finish>",
 "threat_level": 0|1|2, "confidence": 0.0-1.0, "concerns": ["<short>", ...]}

threat_level: 0 = ordinary activity; 1 = something a person should look at; \
2 = a likely intrusion or danger. Judge only what is visible across the frames."""


def build_summary_prompt(
    ctx: AlertContext,
    frame_count: int,
    *,
    arming: str | None = None,
    concerns: list[str] | tuple[str, ...] = (),
    owner_real: list[str] | tuple[str, ...] = (),
) -> str:
    """The per-event half of the whole-event summary question."""
    metadata = {
        "camera": ctx.camera,
        "local_time": safe_strftime(ctx.start_time, "%Y-%m-%d %H:%M:%S"),
        "duration_s": round(ctx.duration, 1),
        "labels": ctx.labels,
        "zones": ctx.zones,
    }
    lines = [
        f"{frame_count} frame(s) spread across one event, in chronological order.",
        "",
        "Detector metadata (the object detector's own labels; it can be wrong):",
        json.dumps(metadata, indent=2, sort_keys=True),
    ]
    if arming in ARMING_NOTES:
        lines += ["", f"Household: {ARMING_NOTES[arming]}"]
    if concerns:
        lines += ["", "Also watch for, and list under concerns if seen:"]
        lines += [f"- {c}" for c in concerns]
    if owner_real:
        lines += ["", "The owner marked earlier events like this one real (worth an alert):"]
        lines += [f'- "{t}"' for t in owner_real]
        lines.append(
            "If this event shows the same kind of thing, say so under concerns and rate "
            "potential_threat_level at least 1."
        )
    lines += ["", "Respond with the JSON object only."]
    return "\n".join(lines)
