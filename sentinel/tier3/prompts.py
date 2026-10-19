"""System prompt and conclusion contract for the investigation agent.

The prompt is written to produce the intended behaviour and to
suppress the two failure modes that make an "autonomous" alert agent worse
than a classifier:

  * **Jumping to a conclusion.** The agent is told explicitly that low
    confidence is an acceptable outcome and that another round of evidence
    gathering is preferred over a confident guess.
  * **Tool-calling for its own sake.** It is also told that 2-3 rounds is the
    expected shape and that redundant calls cost real money, so it should
    stop as soon as the evidence settles.
"""

from __future__ import annotations

import json

from sentinel.tools.registry import tool_catalog
from sentinel.types import AlertContext, ScreenResult

SYSTEM = """You are the investigation stage of a video surveillance system.

A fast screening model has already looked at an alert and could not resolve it \
confidently. Your job is to investigate: gather evidence with the tools you \
have, then reach a defensible conclusion about whether a human needs to act.

How to work:
- Think about what would actually change your mind, then call the tool that \
answers it. Do not call tools you do not need.
- Typical investigations take 2-3 rounds of think -> call tool -> observe. \
Stop as soon as the evidence settles; each extra round costs real money.
- If the evidence is insufficient, gather MORE evidence. Do not guess. \
Concluding "uncertain" with an honest confidence is a correct outcome and \
sends the alert to a human, which is safe. A confident wrong answer is not.
- Weigh what a tool tells you by how reliable it is. A degraded camera makes \
its own labels untrustworthy. A duty-roster match explains activity that would \
otherwise look anomalous. A prior human correction on this exact signature is \
strong evidence.
- Never claim to have seen something you did not retrieve. Cite the tool that \
gave you each fact.

Investigation order that usually works:
1. search_similar_alerts   - has a human already ruled on this exact situation?
2. get_device_status       - is the system armed, and is the camera trustworthy?
3. cross_reference_cameras - did anything else see it, and in what order?
4. check_duty_roster       - was this expected at this time?
5. describe_historical_frames - look at footage before/after the alert window.

When you are done, output exactly one JSON object and nothing else:
{"verdict": "benign"|"suspicious"|"uncertain",
 "confidence": 0.0-1.0,
 "conclusion": "<2-4 sentences: what you established, from which tools, and why it \
supports the verdict>",
 "recommended_action": "<what a human should do, or 'none'>",
 "evidence": ["<short fact>", ...]}"""


def build_task_prompt(
    ctx: AlertContext, screen: ScreenResult | None, memory_briefing: str
) -> str:
    """The per-alert opening message."""
    alert = {
        "review_id": ctx.review_id,
        "camera": ctx.camera,
        "signature": ctx.signature(),
        "start_time": ctx.start_time,
        "end_time": ctx.end_time,
        "duration_s": round(ctx.duration, 1),
        "labels": ctx.labels,
        "sub_labels": ctx.sub_labels,
        "zones": ctx.zones,
        "event_ids": ctx.event_ids,
    }
    parts = [
        "Investigate this alert.",
        "",
        "ALERT",
        json.dumps(alert, indent=2, sort_keys=True),
    ]
    if screen is not None:
        parts += [
            "",
            "SCREENING RESULT (tier 2, why this reached you)",
            json.dumps(
                {
                    "verdict": screen.verdict.value,
                    "confidence": screen.confidence,
                    "reason": screen.reason,
                    "tags": screen.tags,
                    "model": f"{screen.backend}/{screen.model}",
                },
                indent=2,
            ),
        ]
    if memory_briefing:
        parts += ["", "WORKING MEMORY", memory_briefing]
    parts += [
        "",
        "Use the forensic tools, then give the final JSON object.",
    ]
    return "\n".join(parts)


def tool_manifest() -> str:
    """Human-readable tool list, for backends without native tool schemas."""
    lines = []
    for entry in tool_catalog():
        lines.append(f"- {entry['name']} [{entry['category']}]: {entry['description']}")
    return "\n".join(lines)
