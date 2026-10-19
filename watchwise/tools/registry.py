"""The forensic tool set, assembled in one place.

Order matters only as documentation: it is the order a competent human
investigator would work in, and the agent's system prompt refers to it.
"""

from __future__ import annotations

from watchwise.tools import cross_camera, device_status, duty_roster, frames, similar_alerts
from watchwise.tools.base import ForensicTool

ALL_TOOLS: list[ForensicTool] = [
    similar_alerts.TOOL,       # 1. has a human already ruled on this?
    device_status.TOOL,        # 2. can this alert be trusted at all?
    cross_camera.TOOL,         # 3. what did the other cameras see?
    duty_roster.TOOL,          # 4. was this expected?
    frames.GET_FRAMES_TOOL,    # 5. what other footage exists?
    frames.DESCRIBE_FRAMES_TOOL,  # 6. look at it
]

TOOLS_BY_NAME: dict[str, ForensicTool] = {t.name: t for t in ALL_TOOLS}

CATEGORIES = sorted({t.category for t in ALL_TOOLS})


def get_tool(name: str) -> ForensicTool:
    try:
        return TOOLS_BY_NAME[name]
    except KeyError:
        raise KeyError(
            f"unknown tool {name!r}; available: {sorted(TOOLS_BY_NAME)}"
        ) from None


def tool_catalog() -> list[dict[str, str]]:
    """Machine-readable listing, used by the HTTP API and the prompts."""
    return [
        {"name": t.name, "category": t.category, "description": t.description}
        for t in ALL_TOOLS
    ]
