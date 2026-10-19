"""Layer 3: the four forensic tool categories, and their MCP exposure.

Each tool is tested against the fake NVR through the real ToolContext, then
the same tools are checked to serialize correctly for both MCP transports.
"""

from __future__ import annotations

import time

import pytest

from sentinel.tools.base import ToolContext, summarize
from sentinel.tools.cross_camera import TOOL as CROSS_TOOL
from sentinel.tools.device_status import TOOL as DEVICE_TOOL
from sentinel.tools.device_status import ArmingState
from sentinel.tools.duty_roster import TOOL as ROSTER_TOOL
from sentinel.tools.duty_roster import DutyRoster
from sentinel.tools.frames import DESCRIBE_FRAMES_TOOL, GET_FRAMES_TOOL
from sentinel.tools.registry import ALL_TOOLS, CATEGORIES, get_tool, tool_catalog
from sentinel.tools.similar_alerts import TOOL as SIMILAR_TOOL
from tests.conftest import make_alert


@pytest.fixture
def tool_ctx(nvr_client, store, patterns, tmp_path):
    from sentinel.tier2.vlm.stub import StubVLM

    return ToolContext(
        nvr=nvr_client,
        store=store,
        patterns=patterns,
        roster=None,
        config_cache={"vlm": StubVLM(), "arming_path": str(tmp_path / "arming.json")},
    )


# ---------------- registry ----------------


def test_all_four_spec_categories_plus_frames_are_present():
    assert set(CATEGORIES) == {
        "vector_search",
        "device_status",
        "cross_camera",
        "duty_roster",
        "frames",
    }
    assert len(ALL_TOOLS) == 6


def test_every_tool_has_a_valid_json_schema():
    for tool in ALL_TOOLS:
        schema = tool.input_schema
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        for name in schema["required"]:
            assert name in schema["properties"], f"{tool.name}: required {name} undeclared"
        assert len(tool.description) > 80, f"{tool.name} description too thin for a model"


def test_get_tool_rejects_unknown_names():
    with pytest.raises(KeyError):
        get_tool("no_such_tool")


def test_tool_catalog_is_serializable():
    catalog = tool_catalog()
    assert {e["name"] for e in catalog} == {t.name for t in ALL_TOOLS}


# ---------------- category 1: vector search ----------------


async def test_similar_alerts_finds_prior_human_judgement(tool_ctx, patterns):
    alert = make_alert(camera="cam_drive", labels=["car"], zones=["driveway"])
    patterns.upsert(
        kind="false_alarm",
        signature=alert.signature(),
        camera="cam_drive",
        description="headlights sweeping the gate",
    )
    result = await SIMILAR_TOOL.run(
        tool_ctx,
        {"query": "car in driveway", "camera": "cam_drive", "signature": alert.signature()},
    )
    assert result["ok"]
    assert result["pattern_matches"][0]["kind"] == "false_alarm"
    assert "headlights" in result["interpretation"]


async def test_similar_alerts_merges_the_nvr_clip_index(tool_ctx, nvr_state):
    nvr_state.add_event(
        event_id="ev-9",
        camera="cam_yard",
        description="courier leaving a parcel at the gate",
    )
    result = await SIMILAR_TOOL.run(tool_ctx, {"query": "courier parcel gate"})
    assert [h["event_id"] for h in result["similar_events"]] == ["ev-9"]
    assert result["nvr_search_error"] is None


async def test_similar_alerts_survives_a_dead_semantic_index(tool_ctx, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("embeddings disabled")

    monkeypatch.setattr(tool_ctx.nvr, "search_events", boom)
    result = await SIMILAR_TOOL.run(tool_ctx, {"query": "anything"})
    assert result["ok"], "a disabled index must degrade, not fail the tool"
    assert "embeddings disabled" in result["nvr_search_error"]


async def test_similar_alerts_reports_signature_recurrence(tool_ctx, store):
    alert = make_alert(camera="cam_drive", labels=["person"])
    for i in range(3):
        rec = make_alert(review_id=f"r{i}", camera="cam_drive", labels=["person"]).to_dict()
        rec["signature"] = alert.signature()
        rec["start_time"] = time.time() - 100
        store.upsert_alert(rec, "screened")
        store.record_decision(
            {
                "review_id": f"r{i}",
                "disposition": "downgraded",
                "severity": "detection",
                "confidence": 0.9,
                "rationale": "x",
                "screen": None,
                "investigation_id": None,
                "stage": "screened",
            }
        )
    result = await SIMILAR_TOOL.run(
        tool_ctx, {"query": "person", "signature": alert.signature()}
    )
    assert result["signature_recurrence"]["count"] == 3
    assert result["signature_recurrence"]["dispositions"] == {"downgraded": 3}


# ---------------- category 2: device status ----------------


async def test_device_status_reports_healthy_system(tool_ctx):
    result = await DEVICE_TOOL.run(tool_ctx, {"camera": "cam_drive"})
    assert result["ok"]
    assert result["cameras"]["cam_drive"]["healthy"]
    assert result["unhealthy_cameras"] == []
    assert "armed_away" in result["interpretation"]


async def test_device_status_flags_a_dead_camera(tool_ctx, nvr_state):
    nvr_state.camera_stats["cam_drive"]["camera_fps"] = 0.0
    result = await DEVICE_TOOL.run(tool_ctx, {"camera": "cam_drive"})
    assert not result["cameras"]["cam_drive"]["healthy"]
    assert "cam_drive" in result["unhealthy_cameras"]
    assert "degraded" in result["interpretation"]


async def test_device_status_flags_a_stalled_detector(tool_ctx, nvr_state):
    nvr_state.detector_stats["ov"]["inference_speed"] = 0.0
    result = await DEVICE_TOOL.run(tool_ctx, {})
    assert not result["detectors"]["ov"]["healthy"]
    assert "no inference" in result["interpretation"]


async def test_device_status_reflects_arming_state(tool_ctx, tmp_path):
    arming = ArmingState(tmp_path / "arming.json")
    arming.set("disarmed")
    result = await DEVICE_TOOL.run(tool_ctx, {"camera": "cam_drive"})
    assert result["arming"]["state"] == "disarmed"
    assert "Occupant presence is expected" in result["interpretation"]


async def test_device_status_when_nvr_is_down(tool_ctx, monkeypatch):
    async def boom():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(tool_ctx.nvr, "stats", boom)
    result = await DEVICE_TOOL.run(tool_ctx, {})
    assert result["ok"] is False
    assert "connection refused" in result["error"]


def test_arming_state_rejects_nonsense(tmp_path):
    with pytest.raises(ValueError):
        ArmingState(tmp_path / "a.json").set("banana")


# ---------------- category 3: cross-camera ----------------


async def test_cross_reference_finds_nothing_when_the_property_is_quiet(tool_ctx):
    result = await CROSS_TOOL.run(
        tool_ctx, {"start_time": time.time(), "exclude_camera": "cam_drive"}
    )
    assert result["event_count"] == 0
    assert "single camera" in result["interpretation"]


async def test_cross_reference_builds_an_ordered_path(tool_ctx, nvr_state):
    t0 = time.time()
    nvr_state.add_event(event_id="e1", camera="cam_drive", start_time=t0 + 5)
    nvr_state.add_event(event_id="e2", camera="cam_yard", start_time=t0 + 30)
    nvr_state.add_event(event_id="e3", camera="cam_door", start_time=t0 + 60)

    result = await CROSS_TOOL.run(
        tool_ctx,
        {"start_time": t0, "exclude_camera": "cam_drive", "window_after_s": 120},
    )
    assert result["event_count"] == 2
    assert list(result["cameras_with_activity"]) == ["cam_yard", "cam_door"]
    assert "cam_yard -> cam_door" in result["interpretation"]
    assert result["timeline"][0]["offset_s"] == pytest.approx(30.0, abs=1.0)


async def test_cross_reference_excludes_the_alerting_camera(tool_ctx, nvr_state):
    t0 = time.time()
    nvr_state.add_event(event_id="e1", camera="cam_drive", start_time=t0)
    result = await CROSS_TOOL.run(
        tool_ctx, {"start_time": t0, "exclude_camera": "cam_drive"}
    )
    assert "cam_drive" not in result["cameras_searched"]
    assert result["event_count"] == 0


# ---------------- category 4: duty roster ----------------


@pytest.fixture
def roster_file(tmp_path):
    path = tmp_path / "roster.yml"
    path.write_text(
        """
timezone: UTC
entries:
  - name: night_patrol
    kind: shift
    days: [mon, tue, wed, thu, fri, sat, sun]
    start: "22:00"
    end: "06:00"
    cameras: [cam_yard]
    labels: [person]
    expect: rostered guard patrol
    confidence: 0.9
  - name: bin_day
    kind: scheduled_activity
    days: [tue]
    start: "06:00"
    end: "07:30"
    cameras: [cam_drive]
    expect: refuse collection
    confidence: 0.85
""",
        encoding="utf-8",
    )
    return path


def _utc(y, m, d, hh, mm):
    import datetime as dt

    return dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc).timestamp()


async def test_roster_absent_is_reported_honestly(tool_ctx):
    result = await ROSTER_TOOL.run(tool_ctx, {"timestamp": time.time()})
    assert result["configured"] is False
    assert "Do not treat this as evidence" in result["interpretation"]


async def test_roster_matches_a_shift_that_wraps_midnight(tool_ctx, roster_file):
    tool_ctx.roster = DutyRoster(roster_file, "UTC")
    # 2026-03-10 is a Tuesday; 02:30 UTC falls inside the 22:00-06:00 shift.
    result = await ROSTER_TOOL.run(
        tool_ctx,
        {
            "timestamp": _utc(2026, 3, 10, 2, 30),
            "camera": "cam_yard",
            "labels": ["person"],
        },
    )
    assert result["expected"] is True
    assert result["matches"][0]["name"] == "night_patrol"


async def test_roster_does_not_match_the_wrong_camera(tool_ctx, roster_file):
    tool_ctx.roster = DutyRoster(roster_file, "UTC")
    result = await ROSTER_TOOL.run(
        tool_ctx,
        {"timestamp": _utc(2026, 3, 10, 2, 30), "camera": "cam_door", "labels": ["person"]},
    )
    assert result["expected"] is False
    assert "unscheduled" in result["interpretation"]


async def test_roster_day_restriction(tool_ctx, roster_file):
    tool_ctx.roster = DutyRoster(roster_file, "UTC")
    tuesday = await ROSTER_TOOL.run(
        tool_ctx, {"timestamp": _utc(2026, 3, 10, 6, 30), "camera": "cam_drive"}
    )
    assert tuesday["expected"] is True
    wednesday = await ROSTER_TOOL.run(
        tool_ctx, {"timestamp": _utc(2026, 3, 11, 6, 30), "camera": "cam_drive"}
    )
    assert wednesday["expected"] is False


def test_roster_reloads_when_the_file_changes(roster_file):
    roster = DutyRoster(roster_file, "UTC")
    assert len(roster.match(_utc(2026, 3, 10, 2, 30), camera="cam_yard")) == 1
    roster_file.write_text("timezone: UTC\nentries: []\n", encoding="utf-8")
    import os

    os.utime(roster_file, (time.time() + 10, time.time() + 10))
    assert roster.match(_utc(2026, 3, 10, 2, 30), camera="cam_yard") == []


# ---------------- frames ----------------


async def test_get_historical_frames_reports_availability(tool_ctx):
    result = await GET_FRAMES_TOOL.run(
        tool_ctx, {"camera": "cam_drive", "timestamp": 1_700_000_000.0,
                   "offsets_s": [-30, 0, 30]}
    )
    assert result["retrieved"] == 3
    assert result["clip_url"].endswith("clip.mp4")


async def test_get_historical_frames_when_footage_is_gone(tool_ctx, nvr_state):
    nvr_state.recordings_available = False
    result = await GET_FRAMES_TOOL.run(
        tool_ctx, {"camera": "cam_drive", "timestamp": 1.0, "offsets_s": [-10]}
    )
    assert result["retrieved"] == 0
    assert "predate retention" in result["interpretation"]


async def test_describe_historical_frames_actually_runs_the_vlm(tool_ctx):
    result = await DESCRIBE_FRAMES_TOOL.run(
        tool_ctx,
        {
            "camera": "cam_drive",
            "timestamp": 1_700_000_000.0,
            "offsets_s": [-30, 0],
            "question": "Is the same person present earlier?",
        },
    )
    assert result["ok"]
    assert result["frames_examined"] == 2
    assert result["verdict"] in ("benign", "suspicious", "uncertain")
    assert result["model"].startswith("stub/")


# ---------------- error handling & MCP exposure ----------------


async def test_tool_exceptions_become_error_results(tool_ctx, monkeypatch):
    async def boom(*a, **k):
        raise ValueError("kaboom")

    monkeypatch.setattr(tool_ctx.nvr, "list_events", boom)
    result = await CROSS_TOOL.run(tool_ctx, {"start_time": 1.0})
    assert result["ok"] is False
    assert "kaboom" in result["error"]
    assert "latency_ms" in result


def test_summarize_truncates_and_marks_errors():
    assert summarize({"ok": False, "error": "bad"}) == "ERROR: bad"
    long = summarize({"ok": True, "x": "y" * 900}, limit=100)
    assert len(long) == 100 and long.endswith("...")


def test_mcp_stdio_server_lists_every_tool(tool_ctx):
    from sentinel.mcp.stdio_server import build_server

    server = build_server(tool_ctx)
    assert server.name == "sentinel-forensics"
    assert server.get_request_handler("tools/list") is not None
    assert server.get_request_handler("tools/call") is not None


async def test_mcp_stdio_handlers_round_trip(tool_ctx):
    import json

    import mcp.types as mcp_types

    from sentinel.mcp.stdio_server import build_server

    server = build_server(tool_ctx)
    listed = await server.get_request_handler("tools/list").handler(None, None)
    assert {t.name for t in listed.tools} == {t.name for t in ALL_TOOLS}

    called = await server.get_request_handler("tools/call").handler(
        None,
        mcp_types.CallToolRequestParams(
            name="get_device_status", arguments={"camera": "cam_drive"}
        ),
    )
    assert called.is_error is False
    payload = json.loads(called.content[0].text)
    assert payload["cameras"]["cam_drive"]["healthy"] is True


async def test_mcp_stdio_unknown_tool_is_an_error_not_a_crash(tool_ctx):
    import mcp.types as mcp_types

    from sentinel.mcp.stdio_server import build_server

    server = build_server(tool_ctx)
    result = await server.get_request_handler("tools/call").handler(
        None, mcp_types.CallToolRequestParams(name="nope", arguments={})
    )
    assert result.is_error is True


def test_sdk_mcp_server_builds_with_namespaced_tool_names(tool_ctx):
    """The in-process MCP server handed to the Claude Agent SDK."""
    from sentinel.mcp.sdk_server import (
        SERVER_NAME,
        allowed_tool_names,
        build_sdk_mcp_server,
        build_sdk_tools,
    )

    tools = build_sdk_tools(tool_ctx)
    assert len(tools) == len(ALL_TOOLS)

    config = build_sdk_mcp_server(tool_ctx)
    assert config["type"] == "sdk"
    assert config["name"] == SERVER_NAME

    names = allowed_tool_names()
    assert "mcp__forensics__search_similar_alerts" in names
    assert len(names) == len(ALL_TOOLS)


async def test_sdk_mcp_tool_handler_returns_mcp_content(tool_ctx):
    import json

    from sentinel.mcp.sdk_server import build_sdk_tools

    tools = {t.name: t for t in build_sdk_tools(tool_ctx)}
    out = await tools["get_device_status"].handler({"camera": "cam_drive"})
    assert out["isError"] is False
    assert out["content"][0]["type"] == "text"
    assert json.loads(out["content"][0]["text"])["ok"] is True
