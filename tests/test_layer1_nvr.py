"""Layer 1: the NVR integration surface.

These run the real NVRClient against a real ASGI app that implements Frigate's
URL shapes, so query-parameter names and response handling are exercised, not
mocked away.
"""

from __future__ import annotations

import json

import pytest

from watchwise.nvr.client import NVRError
from watchwise.nvr.source import review_payload_to_context
from watchwise.types import Severity


async def test_version_config_and_cameras(nvr_client, nvr_state):
    assert await nvr_client.version() == nvr_state.version
    assert await nvr_client.cameras() == ["cam_door", "cam_drive", "cam_yard"]


async def test_stats_shape_matches_what_device_status_needs(nvr_client):
    stats = await nvr_client.stats()
    assert "cameras" in stats and "detectors" in stats and "service" in stats
    assert stats["cameras"]["cam_drive"]["camera_fps"] == 5.0


async def test_review_and_event_lookup(nvr_client, nvr_state):
    nvr_state.add_review(review_id="rev-a", camera="cam_drive")
    nvr_state.add_event(event_id="ev-a", camera="cam_drive", label="person")

    review = await nvr_client.get_review("rev-a")
    assert review["camera"] == "cam_drive"

    events = await nvr_client.list_events(cameras=["cam_drive"], labels=["person"])
    assert [e["id"] for e in events] == ["ev-a"]

    with pytest.raises(NVRError):
        await nvr_client.get_review("does-not-exist")


async def test_semantic_search_is_reachable(nvr_client, nvr_state):
    nvr_state.add_event(
        event_id="ev-1",
        camera="cam_yard",
        label="person",
        description="courier leaving a parcel by the gate",
    )
    nvr_state.add_event(event_id="ev-2", camera="cam_door", label="cat")

    hits = await nvr_client.search_events("courier parcel gate")
    assert [h["id"] for h in hits] == ["ev-1"]


async def test_historical_frame_retrieval_is_the_forensic_primitive(nvr_client):
    """The agent's 'go look at t-30s' capability."""
    frame = await nvr_client.recording_frame("cam_drive", 1_700_000_000.0)
    assert frame.jpeg.startswith(b"\xff\xd8")
    assert frame.source == "recording"
    assert "bytes=" in repr(frame)  # never dumps image bytes into logs


async def test_missing_recordings_raise_rather_than_return_empty(nvr_client, nvr_state):
    nvr_state.recordings_available = False
    with pytest.raises(NVRError):
        await nvr_client.recording_frame("cam_drive", 1_700_000_000.0)


def test_review_payload_normalization():
    payload = {
        "type": "end",
        "before": {},
        "after": {
            "id": "rev-9",
            "camera": "cam_yard",
            "start_time": 1000.0,
            "end_time": 1012.5,
            "severity": "alert",
            "thumb_path": "/x.webp",
            "data": {
                "objects": ["person"],
                "zones": ["perimeter"],
                "detections": ["ev-1", "ev-2"],
                "sub_labels": ["known_face"],
            },
        },
    }
    ctx = review_payload_to_context(payload)
    assert ctx is not None
    assert ctx.review_id == "rev-9"
    assert ctx.severity is Severity.ALERT
    assert ctx.labels == ["person"]
    assert ctx.zones == ["perimeter"]
    assert ctx.event_ids == ["ev-1", "ev-2"]
    assert ctx.duration == pytest.approx(12.5)
    assert ctx.signature() == "cam_yard|person|perimeter"


def test_review_payload_without_after_falls_back_to_before():
    ctx = review_payload_to_context(
        {"type": "new", "before": {"id": "r", "camera": "c", "start_time": 1.0}}
    )
    assert ctx is not None and ctx.review_id == "r"


async def test_mqtt_source_passes_on_an_events_start_and_end_but_not_its_updates():
    import asyncio
    from types import SimpleNamespace

    from watchwise.config import MQTTSettings
    from watchwise.nvr.source import MQTTAlertSource

    queue: asyncio.Queue = asyncio.Queue()
    source = MQTTAlertSource(queue, MQTTSettings())
    source._loop = asyncio.get_running_loop()
    for kind in ("new", "update", "end"):
        after = {"id": "r", "camera": "c", "start_time": 1.0, "severity": "alert",
                 "data": {"objects": ["person"]}}
        msg = SimpleNamespace(payload=json.dumps({"type": kind, "after": after}).encode(), topic="frigate/reviews")
        source._on_message(None, None, msg)
    await asyncio.sleep(0.05)
    assert queue.qsize() == 2
    assert MQTTSettings().screen_on == ["new", "end"]


def test_unknown_severity_defaults_to_alert():
    ctx = review_payload_to_context(
        {"type": "end", "after": {"id": "r", "camera": "c", "start_time": 1.0,
                                  "severity": "weird"}}
    )
    assert ctx is not None and ctx.severity is Severity.ALERT


async def test_replay_source_feeds_the_same_queue(tmp_path):
    import asyncio

    from watchwise.nvr.source import ReplayAlertSource

    path = tmp_path / "reviews.json"
    path.write_text(
        json.dumps(
            [
                {"type": "end", "after": {"id": f"rev-{i}", "camera": "cam_drive",
                                          "start_time": 1000.0 + i,
                                          "severity": "alert",
                                          "data": {"objects": ["person"]}}}
                for i in range(3)
            ]
        ),
        encoding="utf-8",
    )
    queue: asyncio.Queue = asyncio.Queue()
    source = ReplayAlertSource(queue, path, speed=0.0)
    await source.run()
    assert queue.qsize() == 3
    assert (await queue.get()).review_id == "rev-0"
