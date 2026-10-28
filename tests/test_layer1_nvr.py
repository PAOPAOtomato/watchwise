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


async def test_mqtt_source_passes_on_a_newly_named_person_and_face_attempts():
    """D-057: Frigate names a recognized face in a review update, and reports
    every recognition attempt on <prefix>/tracked_object_update."""
    import asyncio
    from types import SimpleNamespace

    from watchwise.config import MQTTSettings
    from watchwise.nvr.source import MQTTAlertSource

    queue: asyncio.Queue = asyncio.Queue()
    faces: list[dict] = []
    source = MQTTAlertSource(queue, MQTTSettings(), on_face=faces.append)
    source._loop = asyncio.get_running_loop()

    def review(kind, before, after):
        side = lambda subs: {"id": "r", "camera": "c", "start_time": 1.0, "severity": "alert",  # noqa: E731
                             "data": {"objects": ["person"], "sub_labels": subs}}
        payload = {"type": kind, "before": side(before), "after": side(after)}
        return SimpleNamespace(payload=json.dumps(payload).encode(), topic=source.topic)

    source._on_message(None, None, review("update", [], []))  # nothing new: skipped
    source._on_message(None, None, review("update", [], ["rui"]))  # newly named: passed on
    face = {"type": "face", "id": "obj-1", "name": None, "score": 0.0, "camera": "c"}
    source._on_message(None, None, SimpleNamespace(payload=json.dumps(face).encode(), topic=source.face_topic))
    lpr = {**face, "type": "lpr"}
    source._on_message(None, None, SimpleNamespace(payload=json.dumps(lpr).encode(), topic=source.face_topic))
    await asyncio.sleep(0.05)
    assert queue.qsize() == 1 and queue.get_nowait().sub_labels == ["rui"]
    assert faces == [face]
    assert source.face_topic == f"{MQTTSettings().topic_prefix}/tracked_object_update"


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


def _genai_payload(review_id="r", level=1, **metadata):
    meta = {"title": "Person at the door", "shortSummary": "A person walks to the door, looks around and leaves.",
            "scene": "x", "confidence": 0.8, "potential_threat_level": level, **metadata}
    return {"type": "genai", "after": {"id": review_id, "camera": "c", "start_time": 1.0, "end_time": 9.0,
                                       "severity": "alert", "data": {"objects": ["person"], "metadata": meta}}}


def test_frigates_summary_of_a_finished_review_is_read():
    ctx = review_payload_to_context(_genai_payload(level=2))
    assert ctx is not None and ctx.summary is not None
    assert (ctx.summary.threat_level, ctx.summary.confidence) == (2, 0.8)
    assert ctx.summary.text() == "Person at the door. A person walks to the door, looks around and leaves."
    odd = review_payload_to_context(_genai_payload(level="lots", confidence=7))
    assert (odd.summary.threat_level, odd.summary.confidence) == (0, 0.0)
    plain = review_payload_to_context({"type": "end", "after": {"id": "r", "camera": "c", "start_time": 1.0}})
    assert plain.summary is None


async def test_mqtt_source_always_passes_frigates_summaries_on():
    import asyncio
    from types import SimpleNamespace

    from watchwise.config import MQTTSettings
    from watchwise.nvr.source import MQTTAlertSource

    queue: asyncio.Queue = asyncio.Queue()
    source = MQTTAlertSource(queue, MQTTSettings(), screen_on=("end",))
    source._loop = asyncio.get_running_loop()
    source._on_message(None, None, SimpleNamespace(payload=json.dumps(_genai_payload()).encode(), topic="t"))
    await asyncio.sleep(0.05)
    assert queue.get_nowait().summary is not None


def test_a_named_person_is_still_a_person():
    """B-025: Frigate reports a recognized face's person as "person-verified"."""
    ctx = review_payload_to_context({"type": "new", "after": {
        "id": "r", "camera": "c", "start_time": 1.0, "severity": "alert",
        "data": {"objects": ["person-verified", "person", "car"], "sub_labels": ["Ray"]}}})
    assert ctx.labels == ["person", "car"] and ctx.sub_labels == ["Ray"]
    assert ctx.signature() == "c|car,person|nozone"


async def test_the_face_library_lists_people_not_frigates_training_folder(nvr_client, nvr_state):
    """D-060: /setup offers the Face Library's names as residents."""
    assert await nvr_client.faces() == {}
    nvr_state.faces = {"Ray": 4, "alex": 1}
    assert await nvr_client.faces() == {"Ray": 4, "alex": 1}


async def test_cameras_switched_off_in_the_nvr_are_left_out(nvr_client, nvr_state):
    """A camera disabled in Frigate's config (laptopcam after the live test)
    is not asked about in /setup, nor offered for photos."""
    nvr_state.cameras["cam_door"]["enabled"] = False
    assert await nvr_client.cameras() == ["cam_drive", "cam_yard"]
