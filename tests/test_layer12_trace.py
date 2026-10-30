"""Layer 12: the trace export, a recorded run for the web demo (M2-10, M3-9).

A run through the real pipeline, bot and chat (fake Telegram, fake NVR, a
scripted chat model) is exported and read back: every decision in order,
the investigation's steps with evidence ids, every message as the phone
showed it, the owner's taps (one undone), a spot-check, a chat answer with
its tool steps and a confirmed proposal, the digest, arming, and numbers.
Privacy: people pixelated over Frigate's boxes, frames that can't be
blurred safely left out, names replaced, no chat ids, links or Frigate ids.
"""

from __future__ import annotations

import asyncio
import io
import json
import time

import httpx
import pytest
from PIL import Image, ImageStat

from watchwise.config import CameraSettings, TelegramSettings
from watchwise.fakes.telegram_server import FakeTelegramState, build_fake_telegram
from watchwise.llm.base import ChatResult, ToolCallRequest
from watchwise.llm.stub import StubChat
from watchwise.notify.telegram import TelegramAPI
from watchwise.trace import ExportOptions, Scrubber, TraceExporter, blur_people, name_aliases, parse_when
from watchwise.wiring import build_system
from tests.conftest import DAY_TS, NIGHT_TS, make_alert

CHAT = 111


@pytest.fixture
async def house(config, nvr_client, nvr_state, tmp_path):
    app, state = build_fake_telegram(FakeTelegramState())
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    api = TelegramAPI(state.token, base_url="http://fake-telegram", client=http)
    config.notify.telegram = TelegramSettings(enabled=True, chat_ids=[CHAT], min_interval_s=0.0)
    config.faces.household[:] = ["Ray"]
    config.cameras["cam_yard"] = CameraSettings(room="Garden")
    model = StubChat("stub-chat")
    sys_ = build_system(config, nvr_client=nvr_client, start_sources=False, telegram_api=api, chat_model=model)
    sys_.background = [sys_.notifier]
    await sys_.start()
    yield sys_, state, model, nvr_state
    await sys_.bot.stop()
    await sys_.stop()
    await http.aclose()


async def _poll(sys_) -> None:
    await sys_.bot.poll_once()
    await asyncio.gather(*sys_.bot.tasks)


async def _settle(sys_) -> None:
    await sys_.notifier.drain()
    await sys_.investigations.drain(timeout=10)
    await sys_.notifier.drain()


async def _recorded_run(sys_, state, model, nvr_state) -> None:
    """A night: a person by the shed (card, investigation, 🚨 + note), a cat
    held back (spot-checked ✅ from the digest), a second person tapped ✅
    then undone, and a question with a photo and a confirmed proposal."""
    for review_id, offset in (("rev-shed", 0), ("rev-gate", 600)):
        nvr_state.add_event(event_id=f"{review_id}-obj", camera="cam_yard", start_time=NIGHT_TS + offset,
                            box=[0.25, 0.25, 0.5, 0.5], frame_time=NIGHT_TS + offset + 2)
        nvr_state.add_review(review_id=review_id, camera="cam_yard", start_time=NIGHT_TS + offset,
                             detections=[f"{review_id}-obj"])
        await sys_.pipeline.handle(make_alert(review_id=review_id, camera="cam_yard", labels=["person"],
                                              zones=["perimeter"], start_time=NIGHT_TS + offset))
        await _settle(sys_)
    await sys_.pipeline.handle(make_alert(review_id="rev-cat", camera="cam_yard", labels=["cat"],
                                          start_time=NIGHT_TS + 300))
    shed, gate = [m for m in state.sent if m["method"] == "sendPhoto"][:2]

    state.push_button(CHAT, "fb:real:rev-shed", message_id=shed["message_id"])
    await _poll(sys_)
    state.push_message(CHAT, "Ray says nobody should be by the shed at night", reply_to=state.sent[-1]["message_id"])
    await _poll(sys_)
    state.push_button(CHAT, "fb:ok:rev-gate", message_id=gate["message_id"])
    await _poll(sys_)
    state.push_button(CHAT, [b for b in state.buttons(state.edits[-1]) if b.startswith("fb:undo")][0],
                      message_id=gate["message_id"])
    await _poll(sys_)

    sys_.bot.digests.clock = lambda: NIGHT_TS + 3600
    state.push_message(CHAT, "/digest")
    await _poll(sys_)
    check = state.sent[-1]
    state.push_button(CHAT, "fb:ok:rev-cat", message_id=check["message_id"])
    await _poll(sys_)

    model.script = [
        ChatResult(content="", tool_calls=[ToolCallRequest("c1", "show_snapshot", {"event_id": "rev-shed"})]),
        ChatResult(content="", tool_calls=[ToolCallRequest("c2", "propose_arming", {"state": "armed_night"})]),
        ChatResult(content="Here is the shed at night (see https://example.test/x). Tap Confirm to arm Night."),
    ]
    sys_.store.execute("UPDATE alerts SET start_time=start_time")  # (the chat reads the store as it is)
    state.push_message(CHAT, "show me the shed one, and arm night like Ray asked")
    await _poll(sys_)
    offer = state.sent[-1]
    state.push_button(CHAT, state.buttons(offer)[0], message_id=offer["message_id"])
    await _poll(sys_)


async def test_a_recorded_run_exports_as_a_trace_the_demo_can_replay(house, tmp_path):
    sys_, state, model, nvr_state = house
    started = time.time()
    await _recorded_run(sys_, state, model, nvr_state)
    options = ExportOptions(since=NIGHT_TS - 60, until=time.time() + 1, machine="test machine")
    out = tmp_path / "export"
    result = await TraceExporter(sys_.store, sys_.config, sys_.nvr, options).run(out)
    raw = (out / "trace.json").read_text(encoding="utf-8")
    trace = json.loads(raw)

    assert (trace["format"], trace["version"], trace["meta"]["source"]) == ("watchwise-trace", "1.0", "recorded")
    assert trace["meta"]["privacy"]["frames"] == "blurred" and trace["meta"]["machine"] == "test machine"
    events = {e["id"]: e for e in trace["events"]}
    assert len(events) == 3 and {e["camera"] for e in events.values()} == {"cam_yard"}
    assert [c for c in trace["cameras"] if c["id"] == "cam_yard"][0]["room"] == "Garden"

    shed = trace["events"][0]["id"]
    stages = [d["stage"] for d in trace["decisions"] if d["event"] == shed]
    assert stages == ["screened", "investigated"], "every decision in order, not only the last"
    (inv, *_) = [i for i in trace["investigations"] if i["event"] == shed]
    assert inv["steps"] and inv["steps"][0]["evidence_id"] == f"{inv['id']}/1"
    assert inv["observations"] == [] and inv["unknowns"] == [], "M1-6 fields reserved"

    kinds = [(m["kind"], m["action"]) for m in trace["messages"]]
    for expected in [("alert", "sent"), ("alert", "edited"), ("note_request", "sent"), ("digest", "sent"),
                     ("spot_check", "sent"), ("chat", "sent"), ("chat_photo", "sent"), ("proposal", "edited")]:
        assert expected in kinds
    first_card = trace["messages"][0]
    assert first_card["event"] == shed and first_card["frame"] == "f01"
    labels = [b["label"] for row in first_card["buttons"] for b in row]
    assert "✅ Expected" in labels and all("callback" not in json.dumps(b) for row in first_card["buttons"] for b in row)

    taps = [a for a in trace["owner_actions"] if a["kind"] == "button"]
    assert [t["action"] for t in taps] == ["real", "ok", "undo", "ok", "confirm"]
    assert taps[3]["outcome"]["spot_check"] is True
    notes = [a for a in trace["owner_actions"] if a["kind"] == "note"]
    assert notes[0]["text"] == "Resident A says nobody should be by the shed at night"
    corrections = {c["verdict"]: c for c in trace["corrections"] if c["undone"] is None}
    assert corrections["suspicious"]["note"].startswith("Resident A says") and corrections["benign"]["via"] == "spot_check"
    (undone,) = [c for c in trace["corrections"] if c["undone"]]
    assert undone["verdict"] == "benign" and undone["via"] == "card", "Undo deleted it; the taps still tell"

    (turn,) = trace["conversations"][0]["turns"]
    assert [s["tool"] for s in turn["tools"]] == ["show_snapshot", "propose_arming"] and turn["photos"] == [shed]
    assert turn["answer"] == "Here is the shed at night (see [link]). Tap Confirm to arm Night."
    (proposal,) = trace["proposals"]
    assert proposal["status"] == "applied" and turn["proposals"] == [proposal["id"]]
    assert [a["state"] for a in trace["arming"]][-1] == "armed_night"
    (digest,) = trace["digests"]
    assert digest["spot_checks"] == [trace["events"][1]["id"]] and digest["sent"]

    counts = trace["metrics"]["counts"]
    assert counts["events"] == 3 and counts["held_back"] == 1 and counts["spot_checks"]["answered"] == 1
    assert trace["metrics"]["timings"]["label"].startswith("measured (test machine")

    assert result.frames == 2 and result.boxes_blurred == 2 and (out / "frames" / "f01.jpg").exists()
    # The chat id as an identity (digits alone can be part of any timing, e.g. "t": 1111.3).
    for secret in ("Ray", "rev-shed", "telegram:", f'"{CHAT}"', f"{CHAT}:", "https://"):
        assert secret not in raw, f"{secret!r} must not be in the export"
    assert result.names == {"Ray": "Resident A"}
    assert time.time() - started < 30


def test_blurring_pixelates_inside_the_box_only():
    img = Image.new("RGB", (200, 100), (120, 120, 120))
    for x in range(200):
        for y in range(100):
            if (x // 4 + y // 4) % 2:
                img.putpixel((x, y), (255, 255, 255))
    raw = io.BytesIO()
    img.save(raw, "PNG")
    blurred, covered = blur_people(raw.getvalue(), [(0.5, 0.0, 0.25, 0.5)], margin=0.0)
    out = Image.open(io.BytesIO(blurred)).convert("L")
    inside = ImageStat.Stat(out.crop((105, 5, 145, 45))).stddev[0]
    outside = ImageStat.Stat(out.crop((10, 55, 50, 95))).stddev[0]
    assert covered == 1 and inside < 10 < outside, "a checkerboard is unreadable inside the box, intact outside"


async def test_a_frame_that_cant_be_blurred_safely_is_left_out(house, tmp_path):
    sys_, state, model, nvr_state = house
    nvr_state.add_event(event_id="a", camera="cam_yard", start_time=NIGHT_TS, box=[0.1, 0.1, 0.2, 0.2])
    nvr_state.add_event(event_id="b", camera="cam_yard", start_time=NIGHT_TS, box=[0.6, 0.1, 0.2, 0.2])
    nvr_state.add_review(review_id="two", camera="cam_yard", start_time=NIGHT_TS, detections=["a", "b"])
    nvr_state.add_event(event_id="c", camera="cam_yard", start_time=NIGHT_TS + 900)
    nvr_state.add_review(review_id="nobox", camera="cam_yard", start_time=NIGHT_TS + 900, detections=["c"])
    for review_id, offset in (("two", 0), ("nobox", 900)):
        await sys_.pipeline.handle(make_alert(review_id=review_id, camera="cam_yard", labels=["person"],
                                              zones=["perimeter"], start_time=NIGHT_TS + offset))
    await _settle(sys_)
    options = ExportOptions(since=NIGHT_TS - 60, until=time.time() + 1)
    result = await TraceExporter(sys_.store, sys_.config, sys_.nvr, options).run(tmp_path / "x")
    assert result.frames == 0 and set(result.frames_left_out.values()) == {
        "more than one person in view: can't be sure every one is covered", "no best frame with a box from Frigate"}

    none = await TraceExporter(sys_.store, sys_.config, sys_.nvr,
                               ExportOptions(since=NIGHT_TS - 60, until=time.time() + 1, frames="none")).run(tmp_path / "y")
    assert none.frames == 0 and not none.frames_left_out


def test_names_become_aliases_and_roles_stay(config):
    config.faces.household[:] = ["Ray"]
    aliases, roles = name_aliases(config, ["ray", "Jiahua"], ["Alex", "cleaner", "night_security_patrol"],
                                  {"bin_collection": "the bins"})
    assert aliases == {"Ray": "Resident A", "Jiahua": "Resident B", "Alex": "Visitor A", "bin_collection": "the bins"}
    assert roles == ["cleaner", "night_security_patrol"]
    scrub = Scrubber(aliases)
    assert scrub.text("RAY and alex met; see http://x.y/z by telegram:123 (Rayburn)") == (
        "Resident A and Visitor A met; see [link] by owner (Rayburn)")
    assert scrub.any({"a": ["Jiahua", 3], "b": None}) == {"a": ["Resident B", 3], "b": None}
    assert scrub.any({"recognized Ray": 4}) == {"recognized Resident A": 4}, "keys too (live 2026-10-30)"


def test_export_times_can_be_relative_or_local():
    now = DAY_TS
    assert parse_when("-6h", now) == now - 6 * 3600 and parse_when("-90m", now) == now - 5400
    assert parse_when("2026-03-10 14:00") == DAY_TS
    with pytest.raises(ValueError):
        parse_when("yesterday")
