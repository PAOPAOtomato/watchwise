"""Layer 10: alerts that reach a person, and buttons that teach the memory.

Driven over real HTTP against a fake Telegram Bot API (``fakes/telegram_server``):
an alert becomes a card with a snapshot and buttons; the card is edited in
place when the investigation finishes; a tap on "Expected" becomes a human
correction in long-term memory, and "Undo" takes it back exactly. Also: the
allow-list, snooze, "Why", "Look closer", Telegram's error envelopes, MQTT
verdict publishing (B-001), and the `watchwise telegram` commands.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
import yaml

from watchwise import cli
from watchwise.config import TelegramSettings
from watchwise.fakes.telegram_server import FakeTelegramState, build_fake_telegram
from watchwise.notify.presenter import (
    CAPTION_LIMIT,
    AlertCard,
    build_card,
    keyboard_markup,
    parse_callback,
)
from watchwise.notify.telegram import TelegramAPI, TelegramError
from watchwise.setup.render import replace_top_level_block
from watchwise.types import Decision, Disposition, ScreenResult, Severity, Stage, Verdict
from watchwise.wiring import build_system
from tests.conftest import DAY_TS, NIGHT_TS, make_alert

CHAT = 111


@pytest.fixture
def tg():
    app, state = build_fake_telegram(FakeTelegramState())
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    return TelegramAPI(state.token, base_url="http://fake-telegram", client=client), state, client


@pytest.fixture
async def system(config, nvr_client, tg):
    api, state, _ = tg
    config.notify.telegram = TelegramSettings(enabled=True, chat_ids=[CHAT], min_interval_s=0.0)
    sys_ = build_system(config, nvr_client=nvr_client, start_sources=False, telegram_api=api)
    sys_.background = [sys_.notifier]  # tests poll the bot by hand
    await sys_.start()
    yield sys_, state
    await sys_.stop()


def night_person(review_id: str = "rev-a", offset_h: float = 0.0):
    return make_alert(
        review_id=review_id,
        camera="cam_yard",
        labels=["person"],
        zones=["perimeter"],
        start_time=NIGHT_TS + offset_h * 3600,
    )


async def settle(sys_) -> None:
    await sys_.notifier.drain()
    await sys_.investigations.drain(timeout=10)
    await sys_.notifier.drain()


async def press(sys_, state, data: str) -> None:
    state.push_button(CHAT, data, message_id=state.sent[0]["message_id"])
    await sys_.bot.poll_once()


# ---------------------------------------------------------------- cards


async def test_an_alert_arrives_as_a_card_with_a_snapshot_and_buttons(system):
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await sys_.notifier.drain()

    assert len(state.sent) == 1
    card = state.sent[0]
    assert card["method"] == "sendPhoto" and card["chat_id"] == CHAT
    assert card["photo"][:2] == b"\xff\xd8"
    assert card["parse_mode"] == "HTML"
    assert "Person · cam_yard · perimeter" in card["text"]
    assert "Investigating" in card["text"]
    # No "Look closer" while an investigation is already queued.
    assert state.buttons(card) == ["fb:ok:rev-a", "fb:real:rev-a", "fb:snooze:rev-a", "fb:why:rev-a"]


async def test_the_card_is_updated_in_place_when_the_investigation_finishes(system):
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await settle(sys_)

    assert len(state.sent) == 1, "a refined verdict must edit the card, not send another"
    edit = state.edits[-1]
    assert edit["method"] == "editMessageCaption"
    assert edit["message_id"] == state.sent[0]["message_id"]
    assert "after an investigation" in edit["text"]
    assert "Investigating" not in edit["text"]


async def test_a_slow_verdict_first_shows_an_analyzing_card_then_edits_it(system):
    import asyncio

    from watchwise.tier2.vlm.base import VLMBackend

    class Slow(VLMBackend):
        name = "slow"

        async def infer(self, frames, prompt, system):
            await asyncio.sleep(0.7)
            return '{"verdict": "suspicious", "confidence": 0.85, "reason": "a person at the fence"}'

    sys_, state = system
    sys_.screener.vlm = Slow("slow-vlm")  # armed by default: the card goes out at once
    await sys_.pipeline.handle(night_person())
    await sys_.notifier.drain()

    assert len(state.sent) == 1
    assert "Analyzing the frames" in state.sent[0]["text"]
    # No "Look closer" on the placeholder: the verdict may queue an investigation itself.
    assert state.buttons(state.sent[0]) == ["fb:ok:rev-a", "fb:real:rev-a", "fb:snooze:rev-a", "fb:why:rev-a"]
    verdict = state.edits[0]
    assert verdict["message_id"] == state.sent[0]["message_id"]
    assert "a person at the fence" in verdict["text"] and "Analyzing" not in verdict["text"]


async def test_cards_link_to_the_recording_when_frigates_web_page_is_known(config, nvr_client, tg):
    """M2-13: the card judges; Frigate's page shows the whole event."""
    api, state, _ = tg
    config.notify.telegram = TelegramSettings(enabled=True, chat_ids=[CHAT], min_interval_s=0.0)
    config.nvr.web_url = "https://watchwise-dev.example.ts.net:8971/"
    sys_ = build_system(config, nvr_client=nvr_client, start_sources=False, telegram_api=api)
    sys_.background = [sys_.notifier]
    await sys_.start()
    try:
        await sys_.pipeline.handle(night_person())
        await settle(sys_)
        clip = ("▶️ Clip", "https://watchwise-dev.example.ts.net:8971/review?id=rev-a")
        assert state.links(state.sent[0]) == [clip]
        assert state.links(state.edits[-1]) == [clip], "a revised card keeps its link"
        assert "fb:ok:rev-a" in state.buttons(state.sent[0])
    finally:
        await sys_.stop()


async def test_frigates_summary_is_added_to_the_card_and_can_raise_one(system):
    """D-045, seen from the phone."""
    from watchwise.nvr.source import review_payload_to_context

    sys_, state = system

    def summary_of(ctx, level, text):
        after = {"id": ctx.review_id, "camera": ctx.camera, "start_time": ctx.start_time,
                 "end_time": ctx.end_time, "severity": "alert",
                 "data": {"objects": ctx.labels, "zones": ctx.zones,
                          "metadata": {"title": "Person by the fence", "shortSummary": text,
                                       "confidence": 0.8, "potential_threat_level": level}}}
        return review_payload_to_context({"type": "genai", "after": after})

    alert = night_person()
    await sys_.pipeline.handle(alert)
    await settle(sys_)
    await sys_.pipeline.handle(summary_of(alert, 0, "A person walks along the fence and leaves."))
    await sys_.notifier.drain()
    assert len(state.sent) == 1
    assert "What happened: Person by the fence. A person walks along the fence" in state.edits[-1]["text"]

    # A routine event sends nothing, until Frigate's review of it says otherwise.
    cat = make_alert(review_id="cat-2", camera="cam_yard", labels=["cat"])
    await sys_.pipeline.handle(cat)
    await sys_.notifier.drain()
    assert len(state.sent) == 1
    await sys_.pipeline.handle(summary_of(cat, 1, "A person crouches behind the shed."))
    await sys_.notifier.drain()
    assert len(state.sent) == 2 and "A person crouches behind the shed" in state.sent[-1]["text"]


def test_a_card_says_which_verdict_its_confidence_belongs_to():
    """B-023: an alert raised because the house was armed showed "Confidence
    0.95", which was the model's confidence that it looked ordinary."""
    screen = ScreenResult(verdict=Verdict.BENIGN, confidence=0.95, reason="a person at a desk")
    decision = Decision(review_id="r", disposition=Disposition.ALERTED, severity=Severity.ALERT,
                        confidence=0.95, rationale="a person in view while armed away: ...",
                        stage=Stage.SCREENED, screen=screen)
    card = build_card(make_alert(review_id="r"), decision)
    assert "Vision model: benign @ 0.95 · first look by the vision model" in card.lines


def test_a_face_decision_says_who_and_how_sure_frigate_was():
    """D-057: the card names who Frigate recognized, with Frigate's own score."""
    decision = Decision(review_id="r", disposition=Disposition.DOWNGRADED, severity=Severity.DETECTION,
                        confidence=0.93, rationale="recognized as Rui by Frigate's face recognition",
                        stage=Stage.FACE)
    card = build_card(make_alert(review_id="r"), decision)
    assert card.lines == ["Why: recognized as Rui by Frigate's face recognition",
                          "Decided by Frigate's face recognition (match score 0.93)"]
    unknown = Decision(review_id="r", disposition=Disposition.ALERTED, severity=Severity.ALERT, confidence=0.0,
                       rationale="a face Frigate's face recognition could not match to anyone it knows",
                       stage=Stage.FACE)
    assert build_card(make_alert(review_id="r"), unknown).lines[1] == "Decided by Frigate's face recognition"


def test_an_alert_the_owner_asked_for_is_final_and_offers_look_closer():
    """D-056: no investigation runs for a match to what the owner marked
    real, so the card must not promise one; Look closer starts one."""
    from watchwise.notify.presenter import INVESTIGATING, decision_from_row

    screen = ScreenResult(verdict=Verdict.SUSPICIOUS, confidence=0.95, reason="waving both arms",
                          tags=["owner_real"])
    decision = Decision(review_id="r", disposition=Disposition.ALERTED, severity=Severity.ALERT,
                        confidence=0.95, rationale="tier-2 screening: waving both arms",
                        stage=Stage.SCREENED, screen=screen)
    card = build_card(make_alert(review_id="r"), decision)
    assert INVESTIGATING not in card.lines and card.keyboard == "full"
    row = {**decision.to_dict(), "screen_json": __import__("json").dumps(screen.to_dict()),
           "investigation_id": None, "summary": None}
    row["keys"] = lambda: row.keys()  # sqlite3.Row's interface, as decision_from_row uses it
    assert decision_from_row(row).screen.owner_real


async def test_no_link_without_frigates_web_page(system):
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await sys_.notifier.drain()
    assert state.links(state.sent[0]) == []


def test_a_clip_link_quotes_the_review_id():
    from watchwise.notify.presenter import clip_url

    assert clip_url("https://f:8971", "1792.5-ab/c") == "https://f:8971/review?id=1792.5-ab%2Fc"
    assert clip_url(None, "x") is None and clip_url("", "x") is None


async def test_routine_activity_sends_nothing(system):
    sys_, state = system
    await sys_.pipeline.handle(make_alert(review_id="cat", camera="cam_yard", labels=["cat"]))
    await settle(sys_)
    assert state.sent == []


async def test_text_only_mode_sends_no_image(system):
    sys_, state = system
    sys_.config.notify.telegram.send_images = False
    await sys_.pipeline.handle(night_person())
    await sys_.notifier.drain()
    assert state.sent[0]["method"] == "sendMessage" and state.sent[0]["photo"] is None


# ---------------------------------------------------------------- teaching


async def test_a_press_too_old_to_answer_still_updates_the_card(system):
    """B-016: a press made while the service was stopped arrives when it
    restarts; Telegram then refuses the answer, but the press still counts."""
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await settle(sys_)
    state.failures["answerCallbackQuery"] = [
        (400, "Bad Request: query is too old and response timeout expired or query ID is invalid", None)
    ]
    await press(sys_, state, "fb:ok:rev-a")
    assert sys_.store.query("SELECT COUNT(*) AS n FROM corrections")[0]["n"] == 1
    assert "Marked expected (1 of 3)" in state.edits[-1]["text"]


async def test_expected_becomes_a_correction_and_undo_takes_it_back(system):
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await settle(sys_)

    await press(sys_, state, "fb:ok:rev-a")
    corrections = sys_.store.query("SELECT * FROM corrections")
    assert [(c["human_verdict"], c["reviewer"]) for c in corrections] == [("benign", "telegram:tester")]
    assert state.answers[-1]["text"].startswith("Thanks")
    edit = state.edits[-1]
    assert "Marked expected (1 of 3)" in edit["text"]
    undo = state.buttons(edit)[0]
    assert undo.startswith("fb:undo:") and undo.endswith(":rev-a")
    assert len(sys_.patterns.all_patterns()) == 1

    await press(sys_, state, undo)
    assert sys_.store.query("SELECT COUNT(*) AS n FROM corrections")[0]["n"] == 0
    assert sys_.patterns.all_patterns() == []
    assert "Undone" in state.edits[-1]["text"]
    assert "fb:ok:rev-a" in state.buttons(state.edits[-1])


async def test_three_taps_teach_a_pattern_and_the_card_says_so(system):
    sys_, state = system
    for i in range(3):
        rid = f"rev-{i}"
        await sys_.pipeline.handle(night_person(rid, offset_h=i))
        await settle(sys_)
        state.push_button(CHAT, f"fb:ok:{rid}")
        await sys_.bot.poll_once()
    assert "Learned from 3 of your corrections" in state.edits[-1]["text"]
    (pattern,) = sys_.patterns.all_patterns()
    assert pattern["kind"] == "false_alarm" and pattern["corrections"] == 3


async def test_real_marks_it_worth_alerting(system):
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await settle(sys_)
    await press(sys_, state, "fb:real:rev-a")
    assert sys_.store.query("SELECT human_verdict FROM corrections")[0]["human_verdict"] == "suspicious"
    assert "You asked for alerts like this" in state.edits[-1]["text"]


async def test_after_real_the_owner_can_say_what_was_wrong(system):
    """D-053: live, the only Real example was the model's own vague words.
    The owner's reply becomes the example shown to the screener."""
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await settle(sys_)
    await press(sys_, state, "fb:real:rev-a")
    question = state.sent[-1]
    assert "What was wrong?" in question["text"]
    assert question["reply_markup"]["force_reply"] is True

    state.push_message(CHAT, "  waving both arms\nat the camera ", reply_to=question["message_id"])
    await sys_.bot.poll_once()
    assert sys_.store.query("SELECT note FROM corrections")[0]["note"] == "waving both arms at the camera"
    assert "look for: “waving both arms at the camera”" in state.sent[-1]["text"]
    signature = sys_.store.get_alert("rev-a")["signature"]
    assert sys_.store.correction_examples(signature) == [("suspicious", "waving both arms at the camera")]

    # A reply to anything else is not a note.
    state.push_message(CHAT, "hello", reply_to=state.sent[0]["message_id"])
    await sys_.bot.poll_once()
    assert "Questions need a chat model" in state.sent[-1]["text"]  # no chat model in this test
    assert sys_.store.query("SELECT note FROM corrections")[0]["note"] == "waving both arms at the camera"


async def test_the_why_question_can_be_switched_off(system):
    sys_, state = system
    sys_.bot.settings.ask_why_on_real = False
    await sys_.pipeline.handle(night_person())
    await settle(sys_)
    before = len(state.sent)
    await press(sys_, state, "fb:real:rev-a")
    assert len(state.sent) == before
    await press(sys_, state, "fb:ok:rev-a")  # and Expected never asks
    assert len(state.sent) == before


# ---------------------------------------------------------------- other buttons


async def test_buttons_from_strangers_do_nothing(system):
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await settle(sys_)
    state.push_button(999, "fb:ok:rev-a")
    await sys_.bot.poll_once()
    assert state.answers[-1]["text"] == "Not authorized for this chat."
    assert sys_.store.query("SELECT COUNT(*) AS n FROM corrections")[0]["n"] == 0


async def test_start_tells_a_stranger_their_chat_id_and_greets_a_known_chat(system):
    sys_, state = system
    state.push_message(999, "/start")
    state.push_message(CHAT, "/start")
    await sys_.bot.poll_once()
    stranger, known = state.sent[-2], state.sent[-1]
    assert stranger["chat_id"] == 999 and "<code>999</code>" in stranger["text"]
    assert "watchwise telegram link" in stranger["text"]
    assert known["chat_id"] == CHAT and "connected" in known["text"]


async def test_why_replies_with_the_evidence(system):
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await settle(sys_)
    await press(sys_, state, "fb:why:rev-a")
    reply = state.sent[-1]
    assert reply["reply_to"] == state.sent[0]["message_id"]
    assert "Why this alert" in reply["text"]
    assert "Investigation by scripted" in reply["text"]


async def test_snooze_mutes_the_camera_but_alerts_are_still_recorded(system):
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await settle(sys_)
    await press(sys_, state, "fb:snooze:rev-a")
    assert sys_.store.snoozed_until("cam_yard") > time.time() + 3000
    assert "muted until" in state.edits[-1]["text"]

    await sys_.pipeline.handle(night_person("rev-b", offset_h=1))
    await settle(sys_)
    assert len(state.sent) == 1, "a snoozed camera sends no new card"
    assert sys_.store.get_alert("rev-b") is not None, "but the alert is still recorded"


async def test_look_closer_investigates_an_alert_that_was_decided_by_a_rule(system):
    sys_, state = system
    sys_.config.gate.always_alert_labels = ["fire"]
    await sys_.pipeline.handle(make_alert(review_id="fire-1", camera="cam_door", labels=["fire"]))
    await sys_.notifier.drain()
    assert "fb:look:fire-1" in state.buttons(state.sent[0])

    await press(sys_, state, "fb:look:fire-1")
    assert state.answers[-1]["text"] == "Investigating…"
    assert "Investigating at your request" in state.edits[-1]["text"]
    await settle(sys_)
    assert "after an investigation" in state.edits[-1]["text"]
    assert "Investigating at your request" not in state.edits[-1]["text"]


# ---------------------------------------------------------------- presenter


def _decision(reason: str) -> Decision:
    screen = ScreenResult(verdict=Verdict.SUSPICIOUS, confidence=0.9, reason=reason)
    return Decision(
        review_id="r", disposition=Disposition.ALERTED, severity=Severity.ALERT,
        confidence=0.9, rationale="tier-2 screening: x", stage=Stage.SCREENED, screen=screen,
    )


def test_card_text_is_escaped_and_always_fits():
    ctx = make_alert(review_id="r", labels=["person"], start_time=DAY_TS)
    card = build_card(ctx, _decision("<script>alert(1)</script> & co"))
    html = card.html()
    assert "&lt;script&gt;" in html and "&amp; co" in html and "<script>" not in html

    long = build_card(ctx, _decision("&" * 5000))
    long.footer = "✅ " + "x" * 500
    text = long.html(CAPTION_LIMIT)
    assert len(text) <= CAPTION_LIMIT
    assert text.startswith("<b>") and "</b>" in text
    assert "&amp" not in text.replace("&amp;", "")  # no entity cut in half


def test_callback_data_round_trips_and_respects_the_64_byte_limit():
    markup = keyboard_markup("rev-1", "full")
    data = [b["callback_data"] for row in markup["inline_keyboard"] for b in row]
    assert [parse_callback(d) for d in data] == [
        ("ok", "rev-1", None), ("real", "rev-1", None), ("look", "rev-1", None),
        ("snooze", "rev-1", None), ("why", "rev-1", None),
    ]
    undo = keyboard_markup("rev-1", "after_feedback:42")["inline_keyboard"][0][0]["callback_data"]
    assert parse_callback(undo) == ("undo", "rev-1", "42")
    assert keyboard_markup("x" * 70, "full") == {"inline_keyboard": []}
    assert parse_callback("garbage") is None


def test_rendered_card_dataclass_defaults():
    card = AlertCard(review_id="r", camera="c", title="t", lines=["a"])
    assert card.keyboard == "full" and card.footer == ""


# ---------------------------------------------------------------- API envelope


async def test_rate_limits_are_retried_and_unchanged_edits_ignored(tg):
    api, state, _ = tg
    state.failures["sendMessage"] = [(429, "Too Many Requests: retry after 0", 0)]
    sent = await api.send_message(CHAT, "hello")
    assert sent["message_id"] and state.calls.count("sendMessage") == 2

    state.failures["editMessageText"] = [(400, "Bad Request: message is not modified", None)]
    await api.edit(CHAT, sent["message_id"], "hello", has_image=False)  # no exception


async def test_errors_never_carry_the_token(tg):
    api, state, _ = tg
    state.failures["getMe"] = [(401, "Unauthorized", None)]
    with pytest.raises(TelegramError) as info:
        await api.get_me()
    assert state.token not in str(info.value)

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    offline = TelegramAPI(
        "secret-token", base_url="http://fake-telegram",
        client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
    )
    with pytest.raises(TelegramError) as info:
        await offline.get_me()
    assert "secret-token" not in str(info.value) and info.value.__cause__ is None


# ---------------------------------------------------------------- MQTT (B-001)


async def test_verdicts_are_published_on_mqtt_when_enabled(config, nvr_client):
    class Publisher:
        def __init__(self):
            self.messages: list[tuple[str, str]] = []

        def publish(self, topic, payload, qos=0, retain=False):
            self.messages.append((topic, payload))

    publisher = Publisher()
    config.notify.mqtt.enabled = True
    sys_ = build_system(config, nvr_client=nvr_client, start_sources=False, mqtt_publisher=publisher)
    await sys_.start()
    try:
        await sys_.pipeline.handle(night_person())
    finally:
        await sys_.stop()
    topics = {topic for topic, _ in publisher.messages}
    assert {"watchwise/verdicts", "watchwise/cam_yard/verdict"} <= topics


# ---------------------------------------------------------------- CLI


@pytest.fixture
def linked_cli(tg, monkeypatch, tmp_path):
    """`watchwise telegram ...` against the fake, with a generated config."""
    _, state, client = tg

    class Bound(TelegramAPI):
        def __init__(self, token, **kwargs):
            super().__init__(token, base_url="http://fake-telegram", client=client)

    monkeypatch.setattr("watchwise.notify.telegram.TelegramAPI", Bound)
    monkeypatch.setenv("WATCHWISE_TELEGRAM_BOT_TOKEN", state.token)
    assert cli.main(["init", "--profile", "demo", "--root", str(tmp_path), "--yes"]) == 0
    return state, tmp_path / "config" / "watchwise.yml"


def test_telegram_link_connects_the_chat_that_sends_start(linked_cli, capsys):
    state, path = linked_cli
    before = path.read_text(encoding="utf-8")
    state.push_message(555, "/start", first_name="Rui")
    assert cli.main(["telegram", "link", "--config", str(path), "--yes", "--wait", "2"]) == 0

    after = path.read_text(encoding="utf-8")
    telegram = yaml.safe_load(after)["notify"]["telegram"]
    assert telegram["enabled"] is True and telegram["chat_ids"] == [555]
    assert "# Tier 1: the NVR" in after and "# Phone alerts (Telegram)" in after
    head, tail = before.split("\nnotify:\n")[0], before.split("\nmemory:\n")[1]
    assert after.startswith(head + "\nnotify:\n"), "everything above notify is untouched"
    assert after.endswith("\nmemory:\n" + tail), "everything below notify is untouched"
    assert state.sent[-1]["chat_id"] == 555 and "Linked" in state.sent[-1]["text"]
    assert "Found chat 555 Rui" in capsys.readouterr().out


def test_telegram_test_sends_a_sample_card(linked_cli, capsys, monkeypatch):
    state, path = linked_cli
    state.push_message(555, "/start")
    assert cli.main(["telegram", "link", "--config", str(path), "--yes", "--wait", "2"]) == 0
    assert cli.main(["telegram", "test", "--config", str(path)]) == 0
    card = state.sent[-1]
    assert card["method"] == "sendPhoto" and card["chat_id"] == 555
    assert "test card" in card["text"]
    assert state.links(card) == []

    # The generated config reads Frigate's web page from the environment; the
    # test card then carries a link, so the phone can be checked before an alert.
    monkeypatch.setenv("FRIGATE_WEB_URL", "https://watchwise-dev.example.ts.net:8971")
    assert cli.main(["telegram", "test", "--config", str(path)]) == 0
    assert state.links(state.sent[-1]) == [("▶️ Clip", "https://watchwise-dev.example.ts.net:8971/review")]


def test_telegram_commands_explain_a_missing_token(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("WATCHWISE_TELEGRAM_BOT_TOKEN", raising=False)
    assert cli.main(["telegram", "test", "--config", str(tmp_path / "none.yml")]) == 2
    assert "@BotFather" in capsys.readouterr().err


def test_replacing_a_block_keeps_every_other_line():
    text = "# top\na: 1\n\n# about b\nb:\n  x: 1\n  y: 2\n\n# c\nc: 3\n"
    out = replace_top_level_block(text, "b", {"x": 9})
    assert out == "# top\na: 1\n\n# about b\nb:\n  x: 9\n\n# c\nc: 3\n"
    appended = replace_top_level_block("a: 1\n", "notify", {"telegram": {"enabled": True}})
    assert yaml.safe_load(appended) == {"a": 1, "notify": {"telegram": {"enabled": True}}}


# ---------------------------------------------------------------- /face (D-058)


async def _say(sys_, state, text: str) -> dict:
    state.push_message(CHAT, text)
    await sys_.bot.poll_once()
    return state.sent[-1]


async def test_face_takes_the_cameras_frame_and_asks_before_adding_it(system, nvr_state):
    sys_, state = system
    asked = await _say(sys_, state, "/face alex cam_door")
    assert asked["method"] == "sendPhoto" and "Add the face in this frame from cam_door" in asked["text"]
    add, cancel = state.buttons(asked)
    assert add.startswith("fc:add:") and cancel.startswith("fc:cancel:")
    assert nvr_state.faces == {}, "nothing changes before the owner confirms"

    state.push_button(CHAT, add, message_id=asked["message_id"])
    await sys_.bot.poll_once()
    assert nvr_state.faces == {"alex": 1}
    assert "Added a face for alex" in state.sent[-1]["text"] and "resident with /setup" in state.sent[-1]["text"]

    state.push_button(CHAT, add, message_id=asked["message_id"])  # a second tap
    await sys_.bot.poll_once()
    assert "expired or was already used" in state.sent[-1]["text"] and nvr_state.faces == {"alex": 1}


async def test_face_explains_what_it_needs(system, nvr_state):
    sys_, state = system
    assert "Which camera?" in (await _say(sys_, state, "/face alex"))["text"]
    assert "simple name" in (await _say(sys_, state, "/face al-ex! cam_door"))["text"]
    status = (await _say(sys_, state, "/face"))["text"]
    assert "one library for all cameras" in status and "Add photos with /face" in status

    nvr_state.face_in_view = False
    asked = await _say(sys_, state, "/face alex cam_door")
    state.push_button(CHAT, state.buttons(asked)[0], message_id=asked["message_id"])
    await sys_.bot.poll_once()
    assert "No face was detected." in state.sent[-1]["text"] and nvr_state.faces == {}

    nvr_state.face_in_view = True
    asked = await _say(sys_, state, "/face alex cam_door")
    state.push_button(CHAT, state.buttons(asked)[1], message_id=asked["message_id"])  # Cancel
    await sys_.bot.poll_once()
    assert "nothing was added" in state.sent[-1]["text"] and nvr_state.faces == {}


# ---------------------------------------------------------------- /setup and /arm (D-060)


SETUP_CONFIG = """\
# A hand-written test config.
nvr:
  base_url: http://fake-nvr
mqtt:
  enabled: false
vlm:
  backend: stub
  model: stub-vlm
  frames: 2
agent:
  backend: scripted
  max_concurrency: 1
  timeout_s: 30
memory:
  db_path: {db}
  embedding_dim: 128
notify:
  telegram:
    enabled: true
    chat_ids: [{chat}]
    min_interval_s: 0
"""


@pytest.fixture
async def home(nvr_client, nvr_state, tg, tmp_path):
    """The service as `serve --config <file>` builds it, with one camera and
    Ray in Frigate's Face Library."""
    from watchwise.config import WatchwiseConfig

    api, state, _ = tg
    nvr_state.cameras = {"webcam": {"enabled": True}}
    nvr_state.faces = {"Ray": 4}
    path = tmp_path / "watchwise.yml"
    path.write_text(SETUP_CONFIG.format(db=tmp_path / "watchwise.db", chat=CHAT), encoding="utf-8")
    config = WatchwiseConfig.load(path)
    sys_ = build_system(config, nvr_client=nvr_client, start_sources=False, telegram_api=api, config_path=path)
    sys_.background = [sys_.notifier]
    await sys_.start()
    yield sys_, state, path
    await sys_.bot.stop()
    await sys_.stop()


async def _send(sys_, state, text: str, chat: int = CHAT) -> list[dict]:
    n = len(state.sent)
    state.push_message(chat, text)
    await sys_.bot.poll_once()
    return state.sent[n:]


async def _tap(sys_, state, data: str, message: dict) -> list[dict]:
    n = len(state.sent)
    state.push_button(CHAT, data, message_id=message["message_id"])
    await sys_.bot.poll_once()
    return state.sent[n:]


async def test_setup_asks_shows_the_plan_and_saves_only_when_confirmed(home, nvr_state):
    sys_, state, path = home
    before = path.read_text(encoding="utf-8")

    intro, camera = await _send(sys_, state, "/setup")
    assert "nothing changes until you confirm" in intro["text"]
    assert camera["method"] == "sendPhoto" and "1/4 · Camera webcam" in camera["text"]
    assert state.buttons(camera) == ["su:cam:keep:0"]

    noted, who = await _send(sys_, state, "my bedroom workspace; I sit at the desk most of the day")
    assert "Noted for webcam" in noted["text"] and "2/4 · Who lives here?" in who["text"]
    assert state.buttons(who) == ["su:who:0", "su:who:done"]
    await _tap(sys_, state, "su:who:0", who)
    assert state.edits[-1]["message_id"] == who["message_id"] and "✅ Ray" in str(state.edits[-1]["reply_markup"])
    (who,) = await _send(sys_, state, "Alex, Mary Ann")
    assert "I can't use Mary Ann" in who["text"]
    labels = [b["text"] for row in who["reply_markup"]["inline_keyboard"] for b in row]
    assert labels == ["✅ Ray", "✅ Alex", "Done"]

    (always,) = await _tap(sys_, state, "su:who:done", who)
    assert "3/4 · What should always alert you" in always["text"]
    (arming,) = await _send(sys_, state, "anyone at the window\nsomeone carrying a ladder")
    assert "4/4 · When no arming state is set" in arming["text"] and "Now: Away" in arming["text"]

    (plan,) = await _tap(sys_, state, "su:arm:armed_home", arming)
    text = plan["text"]
    assert "<b>The plan</b>" in text and "📷 webcam: “my bedroom workspace; I sit at the desk most of the day” (new)" in text
    assert "👪 Residents: Ray, Alex (+ Ray, + Alex)" in text and "• someone carrying a ladder (new)" in text
    assert "Home (was Away)" in text and "Apply changes cameras, faces, alerts, arming in <code>watchwise.yml</code>" in text
    assert "go into the model prompts (stub stub-vlm)" in text
    assert state.buttons(plan) == ["su:plan:apply", "su:plan:diff", "su:plan:restart", "su:plan:cancel"]
    assert path.read_text(encoding="utf-8") == before, "nothing is written before Apply"

    (diff,) = await _tap(sys_, state, "su:plan:diff", plan)
    assert diff["text"].startswith("<pre>") and "+cameras:" in diff["text"]

    saved, faces = await _tap(sys_, state, "su:plan:apply", plan)
    backup = next(path.parent.glob("watchwise.yml.bak-*"))
    assert f"the previous version is <code>{backup.name}</code>" in saved["text"]
    assert backup.read_text(encoding="utf-8") == before and state.buttons(saved) == ["su:undo"]
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert written["faces"] == {"household": ["Ray", "Alex"]} and written["arming"] == {"default": "armed_home"}
    assert path.read_text(encoding="utf-8").startswith(before), "the rest of the file is untouched"

    # The running service uses the answers at once.
    assert sys_.config.faces.household == ["Ray", "Alex"] and sys_.pipeline.faces.household == ["Ray", "Alex"]
    assert sys_.screener.camera_notes("webcam").startswith("my bedroom workspace")
    assert sys_.screener.alerts.always == ["anyone at the window", "someone carrying a ladder"]
    assert sys_.arming.get()["state"] == "armed_home"
    owner = sys_.investigations.investigator.owner_context(make_alert(camera="webcam"))
    assert "my bedroom workspace" in owner and "someone carrying a ladder" in owner

    assert "• Ray: 4 photos ✅" in faces["text"] and "• Alex: 0 photos" in faces["text"]
    assert state.buttons(faces) == ["su:face:Alex", "su:check", "su:done"]
    (capture,) = await _tap(sys_, state, "su:face:Alex", faces)
    assert capture["method"] == "sendPhoto" and "Face Library as Alex?" in capture["text"]
    assert state.buttons(capture)[0].startswith("fc:add:") and nvr_state.faces == {"Ray": 4}

    (undone,) = await _tap(sys_, state, "su:undo", saved)
    assert "Undone" in undone["text"] and path.read_text(encoding="utf-8") == before
    assert sys_.config.faces.household == [] and sys_.screener.camera_notes("webcam") == ""
    assert sys_.arming.get()["state"] == "armed_away"
    assert "Nothing to undo" in (await _tap(sys_, state, "su:undo", saved))[0]["text"]


async def test_setup_can_be_cancelled_and_ignores_old_buttons(home):
    sys_, state, path = home
    before = path.read_text(encoding="utf-8")
    _, camera = await _send(sys_, state, "/setup")
    (cancelled,) = await _send(sys_, state, "/cancel")
    assert cancelled["text"] == "Cancelled; nothing was changed."
    assert "ended or expired" in (await _tap(sys_, state, "su:cam:keep:0", camera))[0]["text"]

    _, camera = await _send(sys_, state, "/setup")
    (who,) = await _tap(sys_, state, "su:cam:keep:0", camera)
    assert "2/4" in who["text"]
    assert "answered already" in (await _tap(sys_, state, "su:cam:keep:0", camera))[0]["text"]
    (always,) = await _tap(sys_, state, "su:who:done", who)
    await _tap(sys_, state, "su:alw:keep", always)
    (reminder,) = await _send(sys_, state, "home please")
    assert reminder["text"] == "Please use the buttons above, or /cancel."
    assert path.read_text(encoding="utf-8") == before


async def test_setup_plans_again_if_the_file_changed_before_apply(home):
    sys_, state, path = home
    _, camera = await _send(sys_, state, "/setup")
    (who,) = await _tap(sys_, state, "su:cam:keep:0", camera)
    await _tap(sys_, state, "su:who:0", who)
    (always,) = await _tap(sys_, state, "su:who:done", who)
    (arming,) = await _tap(sys_, state, "su:alw:keep", always)
    (plan,) = await _tap(sys_, state, "su:arm:armed_away", arming)
    assert "Apply changes faces in" in plan["text"]

    edited = path.read_text(encoding="utf-8") + "\n# a hand edit meanwhile\n"
    path.write_text(edited, encoding="utf-8")
    (again,) = await _tap(sys_, state, "su:plan:apply", plan)
    assert "changed after I showed you the plan, so I wrote nothing" in again["text"]
    assert path.read_text(encoding="utf-8") == edited and not list(path.parent.glob("*.bak-*"))
    saved, _ = await _tap(sys_, state, "su:plan:apply", again)
    assert "Saved" in saved["text"] and "# a hand edit meanwhile" in path.read_text(encoding="utf-8")


async def test_setup_with_nothing_new_writes_nothing(home):
    sys_, state, path = home
    before = path.read_text(encoding="utf-8")
    _, camera = await _send(sys_, state, "/setup")
    (who,) = await _tap(sys_, state, "su:cam:keep:0", camera)
    (always,) = await _tap(sys_, state, "su:who:done", who)
    (arming,) = await _tap(sys_, state, "su:alw:keep", always)
    (plan,) = await _tap(sys_, state, "su:arm:armed_away", arming)
    assert "Nothing to change" in plan["text"] and path.read_text(encoding="utf-8") == before


async def _check(sys_, state, faces_seen, *, reviews=()) -> dict:
    import asyncio

    sys_.bot.setup.check_s, sys_.bot.setup.quiet_s = 5.0, 0.2
    state.push_button(CHAT, "su:check", message_id=1)
    await sys_.bot.poll_once()
    assert "Walk past a camera" in state.sent[-1]["text"]
    await asyncio.sleep(0.05)  # the check is listening
    for face in faces_seen:
        sys_.pipeline.note_face(face)
    for review in reviews:
        await sys_.pipeline.handle(review)
    await asyncio.gather(*sys_.bot.tasks)
    return state.sent[-1]


async def test_the_recognition_check_says_what_frigate_saw(home):
    sys_, state, _ = home
    sys_.config.faces.household = ["Ray"]
    sys_.arming.set("armed_home")
    named = make_alert(review_id="walk-1", camera="webcam", labels=["person"])
    named.sub_labels = ["ray"]
    ok = await _check(sys_, state, [{"id": "o1", "name": "Ray", "score": 0.95, "camera": "webcam"}], reviews=[named])
    assert "✅ <b>Recognized Ray</b> (0.95) on webcam" in ok["text"] and "held back" in ok["text"]

    close = await _check(sys_, state, [{"id": "o2", "name": "Ray", "score": 0.81, "camera": "webcam"}])
    assert "best guess was <b>Ray</b> at 0.81, below the 0.90 Frigate needs" in close["text"]
    assert "su:face:Ray" in state.buttons(close)

    nobody = await _check(sys_, state, [{"id": "o3", "name": None, "camera": "webcam"}] * 2)
    assert "saw a face 2 times but matched nobody" in nobody["text"]

    sys_.bot.setup.check_s = 0.2
    state.push_button(CHAT, "su:check", message_id=1)
    await sys_.bot.poll_once()
    import asyncio

    await asyncio.gather(*sys_.bot.tasks)
    assert "No face reached Frigate's face recognition" in state.sent[-1]["text"]


async def test_arm_shows_and_switches_the_arming_state(home):
    sys_, state, _ = home
    (status,) = await _send(sys_, state, "/arm")
    assert "Arming: <b>Away</b> (nothing set yet" in status["text"]
    assert state.buttons(status) == ["ar:armed_home", "ar:armed_away", "ar:armed_night", "ar:disarmed"]
    state.push_button(CHAT, "ar:armed_home", message_id=status["message_id"])
    await sys_.bot.poll_once()
    assert sys_.arming.get()["state"] == "armed_home" and sys_.arming.get()["source"] == "telegram:tester"
    assert "Arming: <b>Home</b> (since" in state.edits[-1]["text"] and state.answers[-1]["text"] == "Arming: Home"
    await _send(sys_, state, "/arm night")
    assert sys_.arming.get()["state"] == "armed_night"
    assert "Send /arm home, away, night or off" in (await _send(sys_, state, "/arm banana"))[0]["text"]
    assert sys_.arming.get()["state"] == "armed_night"


async def test_strangers_get_no_setup_and_no_arming(home):
    sys_, state, path = home
    assert await _send(sys_, state, "/setup", chat=999) == []
    assert await _send(sys_, state, "/arm away", chat=999) == []
    assert sys_.arming.get()["source"] == "default"


async def test_setup_explains_a_service_without_a_config_file(system):
    sys_, state = system
    (reply,) = await _send(sys_, state, "/setup")
    assert "running without a config file" in reply["text"]


def test_the_alert_me_button_replaces_real():
    labels = [b["text"] for row in keyboard_markup("r", "full")["inline_keyboard"] for b in row]
    assert "🚨 Alert me" in labels and "🚨 Real" not in labels


async def test_the_diff_never_shows_a_secret_from_the_config(home):
    sys_, state, path = home
    text = path.read_text(encoding="utf-8").replace("mqtt:\n  enabled: false\n", "")
    path.write_text(text + "mqtt:\n  enabled: false\n  password: hunter2\n", encoding="utf-8")
    _, camera = await _send(sys_, state, "/setup")
    (_, who) = await _send(sys_, state, "the hallway")
    (always,) = await _tap(sys_, state, "su:who:done", who)
    (arming,) = await _tap(sys_, state, "su:alw:keep", always)
    (plan,) = await _tap(sys_, state, "su:arm:armed_away", arming)
    (diff,) = await _tap(sys_, state, "su:plan:diff", plan)
    assert "password: •••" in diff["text"] and "hunter2" not in diff["text"]


# ---------------------------------------------------------------- rooms: one card per incident (D-062)


def _in_garden(sys_) -> None:
    from watchwise.config import CameraSettings

    sys_.config.cameras.update({"cam_yard": CameraSettings(room="Garden"), "cam_drive": CameraSettings(room="garden")})


def _person(review_id: str, camera: str, after_s: float = 0.0, label: str = "person"):
    return make_alert(review_id=review_id, camera=camera, labels=[label], start_time=NIGHT_TS + after_s)


def _cards(state) -> list[dict]:
    return [m for m in state.sent if m["method"] in ("sendPhoto", "sendMessage") and m.get("reply_markup")
            and any(b.startswith("fb:") for b in state.buttons(m))]


async def test_two_cameras_in_one_room_share_one_card(system):
    sys_, state = system
    _in_garden(sys_)
    await sys_.pipeline.handle(_person("rev-a", "cam_yard"))
    await sys_.pipeline.handle(_person("rev-b", "cam_drive", after_s=5))
    await settle(sys_)
    assert len(_cards(state)) == 1, "the second camera's alert edits the room's card instead of buzzing again"
    card = state.edits[-1]
    assert card["message_id"] == state.sent[0]["message_id"]
    assert "Person · Garden (cam_yard + cam_drive)" in card["text"] and "Also cam_drive: 🚨" in card["text"]
    assert all(b.endswith(":rev-a") for b in state.buttons(card)), "the buttons are the card's own"
    assert sys_.store.card_members("rev-a", "telegram") == ["rev-b"]


async def test_other_rooms_and_later_events_get_their_own_cards(system):
    sys_, state = system
    _in_garden(sys_)
    await sys_.pipeline.handle(_person("rev-a", "cam_yard"))
    await sys_.pipeline.handle(_person("rev-b", "cam_door", after_s=5))  # no room: its own
    await sys_.pipeline.handle(_person("rev-c", "cam_drive", after_s=600))  # same room, 10 min later
    await settle(sys_)
    assert len(_cards(state)) == 3
    # A room card is joined by the room's *other* cameras; a camera's own next
    # event (here 10 s later, with a new label) is not merged into it.
    assert sys_.notifier._open_card_in_room(_person("rev-d", "cam_yard", after_s=10, label="car")) is None
    assert sys_.notifier._open_card_in_room(_person("rev-e", "cam_drive", after_s=10)) == "rev-a"


async def test_a_new_room_card_shows_what_the_other_camera_already_saw(system):
    sys_, state = system
    _in_garden(sys_)
    await sys_.pipeline.handle(_person("rev-cat", "cam_drive", label="cat"))  # routine: no card
    await sys_.notifier.drain()
    assert _cards(state) == []
    await sys_.pipeline.handle(_person("rev-a", "cam_yard", after_s=3))
    await settle(sys_)
    (card,) = _cards(state)
    text = (state.edits[-1] if state.edits else card)["text"]
    assert "🚨 Person, Cat · Garden (cam_yard + cam_drive)" in text, "the alert leads; the routine cat is listed"
    assert "Also cam_drive: ℹ️" in text and sys_.store.card_members("rev-a", "telegram") == ["rev-cat"]


async def test_a_tap_on_a_room_card_teaches_every_camera_and_undo_takes_all_back(system):
    sys_, state = system
    _in_garden(sys_)
    await sys_.pipeline.handle(_person("rev-a", "cam_yard"))
    await sys_.pipeline.handle(_person("rev-b", "cam_drive", after_s=5))
    await settle(sys_)
    corrected = lambda: sorted((r["review_id"], r["human_verdict"]) for r in sys_.store.query(  # noqa: E731
        "SELECT review_id, human_verdict FROM corrections"))

    await press(sys_, state, "fb:ok:rev-a")
    assert corrected() == [("rev-a", "benign"), ("rev-b", "benign")]
    assert "Also recorded for cam_drive." in state.edits[-1]["text"]
    (undo,) = [b for b in state.buttons(state.edits[-1]) if b.startswith("fb:undo:")]
    await press(sys_, state, undo)
    assert corrected() == []

    await press(sys_, state, "fb:real:rev-a")
    assert corrected() == [("rev-a", "suspicious"), ("rev-b", "suspicious")]
    asked = state.sent[-1]
    assert "What was wrong?" in asked["text"]
    state.push_message(CHAT, "climbing over the fence", reply_to=asked["message_id"])
    await sys_.bot.poll_once()
    notes = {r["review_id"]: r["note"] for r in sys_.store.query("SELECT review_id, note FROM corrections")}
    assert notes == {"rev-a": "climbing over the fence", "rev-b": "climbing over the fence"}

    await press(sys_, state, "fb:snooze:rev-a")
    assert sys_.store.snoozed_until("cam_yard") and sys_.store.snoozed_until("cam_drive")
    assert not sys_.store.snoozed_until("cam_door")
    await press(sys_, state, "fb:why:rev-a")
    why = state.sent[-1]["text"]
    assert "(cam_yard," in why and "(cam_drive," in why


async def test_setup_asks_the_room_of_each_camera_and_rooms_apply_at_once(home, nvr_state):
    sys_, state, path = home
    nvr_state.cameras = {"laptopcam": {"enabled": True}, "webcam": {"enabled": True}}
    _, laptop = await _send(sys_, state, "/setup")
    assert "Camera laptopcam</b> (1 of 2)" in laptop["text"]
    _, room = await _send(sys_, state, "the bed side of the bedroom")
    assert "Which room is <b>laptopcam</b> in?" in room["text"] and state.buttons(room) == ["su:room:0:own"]
    noted, webcam = await _send(sys_, state, "Bedroom")
    assert "laptopcam is in Bedroom" in noted["text"] and "Camera webcam</b> (2 of 2)" in webcam["text"]
    (room,) = await _tap(sys_, state, "su:cam:keep:1", webcam)
    assert state.buttons(room) == ["su:room:1:0", "su:room:1:own"] and "🏠 Bedroom" in str(room["reply_markup"])
    (who,) = await _tap(sys_, state, "su:room:1:0", room)
    (always,) = await _tap(sys_, state, "su:who:done", who)
    (arming,) = await _tap(sys_, state, "su:alw:keep", always)
    (plan,) = await _tap(sys_, state, "su:arm:armed_away", arming)
    assert "📷 laptopcam · room Bedroom: “the bed side of the bedroom” (new)" in plan["text"]
    assert "📷 webcam · room Bedroom: no notes (new)" in plan["text"]
    await _tap(sys_, state, "su:plan:apply", plan)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["cameras"]["webcam"] == {"room": "Bedroom"}

    sent = len(_cards(state))
    await sys_.pipeline.handle(_person("bed-1", "webcam"))
    await sys_.pipeline.handle(_person("bed-2", "laptopcam", after_s=2))
    await settle(sys_)
    assert len(_cards(state)) == sent + 1, "the running service uses the rooms at once"
    assert "Bedroom (webcam + laptopcam)" in state.edits[-1]["text"]


async def test_a_room_card_takes_new_events_for_five_minutes_only(system):
    """B-028: someone staying in the room kept one card open, and a new
    incident there only edited it (Telegram doesn't notify edits)."""
    sys_, state = system
    _in_garden(sys_)
    await sys_.pipeline.handle(_person("rev-a", "cam_yard"))
    staying = _person("rev-b", "cam_drive", after_s=5)
    staying.end_time = None  # still going
    await sys_.pipeline.handle(staying)
    await sys_.pipeline.handle(_person("rev-c", "cam_yard", after_s=400))
    await settle(sys_)
    assert len(_cards(state)) == 2 and sys_.store.card_members("rev-a", "telegram") == ["rev-b"]


async def test_an_event_after_the_owner_answered_the_card_gets_its_own(system):
    sys_, state = system
    _in_garden(sys_)
    await sys_.pipeline.handle(_person("rev-a", "cam_yard"))
    await settle(sys_)
    await press(sys_, state, "fb:ok:rev-a")
    await sys_.pipeline.handle(_person("rev-b", "cam_drive", after_s=10))
    await settle(sys_)
    assert len(_cards(state)) == 2 and sys_.store.card_members("rev-a", "telegram") == []


async def test_an_event_more_serious_than_its_room_card_gets_its_own(system):
    sys_, state = system
    _in_garden(sys_)
    await sys_.pipeline.handle(_person("rev-cat", "cam_drive", label="cat"))  # routine
    await sys_.pipeline.handle(_person("rev-a", "cam_yard", after_s=3))
    await settle(sys_)
    (card,) = _cards(state)
    assert sys_.store.card_members("rev-a", "telegram") == ["rev-cat"]

    # The card was later judged routine (an investigation, say); then the
    # other camera's event turns into an alert: that must buzz.
    downgraded = Decision(review_id="rev-a", disposition=Disposition.DOWNGRADED, severity=Severity.DETECTION,
                          confidence=0.9, rationale="routine", stage=Stage.INVESTIGATED)
    sys_.store.record_decision(downgraded.to_dict())
    alert = Decision(review_id="rev-cat", disposition=Disposition.ALERTED, severity=Severity.ALERT,
                     confidence=0.9, rationale="it was a person after all", stage=Stage.INVESTIGATED)
    sys_.store.record_decision(alert.to_dict())
    await sys_.notifier.process(_person("rev-cat", "cam_drive", label="cat"), alert)
    assert len(_cards(state)) == 2 and sys_.store.card_members("rev-a", "telegram") == []
    assert "Also cam_drive" not in state.edits[-1]["text"], "the old card no longer lists it"


async def test_a_photo_for_the_face_library_asks_which_camera_takes_it(system, nvr_state):
    """Any camera's photo enrolls a resident for every camera (one Face
    Library); the owner picks the camera they can face closely."""
    sys_, state = system
    state.push_button(CHAT, "su:face:alex", message_id=1)
    await sys_.bot.poll_once()
    asked = state.sent[-1]
    assert "Which camera should take the photo of alex?" in asked["text"]
    assert "enrolls you for every camera" in asked["text"] and "own angle" not in asked["text"]
    assert state.buttons(asked) == ["su:face:alex:0", "su:face:alex:1", "su:face:alex:2"]
    state.push_button(CHAT, "su:face:alex:1", message_id=asked["message_id"])
    await sys_.bot.poll_once()
    capture = state.sent[-1]
    assert capture["method"] == "sendPhoto" and "frame from cam_drive" in capture["text"]
    assert nvr_state.faces == {}, "nothing is added before ✅ Add"


# ---------------- faces from the camera that missed (D-071) ----------------


async def test_a_face_nobody_matched_can_be_taught_from_that_cards_own_crop(system, nvr_state):
    """One Face Library serves every camera; the camera that missed teaches it
    with the crop Frigate tried there. Nothing changes before ✅ Add."""
    sys_, state = system
    sys_.config.faces.household[:] = ["alex"]
    nvr_state.add_review(review_id="unk-1", camera="cam_door", start_time=NIGHT_TS, detections=["1793000000.1-abc"])
    nvr_state.face_attempts.update({
        "1793000000.1-abc-1793000001.0-unknown-0.42.webp": b"small",
        "1793000000.1-abc-1793000002.0-unknown-0.61.webp": b"the larger crop",
        "1793000099.9-zzz-1793000100.0-alex-0.95.webp": b"another event, much larger than both",
    })
    ctx = _person("unk-1", "cam_door")
    raised = Decision(review_id="unk-1", disposition=Disposition.ALERTED, severity=Severity.ALERT, confidence=0.0,
                      rationale="a face Frigate's face recognition could not match to anyone it knows", stage=Stage.FACE)
    sys_.store.upsert_alert({**ctx.to_dict(), "signature": ctx.signature()}, Stage.FACE.value)
    sys_.store.record_decision(raised.to_dict())
    await sys_.notifier.process(ctx, raised)
    card = _cards(state)[-1]
    assert "fc:who:alex:unk-1" in state.buttons(card)

    state.push_button(CHAT, "fc:who:alex:unk-1", message_id=card["message_id"])
    await sys_.bot.poll_once()
    preview = state.sent[-1]
    assert preview["method"] == "sendPhoto" and "Add this face from cam_door to alex" in preview["text"]
    assert "matched nobody (score 0.61)" in preview["text"], "the largest crop of that event"
    assert nvr_state.trained == [], "nothing changes before ✅ Add"

    state.push_button(CHAT, state.buttons(preview)[0], message_id=preview["message_id"])
    await sys_.bot.poll_once()
    assert nvr_state.trained == [("alex", "1793000000.1-abc-1793000002.0-unknown-0.61.webp")]
    assert "Added a face for alex" in state.sent[-1]["text"] and "every camera uses it" in state.sent[-1]["text"]

    status = (await _say(sys_, state, "/face"))["text"]
    assert "• alex: 1 photo(s); add 4+ more clear photos" in status


async def test_only_a_face_nobody_matched_offers_the_resident_buttons(system, nvr_state):
    sys_, state = system
    sys_.config.faces.household[:] = ["alex"]
    ctx = _person("plain-1", "cam_door")
    plain = Decision(review_id="plain-1", disposition=Disposition.ALERTED, severity=Severity.ALERT, confidence=0.9,
                     rationale="a person at the door at night", stage=Stage.SCREENED)
    await sys_.notifier.process(ctx, plain)
    assert not any(b.startswith("fc:who") for b in state.buttons(_cards(state)[-1]))

    nvr_state.add_review(review_id="gone-1", camera="cam_door", start_time=NIGHT_TS, detections=["1793000000.5-old"])
    state.push_button(CHAT, "fc:who:alex:gone-1", message_id=1)
    await sys_.bot.poll_once()
    assert "Frigate kept no face crop from this event" in state.sent[-1]["text"]


# ---------------------------------------------------------------- the daily digest (M2-7)


def _decided(store, review_id: str, start: float, *, stage: str, rationale: str, disposition: str = "downgraded",
             labels: tuple[str, ...] = ("person",), camera: str = "cam_yard", memory_ids: tuple[str, ...] = ()) -> None:
    store.upsert_alert({"review_id": review_id, "camera": camera, "start_time": start, "end_time": start + 5,
                        "signature": f"{camera}|{','.join(labels)}|nozone", "labels": list(labels), "zones": []}, stage)
    store.record_decision({"review_id": review_id, "disposition": disposition, "severity": "detection",
                           "confidence": 0.9, "rationale": rationale, "stage": stage, "screen": None,
                           "matched_memory_ids": list(memory_ids)})


def _a_day_held_back(store, patterns) -> str:
    """Eleven events of one night: one card, one muted alert, nine held back
    for five reasons. Returns the learned pattern's id."""
    for _ in range(3):
        pid = patterns.upsert(kind="false_alarm", signature="cam_drive|car|nozone",
                              description="headlights sweeping the gate", camera="cam_drive")
    learned = ("gate/memory_pattern: matches known false-alarm pattern 'headlights sweeping the gate' "
               "(similarity 0.97, 3 corrections, trust 0.75)")
    for i in range(3):
        _decided(store, f"face-{i}", NIGHT_TS + i * 60, stage="face",
                 rationale="recognized as Ray by Frigate's face recognition")
    for i in range(2):
        _decided(store, f"car-{i}", NIGHT_TS + 600 + i * 60, stage="gated", rationale=learned,
                 labels=("car",), camera="cam_drive", memory_ids=(pid,))
    _decided(store, "screen-0", NIGHT_TS + 900, stage="screened", rationale="tier-2 screening: a person sitting")
    _decided(store, "inv-0", NIGHT_TS + 960, stage="investigated", rationale="auto-downgraded: the resident's routine")
    _decided(store, "cat-0", NIGHT_TS + 1000, stage="gated", disposition="suppressed", labels=("cat",),
             rationale="gate/never_screen_label: label(s) ['cat'] are configured as never-screen")
    _decided(store, "dup-0", NIGHT_TS + 1100, stage="gated", rationale="gate/dedupe_window: duplicate of …")
    _decided(store, "alert-0", NIGHT_TS + 1200, stage="screened", disposition="alerted", rationale="a stranger")
    store.add_notification("alert-0", "telegram", "111", "9", has_image=True)
    _decided(store, "muted-0", NIGHT_TS + 1300, stage="screened", disposition="alerted", rationale="while muted")
    return pid


def test_the_digest_counts_what_was_held_back_and_why(store, patterns):
    from watchwise.notify.digest import build_digest, render_digest

    _a_day_held_back(store, patterns)
    window = (NIGHT_TS - 3600, NIGHT_TS + 7200)
    digest = build_digest(store, *window, spot_checks=2)

    assert (digest.events, digest.sent, digest.muted, digest.held_count) == (11, 1, 1, 9)
    groups = {g.label: len(g.review_ids) for g in digest.held}
    assert groups == {
        "recognized Ray": 3,
        "learned from your 3 corrections for “headlights sweeping the gate”": 2,
        "the vision model's first look found it ordinary": 1,
        "an investigation found it ordinary": 1,
        "never screened (cat)": 1,
        "a repeat of the same kind of event just before": 1,
    }
    # At most one per reason; what the owner taught first, then the largest group.
    assert digest.spot_checks[0] in ("car-0", "car-1") and digest.spot_checks[1].startswith("face-")
    assert build_digest(store, *window, spot_checks=2).spot_checks == digest.spot_checks, "the same window, the same picks"

    text = render_digest(digest)
    assert "11 events: 1 sent to you, 9 held back, 1 alerted while muted" in text
    assert "👤 recognized Ray · 3" in text
    assert "🧠 learned from your 3 corrections for “headlights sweeping the gate” · 2" in text
    assert "Spot-check 2 of the 9" in text
    assert len(text) < 4096


def test_spot_checks_skip_what_was_answered_or_checked_and_can_be_switched_off(store, patterns):
    from watchwise.notify.digest import build_digest, render_digest

    _a_day_held_back(store, patterns)
    window = (NIGHT_TS - 3600, NIGHT_TS + 7200)
    for review_id in ("car-0", "car-1", "face-0", "face-1"):
        store.execute("INSERT INTO corrections(review_id, signature, human_verdict, created_at) VALUES (?,?,?,?)",
                      (review_id, "s", "benign", NIGHT_TS + 5000))
    store.add_notification("face-2", "telegram", "111", "10", has_image=False, kind="spot_check")

    digest = build_digest(store, *window, spot_checks=5)
    assert digest.held_count == 9, "a spot-checked event is still one that was held back"
    assert "car-0" not in digest.spot_checks and not any(r.startswith("face-") for r in digest.spot_checks)
    assert len(digest.spot_checks) == len(set(digest.spot_checks)) == 4  # one per remaining reason
    assert digest.answers == {"benign": 4}
    assert build_digest(store, *window, spot_checks=0).spot_checks == []

    record = {"answered": 8, "right_to_hold_back": 7, "accuracy": 0.875}
    text = render_digest(build_digest(store, *window, spot_checks=2, record=record))
    assert "Your answers: 4 ✅ Expected, 0 🚨 Alert me." in text
    assert "Spot-checks so far: 7 of 8 were right to hold back." in text


class _Notifier:
    def __init__(self) -> None:
        self.sent: list = []

    async def send_digest(self, digest, chat_ids=None, *, silent=True):
        self.sent.append((digest, silent))
        return {"111": str(len(self.sent))}


async def test_the_daily_digest_goes_out_once_a_day_and_catches_up_after_a_restart(store, tmp_path):
    from watchwise.config import DigestSettings
    from watchwise.memory.store import Store
    from watchwise.notify.digest import SCHEDULED, DigestScheduler

    def at(day: int, hour: int, minute: int = 0) -> float:
        return time.mktime((2026, 3, day, hour, minute, 0, 0, 0, -1))

    now = {"t": at(10, 7)}
    notifier, settings = _Notifier(), DigestSettings(at="08:00")
    scheduler = DigestScheduler(store, None, notifier, settings, clock=lambda: now["t"])
    _decided(store, "e-1", at(10, 3), stage="screened", rationale="a cat")
    assert not scheduler.due()
    now["t"] = at(10, 8, 1)
    assert scheduler.due()
    digest = await scheduler.send(SCHEDULED)
    assert digest.events == 1 and len(notifier.sent) == 1 and notifier.sent[0][1] is True, "silent by default"
    assert not scheduler.due(), "once a day"
    assert 1.0 <= scheduler.pause() <= 300.0

    now["t"] = at(11, 8, 1)
    assert scheduler.due()
    quiet = await scheduler.send(SCHEDULED)
    assert quiet.empty and len(notifier.sent) == 1, "a day with nothing to say sends nothing"
    assert store.last_digest()["since"] == at(10, 8, 1), "each digest starts where the last one ended"
    assert not scheduler.due(), "but the window moved on"

    now["t"] = at(12, 10)  # stopped over the 08:00 of the 12th
    restarted = DigestScheduler(store, None, notifier, settings, clock=lambda: now["t"])
    assert restarted.due(), "two hours late: sent at start"
    now["t"] = at(12, 21)
    assert not restarted.due(), "thirteen hours late: wait for the next one"
    asked = await restarted.send()
    assert asked.empty and notifier.sent[-1][0] is asked, "/digest always answers, even with nothing new"

    fresh = Store(tmp_path / "fresh.db")
    now["t"] = at(10, 15)
    first = DigestScheduler(fresh, None, notifier, settings, clock=lambda: now["t"])
    assert not first.due(), "the very first one waits for the next 08:00"
    now["t"] = at(11, 8)
    assert first.due()
    first.settings = DigestSettings(enabled=False)
    assert not first.due(), "the schedule can be switched off"
    fresh.close()


async def test_spot_checks_are_cards_whose_answers_feed_the_accuracy_report(system):
    """M2-7 end to end: what the owner taught holds a cat back; /digest sends
    it back with another held-back event as cards; their answers are
    corrections, also after a restart (B-036), and count as spot-checks."""
    from watchwise.notify.digest import build_digest

    sys_, state = system
    for i in range(3):  # three cats on the lawn, each answered "expected"
        await sys_.pipeline.handle(make_alert(review_id=f"cat-{i}", camera="cam_yard", labels=["cat"],
                                              zones=["lawn"], start_time=DAY_TS + i * 3600))
        sys_.feedback.submit(f"cat-{i}", "benign", note="the neighbour's cat on the lawn",
                             influenced_pattern_ids=sys_.bot.handler.influenced(f"cat-{i}"))
    held = await sys_.pipeline.handle(make_alert(review_id="cat-3", camera="cam_yard", labels=["cat"],
                                                 zones=["lawn"], start_time=DAY_TS + 3 * 3600))
    assert held.stage is Stage.GATED and sys_.store.memory_ids("cat-3") == held.matched_memory_ids
    await sys_.pipeline.handle(make_alert(review_id="dog-0", camera="cam_drive", labels=["dog"],
                                          start_time=NIGHT_TS + 3600))
    await sys_.pipeline.handle(night_person("rev-p"))
    await settle(sys_)
    cards_before = len(state.sent)

    sys_.bot.digests.clock = lambda: DAY_TS + 6 * 3600
    state.push_message(CHAT, "/digest")
    await sys_.bot.poll_once()
    digest_msg, *checks = state.sent[cards_before:]
    assert digest_msg["silent"] and "6 events: 1 sent to you, 5 held back" in digest_msg["text"]
    assert "🧠 learned from your 3 corrections for “the neighbour's cat on the lawn” · 1" in digest_msg["text"]
    assert [c["reply_to"] for c in checks] == [digest_msg["message_id"]] * 2
    assert all(c["silent"] and c["method"] == "sendPhoto" for c in checks)
    first, second = checks
    assert first["text"].startswith("<b>🔍 Spot-check · Cat · cam_yard · lawn · Tue 17:00</b>")
    assert state.buttons(first) == ["fb:ok:cat-3", "fb:real:cat-3", "fb:why:cat-3"]
    assert "fb:ok:dog-0" in state.buttons(second)

    sys_.memory_sink.decisions.clear()  # as after a restart: only the store remembers (B-036)
    (pattern_id,) = held.matched_memory_ids
    state.push_button(CHAT, "fb:real:cat-3", message_id=first["message_id"])
    await sys_.bot.poll_once()
    assert sys_.patterns.get(pattern_id)["disagreements"] == 1, "the pattern that held it back lost trust"
    assert state.answers[-1]["text"] == "Noted: that should have alerted."
    assert "🔍 Spot-check: this should have alerted." in state.edits[-1]["text"]
    assert "What was wrong?" in state.sent[-1]["text"]

    state.push_button(CHAT, "fb:ok:dog-0", message_id=second["message_id"])
    await sys_.bot.poll_once()
    assert "🔍 Spot-check: right to hold back." in state.edits[-1]["text"]
    assert sys_.feedback.accuracy_report()["spot_checks"] == {"answered": 2, "right_to_hold_back": 1, "accuracy": 0.5}

    undo = state.buttons(state.edits[-1])[0]
    state.push_button(CHAT, undo, message_id=second["message_id"])
    await sys_.bot.poll_once()
    assert state.buttons(state.edits[-1]) == ["fb:ok:dog-0", "fb:real:dog-0", "fb:why:dog-0"], "a spot-check again"
    assert sys_.feedback.spot_checks()["answered"] == 1

    again = build_digest(sys_.store, DAY_TS - 18 * 3600, DAY_TS + 6 * 3600, spot_checks=2)
    assert again.spot_checks == [], "an event is never spot-checked twice"


# ---------------------------------------------------------------- the trace record (M2-10)


async def test_every_decision_card_and_owner_action_is_kept_in_order(system):
    """What `watchwise export` replays: each decision (the alerts table keeps
    only the last), each card as the phone showed it, each tap and note."""
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await settle(sys_)
    decisions = [json.loads(r["decision"]) for r in sys_.store.query(
        "SELECT decision FROM decision_log WHERE review_id='rev-a' ORDER BY id")]
    assert [d["stage"] for d in decisions] == ["screened", "investigated"]

    await press(sys_, state, "fb:real:rev-a")
    question = state.sent[-1]
    state.push_message(CHAT, "climbing the fence", reply_to=question["message_id"])
    await sys_.bot.poll_once()
    state.push_message(CHAT, "/arm night")
    await sys_.bot.poll_once()

    cards = sys_.store.query("SELECT * FROM card_log ORDER BY id")
    assert [(c["kind"], c["action"]) for c in cards][:4] == [
        ("alert", "sent"), ("alert", "edited"), ("alert", "edited"), ("note_request", "sent")]
    assert cards[0]["text"] == state.sent[0]["text"] and cards[0]["has_image"] == 1
    assert cards[1]["text"] == state.edits[0]["text"] and "after an investigation" in cards[1]["text"]
    assert json.loads(cards[2]["buttons"])["inline_keyboard"][0][0]["text"] == "↩️ Undo"

    owner = sys_.store.query("SELECT * FROM owner_log ORDER BY id")
    assert [(o["kind"], o["data"]) for o in owner] == [
        ("button", "fb:real:rev-a"), ("note", "climbing the fence"), ("command", "/arm night")]
    tapped = json.loads(owner[0]["outcome"])
    assert owner[0]["review_id"] == "rev-a" and tapped["toast"] == "Noted: you want alerts like this."
    assert tapped["corrections"] and tapped["spot_check"] is False
    assert json.loads(owner[1]["outcome"])["replied_to"] == question["message_id"]
    (armed,) = sys_.store.query("SELECT state, source FROM arming_log")
    assert armed["state"] == "armed_night" and armed["source"].startswith("telegram:")


def test_the_trace_record_is_kept_for_trace_days_only(tmp_path):
    from watchwise.memory.store import Store

    path = tmp_path / "w.db"
    store = Store(path)
    store.log_arming("armed_away", "api", at=time.time() - 20 * 86400)
    store.log_arming("armed_home", "api")
    store.close()
    kept = Store(path, log_days=14)
    assert [r["state"] for r in kept.query("SELECT state FROM arming_log")] == ["armed_home"]
    kept.close()


def test_a_digest_with_nothing_new_still_shows_the_spot_check_record(store):
    from watchwise.notify.digest import build_digest, render_digest

    record = {"answered": 2, "right_to_hold_back": 1, "accuracy": 0.5}
    text = render_digest(build_digest(store, NIGHT_TS, NIGHT_TS + 60, record=record))
    assert "Nothing happened on your cameras" in text and "Spot-checks so far: 1 of 2" in text
    assert "Spot-checks so far" not in render_digest(build_digest(store, NIGHT_TS, NIGHT_TS + 60))


async def test_the_learned_footer_says_what_memory_really_does_here(system):
    """B-037: live, with gate.memory_mode model_checked, 36 cards said
    "Watchwise can now hold these back by itself", which only the free mode
    does, and never for a person while armed away or at night."""
    sys_, state = system
    await sys_.pipeline.handle(night_person())
    await settle(sys_)
    await press(sys_, state, "fb:ok:rev-a")
    footer = state.edits[-1]["text"]
    assert "Marked expected (1 of 3)" in footer and "(a person still alerts while armed away or at night)" in footer

    sys_.config.gate.memory_mode = "model_checked"
    await press(sys_, state, [b for b in state.buttons(state.edits[-1]) if b.startswith("fb:undo")][0])
    await press(sys_, state, "fb:ok:rev-a")
    footer = state.edits[-1]["text"]
    assert "by itself" not in footer
    assert "The vision model is now shown your judgements of person on cam_yard as examples" in footer


def test_a_pattern_watchwise_named_itself_reads_as_what_and_where(store, patterns):
    """Live 2026-10-30: the digest quoted 'recurring false alarm on webcam for ["person"]'."""
    from watchwise.notify.digest import build_digest

    pid = patterns.upsert(kind="false_alarm", signature="webcam|person|nozone", camera="webcam",
                          description='recurring false alarm on webcam for ["person"]')
    _decided(store, "s-1", NIGHT_TS, stage="screened", rationale="tier-2 screening: a person at the desk",
             camera="webcam", memory_ids=(pid,))
    (group,) = build_digest(store, NIGHT_TS - 60, NIGHT_TS + 60).held
    assert group.label == "the vision model, shown your 1 correction for person on webcam, found it ordinary"
