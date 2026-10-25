"""Layer 10: alerts that reach a person, and buttons that teach the memory.

Driven over real HTTP against a fake Telegram Bot API (``fakes/telegram_server``):
an alert becomes a card with a snapshot and buttons; the card is edited in
place when the investigation finishes; a tap on "Expected" becomes a human
correction in long-term memory, and "Undo" takes it back exactly. Also: the
allow-list, snooze, "Why", "Look closer", Telegram's error envelopes, MQTT
verdict publishing (B-001), and the `watchwise telegram` commands.
"""

from __future__ import annotations

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
    assert "Marked real" in state.edits[-1]["text"]


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
    assert "later version" in state.sent[-1]["text"]
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
    assert "Added a face for alex" in state.sent[-1]["text"] and "faces.household" in state.sent[-1]["text"]

    state.push_button(CHAT, add, message_id=asked["message_id"])  # a second tap
    await sys_.bot.poll_once()
    assert "expired or was already used" in state.sent[-1]["text"] and nvr_state.faces == {"alex": 1}


async def test_face_explains_what_it_needs(system, nvr_state):
    sys_, state = system
    assert "Which camera?" in (await _say(sys_, state, "/face alex"))["text"]
    assert "simple name" in (await _say(sys_, state, "/face al-ex! cam_door"))["text"]
    assert "Send /face &lt;name&gt;" in (await _say(sys_, state, "/face"))["text"]

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
