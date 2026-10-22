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


def test_telegram_test_sends_a_sample_card(linked_cli, capsys):
    state, path = linked_cli
    state.push_message(555, "/start")
    assert cli.main(["telegram", "link", "--config", str(path), "--yes", "--wait", "2"]) == 0
    assert cli.main(["telegram", "test", "--config", str(path)]) == 0
    card = state.sent[-1]
    assert card["method"] == "sendPhoto" and card["chat_id"] == 555
    assert "test card" in card["text"]


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
