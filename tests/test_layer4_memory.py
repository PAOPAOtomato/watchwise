"""Layer 4: short-term and long-term memory, and the correction loop.

The long-term tests are the ones that matter for the central claim --
that the system learns from human review and downgrades repeat false alarms.
They check that learning happens, and equally that it *stops* happening when
humans start disagreeing.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from watchwise.memory.feedback import FeedbackLoop
from watchwise.memory.short_term import ShortTermMemory
from watchwise.memory.vectors import (
    HashEmbedder,
    NVRCLIPEmbedder,
    build_embedder,
    cosine_search,
    decay_factor,
)
from watchwise.types import ScreenResult, Verdict
from tests.conftest import make_alert

DAY = 86400.0


# ---------------- embeddings & vector search ----------------


def test_hash_embedder_is_deterministic_across_instances():
    a = HashEmbedder(128).embed("person at the side gate")
    b = HashEmbedder(128).embed("person at the side gate")
    assert np.allclose(a, b)
    assert a.shape == (128,)
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)


def test_hash_embedder_ranks_related_text_above_unrelated():
    e = HashEmbedder(256)
    query = e.embed("person at the side gate at night")
    related = e.embed("person near the side gate after dark")
    unrelated = e.embed("delivery van parked on the main road")
    assert float(query @ related) > float(query @ unrelated)


def test_nvr_clip_embedder_refuses_rather_than_faking_a_vector():
    with pytest.raises(NotImplementedError):
        NVRCLIPEmbedder(256).embed("anything")


def test_build_embedder_rejects_unknown_backends():
    with pytest.raises(ValueError):
        build_embedder("magic", 64)


def test_cosine_search_orders_by_similarity():
    matrix = np.eye(4, dtype=np.float32)
    hits = cosine_search(matrix[2], matrix, top_k=2)
    assert hits[0][0] == 2 and hits[0][1] == pytest.approx(1.0)


def test_cosine_search_handles_an_empty_index():
    assert cosine_search(np.zeros(4, dtype=np.float32), np.zeros((0, 0)), 3) == []


def test_decay_halves_at_the_half_life():
    assert decay_factor(30.0, 30.0) == pytest.approx(0.5)
    assert decay_factor(0.0, 30.0) == 1.0


# ---------------- long-term pattern library ----------------


def test_upsert_reinforces_rather_than_duplicating(patterns):
    sig = "cam_drive|car|driveway"
    first = patterns.upsert(kind="false_alarm", signature=sig, description="headlights")
    second = patterns.upsert(kind="false_alarm", signature=sig, description="headlights again")
    assert first == second
    row = patterns.get(first)
    assert row["corrections"] == 2
    assert "headlights again" in row["description"]


def test_descriptions_are_merged_and_bounded(patterns):
    sig = "cam|x|y"
    for i in range(8):
        patterns.upsert(kind="false_alarm", signature=sig, description=f"note {i}")
    description = patterns.get(patterns.upsert(kind="false_alarm", signature=sig,
                                               description="note 8"))["description"]
    assert len(description.split(" | ")) <= 5


def test_exact_signature_match_outranks_wording_similarity(patterns):
    patterns.upsert(kind="false_alarm", signature="cam_a|car|drive", description="alpha")
    patterns.upsert(kind="false_alarm", signature="cam_b|car|drive", description="alpha")
    hits = patterns.search("car drive", signature="cam_b|car|drive", top_k=2)
    assert hits[0].signature == "cam_b|car|drive"
    assert hits[0].similarity >= 0.95


def test_trust_is_laplace_smoothed(patterns):
    pid = patterns.upsert(kind="false_alarm", signature="s", description="d")
    patterns.record_outcome(pid, agreed=True)
    one_of_one = patterns.search("d", signature="s")[0].trust
    for _ in range(8):
        patterns.record_outcome(pid, agreed=True)
    nine_of_nine = patterns.search("d", signature="s")[0].trust
    assert one_of_one < nine_of_nine < 1.0


def test_stale_patterns_score_lower_than_fresh_ones(store, patterns):
    fresh = patterns.upsert(kind="false_alarm", signature="cam|a|z", description="fresh thing")
    stale = patterns.upsert(kind="false_alarm", signature="cam|b|z", description="stale thing")
    store.execute(
        "UPDATE patterns SET last_seen=? WHERE id=?", (time.time() - 120 * DAY, stale)
    )
    hits = {h.pattern_id: h.score for h in patterns.search("thing", top_k=5)}
    assert hits[fresh] > hits[stale]


def test_suppression_match_requires_enough_corrections(patterns):
    sig = "cam_drive|car|driveway"
    patterns.upsert(kind="false_alarm", signature=sig, description="headlights")
    assert patterns.best_suppression_match(
        sig, "car driveway", min_similarity=0.5, min_corrections=3
    ) is None
    patterns.upsert(kind="false_alarm", signature=sig, description="headlights")
    patterns.upsert(kind="false_alarm", signature=sig, description="headlights")
    assert patterns.best_suppression_match(
        sig, "car driveway", min_similarity=0.5, min_corrections=3
    ) is not None


def test_suppression_never_matches_a_true_alarm_pattern(patterns):
    sig = "cam_yard|person|perimeter"
    for _ in range(5):
        patterns.upsert(kind="true_alarm", signature=sig, description="real intruder")
    assert patterns.best_suppression_match(
        sig, "person perimeter", min_similarity=0.5, min_corrections=1
    ) is None


def test_delete_removes_the_pattern_and_its_vector(store, patterns):
    pid = patterns.upsert(kind="false_alarm", signature="s", description="d")
    patterns.delete(pid)
    assert patterns.get(pid) is None
    ids, _ = store.load_vectors(PatternKind := "pattern")
    assert pid not in ids


# ---------------- short-term memory ----------------


def test_seed_pins_the_case_facts():
    memory = ShortTermMemory(max_items=5, max_chars=100_000)
    screen = ScreenResult(verdict=Verdict.UNCERTAIN, confidence=0.4, reason="unclear",
                          backend="stub", model="m")
    memory.seed(make_alert(review_id="rev-x"), screen)
    assert len(memory.items) == 2
    assert all(item.pinned for item in memory.items)
    assert "rev-x" in memory.transcript()


def test_item_budget_evicts_oldest_unpinned_only():
    memory = ShortTermMemory(max_items=4, max_chars=100_000)
    memory.seed(make_alert())
    for i in range(10):
        memory.add("observation", f"obs {i}", iteration=i)
    assert len(memory.items) == 4
    assert memory.items[0].pinned
    assert "obs 9" in memory.transcript()
    assert "obs 0" not in memory.transcript()


def test_char_budget_is_enforced_independently():
    memory = ShortTermMemory(max_items=1000, max_chars=300)
    memory.seed(make_alert())
    for i in range(50):
        memory.add("observation", "x" * 100, iteration=i)
    assert memory._char_count() <= 300 or all(i.pinned for i in memory.items)


def test_pinned_items_survive_even_a_tiny_budget():
    memory = ShortTermMemory(max_items=1, max_chars=10)
    memory.seed(make_alert())
    memory.add("observation", "y" * 500)
    assert any(item.pinned for item in memory.items)


def test_briefing_carries_facts_questions_and_confidence():
    memory = ShortTermMemory()
    memory.seed(make_alert())
    memory.note_fact("cameras_with_activity", 3)
    memory.add_question("Where did they go?")
    memory.set_confidence(0.42)
    briefing = memory.briefing()
    assert "cameras_with_activity" in briefing
    assert "Where did they go?" in briefing
    assert "0.42" in briefing


def test_resolving_a_question_removes_it():
    memory = ShortTermMemory()
    memory.add_question("q1")
    memory.add_question("q1")  # deduplicated
    assert memory.open_questions == ["q1"]
    memory.resolve_question("q1")
    assert memory.open_questions == []


def test_tools_used_is_derived_from_observations():
    memory = ShortTermMemory()
    memory.add_observation("get_device_status", {"ok": True}, 1)
    memory.add_observation("cross_reference_cameras", {"ok": True}, 2)
    memory.add_observation("get_device_status", {"ok": True}, 3)
    assert memory.tools_used() == ["get_device_status", "cross_reference_cameras"]


# ---------------- the correction loop ----------------


def _seed_alert(store, review_id="rev-1", disposition="downgraded", camera="cam_drive"):
    alert = make_alert(review_id=review_id, camera=camera, labels=["car"], zones=["driveway"])
    record = alert.to_dict()
    record["signature"] = alert.signature()
    store.upsert_alert(record, "screened")
    store.record_decision(
        {
            "review_id": review_id,
            "disposition": disposition,
            "severity": "detection",
            "confidence": 0.9,
            "rationale": "r",
            "screen": None,
            "investigation_id": None,
            "stage": "screened",
        }
    )
    return alert


def test_a_benign_correction_creates_a_false_alarm_pattern(store, patterns):
    loop = FeedbackLoop(store, patterns)
    alert = _seed_alert(store)
    result = loop.submit("rev-1", "benign", note="headlights again", reviewer="sam")
    assert result.agreed is True  # system had downgraded it
    assert result.pattern_id is not None
    assert patterns.get(result.pattern_id)["kind"] == "false_alarm"
    assert patterns.get(result.pattern_id)["signature"] == alert.signature()


def test_a_suspicious_correction_creates_a_true_alarm_pattern(store, patterns):
    loop = FeedbackLoop(store, patterns)
    _seed_alert(store, disposition="downgraded")
    result = loop.submit("rev-1", "suspicious", note="that was a real prowler")
    assert result.agreed is False, "system said benign, human said suspicious"
    assert patterns.get(result.pattern_id)["kind"] == "true_alarm"


def test_corrections_score_the_patterns_that_caused_the_decision(store, patterns):
    """The closing of the loop: a pattern that misled us loses trust."""
    loop = FeedbackLoop(store, patterns)
    _seed_alert(store, disposition="suppressed")
    pid = patterns.upsert(kind="false_alarm", signature="cam_drive|car|driveway",
                          description="headlights")
    loop.submit("rev-1", "suspicious", influenced_pattern_ids=[pid])
    row = patterns.get(pid)
    assert row["disagreements"] == 1 and row["agreements"] == 0


def test_undo_removes_a_pattern_that_one_correction_created(store, patterns):
    loop = FeedbackLoop(store, patterns)
    _seed_alert(store)
    result = loop.submit("rev-1", "benign", note="headlights")
    undone = loop.undo(result.correction_id)
    assert (undone.review_id, undone.human_verdict, undone.effects_reversed) == ("rev-1", "benign", 1)
    assert patterns.get(result.pattern_id) is None
    assert store.query("SELECT COUNT(*) AS n FROM corrections")[0]["n"] == 0
    assert store.query("SELECT COUNT(*) AS n FROM correction_effects")[0]["n"] == 0


def test_undo_takes_back_exactly_one_reinforcement_and_its_scores(store, patterns):
    loop = FeedbackLoop(store, patterns)
    _seed_alert(store, review_id="rev-1")
    first = loop.submit("rev-1", "benign", note="headlights")
    _seed_alert(store, review_id="rev-2")
    second = loop.submit("rev-2", "benign", influenced_pattern_ids=[first.pattern_id])
    row = patterns.get(first.pattern_id)
    assert (row["corrections"], row["agreements"]) == (2, 1)

    loop.undo(second.correction_id)
    row = patterns.get(first.pattern_id)
    assert (row["corrections"], row["agreements"]) == (1, 0), "the first correction still stands"


def test_undo_of_an_unknown_correction_is_an_error(store, patterns):
    with pytest.raises(KeyError):
        FeedbackLoop(store, patterns).undo(12345)


def test_repeated_agreement_raises_trust_and_enables_suppression(store, patterns):
    loop = FeedbackLoop(store, patterns)
    pid = None
    for i in range(4):
        _seed_alert(store, review_id=f"rev-{i}", disposition="downgraded")
        result = loop.submit(f"rev-{i}", "benign", note="headlights",
                             influenced_pattern_ids=[pid] if pid else [])
        pid = result.pattern_id
    match = patterns.best_suppression_match(
        "cam_drive|car|driveway", "car driveway", min_similarity=0.5, min_corrections=3
    )
    assert match is not None
    assert match.corrections >= 3
    assert match.trust > 0.5


def test_unknown_review_id_is_rejected(store, patterns):
    with pytest.raises(KeyError):
        FeedbackLoop(store, patterns).submit("nope", "benign")


def test_invalid_verdict_is_rejected(store, patterns):
    _seed_alert(store)
    with pytest.raises(ValueError):
        FeedbackLoop(store, patterns).submit("rev-1", "maybe")


def test_accuracy_is_null_until_auto_decisions_are_reviewed(store, patterns):
    loop = FeedbackLoop(store, patterns)
    report = loop.accuracy_report()
    assert report["auto_downgrade_accuracy"] is None
    assert "null until such reviews exist" in report["note"]


def test_auto_downgrade_accuracy_counts_only_auto_resolved_alerts(store, patterns):
    loop = FeedbackLoop(store, patterns)
    # Three auto-suppressed alerts; a human agrees with two.
    for i, verdict in enumerate(["benign", "benign", "suspicious"]):
        _seed_alert(store, review_id=f"a{i}", disposition="suppressed")
        loop.submit(f"a{i}", verdict)
    # One alert that was surfaced, so it must not count toward the metric.
    _seed_alert(store, review_id="b0", disposition="alerted")
    loop.submit("b0", "suspicious")

    report = loop.accuracy_report()
    assert report["auto_downgraded_reviewed"] == 3
    assert report["auto_downgrade_accuracy"] == pytest.approx(2 / 3, abs=1e-4)


def test_review_load_measures_what_reached_a_human(store, patterns):
    loop = FeedbackLoop(store, patterns)
    for i in range(7):
        _seed_alert(store, review_id=f"s{i}", disposition="suppressed")
    for i in range(3):
        _seed_alert(store, review_id=f"h{i}", disposition="alerted")
    load = loop.review_load()
    assert load["alerts_ingested"] == 10
    assert load["surfaced_to_human"] == 3
    assert load["review_load_ratio"] == pytest.approx(0.3)


# ---------------- persistence ----------------


def test_patterns_and_vectors_survive_a_restart(tmp_path):
    from watchwise.memory.long_term import PatternLibrary
    from watchwise.memory.store import Store

    db = tmp_path / "s.db"
    store = Store(db)
    lib = PatternLibrary(store, HashEmbedder(128))
    pid = lib.upsert(kind="false_alarm", signature="cam|car|drive", description="headlights")
    store.close()

    store2 = Store(db)
    lib2 = PatternLibrary(store2, HashEmbedder(128))
    hits = lib2.search("headlights", signature="cam|car|drive")
    assert hits and hits[0].pattern_id == pid
    store2.close()


def test_an_existing_database_gains_the_summary_column(tmp_path):
    """Databases created before summaries existed are migrated in place."""
    import sqlite3

    from watchwise.memory.store import Store

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE alerts (review_id TEXT PRIMARY KEY, camera TEXT NOT NULL, start_time REAL NOT NULL, "
        "end_time REAL, signature TEXT NOT NULL, labels TEXT NOT NULL DEFAULT '[]', zones TEXT NOT NULL "
        "DEFAULT '[]', stage TEXT NOT NULL, disposition TEXT, severity TEXT, confidence REAL, rationale TEXT, "
        "screen_json TEXT, investigation_id TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    old.execute("INSERT INTO alerts VALUES ('r', 'c', 1, 2, 'c|person|nozone', '[]', '[]', 'screened', "
                "'alerted', 'alert', 0.5, 'why', NULL, NULL, 1, 1)")
    old.commit()
    old.close()

    store = Store(path)
    decision = {"review_id": "r", "disposition": "alerted", "severity": "alert", "confidence": 0.5,
                "rationale": "why", "stage": "screened", "summary": "A person walks past."}
    store.record_decision(decision)
    assert store.get_alert("r")["summary"] == "A person walks past."
    store.record_decision({**decision, "summary": None, "stage": "investigated"})
    assert store.get_alert("r")["summary"] == "A person walks past.", "a later decision keeps it"
    # B-036: the learned patterns behind a decision are kept with it.
    assert store.memory_ids("r") == []
    store.record_decision({**decision, "matched_memory_ids": ["fp-1"]})
    assert store.memory_ids("r") == ["fp-1"]
    store.close()


def test_an_existing_database_gains_card_kinds_and_the_digest_table(tmp_path):
    """M2-7: cards sent before spot-checks existed are alert cards."""
    import sqlite3

    from watchwise.memory.store import Store

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE notifications (review_id TEXT NOT NULL, channel TEXT NOT NULL, chat_id TEXT NOT NULL, "
        "message_id TEXT NOT NULL, has_image INTEGER NOT NULL DEFAULT 0, footer TEXT NOT NULL DEFAULT '', "
        "keyboard TEXT NOT NULL DEFAULT 'full', created_at REAL NOT NULL, PRIMARY KEY (review_id, channel, chat_id))"
    )
    old.execute("INSERT INTO notifications VALUES ('r', 'telegram', '1', '5', 1, '', 'full', 1)")
    old.commit()
    old.close()

    store = Store(path)
    assert store.notifications_for("r", "telegram")[0]["kind"] == "alert"
    store.add_digest("asked", 1.0, 2.0, {"1": "6"}, ["r"])
    assert store.last_digest()["spot_checks"] == '["r"]'
    store.close()


def test_correction_examples_describe_what_was_seen(store):
    """D-051: the owner's judgements as descriptions, freshest first; a later
    whole-event summary beats the first look's reason."""
    import json as _json
    import time as _time

    for i, (verdict, summary, reason) in enumerate([
        ("benign", None, "a person seated at a desk"),
        ("suspicious", "A man gestures wildly at the camera.", "a person at a desk"),
        ("benign", "A man reads at his desk, then leaves.", "a person"),
    ]):
        rid = f"ex-{i}"
        store.upsert_alert({"review_id": rid, "camera": "webcam", "start_time": 1.0 + i,
                            "signature": "webcam|person|nozone", "labels": ["person"], "zones": []}, "screened")
        store.record_decision({"review_id": rid, "disposition": "alerted", "severity": "alert", "confidence": 0.9,
                               "rationale": "x", "stage": "screened", "summary": summary,
                               "screen": {"verdict": "benign", "confidence": 0.9, "reason": reason}})
        store.execute("INSERT INTO corrections(review_id, signature, system_verdict, system_conf, human_verdict, "
                      "note, reviewer, created_at) VALUES (?,?,?,?,?,?,?,?)",
                      (rid, "webcam|person|nozone", "benign", 0.9, verdict, "", "t", _time.time() + i))
    assert store.correction_examples("webcam|person|nozone") == [
        ("benign", "A man reads at his desk, then leaves."),
        ("suspicious", "A man gestures wildly at the camera."),
        ("benign", "a person seated at a desk"),
    ]
    assert store.correction_examples("other|sig") == []


def test_the_owners_words_beat_the_models_and_a_contrary_reason_is_dropped(store):
    """D-053: live, the only Real example was the first look's own reason
    ("...consistent with casual behavior"), which argued the opposite."""
    import time as _time

    sig = "webcam|person|nozone"
    for i, (verdict, note, screen_verdict, reason) in enumerate([
        ("suspicious", "", "benign", "an expressive gesture, consistent with casual behavior"),
        ("suspicious", "", "uncertain", "a person moving quickly"),
        ("suspicious", "waving both arms at the camera", "benign", "a person at a desk"),
    ]):
        rid = f"nt-{i}"
        store.upsert_alert({"review_id": rid, "camera": "webcam", "start_time": 1.0 + i,
                            "signature": sig, "labels": ["person"], "zones": []}, "screened")
        store.record_decision({"review_id": rid, "disposition": "downgraded", "severity": "detection",
                               "confidence": 0.9, "rationale": "x", "stage": "screened",
                               "screen": {"verdict": screen_verdict, "confidence": 0.9, "reason": reason}})
        store.execute("INSERT INTO corrections(review_id, signature, system_verdict, system_conf, human_verdict, "
                      "note, reviewer, created_at) VALUES (?,?,?,?,?,?,?,?)",
                      (rid, sig, screen_verdict, 0.9, verdict, note, "t", _time.time() + i))
    assert store.correction_examples(sig) == [
        ("suspicious", "waving both arms at the camera"),
        ("suspicious", "a person moving quickly"),
    ]


def test_a_note_request_links_a_reply_to_its_correction(store):
    store.upsert_alert({"review_id": "nr-1", "camera": "webcam", "start_time": 1.0,
                        "signature": "webcam|person|nozone", "labels": ["person"], "zones": []}, "screened")
    cid = int(store.execute("INSERT INTO corrections(review_id, signature, human_verdict, created_at) "
                            "VALUES ('nr-1', 'webcam|person|nozone', 'suspicious', 1.0)").lastrowid)
    store.add_note_request(555, 900, [cid])
    assert store.note_request(555, 900) == [cid]
    assert store.note_request(555, 901) == [] and store.note_request(556, 900) == []
    store.add_note_request(555, 902, [cid, 41, 42])  # a room card: one per camera (D-062)
    assert store.note_request(555, 902) == [cid, 41, 42]
    assert store.set_correction_note(cid, "waving both arms")
    store.execute("DELETE FROM corrections WHERE id=?", (cid,))
    assert not store.set_correction_note(cid, "too late"), "an undone correction takes no note"
