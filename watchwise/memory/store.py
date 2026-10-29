"""SQLite persistence for watchwise's own state.

Deliberately a *separate* database from NVR's. We never write to nvr.db --
NVR owns it, migrates it, and vacuums it. Our tables reference NVR rows by id
only, so an NVR upgrade cannot corrupt our memory and our memory cannot bloat
their DB.

Tables
  alerts          one row per screened alert + final decision
  investigations  one row per tier-3 run, with the serialized ReAct trace
  corrections     human review verdicts (the "correct" half of the loop)
  patterns        long-term misjudgment pattern library (the "feed back" half)
  vectors         embedding blobs keyed by (kind, ref_id)
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS alerts (
    review_id       TEXT PRIMARY KEY,
    camera          TEXT NOT NULL,
    start_time      REAL NOT NULL,
    end_time        REAL,
    signature       TEXT NOT NULL,
    labels          TEXT NOT NULL DEFAULT '[]',
    zones           TEXT NOT NULL DEFAULT '[]',
    stage           TEXT NOT NULL,
    disposition     TEXT,
    severity        TEXT,
    confidence      REAL,
    rationale       TEXT,
    screen_json     TEXT,
    investigation_id TEXT,
    summary         TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_sig ON alerts(signature, start_time);
CREATE INDEX IF NOT EXISTS idx_alerts_camera_time ON alerts(camera, start_time);

CREATE TABLE IF NOT EXISTS investigations (
    id              TEXT PRIMARY KEY,
    review_id       TEXT NOT NULL,
    camera          TEXT NOT NULL,
    status          TEXT NOT NULL,
    verdict         TEXT,
    confidence      REAL,
    conclusion      TEXT,
    iterations      INTEGER NOT NULL DEFAULT 0,
    backend         TEXT,
    model           TEXT,
    cost_usd        REAL NOT NULL DEFAULT 0,
    trace_json      TEXT NOT NULL DEFAULT '{}',
    started_at      REAL,
    finished_at     REAL,
    error           TEXT
);
CREATE INDEX IF NOT EXISTS idx_inv_review ON investigations(review_id);

CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id       TEXT NOT NULL,
    signature       TEXT NOT NULL,
    system_verdict  TEXT,
    system_conf     REAL,
    human_verdict   TEXT NOT NULL,
    note            TEXT,
    reviewer        TEXT,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corr_sig ON corrections(signature);

CREATE TABLE IF NOT EXISTS patterns (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    signature       TEXT NOT NULL,
    camera          TEXT,
    description     TEXT NOT NULL,
    corrections     INTEGER NOT NULL DEFAULT 0,
    agreements      INTEGER NOT NULL DEFAULT 0,
    disagreements   INTEGER NOT NULL DEFAULT 0,
    last_seen       REAL NOT NULL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pattern_sig ON patterns(signature);

CREATE TABLE IF NOT EXISTS vectors (
    kind            TEXT NOT NULL,
    ref_id          TEXT NOT NULL,
    dim             INTEGER NOT NULL,
    vec             BLOB NOT NULL,
    PRIMARY KEY (kind, ref_id)
);

-- What each human correction did to memory, so it can be undone exactly.
-- effect: reinforced (created or strengthened a pattern) | agreed | disagreed
CREATE TABLE IF NOT EXISTS correction_effects (
    correction_id   INTEGER NOT NULL,
    pattern_id      TEXT NOT NULL,
    effect          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_effects_corr ON correction_effects(correction_id);

-- Alert cards sent to people, so they can be updated in place.
CREATE TABLE IF NOT EXISTS notifications (
    review_id       TEXT NOT NULL,
    channel         TEXT NOT NULL,
    chat_id         TEXT NOT NULL,
    message_id      TEXT NOT NULL,
    has_image       INTEGER NOT NULL DEFAULT 0,
    footer          TEXT NOT NULL DEFAULT '',
    keyboard        TEXT NOT NULL DEFAULT 'full',
    created_at      REAL NOT NULL,
    PRIMARY KEY (review_id, channel, chat_id)
);

-- Cameras whose notifications are muted until a time (alerts are still
-- processed and recorded; only the phone stays quiet).
CREATE TABLE IF NOT EXISTS snoozes (
    camera          TEXT PRIMARY KEY,
    until           REAL NOT NULL
);

-- "What was wrong?" questions sent after a Real tap: a reply to one of these
-- messages becomes that correction's note (D-053).
CREATE TABLE IF NOT EXISTS note_requests (
    chat_id         TEXT NOT NULL,
    message_id      INTEGER NOT NULL,
    correction_id   INTEGER NOT NULL,
    created_at      REAL NOT NULL,
    PRIMARY KEY (chat_id, message_id)
);

-- One card per incident per room (D-062): an event from another camera in
-- the room joins the card of the event that opened it (the lead), whose
-- notifications rows carry the messages.
-- "Ask Watchwise" (D-065): the conversation, changes waiting for the owner's
-- ✅ Confirm, and "tell me when..." watches the vision model checks.
CREATE TABLE IF NOT EXISTS chat_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation    TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_conversation ON chat_messages(conversation, id);

CREATE TABLE IF NOT EXISTS proposals (
    id              TEXT PRIMARY KEY,
    conversation    TEXT NOT NULL,
    kind            TEXT NOT NULL,
    payload         TEXT NOT NULL,
    summary         TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    result          TEXT NOT NULL DEFAULT '',
    created_at      REAL NOT NULL,
    decided_at      REAL
);

CREATE TABLE IF NOT EXISTS watches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    request         TEXT NOT NULL,
    cameras         TEXT NOT NULL DEFAULT '[]',
    created_at      REAL NOT NULL,
    expires_at      REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    review_id       TEXT
);

CREATE TABLE IF NOT EXISTS card_members (
    review_id       TEXT NOT NULL,
    channel         TEXT NOT NULL,
    lead_review_id  TEXT NOT NULL,
    created_at      REAL NOT NULL,
    PRIMARY KEY (review_id, channel)
);
"""


class Store:
    """Thin, thread-safe SQLite wrapper.

    Uses one connection guarded by a lock rather than a pool: write volume is
    a handful of rows per alert, and a single connection keeps WAL semantics
    trivial to reason about.
    """

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Columns added after a database was created; never drops anything."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(alerts)")}
        if "summary" not in columns:
            self._conn.execute("ALTER TABLE alerts ADD COLUMN summary TEXT")
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(note_requests)")}
        if "also" not in columns:
            # The other cameras' corrections from the same tap on a room card.
            self._conn.execute("ALTER TABLE note_requests ADD COLUMN also TEXT NOT NULL DEFAULT ''")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------- generic ----------------

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    # ---------------- vectors ----------------

    def put_vector(self, kind: str, ref_id: str, vec: np.ndarray) -> None:
        arr = np.asarray(vec, dtype=np.float32)
        self.execute(
            "INSERT INTO vectors(kind, ref_id, dim, vec) VALUES (?,?,?,?) "
            "ON CONFLICT(kind, ref_id) DO UPDATE SET dim=excluded.dim, vec=excluded.vec",
            (kind, ref_id, int(arr.shape[0]), arr.tobytes()),
        )

    def delete_vector(self, kind: str, ref_id: str) -> None:
        self.execute("DELETE FROM vectors WHERE kind=? AND ref_id=?", (kind, ref_id))

    def load_vectors(self, kind: str) -> tuple[list[str], np.ndarray]:
        rows = self.query("SELECT ref_id, dim, vec FROM vectors WHERE kind=?", (kind,))
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        ids = [r["ref_id"] for r in rows]
        mat = np.vstack(
            [np.frombuffer(r["vec"], dtype=np.float32).reshape(r["dim"]) for r in rows]
        )
        return ids, mat

    # ---------------- alerts ----------------

    def upsert_alert(self, ctx_dict: dict[str, Any], stage: str) -> None:
        now = time.time()
        self.execute(
            """
            INSERT INTO alerts(review_id, camera, start_time, end_time, signature,
                               labels, zones, stage, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(review_id) DO UPDATE SET
                stage=excluded.stage, end_time=excluded.end_time,
                labels=excluded.labels, zones=excluded.zones,
                updated_at=excluded.updated_at
            """,
            (
                ctx_dict["review_id"],
                ctx_dict["camera"],
                ctx_dict["start_time"],
                ctx_dict.get("end_time"),
                ctx_dict["signature"],
                json.dumps(ctx_dict.get("labels", [])),
                json.dumps(ctx_dict.get("zones", [])),
                stage,
                now,
                now,
            ),
        )

    def record_decision(self, decision: dict[str, Any]) -> None:
        self.execute(
            """
            UPDATE alerts SET disposition=?, severity=?, confidence=?, rationale=?,
                              screen_json=?, investigation_id=?, stage=?, updated_at=?,
                              summary=COALESCE(?, summary)
            WHERE review_id=?
            """,
            (
                decision["disposition"],
                decision["severity"],
                decision["confidence"],
                decision["rationale"],
                json.dumps(decision.get("screen")),
                decision.get("investigation_id"),
                decision["stage"],
                time.time(),
                decision.get("summary"),
                decision["review_id"],
            ),
        )

    # ---------------- notifications ----------------

    def add_notification(
        self, review_id: str, channel: str, chat_id: str, message_id: str, *, has_image: bool
    ) -> None:
        self.execute(
            "INSERT OR REPLACE INTO notifications(review_id, channel, chat_id, message_id, "
            "has_image, footer, keyboard, created_at) VALUES (?,?,?,?,?,'','full',?)",
            (review_id, channel, str(chat_id), str(message_id), int(has_image), time.time()),
        )

    def notifications_for(self, review_id: str, channel: str) -> list[sqlite3.Row]:
        return self.query(
            "SELECT * FROM notifications WHERE review_id=? AND channel=?", (review_id, channel)
        )

    def set_notification_state(
        self, review_id: str, channel: str, *, footer: str, keyboard: str
    ) -> None:
        self.execute(
            "UPDATE notifications SET footer=?, keyboard=? WHERE review_id=? AND channel=?",
            (footer, keyboard, review_id, channel),
        )

    def set_snooze(self, camera: str, until: float) -> None:
        self.execute(
            "INSERT OR REPLACE INTO snoozes(camera, until) VALUES (?,?)", (camera, until)
        )

    def snoozed_until(self, camera: str) -> float | None:
        row = self.query_one("SELECT until FROM snoozes WHERE camera=?", (camera,))
        return float(row["until"]) if row else None

    def correction_examples(self, signature: str, limit: int = 8) -> list[tuple[str, str]]:
        """(human verdict, what was seen) for recent corrections of this kind of
        event, newest first. What was seen is the owner's own note, else the
        alert's whole-event summary, else its screening reason, read now, so a
        later note or summary counts. A screening reason that called a Real
        event benign is left out: it argues the opposite of the owner (D-053)."""
        rows = self.query(
            "SELECT c.human_verdict AS verdict, c.note AS note, a.summary AS summary, "
            "a.screen_json AS screen "
            "FROM corrections c JOIN alerts a ON a.review_id = c.review_id "
            "WHERE c.signature=? ORDER BY c.created_at DESC LIMIT ?",
            (signature, limit),
        )
        examples = []
        for row in rows:
            text = (row["note"] or "").strip() or row["summary"]
            if not text and row["screen"]:
                try:
                    screen = json.loads(row["screen"]) or {}
                except ValueError:
                    screen = {}
                if not (row["verdict"] == "suspicious" and screen.get("verdict") == "benign"):
                    text = screen.get("reason")
            if text:
                examples.append((str(row["verdict"]), str(text)))
        return examples

    def add_note_request(self, chat_id: int | str, message_id: int, correction_ids: list[int]) -> None:
        """A "What was wrong?" message about these corrections (one per camera
        of a room card; the first is the card's own)."""
        first, *also = [int(c) for c in correction_ids]
        self.execute(
            "INSERT OR REPLACE INTO note_requests(chat_id, message_id, correction_id, created_at, also) "
            "VALUES (?,?,?,?,?)",
            (str(chat_id), int(message_id), first, time.time(), ",".join(map(str, also))),
        )

    def note_request(self, chat_id: int | str, message_id: int) -> list[int]:
        """The corrections a "What was wrong?" message asked about (empty if none)."""
        row = self.query_one(
            "SELECT correction_id, also FROM note_requests WHERE chat_id=? AND message_id=?",
            (str(chat_id), int(message_id)),
        )
        if row is None:
            return []
        return [int(row["correction_id"]), *(int(c) for c in str(row["also"] or "").split(",") if c)]

    # ---------------- room cards (D-062) ----------------

    def add_card_member(self, lead_review_id: str, review_id: str, channel: str) -> None:
        self.execute(
            "INSERT OR IGNORE INTO card_members(review_id, channel, lead_review_id, created_at) VALUES (?,?,?,?)",
            (review_id, channel, lead_review_id, time.time()),
        )

    def review_for_message(self, channel: str, chat_id: int | str, message_id: int | str) -> str | None:
        """The event whose card is this message, if it is one."""
        row = self.query_one(
            "SELECT review_id FROM notifications WHERE channel=? AND chat_id=? AND message_id=?",
            (channel, str(chat_id), str(message_id)),
        )
        return str(row["review_id"]) if row else None

    # ---------------- chat (D-065) ----------------

    def add_chat_message(self, conversation: str, role: str, content: str, at: float | None = None) -> None:
        self.execute(
            "INSERT INTO chat_messages(conversation, role, content, created_at) VALUES (?,?,?,?)",
            (conversation, role, content, time.time() if at is None else at),
        )

    def chat_history(self, conversation: str, limit: int) -> list[sqlite3.Row]:
        """The last ``limit`` messages, oldest first."""
        rows = self.query(
            "SELECT role, content, created_at FROM chat_messages WHERE conversation=? ORDER BY id DESC LIMIT ?",
            (conversation, int(limit)),
        )
        return list(reversed(rows))

    def clear_chat(self, conversation: str) -> None:
        self.execute("DELETE FROM chat_messages WHERE conversation=?", (conversation,))

    def recent_alerts(self, since: float, until: float | None = None, limit: int = 200) -> list[sqlite3.Row]:
        """Events that started in the window, newest first."""
        return self.query(
            "SELECT * FROM alerts WHERE start_time>=? AND start_time<=? ORDER BY start_time DESC LIMIT ?",
            (since, until if until is not None else 1e12, int(limit)),
        )

    def remove_card_member(self, review_id: str, channel: str) -> None:
        self.execute("DELETE FROM card_members WHERE review_id=? AND channel=?", (review_id, channel))

    def card_lead(self, review_id: str, channel: str) -> str | None:
        """The event whose card this one joined, if it joined one."""
        row = self.query_one(
            "SELECT lead_review_id FROM card_members WHERE review_id=? AND channel=?", (review_id, channel)
        )
        return str(row["lead_review_id"]) if row else None

    def card_members(self, lead_review_id: str, channel: str) -> list[str]:
        """The events that joined this one's card, in the order they joined."""
        rows = self.query(
            "SELECT review_id FROM card_members WHERE lead_review_id=? AND channel=? ORDER BY created_at, rowid",
            (lead_review_id, channel),
        )
        return [str(r["review_id"]) for r in rows]

    def recent_cards(self, channel: str, since: float) -> list[sqlite3.Row]:
        """Cards sent since then (one row per card, newest first), with the
        camera and time span of the event each was sent for."""
        return self.query(
            "SELECT a.review_id, a.camera, a.start_time, a.end_time FROM alerts a "
            "WHERE a.review_id IN (SELECT DISTINCT review_id FROM notifications WHERE channel=? AND created_at>=?) "
            "ORDER BY a.start_time DESC",
            (channel, since),
        )

    def alerts_on(self, cameras: list[str], since: float) -> list[sqlite3.Row]:
        """Events on these cameras that started since then."""
        if not cameras:
            return []
        marks = ",".join("?" for _ in cameras)
        return self.query(
            f"SELECT * FROM alerts WHERE camera IN ({marks}) AND start_time>=? ORDER BY start_time",
            (*cameras, since),
        )

    def set_correction_note(self, correction_id: int, note: str) -> bool:
        """False when the correction is gone (undone)."""
        cur = self.execute("UPDATE corrections SET note=? WHERE id=?", (note, int(correction_id)))
        return cur.rowcount > 0

    def get_alert(self, review_id: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM alerts WHERE review_id=?", (review_id,))

    def recent_alerts_by_signature(
        self, signature: str, since: float, exclude_review_id: str | None = None
    ) -> list[sqlite3.Row]:
        rows = self.query(
            "SELECT * FROM alerts WHERE signature=? AND start_time>=? "
            "ORDER BY start_time DESC LIMIT 50",
            (signature, since),
        )
        if exclude_review_id:
            rows = [r for r in rows if r["review_id"] != exclude_review_id]
        return rows

    # ---------------- investigations ----------------

    def upsert_investigation(self, inv: dict[str, Any]) -> None:
        self.execute(
            """
            INSERT INTO investigations(id, review_id, camera, status, verdict,
                confidence, conclusion, iterations, backend, model, cost_usd,
                trace_json, started_at, finished_at, error)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, verdict=excluded.verdict,
                confidence=excluded.confidence, conclusion=excluded.conclusion,
                iterations=excluded.iterations, cost_usd=excluded.cost_usd,
                trace_json=excluded.trace_json, finished_at=excluded.finished_at,
                error=excluded.error
            """,
            (
                inv["id"],
                inv["review_id"],
                inv["camera"],
                inv["status"],
                inv.get("verdict"),
                inv.get("confidence"),
                inv.get("conclusion"),
                inv.get("iterations", 0),
                inv.get("backend"),
                inv.get("model"),
                inv.get("cost_usd", 0.0),
                json.dumps(
                    {
                        "tool_calls": inv.get("tool_calls", []),
                        "thoughts": inv.get("thoughts", []),
                    }
                ),
                inv.get("started_at"),
                inv.get("finished_at"),
                inv.get("error"),
            ),
        )

    def get_investigation(self, inv_id: str) -> sqlite3.Row | None:
        return self.query_one("SELECT * FROM investigations WHERE id=?", (inv_id,))

    # ---------------- stats ----------------

    def funnel_counts(self, since: float = 0.0) -> dict[str, int]:
        rows = self.query(
            "SELECT stage, COUNT(*) c FROM alerts WHERE start_time>=? GROUP BY stage",
            (since,),
        )
        counts = {r["stage"]: r["c"] for r in rows}
        disp = self.query(
            "SELECT disposition, COUNT(*) c FROM alerts "
            "WHERE start_time>=? AND disposition IS NOT NULL GROUP BY disposition",
            (since,),
        )
        for r in disp:
            key = "disposition:" + str(r["disposition"])
            counts[key] = r["c"]
        return counts
