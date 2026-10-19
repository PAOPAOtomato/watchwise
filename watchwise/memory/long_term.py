"""Long-term memory: the library of historical misjudgment patterns.

This is the half of the loop that makes the system improve. Every human
correction is written here; every new alert is checked against it before any
model is called. A pattern only earns the right to auto-suppress once enough
independent human corrections agree with it, and its influence decays if it
stops recurring.

The unit of memory is a *pattern*, not an alert:

    "cam_driveway | car | driveway"  ->  headlights sweeping the gate at
    night are repeatedly reported as a person; 6 corrections, 0 disagreements

Matching is two-signal: an exact signature match (camera+labels+zones) and a
cosine match on the free-text description. Either can fire, but the score
combines them so a near-miss description on the right camera still ranks.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from watchwise.memory.store import Store
from watchwise.memory.vectors import Embedder, cosine_search, decay_factor

SECONDS_PER_DAY = 86400.0


@dataclass(slots=True)
class PatternMatch:
    pattern_id: str
    kind: str
    signature: str
    description: str
    similarity: float
    score: float
    corrections: int
    agreements: int
    disagreements: int
    age_days: float

    @property
    def trust(self) -> float:
        """How much this pattern has earned the right to decide on its own.

        Laplace-smoothed agreement rate, so a pattern with 1/1 agreement does
        not outrank one with 9/10.
        """
        total = self.agreements + self.disagreements
        return (self.agreements + 1.0) / (total + 2.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "kind": self.kind,
            "signature": self.signature,
            "description": self.description,
            "similarity": round(self.similarity, 4),
            "score": round(self.score, 4),
            "corrections": self.corrections,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "trust": round(self.trust, 4),
            "age_days": round(self.age_days, 2),
        }


class PatternLibrary:
    """CRUD + similarity search over the misjudgment pattern table."""

    VECTOR_KIND = "pattern"

    def __init__(self, store: Store, embedder: Embedder, half_life_days: float = 30.0):
        self.store = store
        self.embedder = embedder
        self.half_life_days = half_life_days

    # ---------------- writes ----------------

    def upsert(
        self,
        *,
        kind: str,
        signature: str,
        description: str,
        camera: str | None = None,
        corrections_delta: int = 1,
    ) -> str:
        """Add a correction to an existing pattern, or create a new one.

        Patterns are keyed by (kind, signature) -- the same camera producing
        the same labels in the same zones is the same recurring mistake, so we
        reinforce rather than duplicate.
        """
        now = time.time()
        row = self.store.query_one(
            "SELECT * FROM patterns WHERE kind=? AND signature=?", (kind, signature)
        )
        if row is not None:
            pattern_id = row["id"]
            merged = self._merge_description(row["description"], description)
            self.store.execute(
                "UPDATE patterns SET corrections=corrections+?, last_seen=?, "
                "description=? WHERE id=?",
                (corrections_delta, now, merged, pattern_id),
            )
            self.store.put_vector(
                self.VECTOR_KIND, pattern_id, self.embedder.embed(f"{signature} {merged}")
            )
            return pattern_id

        pattern_id = f"pat-{uuid.uuid4().hex[:12]}"
        self.store.execute(
            "INSERT INTO patterns(id, kind, signature, camera, description, "
            "corrections, agreements, disagreements, last_seen, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (pattern_id, kind, signature, camera, description, corrections_delta, 0, 0, now, now),
        )
        self.store.put_vector(
            self.VECTOR_KIND, pattern_id, self.embedder.embed(f"{signature} {description}")
        )
        return pattern_id

    @staticmethod
    def _merge_description(existing: str, incoming: str) -> str:
        """Keep the description a short, deduplicated list of observations."""
        parts = [p.strip() for p in existing.split(" | ") if p.strip()]
        incoming = incoming.strip()
        if incoming and incoming not in parts:
            parts.append(incoming)
        return " | ".join(parts[-5:])

    def record_outcome(self, pattern_id: str, *, agreed: bool) -> None:
        """Feed back whether acting on this pattern turned out to be right.

        Called when a human reviews an alert the pattern influenced. This is
        what keeps auto-suppression honest: a pattern that starts causing
        missed detections loses trust and stops firing.
        """
        col = "agreements" if agreed else "disagreements"
        self.store.execute(
            f"UPDATE patterns SET {col}={col}+1, last_seen=? WHERE id=?",
            (time.time(), pattern_id),
        )

    def delete(self, pattern_id: str) -> None:
        self.store.execute("DELETE FROM patterns WHERE id=?", (pattern_id,))
        self.store.delete_vector(self.VECTOR_KIND, pattern_id)

    # ---------------- reads ----------------

    def get(self, pattern_id: str) -> dict[str, Any] | None:
        row = self.store.query_one("SELECT * FROM patterns WHERE id=?", (pattern_id,))
        return dict(row) if row else None

    def all_patterns(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind:
            rows = self.store.query(
                "SELECT * FROM patterns WHERE kind=? ORDER BY corrections DESC", (kind,)
            )
        else:
            rows = self.store.query("SELECT * FROM patterns ORDER BY corrections DESC")
        return [dict(r) for r in rows]

    def search(
        self,
        text: str,
        *,
        signature: str | None = None,
        kind: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> list[PatternMatch]:
        """Rank patterns against a free-text query plus an optional signature.

        Score = similarity * recency_decay, with an exact-signature bonus.
        Returned best-first.
        """
        ids, matrix = self.store.load_vectors(self.VECTOR_KIND)
        if not ids:
            return []

        query_vec = self.embedder.embed(f"{signature or ''} {text}".strip())
        ranked = cosine_search(query_vec, matrix, top_k=max(top_k * 3, top_k))

        now = time.time()
        out: list[PatternMatch] = []
        for row_idx, sim in ranked:
            pattern_id = ids[row_idx]
            row = self.store.query_one("SELECT * FROM patterns WHERE id=?", (pattern_id,))
            if row is None:
                continue
            if kind and row["kind"] != kind:
                continue

            age_days = max(0.0, (now - row["last_seen"]) / SECONDS_PER_DAY)
            score = sim * decay_factor(age_days, self.half_life_days)
            if signature and row["signature"] == signature:
                # An exact signature match is much stronger evidence than
                # wording similarity; lift it but keep the score bounded.
                sim = max(sim, 0.95)
                score = max(score, 0.95 * decay_factor(age_days, self.half_life_days))

            if sim < min_similarity:
                continue

            out.append(
                PatternMatch(
                    pattern_id=pattern_id,
                    kind=row["kind"],
                    signature=row["signature"],
                    description=row["description"],
                    similarity=sim,
                    score=score,
                    corrections=row["corrections"],
                    agreements=row["agreements"],
                    disagreements=row["disagreements"],
                    age_days=age_days,
                )
            )

        out.sort(key=lambda m: m.score, reverse=True)
        return out[:top_k]

    def best_suppression_match(
        self, signature: str, text: str, *, min_similarity: float, min_corrections: int
    ) -> PatternMatch | None:
        """The one pattern, if any, that is allowed to suppress this alert.

        Three gates, all of which must hold: it must be a false-alarm pattern,
        it must match closely enough, and it must have enough independent
        human corrections behind it. Trust is checked by the caller so the
        policy stays in one place.
        """
        for match in self.search(text, signature=signature, kind="false_alarm", top_k=5):
            if match.similarity < min_similarity:
                continue
            if match.corrections < min_corrections:
                continue
            return match
        return None
