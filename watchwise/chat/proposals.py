"""Changes the chat may only propose (D-065, standing rule 6).

The model never changes anything. A ``propose_*`` tool stores a proposal
with a plain description; the channel shows it with ✅ Confirm / ✖️ Cancel,
and only the owner's tap applies it, through the same seams a person would
use: the duty roster file (plan -> apply with a backup), the arming state,
snoozes and watches. A proposal not decided within an hour expires.
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from watchwise.memory.store import Store
from watchwise.memory.watches import WatchBook
from watchwise.setup.plan import apply
from watchwise.setup.roster import plan_roster_entry, plan_roster_removal

logger = logging.getLogger(__name__)

KINDS = ("roster_rule", "roster_removal", "arming", "snooze", "watch")
TTL_S = 3600.0


@dataclass(frozen=True)
class Proposal:
    id: str
    kind: str
    summary: str  # plain text, for the owner to judge


@dataclass(frozen=True)
class Decided:
    done: bool  # applied (confirm) or cancelled (cancel)
    text: str  # what happened, in plain words


class ProposalBook:
    def __init__(
        self,
        store: Store,
        *,
        roster_path: Path | None,
        arming: Any = None,
        watches: WatchBook | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.store = store
        self.roster_path = roster_path
        self.arming = arming
        self.watches = watches
        self.clock = clock

    def propose(self, conversation: str, kind: str, payload: dict[str, Any], summary: str) -> Proposal:
        if kind not in KINDS:
            raise ValueError(f"unknown kind of change {kind!r}")
        proposal = Proposal(secrets.token_hex(4), kind, summary)
        self.store.execute(
            "INSERT INTO proposals(id, conversation, kind, payload, summary, created_at) VALUES (?,?,?,?,?,?)",
            (proposal.id, conversation, kind, json.dumps(payload), summary, self.clock()),
        )
        return proposal

    def confirm(self, proposal_id: str) -> Decided:
        """Apply a pending proposal (the owner tapped ✅ Confirm)."""
        row = self._pending(proposal_id)
        if isinstance(row, str):
            return Decided(False, row)
        payload = json.loads(row["payload"])
        try:
            result = getattr(self, f"_apply_{row['kind']}")(payload)
        except Exception as exc:
            logger.exception("proposal %s failed", proposal_id)
            result = f"That didn't work ({exc}); nothing was changed."
            self._decide(proposal_id, "failed", result)
            return Decided(False, result)
        self._decide(proposal_id, "applied", result)
        logger.info("proposal %s (%s) applied: %s", proposal_id, row["kind"], result)
        return Decided(True, result)

    def waiting(self, conversation: str) -> list[Proposal]:
        """Proposals in this conversation still waiting for a tap."""
        rows = self.store.query(
            "SELECT id, kind, summary FROM proposals WHERE conversation=? AND status='pending' AND created_at>? "
            "ORDER BY created_at",
            (conversation, self.clock() - TTL_S),
        )
        return [Proposal(r["id"], r["kind"], r["summary"]) for r in rows]

    def cancel(self, proposal_id: str) -> Decided:
        row = self._pending(proposal_id)
        if isinstance(row, str):
            return Decided(False, row)
        self._decide(proposal_id, "cancelled", "")
        return Decided(True, "Cancelled; nothing was changed.")

    def _pending(self, proposal_id: str) -> Any:
        row = self.store.query_one("SELECT * FROM proposals WHERE id=?", (proposal_id,))
        if row is None:
            return "I don't know that proposal any more; ask again."
        if row["status"] != "pending":
            return f"Already {row['status']}."
        if self.clock() - float(row["created_at"]) > TTL_S:
            self._decide(proposal_id, "expired", "")
            return "That proposal expired (after an hour); ask again if you still want it."
        return row

    def _decide(self, proposal_id: str, status: str, result: str) -> None:
        self.store.execute(
            "UPDATE proposals SET status=?, result=?, decided_at=? WHERE id=?",
            (status, result, self.clock(), proposal_id),
        )

    # ---------------- applying ----------------

    def _apply_roster_rule(self, payload: dict[str, Any]) -> str:
        if self.roster_path is None:
            raise RuntimeError("no duty roster file is configured (duty_roster.path)")
        entry = dict(payload["entry"])
        change = plan_roster_entry(self.roster_path, entry, timezone=entry.get("timezone", "UTC"))
        apply([change], now=self.clock())
        backup = f"; the previous version is {change.backup.name}" if change.backup else ""
        return f"Added “{entry['name']}” to the duty roster ({self.roster_path.name}{backup})."

    def _apply_roster_removal(self, payload: dict[str, Any]) -> str:
        if self.roster_path is None:
            raise RuntimeError("no duty roster file is configured (duty_roster.path)")
        change = plan_roster_removal(self.roster_path, payload["entry"])
        apply([change], now=self.clock())
        backup = f"; the previous version is {change.backup.name}" if change.backup else ""
        return f"Removed “{payload['entry'].get('name')}” from the duty roster ({self.roster_path.name}{backup})."

    def _apply_arming(self, payload: dict[str, Any]) -> str:
        if self.arming is None:
            raise RuntimeError("arming isn't available here")
        self.arming.set(payload["state"], source="chat")
        return f"Arming is now {payload['label']}."

    def _apply_snooze(self, payload: dict[str, Any]) -> str:
        until = self.clock() + float(payload["minutes"]) * 60
        for camera in payload["cameras"]:
            self.store.set_snooze(camera, until)
        return f"Muted {', '.join(payload['cameras'])} until {time.strftime('%H:%M', time.localtime(until))}; alerts are still recorded."

    def _apply_watch(self, payload: dict[str, Any]) -> str:
        if self.watches is None:
            raise RuntimeError("watches aren't available here")
        self.watches.add(payload["request"], payload["cameras"], payload["hours"])
        return f"Watching {payload['where']} for {payload['hours']:g} h: I'll alert you when {payload['request']}."
