"""Forensic tool category 1: vector search over historical alerts.

Two indexes are searched and merged, because they answer different questions:

  * **NVR's CLIP index** (/api/events/search) -- "has this camera seen
    something that *looks* like this before?" Visual similarity over real
    footage. Already built and maintained by NVR; we do not duplicate it.
  * **Sentinel's pattern library** -- "has a human already told us what this
    is?" Semantic similarity over past human corrections.

The second is the one that changes decisions. The first is what lets the
agent say "this is the fourth time this week" with evidence.
"""

from __future__ import annotations

from typing import Any

from sentinel.tools.base import ForensicTool, ToolContext, obj_schema


async def _search_similar_alerts(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    query = str(args.get("query", "")).strip()
    camera = args.get("camera")
    signature = args.get("signature")
    limit = int(args.get("limit", 5))

    # --- sentinel's own corrected-judgement memory ---
    pattern_hits = [
        m.to_dict()
        for m in ctx.patterns.search(
            query or (signature or ""),
            signature=signature,
            top_k=limit,
            min_similarity=0.3,
        )
    ]

    # --- NVR's CLIP index over actual footage ---
    nvr_hits: list[dict[str, Any]] = []
    nvr_error: str | None = None
    if query:
        try:
            raw = await ctx.nvr.search_events(
                query, cameras=[camera] if camera else None, limit=limit
            )
            events = raw.get("results", raw) if isinstance(raw, dict) else raw
            for e in events or []:
                nvr_hits.append(
                    {
                        "event_id": e.get("id"),
                        "camera": e.get("camera"),
                        "label": e.get("label"),
                        "sub_label": e.get("sub_label"),
                        "start_time": e.get("start_time"),
                        "score": e.get("score"),
                        "search_distance": e.get("search_distance"),
                        "description": (e.get("data") or {}).get("description"),
                    }
                )
        except Exception as exc:
            # A missing/disabled semantic index must not fail the whole tool;
            # the pattern hits are still useful on their own.
            nvr_error = str(exc)

    # --- how often this exact signature has fired recently ---
    recurrence: dict[str, Any] = {}
    if signature:
        import time

        window_days = float(args.get("recurrence_window_days", 7))
        since = time.time() - window_days * 86400
        rows = ctx.store.recent_alerts_by_signature(signature, since)
        dispositions: dict[str, int] = {}
        for r in rows:
            key = r["disposition"] or "undecided"
            dispositions[key] = dispositions.get(key, 0) + 1
        recurrence = {
            "window_days": window_days,
            "count": len(rows),
            "dispositions": dispositions,
        }

    return {
        "query": query,
        "pattern_matches": pattern_hits,
        "similar_events": nvr_hits,
        "nvr_search_error": nvr_error,
        "signature_recurrence": recurrence,
        "interpretation": _interpret(pattern_hits, recurrence),
    }


def _interpret(pattern_hits: list[dict[str, Any]], recurrence: dict[str, Any]) -> str:
    """A plain-language read, so the agent does not have to infer it."""
    if not pattern_hits and not recurrence.get("count"):
        return "No prior human judgement or recent recurrence found for this alert."

    parts = []
    if pattern_hits:
        top = pattern_hits[0]
        parts.append(
            f"Closest prior judgement is a {top['kind']} pattern "
            f"(similarity {top['similarity']}, {top['corrections']} human corrections, "
            f"trust {top['trust']}): {top['description']}"
        )
    if recurrence.get("count"):
        parts.append(
            f"This signature fired {recurrence['count']} time(s) in the last "
            f"{recurrence['window_days']:.0f} day(s); dispositions "
            f"{recurrence['dispositions']}."
        )
    return " ".join(parts)


TOOL = ForensicTool(
    name="search_similar_alerts",
    description=(
        "Vector search over historical alerts and past human corrections. Use this "
        "FIRST on any alert that looks routine: it tells you whether a human has "
        "already ruled on this exact situation, how often it recurs, and what was "
        "decided last time. Returns pattern matches from the correction memory, "
        "visually similar past events from the NVR semantic index, and a recurrence "
        "count for the alert's signature."
    ),
    category="vector_search",
    input_schema=obj_schema(
        {
            "query": {
                "type": "string",
                "description": "Free-text description of what to look for, "
                "e.g. 'person at the side gate after dark'.",
            },
            "camera": {"type": "string", "description": "Restrict to one camera."},
            "signature": {
                "type": "string",
                "description": "Alert signature 'camera|labels|zones' for exact "
                "recurrence lookup.",
            },
            "limit": {"type": "integer", "description": "Max hits per index (default 5)."},
            "recurrence_window_days": {
                "type": "number",
                "description": "Recurrence lookback in days (default 7).",
            },
        },
        required=["query"],
    ),
    handler=_search_similar_alerts,
)
