# Watchwise trace format 1.0

A trace is one recorded run of Watchwise: what the cameras reported, every
decision Watchwise made and what it was based on, every message the owner's
phone showed, every answer the owner gave, the chat, and the daily digest.
`watchwise export` writes it from Watchwise's own database; the web demo
replays it. This document is the contract between the two: a reader needs
nothing else.

```
watchwise export --config config/watchwise.yml --since "2026-10-30 12:00" --until "2026-10-30 13:30" \
    --out site-data/run-1 --machine "NVIDIA L4 (cloud VM)"
```

```
run-1/
  trace.json        the whole run (UTF-8 JSON)
  frames/f01.jpg    photos the phone showed, people pixelated (JPEG, longest edge ≤ 960 px)
  frames/f02.jpg
```

## Conventions

- **Ids** are short strings with a kind prefix, unique within one file:
  `e01` event, `d001` decision, `i01` investigation, `f01` frame, `m001`
  message entry, `tg01` phone message, `a001` owner action, `c01`
  correction, `p01` learned pattern, `q01` chat turn, `pr01` proposal,
  `w01` watch, `g01` digest. Ids from Frigate or Telegram never appear;
  where one would (inside a tool's arguments or result), it is replaced by
  the event id, or by `frigate-id` for an event outside the trace.
- **Times**: every entry has `at` (ISO 8601 with the house's UTC offset,
  milliseconds) and `t`, seconds since `meta.window.start`. Drive the replay
  clock with `t`; show `at` (or its local clock time) to people. A few
  fields hold a time object `{"at", "t"}` instead (`started`, `decided`…).
- **Text** in `messages[].text` is Telegram HTML: `<b>`, `<i>`, `<code>`,
  `<a>` are possible, and `&lt; &gt; &amp;` are escaped. Render it with a
  whitelist of those tags (never as raw HTML) and keep line breaks. All other
  strings are plain text.
- `null` means unknown or not applicable; an empty list means none.
- Readers must **ignore fields they don't know**: 1.x versions only add
  fields. A change that would break a 1.0 reader becomes 2.0.

## Privacy

`meta.privacy` says what was done; by default:

- People in frames are pixelated over the box Frigate drew around each
  person (and any face box), grown by a quarter of its size on every side. A
  frame is left out when Frigate gave no box for that moment, or when more
  than one person was in view (their boxes come from other moments).
- Names of residents, of people in Frigate's Face Library, and of people on
  the duty roster read "Resident A", "Resident B", "Visitor A"… Roles such
  as `cleaner` stay. The owner is "owner"; links read `[link]`.
- What the owner typed is kept as typed, names replaced.

Frames show a real room. Don't enlarge, sharpen or otherwise try to undo the
pixelation, and don't add frames that aren't in the trace.

## Top level

| Field | Type | |
|---|---|---|
| `format` | `"watchwise-trace"` | |
| `version` | `"1.0"` | |
| `meta` | object | where, when and how it was recorded |
| `cameras` | array | |
| `events` | array | one per Frigate review (an incident on one camera), by time |
| `decisions` | array | every decision, by time; an event usually has 1–3 |
| `investigations` | array | the tool-calling investigations, step by step |
| `frames` | array | the photos in `frames/` |
| `messages` | array | everything the phone showed, as sent and as edited |
| `owner_actions` | array | buttons, notes and commands of the owner |
| `corrections` | array | the owner's verdicts, and what each taught |
| `patterns` | array | what memory learned from them |
| `conversations` | array | the "Ask Watchwise" chat |
| `proposals` | array | changes the chat offered, and whether the owner confirmed |
| `watches` | array | "tell me when…" requests |
| `digests` | array | the daily digest and its spot-checks |
| `arming` | array | Home / Away / Night / Off over time |
| `metrics` | object | numbers counted or measured in this run, each with a label |

## `meta`

```json
{
  "title": null,
  "source": "recorded",
  "generated_at": "2026-10-30T14:02:11-05:00",
  "generator": {"name": "watchwise", "version": "0.2.0", "commit": "d98d667-dirty"},
  "window": {"start": "2026-10-30T12:00:00.000-05:00", "end": "2026-10-30T13:30:00.000-05:00", "seconds": 5400.0},
  "timezone": "America/Chicago",
  "utc_offset_minutes": -300,
  "models": {"screener": "ollama/qwen3-vl:8b-instruct", "investigator": "ollama/qwen3-vl:8b-instruct", "chat": "ollama/qwen3-vl:8b-instruct"},
  "machine": "NVIDIA L4 (cloud VM)",
  "privacy": {"frames": "blurred", "names": "replaced", "people_in_frames": "…", "identities": "…"}
}
```

`source` is `recorded` (a real run on real cameras) or `simulated` (a
scripted day; reserved for a later version). `commit` ends in `-dirty` when
the code that ran had changes not yet committed. `privacy.frames` is
`blurred`, `none` (no frames) or `full` (private use only; never publish).

## `cameras`

`{"id": "webcam", "room": "bedroom", "notes": "my desk; I work here"}`. `room`
groups cameras that watch one room (they share one card per incident);
`null` when not set.

## `events`

```json
{
  "id": "e03", "camera": "webcam", "at": "…", "t": 812.4, "end": "…", "duration_s": 19.0,
  "labels": ["person"], "zones": [],
  "outcome": {"disposition": "alerted", "stage": "investigated", "confidence": 0.85,
              "reason": "A person entered while the house is armed away…", "summary": "A person walks in…"},
  "frame": "f02"
}
```

`outcome` is the final decision, the same as the event's last entry in
`decisions`:

- `disposition`: `alerted` (a card asked for attention), `escalated_human`
  (a card: "you decide"), `downgraded` (held back, kept as a detection),
  `suppressed` (held back by a rule or a trusted pattern).
- `stage`, who decided: `gated` (a rule or a learned pattern, no model
  call), `face` (Frigate's face recognition), `screened` (the vision
  model's first look), `investigated` (after an investigation),
  `summarized` (raised by the review of the whole event), `ingested`
  (nothing decided yet).
- `reason` is the one-line reason a card shows; `summary`, what happened
  over the whole event, written after it ended (or `null`).
- `frame` is the event's photo, if the phone showed one.

## `decisions`

```json
{
  "id": "d007", "event": "e03", "at": "…", "t": 813.1,
  "stage": "screened", "disposition": "alerted", "severity": "alert", "confidence": 0.85,
  "reason": "A person entered the room…", "rationale": "tier-2 screening: A person entered the room…",
  "analyzing": false,
  "screen": {"verdict": "suspicious", "confidence": 0.85, "reason": "…", "tags": ["person"],
             "model": "ollama/qwen3-vl:8b-instruct", "latency_ms": 1240.0, "frames_used": 2, "error": null},
  "investigation": "i02", "summary": null, "patterns": ["p01"], "history": "complete"
}
```

- `analyzing: true` is the placeholder sent when the first look takes
  long: the card said "⏳ Analyzing…" until the next decision.
- `screen` is the vision model's own verdict. It can differ from the
  decision: while armed away, a person is an alert even when the model found
  them ordinary.
- `patterns` are the learned patterns behind the decision (see `patterns`).
- `history` is `latest_only` for an event decided before Watchwise kept
  every decision; then only the last one is known.

## `investigations`

```json
{
  "id": "i02", "event": "e03", "status": "done", "verdict": "suspicious", "confidence": 0.85,
  "conclusion": "The person was not present 30 s earlier…",
  "model": "tools/qwen3-vl:8b-instruct",
  "started": {"at": "…", "t": 813.3}, "finished": {"at": "…", "t": 835.0}, "seconds": 21.7,
  "steps": [
    {"evidence_id": "i02/1", "iteration": 1, "tool": "get_device_status", "arguments": {"camera": "webcam"},
     "result": "webcam online, 5.0 fps …", "latency_ms": 41.0, "error": null},
    {"evidence_id": "i02/2", "iteration": 1, "tool": "describe_historical_frames", "arguments": {"offsets": [-30, 0]},
     "result": "…", "latency_ms": 2310.0, "error": null}
  ],
  "thoughts": [],
  "observations": [], "inferences": [], "unknowns": [],
  "error": null
}
```

Each step is one tool call. `evidence_id` is stable, so an evidence pop-up
can link a conclusion to the steps behind it. `observations`, `inferences`
and `unknowns` are reserved: they will hold `{"fact", "evidence_id"}`,
strings and strings when the investigator reports them; today they are
empty, which means "not reported", not "none".

## `frames`

`{"id": "f02", "event": "e03", "file": "frames/f02.jpg", "at": "…", "t": 815.2,
"source": "recording at the person's best frame", "width": 960, "height": 540,
"privacy": "blurred", "boxes": 1}`. This is the event's own frame from Frigate's
recording, at the moment Frigate saw the person best. The phone showed a frame
of the same event, possibly from a moment earlier.

## `messages`: the phone

Every message Watchwise sent, and every edit of one, in order:

```json
{
  "id": "m004", "at": "…", "t": 836.0, "message": "tg03", "action": "edited",
  "kind": "alert", "event": "e03", "members": [],
  "text": "<b>🚨 Person · webcam · 12:14</b>\nWhy: …\nConfidence 0.85 · after an investigation",
  "buttons": [[{"label": "✅ Expected", "action": "ok", "event": "e03"},
               {"label": "🚨 Alert me", "action": "real", "event": "e03"}],
              [{"label": "🔇 Snooze", "action": "snooze", "event": "e03"},
               {"label": "❓ Why", "action": "why", "event": "e03"}],
              [{"label": "▶️ Clip", "link": true}]],
  "has_photo": true, "frame": "f02"
}
```

- `message` names one phone message. `action: "sent"` creates it;
  `"edited"` replaces its text and buttons in place, so draw the latest
  version for that `message`. Edits don't make the phone buzz.
- `kind`:

  | `kind` | The message |
  |---|---|
  | `alert` | an alert card (`event`; `members` are the room's other cameras' events on the same card) |
  | `spot_check` | a held-back event the digest sent back to check |
  | `digest` | the daily digest |
  | `note_request` | "What was wrong?" after 🚨 Alert me |
  | `chat` | an answer in the chat (`event`: the card it was a reply to, if any) |
  | `chat_photo` | a photo sent with an answer |
  | `proposal` | a change offered with ✅ Confirm / ✖️ Cancel |
- `buttons` are rows of buttons. `action` is one of `ok` (✅ Expected),
  `real` (🚨 Alert me), `look` (🔍 Look closer), `snooze`, `why`, `undo`
  (with the `corrections` it takes back), `confirm` / `cancel` (with
  `proposal`), `arm` (with `state`), `that_was` (👤 That was …, with
  `name`), `face_add` / `face_cancel` (a photo for Frigate's Face Library).
  A link button (`"link": true`) opened the recording in Frigate. It has no
  address in the trace.
- `has_photo`: the message carried a photo; `frame` is that photo when the
  trace has it (else `null`: show the illustration or a placeholder).

## `owner_actions`

```json
{"id": "a002", "at": "…", "t": 860.5, "kind": "button", "on_message": "tg03",
 "action": "ok", "event": "e03",
 "outcome": {"toast": "Thanks, noted as expected.", "footer": "✅ Marked expected (1 of 3)…",
             "reply": null, "done": null, "spot_check": false, "corrections": ["c01"]}}
```

- `kind: "button"` is a tap on `on_message`. The `action` fields are as in
  `buttons`. `outcome.toast` is the short pop-up the phone showed and
  `footer` the line added to the card. `reply` is a message sent in answer
  (the evidence for ❓ Why) and `corrections` the verdicts the tap recorded.
  On `confirm` / `cancel`, `toast` is the result and `done` whether it was
  applied.
- `kind: "note"` is a typed reply to "What was wrong?": `text`, `reply`
  (Watchwise's thanks), `replied_to` (the question's `message`).
- `kind: "command"` is a command such as `/digest`, `/arm night` or
  `/new`, in `text`.

## `corrections` and `patterns`: the learning

```json
{"id": "c01", "event": "e03", "at": "…", "t": 860.5, "verdict": "benign", "system_verdict": "suspicious",
 "note": null, "via": "card", "undone": null,
 "effects": [{"pattern": "p01", "effect": "reinforced"}]}
```

`verdict` is the owner's: `benign` (✅ Expected) or `suspicious` (🚨 Alert
me). `via` is `card`, `spot_check` or `other`. `undone` is a time object if
the owner took it back with ↩️ Undo; its `effects` are then empty.
`effects`:

- `reinforced`: the correction created or strengthened that pattern.
- `agreed` / `disagreed`: the pattern had influenced the decision, and the
  owner's verdict agreed with it or not.

```json
{"id": "p01", "kind": "false_alarm", "camera": "webcam", "signature": "webcam|person|nozone",
 "description": "sitting at my desk", "corrections": 3, "agreements": 2, "disagreements": 0,
 "trust": 0.75, "first_seen": {"at": "…", "t": 860.5}, "last_seen": {"at": "…", "t": 1702.0},
 "state": "at export"}
```

A pattern's numbers are as they stood at export. `trust` is
(agreements + 1) / (agreements + disagreements + 2). To show "you taught it
→ it held the next one back", follow a correction's `effects` to a
pattern, then find later `decisions` whose `patterns` include it. Where
memory may act alone (the default `gate.memory_mode: free`), a pattern
learned from at least 3 corrections, with trust ≥ 0.7, can hold an event
back by itself (`stage: "gated"`). Otherwise, as in `model_checked` mode, it
informs the vision model (`stage: "screened"`, the pattern listed). While
armed away or at night, memory never holds back a person.

## `conversations` and `proposals`: the chat

```json
{"id": "owner", "turns": [
  {"id": "q01", "at": "…", "t": 1500.2,
   "question": "why didn't you tell me about the 12:14 one? show me a photo",
   "answer": "It was held back: Frigate recognized Resident A…",
   "about": null,
   "tools": [{"tool": "explain_decision", "arguments": {"time": "12:14"}, "ok": true, "result": "{…}"},
             {"tool": "show_snapshot", "arguments": {"event_id": "e01"}, "ok": true, "result": "{…}"}],
   "photos": ["e01"], "proposals": [], "seconds": 5.3, "prompt_tokens": 2487,
   "guard": null, "steps_recorded": true}
]}
```

`tools` are the tool calls behind the answer, each with what it returned
(JSON text, possibly cut short), as the model saw it. `photos` lists the
events whose photos followed the answer (`chat_photo` messages). `guard`
is set when a safety rule shaped the answer:

- `claimed_change`: the model said it changed something without proposing
  it, and was asked again.
- `nothing_changed`: an honest fixed text replaced such an answer.
- `typed_confirmation`: a typed "yes" was answered without the model,
  because only the button applies a change.

`steps_recorded: false` marks an answer from before tool steps were kept.

```json
{"id": "pr01", "at": "…", "t": 1530.0, "kind": "arming", "summary": "Switch arming to Night…",
 "payload": {"state": "armed_night", "label": "Night"}, "status": "applied", "result": "Arming is now Night.",
 "decided": {"at": "…", "t": 1541.7}}
```

`kind` is `arming`, `snooze`, `roster_rule`, `roster_removal` or `watch`.
`status` is `pending`, `applied`, `cancelled`, `expired` or `failed`. Nothing
changes before the owner taps ✅ Confirm.

## `watches`, `digests`, `arming`

- **Watch:** `{"id": "w01", "request": "a parcel appears on the chair",
  "cameras": ["webcam"], "created": {…}, "expires": {…}, "status": "matched",
  "matched": "e07"}`. The vision model checks each event against it when the
  event ends; a match sends one card.
- **Digest:** `{"id": "g01", "kind": "asked", "covers": {"start", "end"}, "at",
  "t", "messages": ["tg09"], "spot_checks": ["e02", "e05"], "sent": true}`.
  `kind` is `scheduled` (the daily one) or `asked` (`/digest`). `sent` is
  false for a scheduled digest with nothing to report. The spot-checks are
  held-back events sent back as `spot_check` messages.
- **Arming:** `[{"at", "t", "state": "armed_home", "source": "telegram"}, …]`.
  The first entry may come before the window: the state the run began in.
  `state` is `armed_home`, `armed_away`, `armed_night` or `disarmed`.

## `metrics`

```json
{
  "counts": {"label": "counted from this recording", "events": 14, "sent_to_owner": 4, "held_back": 10,
             "alerted_while_muted": 0, "held_back_by_reason": {"recognized Resident A": 6},
             "final_stage": {"gated": 0, "face": 6, "screened": 5, "investigated": 3, "summarized": 0, "ingested": 0},
             "model_calls": {"screenings": 9, "investigations": 3, "chat_answers": 4},
             "owner_answers": {"benign": 3, "suspicious": 1},
             "spot_checks": {"answered": 2, "right_to_hold_back": 1, "accuracy": 0.5}},
  "timings": {"label": "measured (NVIDIA L4 (cloud VM), 2026-10-30)",
              "first_card_after_event_start_s": {"n": 4, "median": 2.9, "max": 4.1},
              "screen_ms": {"n": 9, "median": 1310.0, "max": 2620.0},
              "investigation_s": {"n": 3, "median": 21.5, "max": 24.0},
              "chat_answer_s": {"n": 4, "median": 3.1, "max": 5.3}}
}
```

Show every number with its block's `label`. The values above illustrate the
shape. They describe one run on one setup; they are not a claim about
accuracy or speed elsewhere. Don't compute rates the trace doesn't give.

- `model_calls.screenings` counts the vision model's first looks whose
  verdict was used, once each (later decisions repeat the first look's
  result). The whole-event summaries and watch checks are not counted.
- `first_card_after_event_start_s` runs from the moment Frigate's event
  began to the first card sent for it.
- `spot_checks` counts the owner's answers to spot-check cards in the window.

## Replaying a trace

1. Sort `messages`, `owner_actions`, `decisions` and `arming` by `t`, and
   advance a clock through them (faster than real time is fine; say so).
2. Phone: a `sent` message appears; an `edited` one changes in place. A
   button's tap appears at its `owner_actions` entry, with its `toast` and
   the card's next edit.
3. Evidence: from a card, open its event's `decisions` (the latest is the
   card's verdict), then its `investigation` steps and `frame`.
4. Learning: when a correction `reinforced` a pattern, mark that pattern,
   and highlight later decisions that list it (often `downgraded` with no
   card: "held back, no message sent").
5. Held-back events with no message still belong on the timeline: they are
   what the digest counts and what its spot-checks sample.
