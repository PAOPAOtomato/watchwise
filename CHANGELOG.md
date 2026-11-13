# Changelog

All notable changes to Watchwise are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-13

The first release: an AI guard on top of an unmodified
[Frigate](https://frigate.video) installation, with alert cards and chat in
Telegram. Install it from a clone of this repository (see the README).

### Added

- **Event cascade.** Frigate review events pass a model-free gate (configured
  rules, dedupe, trusted recurring patterns), then a vision model screens what
  is left, and a tool-calling investigator looks into ambiguous cases. Errors,
  timeouts, refusals and unreadable replies become `uncertain` and go to a
  human.
- **Alert cards in Telegram.** Each card says why it was sent, with the
  confidence and the deciding stage. An "Analyzing…" card goes out early and is
  edited in place when the verdict arrives. Buttons: ✅ Expected, 🚨 Alert me,
  🔍 Look closer, 🔇 Snooze, ❓ Why, ▶️ Clip, and ↩️ Undo for every correction.
- **Teaching.** Corrections build patterns with trust scores, decay and undo.
  After 🚨 Alert me, Watchwise asks "What was wrong?" and uses your answer as an
  example for the vision model. A recent 🚨 on the same kind of event stops
  memory from holding it back.
- **Arming.** `/arm` (home, away, night, off). The vision model is told the
  arming state. While armed away or at night, a person in view alerts and is
  investigated; only a duty-roster match or a resident recognized by Frigate
  can clear them, and the card says which.
- **Faces.** With Frigate's face recognition, residents listed in
  `faces.household` are held back while you're home, and an unmatched face
  raises the event. `/face <name>` enrolls a resident from a camera's current
  frame; 👤 on an unknown-face card teaches Frigate from that event.
- **Rooms.** Cameras grouped into rooms share one card per incident, and one
  tap teaches every camera on it.
- **Whole-event summaries.** When an event ends, the vision model describes it
  ("What happened" on the card); a summary can raise an event, never lower it.
- **Ask Watchwise.** Questions in Telegram: "anything while I was away?",
  "why didn't you tell me about the 13:15 one?", photos on request. Changes
  (arming, duty-roster rules, snoozes, "tell me when…" watches) come back as
  proposals that run only after ✅ Confirm.
- **`/setup` in Telegram.** Describe each camera and its room, the residents,
  what should always alert and the default arming state; the plan is shown
  before anything is written, with a backup and Undo.
- **Daily digest.** A silent summary at 08:00 (or `/digest`): what was sent,
  what was held back and why, and two spot-checks of held-back events to
  confirm or correct.
- **Run-anywhere models.** Ollama, any OpenAI-compatible server (LM Studio,
  llama.cpp, vLLM, OpenAI, Gemini, Groq, OpenRouter) or Claude, set per role
  (screener, investigator, chat).
- **Hardware-aware setup.** `watchwise doctor` detects the hardware, model
  servers and keys and recommends a profile; `watchwise init` shows a plan,
  then writes the configuration (and, with `--deploy`, the Docker Compose
  selection and a Frigate config); `watchwise check` calls each configured
  model once. Profiles for CPU, NVIDIA, AMD, Apple Silicon and cloud models.
- **Docker Compose deployment.** Frigate, MQTT, Ollama and Watchwise, with
  hardware overlays.
- **Trace export.** `watchwise export` writes a recorded run (decisions, cards,
  taps, chat steps) as `watchwise-trace` 1.0, with people pixelated and names
  replaced; see `docs/trace-format.md`.
- **Interfaces.** HTTP API (`/health`, `/metrics`, `/feedback`, …), MQTT
  verdicts, webhooks, and MCP tools for other agents.
- `watchwise demo`: a simulated day through the real pipeline, with no
  cameras, models or keys.

Tested with Frigate 0.18.2 and Ollama 0.40.8, on Python 3.11 and 3.12 (Linux,
macOS, Windows).

[0.2.0]: https://github.com/PAOPAOtomato/watchwise/releases/tag/v0.2.0
