# Watchwise

**An AI guard for the cameras you already have.** Alerts that explain
themselves, an agent you can question, and fewer false alarms every time you
correct it, running on the hardware you already own.

[![ci](https://github.com/PAOPAOtomato/watchwise/actions/workflows/ci.yml/badge.svg)](https://github.com/PAOPAOtomato/watchwise/actions/workflows/ci.yml)
![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![status: alpha](https://img.shields.io/badge/status-alpha-orange.svg)

Cameras are cheap; attention isn't. Motion alerts cry wolf (headlights, cats,
the courier) until you mute them, and then the one that mattered is buried.
Watchwise sits on top of [Frigate](https://frigate.video), the open-source NVR,
and decides which alerts deserve a human. When it isn't sure, it investigates
with tools before it answers, and it learns from every correction you make.
Models run locally by default; cloud models are optional.

> **Status: alpha.** The cascade, the investigation agent, the learning loop
> and hardware-aware setup work and are tested; phone alerts with teaching
> buttons are built and tested against a fake Telegram, and a real-phone check
> is next. Chat follows (see [Roadmap](#roadmap)). Watchwise has not yet been
> evaluated on a long-running real deployment; every number below is labeled
> with how it was produced.

---

## Try it in two minutes (no cameras, no GPU, no API key)

```bash
git clone https://github.com/PAOPAOtomato/watchwise.git && cd watchwise
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[anthropic]"

watchwise demo                     # a simulated day through the full cascade
```

`watchwise demo` runs 60 alerts through the real pipeline twice: once cold, then
again after a simulated reviewer has corrected what reached them. A simulated
NVR and stand-in models are used, so the output measures routing, not
perception:

| Simulated day (`watchwise demo`) | Before corrections | After |
|---|---|---|
| Alerts reaching a human | 38 of 60 | 6 of 60 |
| Alerts resolved with no model call | 22 | 54 |
| Genuine alerts hidden | 0 | 0 |

## Set it up for your hardware

```bash
watchwise doctor     # read-only: what is this machine, what is running, what fits
watchwise init       # writes config/watchwise.yml for that profile (shows a plan first)
watchwise check      # talks to each configured model once: latency, tool calling
watchwise serve      # runs against your Frigate (MQTT + HTTP API)
```

`doctor` detects your accelerator (NVIDIA / AMD / Apple Silicon / none), local
model servers (Ollama, LM Studio, llama.cpp, vLLM), Frigate and MQTT, and which
cloud API keys are present (presence only, never values). It then recommends a
profile:

| Profile | For | Models (starting points) | Leaves this machine |
|---|---|---|---|
| `cpu-local` | no usable GPU, everything local | `qwen3-vl:4b` on Ollama for every role | nothing |
| `cpu-hybrid` | no GPU + a cloud key (**recommended without a GPU**) | local `qwen3-vl:2b` screens frames; your cloud model investigates and chats | text only (no images) |
| `nvidia` | NVIDIA, 8-15 GB | `qwen3-vl:8b` for every role | nothing |
| `nvidia-16gb` | NVIDIA, 16 GB+ | `qwen3-vl:8b` screening, `qwen3:14b` reasoning | nothing |
| `apple-silicon` | Mac, 16 GB+ | `qwen3-vl:8b` via native Ollama (Metal) | nothing |
| `apple-silicon-8gb` | Mac, 8 GB | `qwen3-vl:4b` | nothing |
| `amd-rocm` | AMD GPU (Linux) | `qwen3-vl:8b` via Ollama ROCm | nothing |
| `cloud` | nothing to install | your cloud model for every role | images and text |
| `demo` | trying it out | stand-ins, no models | nothing |

Profiles are data ([`watchwise/hw/profiles.yml`](watchwise/hw/profiles.yml)), and
the generated config is meant to be edited: any role can point at any model on
any server. Re-running `init` shows a diff and backs up what it replaces.
Details, memory sizes and per-platform notes: [docs/hardware.md](docs/hardware.md).

**Cloud providers** are picked by which key is in `.env`: Anthropic (Claude),
OpenAI, Google Gemini, Groq, OpenRouter, or anything OpenAI-compatible. On a
CPU-only machine a cloud model is what makes conversation feel instant; in
`cpu-hybrid`, frames still never leave the machine.

**Use a key just for Watchwise.** Put it in `.env` as
`WATCHWISE_<PROVIDER>_API_KEY` (for example `WATCHWISE_OPENAI_API_KEY`). It
takes precedence over the provider's usual variable, so Watchwise never uses a
key that other tools on your machine share, and its spend is easy to cap and
track on the provider's dashboard.

### With Docker (Frigate + MQTT + Ollama + Watchwise)

```bash
watchwise init --deploy          # also writes deploy/.env and deploy/frigate/config.yml
cd deploy && docker compose up -d
```

`deploy/.env` selects the hardware overlays (`docker-compose.nvidia.yml`,
`.amd.yml`, `.apple.yml`, ...), so `docker compose up -d` is the same command on
every machine. Add your cameras to `deploy/frigate/config.yml`.

### Phone alerts (Telegram)

1. In Telegram, message [@BotFather](https://t.me/BotFather), send `/newbot`,
   and copy the token it gives you.
2. Put it in `.env` as `WATCHWISE_TELEGRAM_BOT_TOKEN=<token>`.
3. `watchwise telegram link`, then send `/start` to your bot. Only chats linked
   this way receive alerts or can press buttons.
4. `watchwise telegram test` sends a sample card; `watchwise serve` sends real ones.

Each alert that needs you arrives as one card: a snapshot, what was seen where
and when, one line of *why*, the confidence and which stage decided. While the
agent investigates, the card says so, and it updates itself in place when the
investigation finishes. The buttons feed straight back into memory:

| Button | What it does |
|---|---|
| ✅ Expected | records a human correction; after a few on the same kind of alert, Watchwise holds those back by itself, and the card tells you when that happens |
| 🚨 Real | confirms it was worth an alert |
| 🔍 Look closer | runs a full investigation now (offered when none ran yet) |
| 🔇 Snooze | mutes that camera's cards for an hour; its alerts are still processed and recorded |
| ❓ Why | replies with the evidence: decision, reason, matched memory, the investigation's tools and conclusion |
| ↩️ Undo | takes back your answer, including everything it taught the memory |

Telegram is a cloud relay: card text, and the snapshot unless
`notify.telegram.send_images` is `false`, passes through Telegram's servers.
The bot uses long polling, so no port is opened and no public URL is needed.

---

## How it works

```
 RTSP -> [ Frigate ]  motion -> object detection -> review item
               | MQTT <prefix>/reviews
               v
   GATE (free, no model)        rules - dedupe - learned false-alarm patterns
               | unresolved
               v
   TIER 2  small vision model   benign / suspicious / uncertain + confidence
               |
      confident -> decide and dispatch
      uncertain -> TIER 3 (async, off the alert path)
               v
   TIER 3  tool-calling agent   ReAct loop over forensic tools:
                                similar past alerts, camera health, other
                                cameras' timeline, duty roster, older footage
               v
   DECISION POLICY              suppress / downgrade / alert / escalate
               v
   human correction -> long-term memory -> the gate resolves it free next time
```

* **Cost-aware.** The gate resolves the repetitive majority without any model
  call; the expensive agent sees only what the cheap tiers could not settle.
* **Fails open.** Errors, timeouts, refusals and unparseable model output all
  become "uncertain", which goes to a human. A failure never hides an alert.
* **Every silence is auditable.** Held-back alerts are kept, learned patterns
  carry trust scores and decay, and a pattern that humans start overruling
  loses its authority.
* **Evidence, not vibes.** Every verdict carries a confidence and the full
  investigation trace behind it.
* **Additive.** Watchwise never writes to Frigate. It reads Frigate's HTTP API
  and MQTT, keeps its verdicts in its own database and serves them over its
  own API; removing it leaves Frigate unchanged.

### Model layer

One provider-neutral interface (`watchwise/llm/`) serves every role:

| Provider | Covers |
|---|---|
| `ollama` | local models through Ollama's native API (sets the context length explicitly) |
| `openai_compat` | OpenAI, Groq, Gemini, OpenRouter, LM Studio, llama.cpp, vLLM, OpenVINO Model Server |
| `anthropic` | Claude via the official SDK, with prompt caching and server-side refusal fallback |
| `stub` | deterministic stand-in for tests and demos |

Small local models are untidy, and that is handled once for every provider:
tool calls written into prose are recovered, `<think>` blocks are separated, and
malformed arguments are sent back to the model as errors instead of being run.

---

## Interfaces

| Endpoint | Purpose |
|---|---|
| `GET /health` | per-tier status, queue depth, NVR reachability |
| `GET /metrics` | measured cost funnel, review load, accuracy |
| `POST /alerts/simulate` | push a synthetic alert through the real cascade |
| `POST /alerts/ingest` | accept a Frigate review payload over HTTP instead of MQTT |
| `GET /alerts/{review_id}` | an alert with its full decision history |
| `GET /investigations/{id}` | the complete ReAct trace behind a conclusion |
| `POST /feedback` | a human verdict: this is what closes the learning loop |
| `GET /patterns`, `/patterns/search` | inspect long-term memory |
| `GET /tools`, `POST /tools/{name}` | list and directly invoke forensic tools |
| `GET` / `POST /arming` | arming state |

With `notify.mqtt.enabled`, every verdict is also published on MQTT
(`watchwise/verdicts` and `watchwise/<camera>/verdict`), and `notify.webhook_url`
POSTs each one as JSON, for Home Assistant or any other automation.

The forensic tools are also exposed over **MCP**: in-process to the agent, and
as a stdio server (`watchwise-mcp`, needs `pip install -e ".[mcp]"`) for Claude
Desktop, Claude Code, Cursor or any MCP client.

---

## Roadmap

| | Milestone | Status |
|---|---|---|
| 1 | Cascade, investigation agent, MCP tools, learning loop | done |
| 2 | Run anywhere: model layer, open-model investigator, hardware profiles, `doctor` / `init` / `check` | done |
| 3 | Smart alerts on your phone (Telegram): snapshot, a one-line "why", one-tap Expected / Real / Snooze that teaches the memory, Undo | done (tested against a fake Telegram); daily digest with spot-checks next |
| 4 | Ask Watchwise: chat about your cameras ("why didn't you alert me?", "the cleaner comes Thursdays"), with every change confirmed first | planned |
| 5 | Setup assistant, agent-native setup via SKILL.md, measured hardware benchmarks | planned |
| 6 | Interactive web demo | planned |

## Known limitations

* The stand-in models in `demo` cannot read a scene; they prove routing only.
* Not yet evaluated over a long real deployment; real false-alarm and accuracy
  figures will be published from `GET /metrics` once they exist.
* A learned rule could in principle hide a genuine event with the same
  signature. Mitigations: the corrections floor, trust scores, age decay, and
  demotion rather than deletion. Thresholds are conservative by default.
* On a CPU-only machine, local screening takes seconds per alert. Tier 3 is
  off the alert path, but tier 2 is not.

## Development

```bash
pip install -e ".[dev,anthropic,mcp,agent]"
pytest -q
```

The suite runs against a fake Frigate, a fake model server and a fake Telegram
Bot API (all real ASGI apps), plus recorded hardware-detection output from
machines it never runs on, so it needs no GPU, cameras, bot or API keys. Layout:

```
watchwise/    the package: tier2/ (gate, VLM), tier3/ (agent), tools/, memory/,
              llm/ (model layer), hw/ (detection + profiles), setup/,
              notify/ (alert cards, Telegram), cli.py
tests/        one file per layer
deploy/       docker compose base + hardware overlays, Frigate template
docs/         hardware and model guide
```

## Credits

Watchwise is the agent layer; the hard parts underneath are other people's work:

* [Frigate](https://github.com/blakeblackshear/frigate) (MIT, © Frigate, Inc.):
  stream ingest, motion and object detection, recording, the event database.
  Watchwise only talks to it over its public HTTP API and MQTT.
* [Ollama](https://ollama.com) and the open-weight models it serves.
* [Model Context Protocol](https://modelcontextprotocol.io) and the Anthropic SDKs.

## License

MIT. See [LICENSE](LICENSE).
