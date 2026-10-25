# Watchwise

[English](README.md) | [简体中文](README.zh-CN.md)

**An AI guard for the cameras you already have.** Fewer distracting alerts,
decisions backed by evidence, and an agent that learns from human corrections.

[![ci](https://github.com/PAOPAOtomato/watchwise/actions/workflows/ci.yml/badge.svg)](https://github.com/PAOPAOtomato/watchwise/actions/workflows/ci.yml)
![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

**[Explore Watchwise →](https://watchwise-six-ebon.vercel.app/#experience)**

Select illustrated camera events, inspect their evidence, and see how a correction
changes a later matching event. The website uses a deterministic browser scenario;
its counts and confidence values belong to that story. No cameras or API keys are
needed. Its question panel uses authored answers; installed conversational chat
is still [planned](#roadmap).

Cameras are cheap; attention isn't. Watchwise sits on top of
[Frigate](https://frigate.video), the open-source network video recorder (NVR),
and decides which events need a human. A model-free gate handles configured rules
and trusted recurring patterns, a vision model screens unresolved events, and a
tool-calling agent investigates ambiguous cases. Models run locally by default;
cloud providers are optional. You can inspect the decision history and teach the
memory through Telegram or the feedback API.

> **Live test completed, October 22–23, 2026.** A real USB camera fed Frigate;
> Watchwise processed its events, delivered Telegram cards to a phone, and learned
> from corrections to hold back a later matching event without another model call.
> This verifies that specific end-to-end flow, not long-term reliability,
> detection accuracy, or compatibility with every camera. See the scope below.

## What has been verified

The live setup used an external USB camera on a Windows laptop, go2rtc 1.9.14,
RTSP over Tailscale, Frigate 0.18.2 with an OpenVINO CPU detector, and a Mosquitto
broker. Watchwise ran on an Ubuntu CPU VM; Ollama 0.40.8 served
`qwen3-vl:8b-instruct` (screening) and `qwen3:14b` (investigation) on a separate
NVIDIA L4 GPU VM with 24 GB VRAM. These were operator-managed cloud VMs, so this
particular test was not an entirely on-premises deployment. The USB camera model
was not recorded; it does not establish support for a particular camera brand.

| Capability | Evidence and scope |
|---|---|
| Live ingestion and screening | Real indoor person events traveled through Frigate MQTT and HTTP into Watchwise. Early screening used a snapshot when recordings were not yet available. |
| Phone alerts | A real Telegram phone received cards, including an initial “Analyzing…” state followed by a verdict or investigation update. |
| Learning from corrections | Expected and Real feedback affected pattern trust. A later event with the same camera/label/zone signature was downgraded by the gate, with no model call or new phone card. After a memory reset, repeated Expected feedback also taught a new pattern. |
| Deeper investigation | Ambiguous live events invoked the tool-calling investigator. The test does not validate every forensic tool or the correctness of every conclusion. |
| Recording links | Cards displayed a Clip button. Playback on the phone remains unverified: that phone lacked access to the private network hosting Frigate. |
| Frigate review summaries | Parsing, persistence, card text, and raise-only decisions are implemented and covered by automated tests. Live generation of longer-event summaries exceeded the model context; Frigate GenAI was disabled in that setup. |
| Other platforms and providers | Profiles and provider adapters exist. Automated stand-ins and hardware fixtures do not establish live compatibility for every listed option. |

The test also exposed unresolved limits: memory does not distinguish different
behaviors sharing a signature, the screener prompt does not include arming state,
and an early snapshot does not describe an entire event. These are detailed under
[Known limitations](#known-limitations). No real-world accuracy rate, false-alarm
reduction percentage, or latency guarantee is claimed.

### Source and availability

The installation instructions and feature descriptions below refer to the
published Python implementation at
[`7e7c4aa`](https://github.com/PAOPAOtomato/watchwise/commit/7e7c4aa9f0a6c1c8ade631fd06209fafcaaecf3f).
As of October 24, separate, uncommitted implementation work adds arming-aware
screening, a recent-Real veto for gate memory, and Watchwise-generated event
summaries. Those changes are still being verified and are **not included** in
this documentation/website publication or the installation described here.
They have not been established by the October 22–23 live test. Behavior-aware
matching remains further work.

## Try the routing simulation (no cameras, no GPU, no API key)

Use Python 3.11 or newer:

```bash
git clone https://github.com/PAOPAOtomato/watchwise.git && cd watchwise
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[anthropic]"

watchwise demo                    # a simulated day through the full cascade
```

`watchwise demo` runs 60 synthetic alerts through the real pipeline twice: once
without learned patterns, then again after a simulated reviewer supplies
corrections. It uses a fake NVR and stand-in models. These reproducible results
measure routing, not visual perception or real-world security performance:

| Simulated day (`watchwise demo`) | Before corrections | After |
|---|---|---|
| Alerts reaching a human | 38 of 60 | 6 of 60 |
| Alerts resolved with no model call | 22 | 54 |
| Scenario-designated genuine alerts hidden | 0 | 0 |

## Set it up for your hardware

Run the CLI from the repository root so it can read `.env` and the generated
configuration:

```bash
watchwise doctor     # read-only hardware, server, and provider-key detection
watchwise init       # review a plan, then write config/watchwise.yml
```

Start the chosen model server and pull the models listed by `init`. Configure
Frigate's HTTP API and MQTT connection in `config/watchwise.yml` or through the
environment variables in [.env.example](.env.example). With an existing Frigate
installation, confirm object detection is enabled and `mqtt.topic_prefix` agrees
on both sides. Then run:

```bash
watchwise check      # calls each configured model; cloud calls may incur charges
watchwise serve      # connect to Frigate and process its review events
```

Use `--config <path>` on `check`, `serve`, and the Telegram commands when using a
non-default configuration. `check` probes screening and tool calling; a successful
probe is not a camera-accuracy test. A configured `chat` role can be probed even
though the installed conversational interface is not implemented.

`doctor` detects NVIDIA, AMD, Apple Silicon, or CPU hardware; probes local model
servers and Frigate/MQTT reachability; and checks API-key presence without
printing values. Its profiles are starting points, not hardware certifications:

| Profile | Intended starting point | Models | Model data sent to a cloud provider |
|---|---|---|---|
| `cpu-local` | no usable GPU | `qwen3-vl:4b` via Ollama for all configured roles | none with a local server |
| `cpu-hybrid` | CPU screening plus cloud investigation | local `qwen3-vl:2b`; cloud reasoning model | text and tool results, not screening images |
| `nvidia` | NVIDIA, roughly 8–15 GB | `qwen3-vl:8b-instruct` for all configured roles | none with a local server |
| `nvidia-16gb` | 16 GB class (selected from 15 GB reported), with the caveat below | `qwen3-vl:8b-instruct` screening, `qwen3:14b` reasoning | none with a local server |
| `apple-silicon` | Mac, roughly 16 GB+ unified memory | `qwen3-vl:8b-instruct` via native Ollama | none with a local server |
| `apple-silicon-8gb` | Mac, roughly 8 GB unified memory | `qwen3-vl:4b` | none with a local server |
| `amd-rocm` | supported AMD GPU on Linux | `qwen3-vl:8b-instruct` via Ollama ROCm | none with a local server |
| `cloud` | no local model server | cloud model for all configured roles | screening images, text, and tool results |
| `demo` | routing simulation | deterministic stand-ins | none |

**The `nvidia-16gb` name overstates what fits.** The configured screener and
investigator together exceeded 16 GB in the L4 test; that test used a 24 GB GPU.
The selector has not yet been corrected. On a smaller GPU, use a smaller/single
model configuration and validate it on that machine; do not assume both models
can stay loaded. CPU screening can also be slow or exhaust its output budget.

Profiles live in [`watchwise/hw/profiles.yml`](watchwise/hw/profiles.yml).
Any role can point at a different model or server. Re-running `init` shows a diff
and backs up replaced configuration; review it before applying so customized
settings are preserved. See [docs/hardware.md](docs/hardware.md) for platform
setup and [config.example.yml](config.example.yml) for configuration details.

Cloud presets include Anthropic, OpenAI, Google Gemini, Groq, and OpenRouter;
other OpenAI-compatible endpoints can be configured manually. Validate the model
identifier and required capabilities with `watchwise check`. Preset availability
does not mean every provider has passed a live camera test.

**Use a key just for Watchwise.** Put it in `.env` as
`WATCHWISE_<PROVIDER>_API_KEY`, such as `WATCHWISE_OPENAI_API_KEY`. During setup it
takes precedence over the provider's usual variable; without a dedicated key,
setup can select the usual variable. Inspect `api_key_env` in the generated
configuration. A configured missing key is reported rather than replaced with
another credential.

The table describes model traffic. Pointing a role at a remote server sends its
inputs there. Telegram, webhooks, and MQTT outputs have their own data paths;
local inference alone does not make all notifications local.

### With Docker (Frigate + MQTT + Ollama + Watchwise)

```bash
watchwise init --deploy          # also generates the Compose selection and Frigate config
# Edit deploy/frigate/config.yml: add your camera stream and enable detection.
cd deploy
# Start the selected services after reviewing the generated configuration:
docker compose up -d
```

`deploy/.env` selects the hardware overlays (`docker-compose.nvidia.yml`,
`.amd.yml`, `.apple.yml`, and others). Pull the selected models on the model
server used by that overlay; on macOS, Ollama runs natively. The Frigate template
now explicitly sets `detect.enabled: true`; previously generated configurations
need that setting checked manually. Replace stream placeholders before starting.
The live test used Frigate/MQTT in Docker and Watchwise in a Python environment;
it did not validate every complete Compose overlay.

### Phone alerts (Telegram)

1. Message [@BotFather](https://t.me/BotFather) in Telegram, send `/newbot`, and
   copy the token it gives you.
2. Put it in `.env` as `WATCHWISE_TELEGRAM_BOT_TOKEN=<token>`.
3. Run `watchwise telegram link`, then send `/start` to your bot. Only linked
   chats receive cards or can use their buttons.
4. Run `watchwise telegram test` to send a sample card, then `watchwise serve`
   for real events. Restart a running service after changing its configuration.

Cards show a snapshot, camera and time, reason, confidence, and decision stage.
They update in place as screening or investigation finishes. Confidence is a
model or policy score, not a measured probability of correctness.

| Button | What it does |
|---|---|
| ✅ Expected | records benign feedback and reinforces a pattern; holding back later events also requires the correction, similarity, and trust thresholds below |
| 🚨 Real | records suspicious feedback and reduces trust in an influenced false-alarm pattern |
| 🔍 Look closer | requests a full investigation when one has not already run |
| 🔇 Snooze | mutes that camera's cards for an hour by default; processing and recording continue |
| ❓ Why | explains the stored decision, matched memory, and available investigation evidence |
| ↩️ Undo | reverses the correction and its count/trust effects; merged descriptive text can remain |
| ▶️ Clip | opens Frigate's review page when `nvr.web_url` is configured |

Set `nvr.web_url` (or `FRIGATE_WEB_URL` in generated configurations) to the
Frigate web address your phone can reach. This is separate from `nvr.base_url`,
which serves backend API requests. The phone still needs network access and any
Frigate login; a link does not proxy video or bypass authentication.

Telegram relays card text, the Clip URL when configured, and snapshots unless
`notify.telegram.send_images` is `false`. Long polling needs no incoming public
endpoint. By default, only `alerted` and `escalated_human` decisions start cards;
a card already sent while analyzing can later update to a downgraded result.

## How it works

```text
RTSP -> [ Frigate ] -> motion / object detection -> review event
                            | MQTT <prefix>/reviews
                            v
GATE          configured rules, dedupe, trusted recurring patterns
                            | unresolved
                            v
TIER 2        vision model: benign / suspicious / uncertain + confidence
                            | unresolved or uncertain
                            v
TIER 3        asynchronous tool-calling investigation:
              past alerts, camera health, other cameras, duty roster, footage
                            v
POLICY        suppress / downgrade / alert / escalate
                            v
HUMAN REVIEW  corrections -> pattern memory -> later matching decisions
```

* **Selective model use.** Events resolved by the gate need no model call;
  unresolved events receive screening and, when needed, investigation.
* **Early notification.** Screening starts on a new Frigate review. By default,
  a slow screen produces an “Analyzing…” card after a 10-second budget; armed-away
  and night modes use a zero-second budget with a 0.5-second dispatch floor.
  These are scheduling settings, not delivery guarantees. A later report
  re-screens only if labels or zones changed.
* **Model warm-up.** `service.warm_up_models` loads local models at startup and
  keeps the screener warm. Startup and memory pressure can still cause delays.
* **Failure handling.** Screening errors, timeouts, refusals, and unreadable
  replies become uncertain; unresolved investigations are surfaced for human
  review. This does not protect against an incorrect benign judgment or a bad
  gate match.
* **Auditable decisions.** Held-back events remain recorded. Decisions include
  reasons, confidence scores, matched memory, and investigation traces when an
  investigation ran. Patterns retain correction and agreement/disagreement counts.
* **Read-only Frigate integration.** Watchwise reads Frigate's HTTP API and MQTT,
  writes its own database, and exposes its own API. Removing it leaves Frigate's
  recordings and configuration intact.

### Learning thresholds and safeguards

By default, a false-alarm pattern needs at least **3 corrections**, matching
similarity **0.86**, and trust **0.70** to resolve an event at the gate. Trust is
`(agreements + 1) / (agreements + disagreements + 2)`; correction count alone is
not enough. Real feedback can lower trust, and Undo reverses its recorded effects.

The gate's confidence is capped at `0.95` and otherwise equals trust × similarity.
The policy fully suppresses a learned match only at confidence `0.85` or above;
a lower qualifying score downgrades it to a detection. Both normally avoid a new
phone card while retaining the event for inspection. Configured always-alert
labels take precedence over memory and deduplication.

Patterns are keyed by camera, object labels, and zones. The gate does not inspect
a frame, recognize an identity, or learn a person's schedule. A correction does
not edit the duty roster. Age decay affects search ranking; it does **not** expire
a pattern or independently remove its authority to hold back alerts.

### Whole-event descriptions

When Frigate supplies a GenAI review summary, Watchwise can attach “What happened”
to the card and decision history. With `decision.summary_can_raise: true` (the
default), a suspicious or critical summary raises a held-back event to an alert;
it cannot lower a decision. This integration is implemented and tested with
stand-ins. Longer-event generation failed in the live setup because the image
input exceeded the model's context. Watchwise's own whole-event summarizer is
being implemented separately; it is not in the published baseline above, whose
early verdict can describe only a snapshot.

### Model layer

One provider-neutral interface (`watchwise/llm/`) serves the configured roles:

| Provider | Covers |
|---|---|
| `ollama` | Ollama's native API with explicit context length |
| `openai_compat` | OpenAI-compatible APIs, including configured OpenAI, Groq, Gemini, OpenRouter, LM Studio, llama.cpp, vLLM, and OpenVINO Model Server endpoints |
| `anthropic` | Claude via the official SDK, with prompt caching and refusal fallback support |
| `stub` | deterministic stand-ins for tests and routing simulations |

The model layer recovers supported tool calls written into prose, separates
`<think>` blocks, and returns malformed tool arguments as errors rather than
executing them. Output truncation and empty or unparseable replies are reported
by the screener and `watchwise check`.

## Interfaces

| Endpoint | Purpose |
|---|---|
| `GET /health` | per-tier status, model warm-up, queue depth, NVR reachability |
| `GET /metrics` | routing/cost counters and agreement statistics for reviewed events; not an unbiased accuracy benchmark |
| `POST /alerts/simulate` | push a synthetic alert through the real cascade |
| `POST /alerts/ingest` | accept a Frigate review payload over HTTP instead of MQTT |
| `GET /alerts/{review_id}` | an event with its decision history |
| `GET /investigations/{id}` | the investigation trace behind a conclusion |
| `POST /feedback` | submit a human verdict to the learning loop |
| `GET /patterns`, `/patterns/search` | inspect long-term memory |
| `GET /tools`, `POST /tools/{name}` | list and invoke forensic tools |
| `GET` / `POST /arming` | inspect or change arming state |

With `notify.mqtt.enabled`, verdicts are published on `watchwise/verdicts` and
`watchwise/<camera>/verdict`. `notify.webhook_url` POSTs verdicts as JSON to your
chosen destination, such as a Home Assistant integration.

Forensic tools are also exposed through **MCP**: in-process to the agent and as
a stdio server (`watchwise-mcp`, requiring `pip install -e ".[mcp]"`) for Claude
Desktop, Claude Code, Cursor, or another MCP client. These tools do not constitute
an installed Watchwise chat interface.

## Roadmap

| Milestone | Status |
|---|---|
| Cascade, investigation agent, forensic/MCP tools, learning loop | implemented; the core live camera → phone → correction → later gate flow verified |
| Model adapters, hardware profiles, `doctor` / `init` / `check` | implemented; broader hardware/provider validation remains |
| Telegram cards, teaching buttons, Undo, Snooze, Why, Look closer | implemented and covered by automated tests; live delivery and Expected/Real learning verified |
| Clip links and Frigate summary ingestion | implemented; phone playback remains unverified, live longer-event summary generation failed |
| Arming-aware screening and a recent-Real veto for gate memory | local implementation in progress; not part of the published installation above |
| Behavior-aware memory matching | planned |
| Watchwise whole-event summaries | local implementation in progress; not published or live-verified |
| Daily digest and review spot-checks | planned |
| Ask Watchwise: installed conversational chat, with confirmed configuration changes | planned; configured model roles and website question responses are not this interface |
| Setup assistant, agent setup through SKILL.md, broader measured hardware benchmarks | planned |
| [Interactive product website](https://watchwise-six-ebon.vercel.app/#experience) | available; illustrated scenario and browser-local state |

## Known limitations

These apply to the published baseline identified under [Source and availability](#source-and-availability).

* The live test was a short, single-camera indoor exercise. It does not establish
  long-running reliability, multi-camera performance, or general detection accuracy.
  Stand-in models in `watchwise demo` cannot read a real scene.
* Memory can hold back a genuine event with the same signature as an expected one.
  Different gestures or behaviors are not separated, and Real feedback is not a
  hard veto on later suppression. Trust thresholds, review records, and Undo help
  with correction but do not eliminate this risk.
* The screener is not told the arming state. An ordinary-looking indoor person
  can be judged benign even when armed away. Arming affects provisional timing
  and is available to investigation tools; it does not currently fix that prompt.
  Without an arming file, the default is `armed_away`.
* Early screening may use a single snapshot. A later end report with unchanged
  labels/zones does not trigger another screen. Whole-event summaries are not
  guaranteed, and camera placement or blur can affect judgments.
* Investigation can be slow on real models. Warm-up helps avoid cold starts but
  does not guarantee model residency or response time; the two-model
  `nvidia-16gb` configuration can exceed its nominal memory class.
* CPU screening and thinking models can exhaust their time or output budget.
  `watchwise check --role screener` helps diagnose this on the chosen machine.
* Telegram delivery needs network access. Recording links additionally require
  the phone to reach and authenticate to Frigate; that playback check is outstanding.
* Installed chat, automatic schedule editing, and Watchwise-generated whole-event
  summaries are unavailable in that baseline. Summary work remains local and
  unpublished. The website's question panel sends no input to a model.

## Development

```bash
pip install -e ".[dev,anthropic,mcp,agent]"
pytest -q
```

The automated Python suite uses fake Frigate, model, and Telegram Bot API servers
(real ASGI apps), plus hardware-detection fixtures. It requires no GPU, cameras,
bot, or API keys. Tests cover routing, feedback effects and Undo, notification
updates, warm-up and timeout behavior, recording URLs, and summary escalation.
These checks supplement the live observations; they do not replace them.

The frontend is isolated from the Python application:

```bash
cd website
npm ci
npm run build
npx playwright install chromium
npm test
```

Use Node.js 24 as in CI. See [website/README.md](website/README.md) for preview,
browser checks, and maintenance of the existing Vercel deployment.

```text
watchwise/    Python package: tier2/, tier3/, tools/, memory/, llm/, hw/,
              setup/, notify/, cli.py
tests/        Python tests by layer
deploy/       Docker Compose base, hardware overlays, Frigate template
docs/         hardware and model guide
website/      static product experience, local assets, browser tests
```

## Credits

Watchwise is the agent layer; the foundations are other people's work:

* [Frigate](https://github.com/blakeblackshear/frigate) (MIT, © Frigate, Inc.):
  stream ingestion, motion/object detection, recording, and the event database.
  Watchwise communicates through its public HTTP API and MQTT.
* [Ollama](https://ollama.com) and the open-weight models it serves.
* [Model Context Protocol](https://modelcontextprotocol.io) and the Anthropic SDKs.
* The website bundles Manrope under the
  [SIL Open Font License](website/public/licenses/manrope.txt) and uses original
  SVG illustrations, not live camera footage.

## License

MIT. See [LICENSE](LICENSE).
