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
needed. Its question panel uses authored answers; with an installation, you ask
in Telegram (see [Ask Watchwise](#ask-watchwise)).

Cameras are cheap; attention isn't. Watchwise sits on top of
[Frigate](https://frigate.video), the open-source network video recorder (NVR),
and decides which events need a human. A model-free gate handles configured rules
and trusted recurring patterns, a vision model screens unresolved events, and a
tool-calling agent investigates ambiguous cases. Models run locally by default;
cloud providers are optional. In Telegram you get alert cards that explain
themselves, teach the memory with one tap, and ask questions about what happened;
the feedback API offers the same teaching.

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
non-default configuration; `/setup` in Telegram saves its answers to the file
`serve` was started with. `check` probes screening, tool calling, and the `chat`
role that answers questions; a successful probe is not a camera-accuracy test.

`doctor` detects NVIDIA, AMD, Apple Silicon, or CPU hardware; probes local model
servers and Frigate/MQTT reachability; and checks API-key presence without
printing values. Its profiles are starting points, not hardware certifications:

| Profile | Intended starting point | Models | Model data sent to a cloud provider |
|---|---|---|---|
| `cpu-local` | no usable GPU | `qwen3-vl:4b` via Ollama for all configured roles | none with a local server |
| `cpu-hybrid` | CPU screening plus cloud investigation and chat | local `qwen3-vl:2b`; cloud reasoning model | text and tool results, including chat questions; not screening images |
| `nvidia` | NVIDIA, roughly 8–15 GB | `qwen3-vl:8b-instruct` for all configured roles | none with a local server |
| `nvidia-16gb` | 16 GB class (selected from 15 GB reported) | `qwen3-vl:8b-instruct` for all configured roles, 16k context | none with a local server |
| `apple-silicon` | Mac, roughly 16 GB+ unified memory | `qwen3-vl:8b-instruct` via native Ollama | none with a local server |
| `apple-silicon-8gb` | Mac, roughly 8 GB unified memory | `qwen3-vl:4b` | none with a local server |
| `amd-rocm` | supported AMD GPU on Linux | `qwen3-vl:8b-instruct` via Ollama ROCm | none with a local server |
| `cloud` | no local model server | cloud model for all configured roles | screening images, text, tool results, and chat questions |
| `demo` | routing simulation | deterministic stand-ins | none |

**One model per machine.** The GPU and Apple Silicon profiles use one vision
model for every role (screening, investigation, chat), so only one model has to
stay loaded; validate its context length and memory on your machine. These
profiles also let the model check learned routines (`gate.memory_mode:
model_checked`) and write whole-event summaries. The CPU profiles keep the
model-free gate, write no summaries, and can be slow or exhaust their output
budget.

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
sets `detect.enabled: true`; check this setting in your own configuration.
Replace stream placeholders before starting.

### Phone alerts (Telegram)

1. Message [@BotFather](https://t.me/BotFather) in Telegram, send `/newbot`, and
   copy the token it gives you.
2. Put it in `.env` as `WATCHWISE_TELEGRAM_BOT_TOKEN=<token>`.
3. Run `watchwise telegram link`, then send `/start` to your bot. Only linked
   chats receive cards, use their buttons, or ask questions.
4. Run `watchwise telegram test` to send a sample card, then `watchwise serve`
   for real events. Restart a running service after editing its configuration by
   hand; changes made in Telegram apply at once.

Then send `/setup` to the bot. It asks what each camera looks at (and, with
several cameras, which room it is in), who lives here, what should always alert
you, and what to assume while no arming state is set. It shows the plan and
writes it to your configuration only when you tap ✅ Apply, after backing up the
file; ↩️ Undo puts the old file back. The running service uses the answers at
once. It then offers to add face photos to the library shared by all cameras and
a recognition check: walk past a camera, and it tells you whether Frigate
recognized you.

| Command | What it does |
|---|---|
| `/setup` | describe your cameras, rooms, residents, always-alert rules, and arming default |
| `/arm` | show or switch arming: Home, Away, Night, Off (also `/arm away`) |
| `/face` | list residents' photo counts in the shared Face Library, hints below 5 photos, and other names in the library |
| `/face <name> [camera]` | preview the camera's current frame in Telegram; ✅ Add writes a photo to Frigate's Face Library for all cameras |
| `/new` | start a new conversation with Ask Watchwise |
| `/cancel` | stop a `/setup` in progress |
| `/help` | the buttons and commands |

Cards show a snapshot, camera and time, reason, confidence, and decision stage.
They update in place as screening or investigation finishes. Confidence is a
model or policy score, not a measured probability of correctness. Cameras in the
same room share one card per incident: another camera's event in that room edits
the card instead of buzzing again. A card takes a room's events for five minutes
from its first one; a later event, a more serious one, or one after you answered
the card gets its own card.

When an investigation clears a person while armed Away or Night, the card names
the reason that allowed it: the duty-roster entry with its days and time window,
or the resident recognized by Frigate. A roster match gives the investigation
context; it does not clear a person on its own. During a rostered visit, a face
Frigate cannot match is noted on a held-back event instead of raising it by itself.

| Button | What it does |
|---|---|
| ✅ Expected | records benign feedback and reinforces a pattern; holding back later events also requires the correction, similarity, and trust thresholds below |
| 🚨 Alert me | records suspicious feedback (you want alerts like this) and reduces trust in an influenced false-alarm pattern, then asks “What was wrong?”: your reply becomes the example the vision model looks for |
| 🔍 Look closer | requests a full investigation when one has not already run |
| 🔇 Snooze | mutes the cards of that camera and its room for an hour by default; processing and recording continue |
| ❓ Why | explains the stored decision, matched memory, and available investigation evidence |
| 👤 That was &lt;resident&gt; | on a card raised for an unmatched face, preview a face crop from that event; ✅ Add teaches Frigate's shared Face Library |
| ↩️ Undo | reverses the correction and its count/trust effects; merged descriptive text can remain |
| ▶️ Clip | opens Frigate's review page when `nvr.web_url` is configured |

On a room card, ✅, 🚨 and ↩️ act on every camera's event on the card, so teaching
one view teaches the other.

Set `nvr.web_url` (or `FRIGATE_WEB_URL` in generated configurations) to the
Frigate web address your phone can reach. This is separate from `nvr.base_url`,
which serves backend API requests. The phone still needs network access and any
Frigate login; a link does not proxy video or bypass authentication.

Telegram relays card text, the Clip URL when configured, and snapshots unless
`notify.telegram.send_images` is `false`, as well as your questions, the answers,
and what you type in `/setup`. Long polling needs no incoming public endpoint. By
default, only `alerted` and `escalated_human` decisions start cards; a card
already sent while analyzing can later update to a downgraded result.

### Faces across your cameras

Frigate keeps **one Face Library for all cameras**. Each person's folder holds
face crops; Frigate averages their embeddings into one face embedding for
comparisons from every camera.
Enroll a resident with a clear, front-facing photo from any camera; the enrollment
is available to every camera. Add a few more clear photos with some variety in
camera view or lighting to help recognition. [Frigate's guidance](https://docs.frigate.video/configuration/face_recognition/#creating-a-robust-training-set)
recommends 5–10 images per person as a minimum starting point; photo quality
matters, and enrollment does not guarantee a match in every view. Enable face
recognition in Frigate and list residents through `/setup`.

Send `/face` alone to see each resident's photo count, a prompt for more clear
photos when the count is below 5, and other names in the library. To enroll,
send `/face <name> [camera]`, face the camera alone in good light, and inspect the
current-frame preview. ✅ Add sends that frame to Frigate's registration API to
add the face to the shared library; ✖️ Cancel leaves it unchanged.

A card raised because Frigate matched a face to nobody offers
**👤 That was &lt;resident&gt;** for up to 3 configured residents. Tap a name to
preview an available face crop Frigate tried on that card's event, then check
that it clearly shows the right person. ✅ Add calls Frigate's training API to
move that crop into the resident's Face Library folder; ✖️ Cancel adds nothing.
If Frigate no longer has a crop from the event, use `/face` for a fresh photo.
Both enrollment paths send their previews through Telegram unless
`notify.telegram.send_images` is `false` (then only text and buttons are sent).
Both write to Frigate only after ✅ Add.

### Ask Watchwise

Send the bot an ordinary message, or reply to an alert card to ask about that
event:

* “Anything in the kitchen today?”: a short summary from Watchwise's records,
  with photos when useful.
* “Why didn't you tell me about the 21:31 one?”: the decision, the stage that
  made it, and the evidence.
* “The cleaner comes Thursdays 1–4 pm”: a proposed duty-roster rule.
* “Remove the cleaner rule”: a proposal to remove that duty-roster rule.
* “Tell me when the parcel is picked up”: a proposed watch.

The `chat` role's model answers by looking things up: Watchwise's own records
(every event it judged, with its verdict and summary) and Frigate's event list.
It cannot change anything itself. Adding or removing a duty-roster rule, switching
arming, muting a camera or room, or setting a watch arrives as a proposal with
✅ Confirm and ✖️ Cancel. **Only tapping ✅ Confirm applies the proposed change.**
Typing “confirm”, “yes” or “ok” changes nothing; the bot replies with an instruction
to use the button. Unconfirmed proposals expire after an hour.

New roster rules use your local time zone. You can remove any rule by name; if
several share a name, Watchwise asks which one. A confirmed removal removes only
the selected rule. Roster edits keep a backup of the file. A rule tells the
investigator that a person is expected at those cameras and times, so while
armed it may judge the visit routine instead of asking you. The rule itself does
not clear the person; the investigation can still request your attention.

After 6 quiet hours, your next question starts a fresh conversation. `/new` starts
over at any time. In long conversations, the oldest turns are left out of the
model's prompt to make room within its context budget. Your questions, answers,
and any answer photos pass through Telegram; when the `chat` role uses a cloud
provider, its prompts and retrieved text/tool results also go to that provider.

A watch is checked once per Frigate event on the cameras it names, when the
event ends: the vision model is asked, on a few frames spread across the event,
whether they clearly show what you asked for. A match alerts once and ends the
watch; otherwise it runs for the time you chose (24 hours by default, at most a
week). Watches include residents Frigate recognizes. A matching card says
“👀 You asked to be told when …”; matching your request does not mean the event
was judged suspicious.

## How it works

```text
RTSP -> [ Frigate ] -> motion / object / face detection -> review event
                            | MQTT <prefix>/reviews
                            v
FACES         a resident Frigate recognizes while you're home: held back
GATE          configured rules, dedupe, trusted recurring patterns
                            | unresolved
                            v
TIER 2        vision model: benign / suspicious / uncertain + confidence,
              told the arming state, camera notes, and your always-alert rules
                            | unresolved or uncertain
                            v
TIER 3        asynchronous tool-calling investigation:
              past alerts, camera health, other cameras, duty roster, footage
                            v
POLICY        suppress / downgrade / alert / escalate
                            v
EVENT END     whole-event summary and watches: can raise, never lower
                            v
HUMAN REVIEW  one card per room -> corrections -> memory; questions in chat
```

* **Selective model use.** Events resolved by the gate need no model call;
  unresolved events receive screening and, when needed, investigation.
* **Early notification.** Screening starts on a new Frigate review. By default,
  a slow screen produces an “Analyzing…” card after a 10-second budget; armed-away
  and night modes use a zero-second budget with a 0.5-second dispatch floor.
  These are scheduling settings, not delivery guarantees. A later report
  re-screens only if labels or zones changed.
* **Arming-aware.** The vision model is told the arming state. While armed away
  or for the night, a person in view alerts and is investigated. A benign
  investigation can clear the person only with a matching duty-roster entry or
  a resident recognized by Frigate; that reason appears on the card. A roster
  rule alone does not dismiss an event. Without an arming state, Watchwise assumes
  `armed_away`; switch with `/arm` or set the default in `/setup`.
* **Faces.** With Frigate's face recognition on and residents listed in
  `faces.household` (by `/setup`), a resident Frigate names while you're home is
  held back with no model call. A watch, or an always-alert rule worded “anyone”
  or “any person”, is still asked about them once when the event ends. A face
  Frigate saw but matched to nobody raises a held-back event when it ends, except
  during a rostered visit, when it is noted instead. Recognition runs in Frigate
  against the shared Face Library; a recognized name appears in card text.
* **Rooms.** `cameras.<name>.room` groups cameras into rooms: one card per
  incident per room, a tap teaches every camera on the card, and an
  investigation reads another camera in the same room as another angle, not as
  movement through the property.
* **Model warm-up.** `service.warm_up_models` loads local models at startup and
  keeps the screener warm. Startup and memory pressure can still cause delays.
* **Failure handling.** Screening errors, timeouts, refusals, and unreadable
  replies become uncertain; unresolved investigations are surfaced for human
  review. This does not protect against an incorrect benign judgment or a bad
  gate match.
* **Auditable decisions.** Held-back events remain recorded. Decisions include
  reasons, confidence scores, matched memory, and investigation traces when an
  investigation ran. Patterns retain correction and agreement/disagreement counts.
* **Frigate stays in charge of video.** Watchwise reads Frigate's HTTP API and
  MQTT, writes its own database, and exposes its own API. `/face` enrollment and
  👤 face teaching write to Frigate's Face Library only after ✅ Add; their
  previews pass through Telegram when image sending is enabled. Removing
  Watchwise leaves Frigate's recordings and configuration intact.

### Learning thresholds and safeguards

By default, a false-alarm pattern needs at least **3 corrections**, matching
similarity **0.86**, and trust **0.70** to resolve an event at the gate. Trust is
`(agreements + 1) / (agreements + disagreements + 2)`; correction count alone is
not enough. 🚨 Alert me feedback can lower trust, and Undo reverses its recorded effects.
A 🚨 Alert me on a kind of event (camera, labels, zones) also stops memory from
holding back that kind for 30 days (`gate.real_veto_days`).

The gate's confidence is capped at `0.95` and otherwise equals trust × similarity.
The policy fully suppresses a learned match only at confidence `0.85` or above;
a lower qualifying score downgrades it to a detection. Both normally avoid a new
phone card while retaining the event for inspection. Configured always-alert
labels take precedence over memory and deduplication.

Patterns are keyed by camera, object labels, and zones. On the CPU and cloud
profiles the gate holds back trusted routines itself, without inspecting a frame,
so it cannot tell behaviours apart. On the GPU and Apple Silicon profiles
(`gate.memory_mode: model_checked`) it never does: the vision model compares each
event with descriptions of what you marked expected or alert-worthy, in your own
words when you answered “What was wrong?”. A correction does not edit the duty
roster; a roster rule comes from the chat, with your confirmation. Age decay
affects search ranking; it does **not** expire a pattern or independently remove
its authority to hold back alerts.

### Whole-event descriptions

When an event ends, Watchwise asks the vision model what happened across all of
it, from frames spread over the event (`vlm.summary_frames`; off on the CPU
profiles), and attaches “What happened” to the card and decision history. A GenAI
review summary sent by Frigate is used the same way. With
`decision.summary_can_raise: true` (the default), a suspicious or critical
summary raises a held-back event to an alert; it cannot lower a decision. The
summary arrives after Frigate closes the review, some seconds after the activity
stops; an early verdict may describe only a snapshot.

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
Desktop, Claude Code, Cursor, or another MCP client. Ask Watchwise in Telegram
uses its own tool set; it has no HTTP endpoint.

## Run the checks

```bash
pip install -e ".[dev,anthropic,mcp,agent]"
pytest -q
```

The automated Python suite uses fake Frigate, model, and Telegram Bot API servers
(real ASGI apps), plus hardware-detection fixtures. It requires no GPU, cameras,
bot, or API keys. Tests cover routing, feedback effects and Undo, notification
updates, rooms, the `/setup` conversation and its file changes, chat answers and
confirmed changes (with a scripted model), warm-up and timeout behavior,
recording URLs, and summary escalation.
These checks exercise application behavior; they do not measure visual accuracy.

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
              setup/, notify/, chat/, cli.py
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
