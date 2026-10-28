# Hardware and models

Watchwise configures three model roles:

| Role | Job | What the model needs |
|---|---|---|
| `screener` | tier 2: look at 1-3 frames, answer benign / suspicious / uncertain; when an event ends, describe all of it and check watches | vision, reliable JSON |
| `investigator` | tier 3: a tool-calling agent for the hard cases (async) | reliable tool calling |
| `chat` | Ask Watchwise in Telegram: answer questions with lookup tools, propose changes | reliable tool calling |

On the local profiles one vision model plays all three roles.

A **profile** is a starting point for which model plays which role on a given
kind of machine. It is not a compatibility guarantee or a measurement.
`watchwise doctor` picks one; `watchwise init`
writes it to `config/watchwise.yml`; `watchwise check` tells you how it actually
responds to a synthetic image and a tool-call probe on your machine. It does
not measure scene accuracy or sustained camera capacity. Everything is editable
afterwards. `init` prints model-pull commands; it does not install a model server
or download models for you.

## How a profile is chosen

`watchwise doctor` looks for, in order:

1. **NVIDIA** (via `nvidia-smi`): the current catalog selects `nvidia-16gb`
   from 15 GB of reported memory and `nvidia` from 7 GB. Both use one
   `qwen3-vl:8b-instruct` for every role; `nvidia-16gb` gives it a 16k context,
   `nvidia` 8k.
2. **Apple Silicon** (unified memory): 15 GB+ gives `apple-silicon`, otherwise `apple-silicon-8gb`.
3. **AMD** (via `rocm-smi`, Linux) gives `amd-rocm`.
4. Otherwise the CPU profiles: `cpu-hybrid` if a cloud API key is set, `cpu-local` if not.

A GPU too small for the GPU profiles (or whose memory cannot be read) is
treated as no GPU, and `doctor` says so. Integrated Intel graphics and NPUs are
reported but not used for local LLMs by default, because Ollama does not
accelerate them; see [Intel iGPU / NPU](#intel-igpu--npu) for the alternatives.

Override any choice: `watchwise init --profile nvidia`, `--cloud anthropic`,
`--cloud-model <id>`.

## Memory: what fits

Approximate download size at Ollama's default quantization. Check the real
numbers with `ollama show <model>`; the context window adds more on top.

| Model | Approx. size | Used by |
|---|---|---|
| `qwen3-vl:2b` | ~2 GB | `cpu-hybrid` screener |
| `qwen3-vl:4b` | ~3-4 GB | `cpu-local`, `apple-silicon-8gb` |
| `qwen3-vl:8b-instruct` | ~6 GB | `nvidia`, `nvidia-16gb`, `apple-silicon`, `amd-rocm` |

One vision+tools model (`qwen3-vl`) serves every local role, so only one model
has to stay loaded. The context window comes on top of the download size: a
16k context needs noticeably more memory than 8k. Leave room for Frigate and for
concurrent requests, and check what is actually loaded with `ollama ps`.

The `-instruct` tags answer directly. The plain `qwen3-vl` tags used by the CPU
profiles are thinking models: they reason before they answer, which costs time
and output tokens (see [Tuning for a slow machine](#tuning-for-a-slow-machine)).

**Whole-event summaries.** When an event ends, Watchwise asks its screener model
what happened across the whole event, from a few frames spread over it
(`vlm.summary_frames`, 4 by default; 0 on the CPU profiles). Frigate's own GenAI
review summaries are also used when Frigate sends them, but they are not needed:
Frigate's requests for longer events can exceed a small model's context, so on a
local model you can leave Frigate's GenAI off.

**Context length matters.** Watchwise sets `num_ctx` on every Ollama request
(`context_window` in the config). Ollama's default is small enough to cut a
tool-calling prompt short silently, which shows up as a model that "ignores"
its tools.

## Per platform

### Windows

* Install Ollama (native app), or run it in Docker Desktop (WSL2 backend).
* NVIDIA GPUs work in Docker Desktop through WSL2; the `nvidia` overlay
  requests them.
* Intel/AMD integrated graphics are not passed into Docker Desktop
  containers, so `/dev/dri` overlays do not apply on Windows.

### Linux

* NVIDIA: install the NVIDIA Container Toolkit; the `nvidia` overlay requests the GPU.
* AMD: the `amd` overlay passes `/dev/kfd` and `/dev/dri` and uses Ollama's ROCm image.
* Intel iGPU/NPU for Frigate's OpenVINO detector: add
  `docker-compose.intel.yml` to `COMPOSE_FILE` in `deploy/.env`.

### macOS (Apple Silicon)

* Docker containers on a Mac cannot use the GPU, so Ollama must run
  natively (install the Ollama app). The `apple` overlay points Watchwise's
  container at `host.docker.internal:11434`.
* Frigate uses the `stable-standard-arm64` image; for detection, use Frigate's
  Apple Silicon detector client (it also runs natively).

### Intel iGPU / NPU

Ollama does not reliably accelerate Intel graphics. If you want them for
models, run a server that does and point a role at it as `openai_compat`:

* **llama.cpp server** built with Vulkan or SYCL: `base_url: http://localhost:8080`
* **OpenVINO Model Server**: its OpenAI-compatible endpoint
* **LM Studio** (Vulkan runtime): `base_url: http://localhost:1234`

For Frigate's object detector, OpenVINO already uses Intel CPU, GPU and NPU
(`device: CPU | GPU | NPU`).

## Using any model server

Any role can point at any endpoint:

```yaml
models:
  lmstudio:
    provider: openai_compat
    base_url: http://localhost:1234        # with or without /v1
    model: qwen2.5-vl-7b-instruct
  vllm:
    provider: openai_compat
    base_url: http://gpu-box:8000/v1
    model: Qwen/Qwen3-14B
roles:
  screener: lmstudio
  investigator: vllm
  chat: vllm
```

Then run `watchwise check`: it sends a test image to the screener and a
one-tool conversation to the other roles, and reports latency and whether tool
calling actually worked. The `chat` probe tests the model that answers
questions in Telegram.

## Cloud models and privacy

`watchwise init` picks a cloud provider by which key is in `.env`, in this
order: Anthropic, OpenAI, Gemini, Groq, OpenRouter. The suggested model for
each is in [`watchwise/hw/profiles.yml`](../watchwise/hw/profiles.yml); for
non-Anthropic providers it is a suggestion that `watchwise check` verifies, and
`init` asks you to confirm it.

Each provider can be given **a key just for Watchwise**:
`WATCHWISE_OPENAI_API_KEY`, `WATCHWISE_ANTHROPIC_API_KEY`, and so on. When
set, it wins over the provider's usual variable (`OPENAI_API_KEY`, ...), and
the generated config names it, so Watchwise never quietly uses a key that other
tools share. If a configured key is missing, `watchwise check` says which
variable to set instead of falling back to another credential. Most providers
let you create a separate project or key with its own spending limit; doing so
for Watchwise keeps experiments bounded.

What model requests leave your machine depends on the profile, and `init`
prints it. These rows assume local model endpoints; a remote Ollama or
OpenAI-compatible server receives the inputs for the role assigned to it.

| Profile | Sent to the cloud |
|---|---|
| `cpu-hybrid` | text only: detector metadata, tool results, frame *descriptions* made by the local model, and chat questions with the records looked up to answer them. Never images. |
| `cloud` | frames for screening, summaries and watches, plus the above |
| local-model profiles | no model requests to a cloud provider when endpoints stay local |

What you tell `/setup` (camera notes, always-alert rules) and your watches
become part of the prompts, so they go wherever the screener and investigator
models run.

Notifications are separate from model routing: Telegram relays card text and,
unless `notify.telegram.send_images: false`, snapshots, as well as your
questions, the chat's answers and what you type in `/setup`. A configured
`nvr.web_url` also appears in clip links sent through Telegram. Face recognition
runs in Frigate; recognized names appear in card text. Webhooks and remote MQTT
brokers receive the data sent to them.

Notes:

* **Provider verification:** OpenAI model calls were checked against the real
  API. Anthropic integration has automated transport tests; suggested Gemini,
  Groq and OpenRouter models have not been verified against their live APIs.
  An OpenAI-compatible transport does not establish that every suggested model
  supports the required vision, parameters and tool calls.
* **Claude**: the adapter supports prompt caching and configured server-side
  refusal fallback on supported models. A remaining refusal becomes
  "uncertain"; it does not clear the event as benign. Effort is configured per
  endpoint with `effort:`.
* Cloud pricing, rate limits and data-use terms belong to the selected provider.
  Review those terms before sending real camera data; `watchwise check` and
  `watchwise demo --real-models` can make billable calls.
* Keys live only in `.env` (git-ignored). The config names the variable, never the key.

## Tuning for a slow machine

In `config/watchwise.yml`:

* `vlm.frames: 1` and `vlm.max_image_edge: 448`: fewer and smaller images per alert.
* The screener's output cap is its model's `max_tokens`. Thinking models (the
  plain `qwen3-vl` tags of the CPU profiles) reason before they answer, so keep
  it at 1024 or more: `watchwise check` warns when a reply is cut off before the
  verdict.
* `vlm.max_concurrency: 1`: one screening at a time; bursts queue instead.
* `agent.max_iterations: 4`: shorter investigations.
* `gate.never_screen_labels: [cat, dog]`: never spend a model call on these.
* The `cpu-hybrid` profile keeps `screener` local and points `investigator`
  and `chat` at a cloud model. It does not make local CPU
  screening faster. In CPU tests, verdicts could take tens of seconds and
  thinking-model replies sometimes exhausted their output cap or timed out.
* Model warm-up runs at service startup by default. While screening runs,
  provisional "Analyzing…" cards can reach you first; warm-up and asynchronous
  investigation do not guarantee a quick final verdict.

## More cameras and phones

Frigate decodes every camera's stream on its own machine, so the number of
cameras is bounded by that machine's CPU (or its hardware decoder), and by the
upload bandwidth when the cameras and Frigate are in different places. Per
camera, at the source:

* keep the stream at 1280×720 and 10–15 fps; Frigate detects at 5 fps, and
  higher resolutions and frame rates cost decoding time and bandwidth for
  nothing;
* cap the bitrate where the camera or encoder allows it: go2rtc's software
  encoder sets no cap of its own, while phone streaming apps usually let you set
  one.

**Phones as cameras.** An app that pushes its camera as RTMP can feed a go2rtc
on the same network, which Frigate then reads like any other stream. For
example, with Larix Broadcaster on an iPhone:

```yaml
# go2rtc.yaml on the machine the phones can reach
rtmp:
  listen: ":1935"          # allow it from the local network only
streams:
  phone_kitchen:           # empty: the phone's push fills it
```

In the app, set the URL to `rtmp://<that machine>:1935/phone_kitchen/live`:
go2rtc takes the stream name from the first part of the path. Use H.264, not
HEVC. A phone keeps sending only while the app is open, the screen is on and no
call comes in, so keep it on a charger with auto-lock off.

**In Watchwise**, run `/setup` again after adding cameras: it asks what each one
looks at and which room it is in. Cameras in one room share one card per
incident, and each camera recognizes residents best with a face photo from its
own angle (`/face <name> <camera>`).

