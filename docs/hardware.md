# Hardware and models

This guide describes the published Python baseline identified in the
[verification notes](verification.md#source-and-availability). Concurrent local implementation
work is separate from the installation documented here.

Watchwise configures three model roles. Screening and investigation are
implemented; `chat` reserves a model for the planned installed chat service:

| Role | Job | What the model needs |
|---|---|---|
| `screener` | tier 2: look at 1-3 frames, answer benign / suspicious / uncertain | vision, reliable JSON |
| `investigator` | tier 3: a tool-calling agent for the hard cases (async) | reliable tool calling |
| `chat` | model configuration and connectivity probe; installed conversation is planned | tool calling |

A **profile** is a starting point for which model plays which role on a given
kind of machine. It is not a compatibility guarantee or a measurement.
`watchwise doctor` picks one; `watchwise init`
writes it to `config/watchwise.yml`; `watchwise check` tells you how it actually
responds to a synthetic image and a tool-call probe on your machine. It does
not measure scene accuracy or sustained camera capacity. Everything is editable
afterwards. `init` prints model-pull commands; it does not install a model server
or download models for you.

The successful live camera-to-phone test on October 22–23, 2026 used an external
USB webcam through go2rtc and Tailscale, Frigate 0.18.2 with an OpenVINO CPU detector, and Ollama
on an NVIDIA L4 with 24 GB of memory. Screening used
`qwen3-vl:8b-instruct`; investigation used `qwen3:14b`. That setup does not
establish compatibility with every listed GPU, camera or model provider.

## How a profile is chosen

`watchwise doctor` looks for, in order:

1. **NVIDIA** (via `nvidia-smi`): the current catalog selects `nvidia-16gb`
   from 15 GB of reported memory and `nvidia` from 7 GB. The paired models in
   `nvidia-16gb` exceed a 16 GB card's memory at their configured contexts;
   review the caveat below before accepting the recommendation.
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
| `qwen3-vl:8b-instruct` | ~6 GB | `nvidia`, `nvidia-16gb` screener, `apple-silicon`, `amd-rocm` |
| `qwen3:14b` | ~9 GB | `nvidia-16gb` reasoning |

One vision+tools model (`qwen3-vl`) serves every local role on small machines,
so only one model has to stay loaded.

**The `nvidia-16gb` profile needs a sizing correction.** Its configured vision
and reasoning models were run together on the 24 GB L4 on October 23, 2026.
Their combined resident memory exceeded 16 GB, despite the profile's name and selection
threshold. Do not assume both models stay loaded on a 16 GB card; eviction and
reloads can delay later verdicts. The single-model `nvidia` profile is an
alternative to evaluate with `watchwise check`, while leaving memory for
Frigate, context windows and concurrent requests. It is not a guarantee that
every nominal 8 GB card will fit the full workload.

Frigate can be configured to call the same Ollama server for its own GenAI
summaries, but sharing a model does not make its requests fit the model's
context. In the live setup, longer review summaries exceeded the configured
context and Frigate GenAI was disabled. Watchwise accepts Frigate summaries
when available; its own whole-event summaries are being implemented separately
and are not included in that published baseline.

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
calling actually worked. The `chat` probe tests the model endpoint, not an
installed Ask Watchwise conversation interface.

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
| `cpu-hybrid` | text only: detector metadata, tool results, frame *descriptions* made by the local model. Never images. |
| `cloud` | frames for screening, plus the above |
| local-model profiles | no model requests to a cloud provider when endpoints stay local |

Notifications are separate from model routing: Telegram relays card text and,
unless `notify.telegram.send_images: false`, snapshots. A configured
`nvr.web_url` also appears in clip links sent through Telegram. Webhooks and
remote MQTT brokers receive the data sent to them.

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
  default `qwen3-vl` tags) reason before they answer, so keep it at 1024 or
  more: `watchwise check` warns when a reply is cut off before the verdict.
* `vlm.max_concurrency: 1`: one screening at a time; bursts queue instead.
* `agent.max_iterations: 4`: shorter investigations.
* `gate.never_screen_labels: [cat, dog]`: never spend a model call on these.
* The `cpu-hybrid` profile keeps `screener` local and points `investigator`
  and the reserved `chat` role at a cloud model. It does not make local CPU
  screening faster. In CPU tests, verdicts could take tens of seconds and
  thinking-model replies sometimes exhausted their output cap or timed out.
* Model warm-up runs at service startup by default. While screening runs,
  provisional "Analyzing…" cards can reach you first; warm-up and asynchronous
  investigation do not guarantee a quick final verdict.
