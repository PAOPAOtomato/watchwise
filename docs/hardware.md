# Hardware and models

Watchwise runs three model roles, and each can run anywhere:

| Role | Job | What the model needs |
|---|---|---|
| `screener` | tier 2: look at 1-3 frames, answer benign / suspicious / uncertain | vision, reliable JSON |
| `investigator` | tier 3: a tool-calling agent for the hard cases (async) | reliable tool calling |
| `chat` | conversation with you (milestone 4) | tool calling, and speed |

A **profile** is a starting point for which model plays which role on a given
kind of machine. It is not a measurement. `watchwise doctor` picks one; `watchwise init`
writes it to `config/watchwise.yml`; `watchwise check` tells you how it actually
behaves on your machine. Everything is editable afterwards.

## How a profile is chosen

`watchwise doctor` looks for, in order:

1. **NVIDIA** (via `nvidia-smi`): 16 GB+ gives `nvidia-16gb`, 7 GB+ gives `nvidia`.
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
| `qwen3-vl:8b` | ~6 GB | `nvidia`, `apple-silicon`, `amd-rocm` |
| `qwen3:14b` | ~9 GB | `nvidia-16gb` reasoning |

One vision+tools model (`qwen3-vl`) serves every local role on small machines,
so only one model has to stay loaded. If Frigate's own GenAI features use the
same Ollama server, it can share that model too.

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
calling actually worked.

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

What leaves your machine depends on the profile, and `init` prints it:

| Profile | Sent to the cloud |
|---|---|
| `cpu-hybrid` | text only: detector metadata, tool results, frame *descriptions* made by the local model. Never images. |
| `cloud` | frames for screening, plus the above |
| everything else | nothing |

Notes:

* **Claude**: requests use prompt caching and server-side refusal fallback
  (a declined request is retried on Anthropic's recommended fallback model).
  Anything still declined is treated as "uncertain" and goes to a human. Chat
  uses a lower effort setting than investigation, for speed; both are
  `effort:` values in the config.
* **Free tiers** (for example Groq, or Google AI Studio for Gemini) work through
  the same `openai_compat` provider. They are rate-limited, and some providers'
  free-tier terms allow using your inputs to improve their models. That is fine
  for the demo's synthetic data; check the terms before sending real camera data.
* Keys live only in `.env` (git-ignored). The config names the variable, never the key.

## Tuning for a slow machine

In `config/watchwise.yml`:

* `vlm.frames: 1` and `vlm.max_image_edge: 448`: fewer and smaller images per alert.
* `vlm.max_concurrency: 1`: one screening at a time; bursts queue instead.
* `agent.max_iterations: 4`: shorter investigations.
* `gate.never_screen_labels: [cat, dog]`: never spend a model call on these.
* Point `chat` (and optionally `investigator`) at a cloud model and keep
  `screener` local: that is the `cpu-hybrid` profile.
