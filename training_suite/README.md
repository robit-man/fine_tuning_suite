# Fine-Tuning Suite Python Package

This directory contains the executable training, GGUF, Ollama, dashboard, and
multimodal tooling for the repository. The authoritative project guide is the
[root README](../README.md); operational rules for automated agents are in
[AGENTS.md](AGENTS.md).

## Entry points

Run commands from the repository root:

```bash
python3 training_suite/app.py bootstrap
training_suite/.venv/bin/python -m training_suite --help
```

The package CLI exposes model intake, Ollama inspection, Modelfile generation,
job state, capability gates, Qwen3-Omni planning, monolithic GGUF packing, and
audio routing smoke tests. `app.py` contains the Qwen3.5-9B LoRA distillation,
evaluation, export, and Hugging Face upload harness.

## Dashboard

```bash
training_suite/.venv/bin/python -m training_suite db-init
training_suite/.venv/bin/python -m training_suite web \
  --host 127.0.0.1 \
  --port 7860
```

The Flask application provides model and dataset intake, background actions,
job logs, export controls, evaluation history, and comparison views backed by
SQLite.

Important API endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`, `POST` | `/api/models` | Model inventory and intake |
| `GET`, `POST` | `/api/datasets` | Dataset inventory and registration |
| `GET`, `POST` | `/api/jobs` | Background job control |
| `GET`, `POST` | `/api/evals` | Evaluation control and reports |
| `GET` | `/api/actions` | Available job actions |
| `GET` | `/api/ollama/show?model=<tag>` | Ollama model inspection |
| `GET` | `/api/omni/audio/contract` | PCM WAV transport requirements |
| `GET` | `/api/omni/router/contract` | Custom Ollama audio routing extension |
| `POST` | `/api/omni/audio/validate` | Validate tagged base64 audio |
| `POST` | `/api/omni/plan` | Plan native fusion or a monolithic router |
| `POST` | `/api/omni/cascade` | HTTP reference implementation for audio → language → TTS |

The API has no built-in authentication; keep it on loopback unless it is placed
behind an authenticated reverse proxy.

## One-file audio/video comprehension and TTS experiment

The required deliverable is one physical GGUF that Ollama imports through one
`FROM` line. It contains three namespaced execution graphs:

```text
unprefixed   Qwen3.8 or Ornith language, thinking, and tools
a.c.*        self-contained audio/video comprehension
s.t.*        independently text-conditioned TTS
```

The custom Ollama compatibility/runner hook filters those views and performs
the routing. This avoids falsely treating incompatible Qwen3-Omni and
Qwen3.8/Ornith hidden states as interchangeable.

Plan a configuration from model metadata:

```bash
training_suite/.venv/bin/python -m training_suite omni-plan \
  --text-source manitcor/Qwen3.8-27B-Obliterated-E03 \
  --omni-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --out training_suite/outputs/omni/qwen38-27b-experiment
```

Pack already converted/quantized component GGUFs:

```bash
training_suite/.venv/bin/python -m training_suite omni-pack \
  --base-gguf ./components/language.gguf \
  --comprehension-gguf ./components/comprehension.gguf \
  --tts-gguf ./components/tts.gguf \
  --out ./release/model.gguf

training_suite/.venv/bin/python -m training_suite omni-inspect \
  ./release/model.gguf
```

`omni-pack` preserves the base tensor namespace, copies embedded component
metadata into filtered views, adds component digests and the router contract,
and writes a one-`FROM` Modelfile plus a JSON report.

The custom `/api/chat` extension accepts `messages[].audios[]` entries containing
base64 16 kHz mono PCM16 WAV. It returns generated speech under
`message.audio` as base64 24 kHz mono PCM16 WAV. `response_modalities` and
`speech_mode` determine whether TTS runs.

The Flask cascade is a reference implementation using external HTTP stages.
The custom Ollama loader/runner and normalized video transport are still under
construction. Do not claim live audio/video/TTS capability from the presence of
embedded weights alone.

See [ADR-002](../.aiwg/architecture/adr-002-monolithic-ollama-audio-router-gguf.md)
for the file layout, request schema, runtime hooks, and release gates.

## Main modules

| Path | Responsibility |
|---|---|
| `app.py` | LoRA SFT, fixed-split evaluation, merge, GGUF export, upload |
| `cli.py` | Package command line |
| `web.py` | Flask UI and JSON API |
| `core/state.py` | SQLite inventory |
| `core/jobs.py` | Background subprocesses and persistent logs |
| `models/intake.py` | HF/GGUF/Ollama architecture and capability inspection |
| `models/gguf.py` | GGUF metadata and tensor inspection |
| `models/ollama.py` | Ollama metadata and Modelfile helpers |
| `models/audio.py` | Strict base64 PCM WAV validation |
| `models/omni.py` | Qwen3-Omni signature and compatibility planning |
| `models/single_gguf.py` | Monolithic packer, inspector, and router contract |
| `omni_runtime.py` | HTTP audio-comprehension/language/TTS reference route |
| `tool_splice.py` | Tool-enabled Ollama packaging for HF GGUFs |
| `ornith_vision_splice.py` | Shape-gated monolithic vision tensor transplant |
| `gguf_text_surgery.py` | GGUF text tensor substitution |
| `evals/runner.py` | Capability, tool, and audio smoke gates |

## Safety and lifecycle

- Before starting or changing a CUDA-backed container or service on this host,
  run `docker gpu discover` and use the scoped reservation protocol documented
  in `/usr/local/share/ollama-unify/AGENTS.md`.
- Never bypass tensor-shape, architecture, vocabulary, projector, or component
  gates to force a build.
- Publish only after every claimed capability passes a live test.
- Keep credentials in environment variables and never log or commit them.
- After verified publication, remove run-local safetensors, merged checkpoints,
  full-precision GGUF intermediates, and redundant conversions. Preserve the
  final GGUF, Modelfile, source revisions, hashes, licenses, and reports.
- Never edit or delete Ollama blob-store files directly; use `ollama rm` for
  obsolete local tags.

The complete cleanup procedure is in
[INSTRUCTIONS.md](INSTRUCTIONS.md#phase-6-end-of-session-cleanup).

## Tests

```bash
training_suite/.venv/bin/python -m pytest -q
```

Unit/API tests do not start CUDA services. Live Ollama, vision, audio, video,
and TTS gates require the corresponding runtime and fixtures.
