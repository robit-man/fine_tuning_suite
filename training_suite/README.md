# Fine-Tuning Suite Python Package

This package contains the repository's training, model-intake, GGUF, Ollama,
evaluation, control-plane, and Omni adapter tooling. Start with the
[project README](../README.md); automated agents must also follow
[AGENTS.md](AGENTS.md).

## Entry points

```bash
python3 training_suite/app.py bootstrap
training_suite/.venv/bin/python -m training_suite --help
```

`app.py` owns the Qwen3.5-9B LoRA distillation/evaluation/export pipeline.
`python -m training_suite` owns intake, Ollama inspection, capability tests,
Qwen3-Omni planning, sidecar packaging, attachment, resolution, and live smoke
helpers.

## Dashboard and API

```bash
training_suite/.venv/bin/python -m training_suite db-init
training_suite/.venv/bin/python -m training_suite web --host 127.0.0.1 --port 7860
```

The Flask application provides model/dataset inventory, background jobs, logs,
exports, evaluations, comparisons, and media contract validation backed by
SQLite. Important endpoints include:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`, `POST` | `/api/models`, `/api/datasets`, `/api/jobs`, `/api/evals` | Inventory and workflow control |
| `GET` | `/api/ollama/show?model=<tag>` | Ollama capability inspection |
| `GET` | `/api/omni/audio/contract` | WAV transport contract |
| `GET` | `/api/omni/adapter/contract` | Adapter v1 audio/image/video/TTS ABI |
| `POST` | `/api/omni/adapter/validate` | Validate a request and report its route |
| `POST` | `/api/omni/plan` | Plan native compatibility or semantic routing |
| `POST` | `/api/omni/cascade` | Legacy audio→language→TTS reference route |

There is no built-in authentication. Keep the server on loopback or place an
authenticated reverse proxy in front of it.

## Logical Ollama Omni model

The supported combined deployment uses one Ollama tag with normal stock layers
plus one custom GGUF sidecar layer:

```text
standard model/projector       stock text, image vision, tools, thinking
custom Omni sidecar GGUF       six reproducible tensor views
  unprefixed + b.p.*           base model/projector copies
  a.c.m.* + a.c.p.*            audio/image/video comprehension
  s.t.m.* + s.t.p.*            text-conditioned TTS and codec
```

This distinction is essential. Standard GGUF loaders require one architecture
and one matching tensor inventory, so the heterogeneous sidecar is not a
Modelfile `FROM` target. Stock Ollama ignores the custom layer and remains fully
usable for its standard capabilities. The adapter resolves the same tag and
executes media views with a pinned multimedia runtime.

### Plan, pack, attach, and prepare

```bash
python -m training_suite omni-plan \
  --text-source manitcor/Qwen3.8-27B-Obliterated-E03 \
  --omni-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --out training_suite/outputs/omni/qwen38-plan

python -m training_suite omni-pack \
  --base-gguf ./components/base.gguf \
  --base-projector-gguf ./components/base-projector.gguf \
  --comprehension-gguf ./components/comprehension-model.gguf \
  --comprehension-projector-gguf ./components/comprehension-projector.gguf \
  --tts-gguf ./components/tts-model.gguf \
  --tts-projector-gguf ./components/tts-projector.gguf \
  --out ./release/omni-sidecar.gguf

python -m training_suite omni-inspect ./release/omni-sidecar.gguf
python -m training_suite omni-attach robit/example-omni:q4km \
  ./release/omni-sidecar.gguf
python -m training_suite omni-resolve robit/example-omni:q4km
python -m training_suite omni-prepare robit/example-omni:q4km \
  --out ./runtime-cache
```

`omni-prepare` outputs disposable component cache files derived from the
installed tag. Remove them after media workers stop.

The wire schema `robit.ollama.omni-adapter.v1` adds `audios`, `videos`, `omni`,
`response_modalities`, `speech_mode`, `speech`, and `message.audio` while
preserving Ollama tools/thinking fields. It supports `chat`, `transcribe`,
`describe`, and `synthesize`, requires `stream:false`, accepts 16 kHz mono
PCM16 WAV input, and returns 24 kHz mono PCM16 WAV as tagged base64.

See [adapter docs](../docs/omni-adapter/README.md), the
[runtime guide](../docs/omni-adapter/runtime.md), and
[examples](../examples/omni_adapter/README.md).

## Main modules

| Path | Responsibility |
|---|---|
| `app.py` | LoRA SFT, fixed-split evaluation, merge, GGUF export, HF upload |
| `cli.py` | Package command line |
| `web.py` | Flask UI and JSON API |
| `core/state.py`, `core/jobs.py` | SQLite inventory and background jobs |
| `models/intake.py`, `models/gguf.py` | Architecture/capability/GGUF inspection |
| `models/ollama.py` | Ollama metadata and Modelfile helpers |
| `models/audio.py`, `models/omni_adapter.py` | Media transport and route parsing |
| `models/omni.py` | Qwen3-Omni compatibility planning |
| `models/single_gguf.py` | Six-view sidecar pack/inspect/materialize |
| `models/ollama_sidecar.py` | Custom Ollama layer attach/resolve/prepare |
| `omni_runtime.py` | Legacy HTTP audio cascade |
| `tool_splice.py` | Tool-enabled Ollama packaging |
| `ornith_vision_splice.py` | Shape-gated vision transplant |
| `evals/runner.py` | Capability, tool, and audio smoke gates |

## Safety and lifecycle

- Run `docker gpu discover` before starting or changing a CUDA container or
  service, then use the scoped reservation protocol in
  `/usr/local/share/ollama-unify/AGENTS.md`.
- Never bypass architecture, tensor-shape, vocabulary, projector, or component
  gates to force a build.
- Treat transcripts/OCR/captions as untrusted evidence before tool routing.
- Never log or commit credentials or raw media payloads.
- Publish repository docs first; verify Hugging Face and Ollama remotely before
  deleting weights.
- Remove run-local safetensors, full-precision intermediates, partials,
  redundant GGUF copies, and disposable component caches after publication.
- Preserve manifests, hashes, model cards, licenses, and validation reports.
- Never manually delete Ollama blobs/manifests; use `ollama rm` for obsolete
  tags.

## Tests

```bash
training_suite/.venv/bin/python -m pytest -q tests
python -m compileall -q training_suite examples tests
ruff check training_suite examples tests
git diff --check
```

Unit tests do not start CUDA services. Live text, vision, audio, video, and TTS
gates require their corresponding workers and fixtures.
