# Fine-Tuning Suite

Fine-Tuning Suite is a local-first toolkit for adapting, inspecting, evaluating,
converting, and publishing Qwen-family language models. It combines a
reproducible LoRA distillation harness with GGUF surgery, Ollama packaging,
capability verification, a Flask control plane, and compatibility-gated
multimodal experiments.

The project is intentionally conservative about capability claims. A model is
not considered to support vision, tools, thinking, audio, video comprehension,
or speech output until the relevant artifact and live runtime path have passed
their gates.

## What the suite provides

| Area | Capability | Maturity |
|---|---|---|
| Training | Qwen3.5-9B LoRA SFT with completion-only loss, DDP, validation, early stopping, and configurable training rounds | Implemented |
| Data | Deterministic train/validation/test splits; R5, R6, and R7 curation recipes; anti-repetition data generation | Implemented |
| Evaluation | Frozen baseline/tuned deltas, GSM8K, MMLU, diverse behavior, repetition, image, structured-tool, and adversarial-tool probes | Implemented |
| Intake | Inspect Hugging Face repositories, local GGUF files, and Ollama models; compare detected and requested capabilities | Implemented |
| Tools and thinking | Generate Ollama Modelfiles with Qwen renderers/parsers and verify structured tool calls | Implemented |
| Vision | Preserve a compatible vision tower during LoRA export or transplant text tensors into a shape-compatible multimodal GGUF | Implemented for supported Qwen/Ornith layouts |
| GGUF | Inspect metadata/tensors, patch vision flags and RoPE sections, substitute text tensors, convert, and quantize | Implemented |
| Ollama | Create, inspect, test, copy, and publish registry tags | Implemented |
| Control plane | Dark-theme dashboard, SQLite inventory, background jobs, logs, REST API, and model/evaluation comparisons | Implemented |
| Monolithic multimodal GGUF | Pack a base language GGUF, a self-contained comprehension GGUF, and a text-conditioned TTS GGUF into one Ollama-importable file | Implemented packer and inspector |
| Audio comprehension | Validate base64 PCM WAV and route it through the embedded comprehension graph before the language graph | Versioned adapter/parser and HTTP reference runtime implemented; custom Ollama hook pending |
| Video comprehension | Validate MP4/WebM envelopes, sampling policy, and route temporal media through comprehension | Versioned adapter/parser and reference sidecar implemented; component converter and custom Ollama hook pending |
| TTS | Route language or direct text through the embedded text-conditioned speech graph and return validated base64 PCM WAV | Versioned adapter/parser and reference sidecar implemented; component converter and custom Ollama hook pending |
| Native Qwen3-Omni grafting | Reject unsafe tensor substitution and describe the exact component/runtime boundary | Compatibility-gated research path |

“Implemented” means the suite contains executable code and tests. It does not
mean every external model server is installed or running on a given host.

## Architecture

The repository has two related execution surfaces:

```text
                                    ┌──────────────────────────────┐
Hugging Face / GGUF / Ollama ──────▶│ Intake + compatibility gate │
                                    └──────────────┬───────────────┘
                                                   │
                      ┌────────────────────────────┼──────────────────────────┐
                      ▼                            ▼                          ▼
             LoRA training/evals          GGUF + Modelfile           Omni bundle plan
                      │                            │                          │
                      └───────────────┬────────────┘                          │
                                      ▼                                       ▼
                             Ollama create/push                 monolithic GGUF + custom router

Dashboard / REST API ──▶ SQLite state ──▶ background job runner ──▶ logs and reports
```

For combined audio/video comprehension and spoken output, the target deployment
is one physical GGUF containing multiple namespaced execution graphs:

```text
one GGUF
  ├── unprefixed tensors ──▶ Qwen3.8 or Ornith language graph
  ├── a.c.* tensors ───────▶ audio/video comprehension graph
  └── s.t.* tensors ───────▶ text-conditioned TTS graph
             │
             ▼
      custom Ollama router
        audio/video ──▶ comprehension ──▶ semantic text ──▶ language
        language text ──▶ optional TTS ──▶ tagged 24 kHz PCM16 WAV
```

The result is one `.gguf` file and one Ollama model layer. It is still a
multi-graph runtime internally: a custom Ollama compatibility/runner hook is
responsible for filtered component views, routing, and audio I/O.

## Design boundaries

### GGUF does not define the execution graph

GGUF can store all tensors and metadata in one file. A loader still needs code
for each encoder, projector, language trunk, speech model, codec, and scheduling
relationship. The suite therefore provides a monolithic container format and
requires a custom Ollama handler to execute it.

### Vision splicing works only when interfaces match

The existing vision pipeline uses a supported donor architecture and verifies
tensor names, shapes, counts, and projector width before producing a combined
GGUF. It preserves quantized bytes where possible and blocks incompatible
substitutions.

### Qwen3-Omni and Qwen3.8/Ornith are not direct tensor substitutes

The current Qwen3-Omni donor uses a different Thinker architecture, hidden
width, layer count, vocabulary, and Talker conditioning interface from the
Qwen3.8 and Ornith configurations targeted by this repository. The
`omni-plan` gate therefore selects a `monolithic-router` layout for those
combinations instead of claiming native hidden-state fusion.

A true native graft would require training alignment components—for example,
an audio/vision sequence bridge from the donor encoder width into the target
language width—plus special-token alignment, multimodal instruction data, and
runtime support for the new graph. Padding, reshaping, or copying incompatible
tensors is not a valid substitute for that training.

### The current multimodal scope

The experiment covers:

- audio understanding;
- video understanding;
- text reasoning, thinking, and tools in Qwen3.8 or Ornith;
- TTS audio generation.

Video generation is not part of the experiment. The versioned adapter parser
and reference sidecar implement turn-based audio/image/video input, semantic
routing, and tagged TTS output. The custom Ollama loader/handler and production
component converters remain to be implemented.

## Requirements

- Linux with Python 3.12 recommended;
- `uv` recommended, or standard-library `venv` and `pip`;
- CUDA-capable PyTorch and NVIDIA GPUs for training or GPU inference;
- Ollama for local model creation, capability inspection, and publishing;
- llama.cpp conversion/quantization tools for GGUF export workflows;
- enough disk space for source safetensors, merged weights, full-precision
  GGUF intermediates, quantized GGUFs, and Ollama blobs;
- Hugging Face and Ollama credentials only when downloading restricted assets
  or publishing.

Model training and large conversion jobs can temporarily require several times
the final quantized model size. Review the cleanup policy before starting.

## Installation

Run commands from the repository root:

```bash
python3 training_suite/app.py bootstrap
training_suite/.venv/bin/python -m training_suite db-init
```

`bootstrap` creates `training_suite/.venv`, installs
`training_suite/requirements.txt`, records the requirements hash, and reuses the
environment until the dependency file changes.

To start the dashboard:

```bash
training_suite/.venv/bin/python -m training_suite web \
  --host 127.0.0.1 \
  --port 7860
```

Open `http://127.0.0.1:7860`. Binding to a non-loopback address exposes a job
control API; place authentication and a trusted reverse proxy in front of it.

## Primary workflows

### 1. Reasoning distillation

The harness freezes evaluation splits before training, evaluates the untouched
base model, trains a LoRA adapter, compares tuned results against the same
samples, merges the adapter, exports GGUF, and creates an Ollama model.

```bash
python3 training_suite/app.py prepare
python3 training_suite/app.py baseline
python3 training_suite/app.py train
python3 training_suite/app.py eval
python3 training_suite/app.py export
```

`all` runs the local pipeline through export; upload remains opt-in:

```bash
python3 training_suite/app.py all
HF_TOKEN=... python3 training_suite/app.py upload
```

The token is read only from the environment. Do not put credentials in command
arguments, logs, Modelfiles, manifests, or commits.

Training-round overrides:

| Variable | Purpose |
|---|---|
| `DISTILL_GPUS` | Comma-separated CUDA devices used by the DDP harness |
| `DISTILL_TRAIN_FILE` / `DISTILL_VAL_FILE` | Split filenames without `.jsonl` |
| `DISTILL_OUTPUT_SUFFIX` | Distinguish checkpoint rounds |
| `DISTILL_LR` | Learning rate |
| `DISTILL_LORA_R` / `DISTILL_LORA_ALPHA` | LoRA rank and scale |
| `DISTILL_EPOCHS` | Epoch count |
| `DISTILL_PATIENCE` | Early-stopping patience |
| `DISTILL_HF_REPO` | Override the Hugging Face upload target |

The R5/R6/R7 recipes are executable directly or from the dashboard:

```bash
training_suite/.venv/bin/python training_suite/curate_r5.py
training_suite/.venv/bin/python training_suite/curate_r6.py
training_suite/.venv/bin/python training_suite/curate_r7.py
```

See [the operational instructions](training_suite/INSTRUCTIONS.md) for the
round-specific dataset and export sequence.

### 2. Model intake and repair planning

Intake can combine remote metadata, raw-weight configuration, a local GGUF, and
an existing Ollama tag. The result records architecture, quantization, context,
tensor count, detected capabilities, missing capabilities, blockers, and a
repair mode.

```bash
training_suite/.venv/bin/python -m training_suite intake \
  --source OBLITERATUS/Ornith-1.5-9B-OBLITERATED \
  --raw-source OBLITERATUS/Ornith-1.5-9B-OBLITERATED \
  --target-capability completion \
  --target-capability vision \
  --target-capability tools \
  --target-capability thinking \
  --save
```

Use `--gguf-path`, `--ollama-model`, and `--donor-model` when those artifacts
are available. Intake is diagnostic: it does not silently download all weights
or perform a splice.

### 3. Enable and verify tool calling in Ollama

`tool_splice.py` pulls an HF GGUF through Ollama, writes a Modelfile with the
Qwen renderer/parser, creates a local tag, and runs capability and structured
tool-call checks.

```bash
# Built-in Ornith 1.0 presets
training_suite/.venv/bin/python training_suite/tool_splice.py 9b
training_suite/.venv/bin/python training_suite/tool_splice.py 35b

# Any compatible Hugging Face GGUF repository
training_suite/.venv/bin/python training_suite/tool_splice.py \
  --model org/model-GGUF \
  --tag local-model-tools:q4km \
  --gguf model-q4_k_m.gguf
```

This is packaging and runtime configuration, not weight fine-tuning. The
result must still return structured `tool_calls` in a live smoke test.

### 4. Preserve or add compatible vision weights

The vision tools support two strategies:

- merge a LoRA adapter into a compatible multimodal Hugging Face base before
  GGUF conversion;
- use `ornith_vision_splice.py` or `gguf_text_surgery.py` to replace the text
  tensors in a compatible multimodal GGUF while retaining the donor vision
  tensors and metadata.

The automated Ornith path performs shape validation, writes a splice report,
creates an Ollama tag, and can run vision/tools/thinking smoke tests:

```bash
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py \
  9b --create --test

training_suite/.venv/bin/python training_suite/ornith_vision_splice.py \
  9b --reuse-existing --copy-remote --push
```

The built-in presets target the layouts encoded in the script. For a different
Ornith or Qwen release, supply explicit source/donor paths and do not lower the
tensor-replacement or shape-safety gates merely to force a build.

### 5. Plan and pack audio/video comprehension and TTS combinations

`omni-plan` fetches only model configuration metadata, extracts architecture
signatures, compares the target language model with the Qwen3-Omni donor, and
writes a reproducible component manifest.

Qwen3.8 example:

```bash
training_suite/.venv/bin/python -m training_suite omni-plan \
  --text-source manitcor/Qwen3.8-27B-Obliterated-E03 \
  --omni-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --target-tag robit/qwen3.8-27b-omni-experiment:latest \
  --out training_suite/outputs/omni/qwen38-27b-experiment
```

Ornith example:

```bash
training_suite/.venv/bin/python -m training_suite omni-plan \
  --text-source OBLITERATUS/Ornith-1.5-9B-OBLITERATED \
  --omni-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --target-tag robit/ornith-1.5-omni-experiment:latest \
  --out training_suite/outputs/omni/ornith15-9b-experiment
```

Each output directory contains:

- `omni_bundle.json`: source signatures, mismatches, requested capabilities,
  runtime support, component assignments, and artifact policy;
- `audio_contract.json`: the validated request/response format;
- `Modelfile`: written when `--text-gguf` is supplied. An incompatible Omni
  projector is never attached to a Qwen3.8 or Ornith language GGUF.

For Qwen3.8 and Ornith, the expected result is `monolithic-router`: direct
hidden-state fusion is blocked, while one physical multi-graph GGUF remains a
valid packaging target. Use `--require-native` only when the experiment truly
requires tensor-level substitution into the Omni Thinker; it exits with status
2 when those signatures do not match.

#### Build the one-file GGUF

Convert and quantize each executable graph to GGUF first, then pack them:

```bash
training_suite/.venv/bin/python -m training_suite omni-pack \
  --base-gguf ./components/qwen38-or-ornith.q4_k_m.gguf \
  --base-source manitcor/Qwen3.8-27B-Obliterated-E03 \
  --comprehension-gguf ./components/qwen3-omni-comprehension.q4_k_m.gguf \
  --comprehension-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --tts-gguf ./components/qwen3-tts.q4_k_m.gguf \
  --tts-source Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --out ./release/model.gguf \
  --renderer qwen3.8 \
  --parser qwen3.5
```

The packer leaves the base language tensors unprefixed, rewrites comprehension
tensors under `a.c.*`, rewrites TTS tensors under `s.t.*`, copies each embedded
component's metadata into a namespaced view, adds source digests and the router
contract, and writes one GGUF. It also writes `model.gguf.report.json` and a
single-`FROM` Modelfile.

Inspect the result without running inference:

```bash
training_suite/.venv/bin/python -m training_suite omni-inspect \
  ./release/model.gguf
```

The comprehension input must be self-contained: a bare Qwen3-Omni audio/vision
encoder cannot feed Qwen3.8 or Ornith without a trained bridge. The TTS input
must be text-conditioned; an unmodified Omni Talker tied to another Thinker is
not sufficient.

#### Audio wire contract

The custom Ollama handler accepts audio as a tagged message field analogous to
`images`:

```json
{
  "model": "robit/combined:latest",
  "messages": [{
    "role": "user",
    "content": "Summarize the conversation and answer the final question.",
    "audios": [{
      "mime_type": "audio/wav",
      "encoding": "base64",
      "sample_rate_hz": 16000,
      "channels": 1,
      "sample_width_bits": 16,
      "data": "<base64 RIFF/WAVE bytes>"
    }]
  }],
  "response_modalities": ["text", "audio"],
  "speech_mode": "auto",
  "stream": false
}
```

The validator accepts at most 32 MiB decoded input. Output is base64 24 kHz,
mono, PCM16 WAV in the same JSON-safe envelope. Base64 is a transport encoding;
audio is not represented as a bitmap.

When speech is selected, the handler returns the tagged audio under
`message.audio` while preserving normal text, thinking, and tool-call fields.
`speech_mode=always` and `speech_mode=never` override automatic model routing.

Validate a request without running inference:

```bash
curl -s http://127.0.0.1:7860/api/omni/audio/validate \
  -H 'Content-Type: application/json' \
  --data-binary @request.json
```

The full extension contract is available at
`GET /api/omni/router/contract`.

The stable wire schema is `robit.ollama.omni-adapter.v1`. It also defines
`omni.task=chat|transcribe|describe|synthesize`, validated image/video
envelopes, preservation of normal Ollama tools/thinking fields, and the rule
that TTS is deferred while unresolved tool calls are present. See the complete
[adapter documentation](docs/omni-adapter/README.md) and
[runnable clients/server](examples/omni_adapter/README.md).

#### HTTP reference runtime

Before the custom Ollama loader/runner is complete, the Flask endpoint exercises
the same routing semantics across separate HTTP processes. It is a development
and evaluation harness, not the final one-file execution path.

Configure three external stages:

```bash
export TRAINING_SUITE_OMNI_ASR_URL=http://127.0.0.1:8080/v1/chat/completions
export TRAINING_SUITE_OMNI_ASR_MODEL=qwen3-omni
export TRAINING_SUITE_OMNI_LANGUAGE_MODEL=robit/qwen3.8-27b-obliterated-e03:27b
export TRAINING_SUITE_OMNI_TTS_URL=http://127.0.0.1:8081/synthesize
export OLLAMA_URL=http://127.0.0.1:11434
```

The first endpoint receives an OpenAI-compatible `input_audio` content part and
must return semantic text. Ollama receives that text and runs with `think=true`.
The TTS endpoint receives `{text, output}` and must return either `audio/wav`
bytes or the documented JSON envelope.

After starting the dashboard, run the publication gate:

```bash
training_suite/.venv/bin/python -m training_suite omni-audio-smoke \
  --audio ./fixture-16khz-mono.wav
```

The gate fails unless the complete request succeeds and the returned waveform
is 24 kHz mono PCM16 WAV.

#### Video comprehension

The planner detects the donor's `video-input` path and records a
`video_understanding` component. A native Omni runtime should decode the video,
preserve frame order and temporal metadata, and return semantic text to the
same Ollama language stage used by audio.

The adapter v1 parser accepts tagged MP4/WebM, validates container signatures,
and bounds FPS/frame sampling. The reference sidecar translates that envelope
to a multimodal comprehension service. The final custom Ollama runner must
perform bounded decoding, preserve frame timestamps, and keep video audio
aligned. Do not advertise the monolithic tag as live video input until its
embedded graph passes those end-to-end probes.

### 6. Import and publish the monolithic GGUF to Ollama

The generated Modelfile has one model reference:

```text
FROM ./model.gguf
```

Import with the custom Ollama build, then test the local tag before copying it
into the account namespace:

```bash
training_suite/.venv/bin/python -m training_suite capability-gate \
  local-model:q4km \
  --capability tools \
  --capability thinking

training_suite/.venv/bin/python -m training_suite tool-smoke local-model:q4km

training_suite/.venv/bin/python -m training_suite omni-inspect ./release/model.gguf

ollama signin
ollama cp local-model:q4km robit/model:q4km
ollama push robit/model:q4km
```

After pushing, fetch and inspect the remote tag or registry manifest. A
successful local `ollama create` is not proof that the remote publication is
complete.

## CLI reference

The package CLI provides control-plane and inspection commands:

| Command | Purpose |
|---|---|
| `web` | Start the Flask dashboard/API |
| `db-init` | Initialize the SQLite state database |
| `intake` | Inspect a model and optionally store its repair plan |
| `ollama-show` | Parse local Ollama model metadata and capabilities |
| `modelfile` | Generate an Ollama Modelfile |
| `job-list` | List tracked background jobs |
| `tool-smoke` | Verify a structured Ollama tool call |
| `capability-gate` | Compare Ollama-advertised capabilities with requirements |
| `ornith-seed` | Register the canonical Ornith intake case |
| `omni-plan` | Generate a native-graft or monolithic-router plan |
| `omni-pack` | Pack the base, comprehension, and TTS GGUFs into one file |
| `omni-inspect` | Validate schema, metadata, and tensor namespaces in that file |
| `omni-audio-smoke` | Validate live audio understanding, reasoning, and TTS |

The training harness provides:

| Command | Purpose |
|---|---|
| `bootstrap` | Create/update the project virtual environment |
| `prepare` | Download data and create deterministic splits |
| `baseline` | Evaluate the untouched base model |
| `train` | Run DDP LoRA SFT |
| `eval` | Evaluate the adapter and write a delta report |
| `export` | Merge, convert, quantize, and create an Ollama model |
| `prepare-tools` | Create the locked tool benchmark |
| `baseline-tools` / `eval-tools` | Compare base and tuned tool behavior |
| `verify-ollama` | Apply the Ollama release gate |
| `upload` | Publish artifacts to Hugging Face |
| `all` | Run prepare through export; upload is excluded |

Use `--help` on any command for its exact arguments.

## Dashboard and REST API

The dashboard manages model and dataset intake, actions, background job logs,
exports, evaluations, and comparisons. State is stored in
`training_suite/state/suite.sqlite3` by default; job logs are written under
`training_suite/logs/jobs/`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET`, `POST` | `/api/models` | List or intake models |
| `GET` | `/api/models/<id>` | Model detail and repair plan |
| `GET`, `POST` | `/api/datasets` | List or register datasets |
| `GET`, `POST` | `/api/jobs` | List or start background actions |
| `GET` | `/api/jobs/<id>` | Job status and log tail |
| `POST` | `/api/jobs/<id>/cancel` | Request job termination |
| `GET`, `POST` | `/api/evals` | List or start evaluations |
| `GET` | `/api/actions` | Discover supported actions |
| `GET` | `/api/ollama/show?model=<tag>` | Inspect an Ollama tag |
| `GET` | `/api/compare/models?id=1&id=2` | Compare model records |
| `GET` | `/api/compare/evals?id=1&id=2` | Compare evaluation records |
| `GET` | `/api/omni/audio/contract` | Read the audio contract |
| `GET` | `/api/omni/router/contract` | Read the custom Ollama request/routing/response contract |
| `GET` | `/api/omni/adapter/contract` | Read the versioned wire-only adapter contract |
| `POST` | `/api/omni/adapter/validate` | Validate an ASR/TTS/image/video request and report its route |
| `POST` | `/api/omni/audio/validate` | Validate audio without inference |
| `POST` | `/api/omni/plan` | Generate an Omni compatibility plan |
| `POST` | `/api/omni/cascade` | Run the configured audio/Ollama/TTS cascade |

The API is designed for local agents and automation. It currently has no
built-in authentication or multi-tenant authorization.

## Evaluation and release gates

The suite includes:

- held-out loss and perplexity with fixed splits;
- a fixed-seed GSM8K honesty probe;
- MMLU and Ollama-based model checks;
- diverse instruction, formatting, reasoning, concision, and tangent tests;
- repetition and cyclic-pattern detection;
- structured single, parallel, no-tool, and post-tool turns;
- adversarial tool-use tests;
- rendered-image probes;
- Ollama capability inspection;
- strict audio request and waveform-output checks.

Reports are machine-readable JSON under `training_suite/outputs/reports/` or
`training_suite/logs/`. Capability metadata is necessary but insufficient: a
release must pass behavioral tests for each advertised modality.

## Guidance for coding agents and operators

Read [training_suite/AGENTS.md](training_suite/AGENTS.md) before changing or
running the suite. The following rules are especially important:

1. **Respect the GPU broker.** Before changing or starting any CUDA-backed
   Docker container or service on this host, run `docker gpu discover`. Use
   scoped GPU reservations exactly as described in
   `/usr/local/share/ollama-unify/AGENTS.md`; never select devices from a
   one-time free-VRAM scan.
2. **Separate inspection from execution.** `intake` and `omni-plan` are
   diagnostic. Do not turn them into implicit multi-gigabyte downloads or start
   inference services without making that operation explicit.
3. **Preserve compatibility gates.** Never bypass architecture, tensor-shape,
   vocabulary, projector-width, or component-completeness failures. A GGUF that
   can be written is not necessarily a model that can be loaded correctly.
4. **Keep capabilities honest.** Renderer/parser settings may enable Ollama to
   expose tools and thinking, but live structured calls still need testing.
   Likewise, a projector file is not proof of audio or video support.
5. **Protect secrets.** Use `HF_TOKEN` and registry login state through the
   environment or official clients. Never print, persist, or commit tokens.
6. **Preserve provenance.** Record source repositories and revisions,
   quantization, licenses, Modelfiles, component digests, compatibility reports,
   and evaluation results for every published artifact.
7. **Avoid collateral changes.** The workspace may contain work from other
   users or agents. Do not discard unrelated edits or overwrite shared model
   outputs.
8. **Publish only after gates pass.** Create locally, test every claimed
   capability, copy to the remote tag, push, and verify the remote artifact in
   that order.
9. **Treat embedded components as independent graphs.** Audio/video
   comprehension and TTS have their own weights, revisions, loaders, health
   checks, memory budgets, and failure modes even though they share one file.
   Record them in the bundle manifest.
10. **Clean up completed sessions.** Large temporary weights must not accumulate
    indefinitely.

## Artifact lifecycle and storage cleanup

Use one explicit output directory per run. Cleanup is permitted only after:

1. the final local model loads;
2. required capability and behavioral tests pass;
3. every requested registry push succeeds;
4. the remote tag or manifest is verified; and
5. compact reproducibility metadata has been saved.

Then remove run-local downloaded safetensor shards, merged safetensors, LoRA
checkpoints already distilled into the verified deliverable, F16/BF16 GGUF
intermediates, partial downloads, and redundant conversion outputs. Retain
Modelfiles, manifests, source revisions, component digests, licenses, splice
reports, evaluation reports, and any final GGUF that is itself a required
deliverable.

Inspect exact paths and sizes before deletion. Shared Hugging Face caches and
donor weights may still be in use by another run. Never recursively clean the
repository root, the entire `outputs/` tree, a cache root, a home directory, or
a path built from an unset variable.

Do not delete files from Ollama's blob or manifest store directly. Use
`ollama rm <obsolete-local-tag>` for unneeded local tags. Record disk usage
before and after cleanup in the session handoff.

The full checklist is in
[End-of-Session Cleanup](training_suite/INSTRUCTIONS.md#phase-6-end-of-session-cleanup).

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TRAINING_SUITE_DB` | `training_suite/state/suite.sqlite3` | SQLite database path |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `OLLAMA_MODELS` | Host-dependent | Override the Ollama model store used by vision tooling |
| `TRAINING_SUITE_OLLAMA_RENDERER` | `qwen3.5` | Generated Modelfile renderer |
| `TRAINING_SUITE_OLLAMA_PARSER` | `qwen3.5` | Generated Modelfile parser |
| `TRAINING_SUITE_OMNI_ASR_URL` | unset | Audio-understanding chat endpoint |
| `TRAINING_SUITE_OMNI_ASR_MODEL` | `qwen3-omni` | Model name sent to that endpoint |
| `TRAINING_SUITE_OMNI_LANGUAGE_MODEL` | unset | Ollama Qwen3.8/Ornith language tag |
| `TRAINING_SUITE_OMNI_TTS_URL` | unset | Speech-synthesis endpoint |
| `OMNI_COMPREHENSION_URL` | `http://127.0.0.1:8901/v1/chat/completions` | Example adapter comprehension endpoint |
| `OMNI_COMPREHENSION_MODEL` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | Example adapter comprehension model |
| `OMNI_LANGUAGE_URL` | `http://127.0.0.1:11434` | Example adapter Ollama base URL |
| `OMNI_TTS_URL` | `http://127.0.0.1:8091/synthesize` | Example adapter TTS endpoint |
| `HF_TOKEN` | unset | Hugging Face upload credential |

## Repository map

| Path | Responsibility |
|---|---|
| `training_suite/app.py` | Self-bootstrapping distillation, evaluation, export, and upload harness |
| `training_suite/cli.py` | Package CLI |
| `training_suite/web.py` | Flask UI and REST API |
| `training_suite/core/` | Paths, SQLite state, and background jobs |
| `training_suite/datasets/` | Dataset registry and curation recipes |
| `training_suite/evals/` | Reusable capability and smoke gates |
| `training_suite/models/intake.py` | HF/GGUF/Ollama inspection and repair planning |
| `training_suite/models/gguf.py` | GGUF metadata and capability inspection |
| `training_suite/models/ollama.py` | Ollama inspection and Modelfile helpers |
| `training_suite/models/audio.py` | Strict base64 PCM WAV contract |
| `training_suite/models/omni.py` | Qwen3-Omni signatures, compatibility gate, and bundle writer |
| `training_suite/models/omni_adapter.py` | Versioned audio/image/video/TTS request parser and route planner |
| `training_suite/models/single_gguf.py` | Monolithic GGUF schema, packer, inspector, and custom audio contract |
| `training_suite/omni_runtime.py` | Audio-understanding → Ollama → TTS HTTP cascade |
| `docs/omni-adapter/` | Wire ABI, GGUF ABI, runtime patch guide, release runbook, schemas, and tests |
| `examples/omni_adapter/` | Runnable reference server plus Python and JavaScript clients |
| `training_suite/tool_splice.py` | Tool-enabled Ollama packaging for HF GGUFs |
| `training_suite/ornith_vision_splice.py` | Shape-gated Ornith vision transplant and publish workflow |
| `training_suite/gguf_text_surgery.py` | GGUF text-tensor substitution |
| `training_suite/splice_vision_*.py` | Vision-preserving Hugging Face merge workflows |
| `training_suite/templates/`, `static/` | Dashboard presentation |
| `tests/` | Unit and API tests |
| `.aiwg/architecture/` | Multimodal ADR and impact analysis |

Additional documentation:

- [Agent operating guide](training_suite/AGENTS.md)
- [End-to-end operational instructions](training_suite/INSTRUCTIONS.md)
- [Model cards](training_suite/MODEL_CARDS.md)
- [Ollama registry README content](training_suite/OLLAMA_MODEL_READMES.md)
- [Qwen3-Omni bundle ADR](.aiwg/architecture/adr-001-qwen3-omni-audio-bundles.md)
- [Monolithic Ollama audio-router GGUF ADR](.aiwg/architecture/adr-002-monolithic-ollama-audio-router-gguf.md)
- [Omni adapter documentation](docs/omni-adapter/README.md)
- [Omni adapter examples](examples/omni_adapter/README.md)

## Tests

```bash
training_suite/.venv/bin/python -m pytest -q
```

For documentation-only or CPU environments, unit tests can run without
starting CUDA workloads. Live Ollama, vision, audio, and TTS gates require their
respective services and fixtures.

## Limitations

- The core training harness is specialized for Qwen3.5-9B. The inspection,
  packaging, evaluation, and GGUF utilities cover a broader set of compatible
  models, but that does not make the SFT recipe architecture-agnostic.
- Modelfile renderer/parser settings cannot restore tool behavior removed by
  fine-tuning.
- Direct GGUF tensor surgery is safe only for explicitly verified layouts.
- The monolithic file requires a custom Ollama compatibility/runner hook for
  component views, routing, and audio fields. Stock Ollama does not gain those
  behaviors merely by importing the file.
- The versioned parser/reference sidecar implements audio/image/video envelopes
  and all four routes, but the custom monolithic runner, production component
  converters, and in-process video temporal normalization remain follow-up work.
- Model and dataset licenses vary. Review every source license and acceptable-use
  condition before training, merging, or publishing derived artifacts.

## License and attribution

Repository code and model/data artifacts may have different licenses. Qwen base
models, Ornith derivatives, datasets, donor projectors, and generated outputs
must retain their original notices and comply with their respective terms. Do
not infer that the repository's code license grants redistribution rights for
third-party weights or datasets.
