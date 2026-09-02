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
| Omni GGUF sidecar | Pack six byte-preserving model/projector views into one namespaced GGUF, attach it as a custom Ollama layer, resolve it, and materialize disposable runtime views | Implemented and tested |
| Audio comprehension | Validate base64 PCM WAV and route installed-sidecar Qwen3-Omni output into the language graph | Reference runtime live-tested |
| Video comprehension | Validate MP4/WebM, demux optional audio, and route temporal media through installed-sidecar Qwen3-Omni | Reference runtime live-tested |
| TTS | Route language or direct text through installed-sidecar Qwen3-TTS and return tagged base64 PCM WAV | Reference runtime live-tested |
| Phone validation harness | Authenticated mobile chat/call UI with adaptive VAD, camera clips, streamed TTS, voice cloning, reasoning control, isolated multi-user queuing, and expiring timing diagnostics | Implemented and live-tested |
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
                             Ollama create/push                 namespaced sidecar + adapter

Dashboard / REST API ──▶ SQLite state ──▶ background job runner ──▶ logs and reports
```

For combined audio/video comprehension and spoken output, the deployment unit
is one Ollama model tag. Standard layers remain stock-runnable; one custom
layer carries a namespaced multi-graph GGUF:

```text
one Ollama tag
  ├── standard model/projector ──▶ stock text, vision, tools, thinking
  └── custom Omni layer ─────────▶ one six-view GGUF sidecar
       ├── a.c.m.* + a.c.p.* ───▶ audio/image/video comprehension
       └── s.t.m.* + s.t.p.* ───▶ text-conditioned TTS
                        │
                        ▼
                 adapter router
                   media ──▶ semantic text ──▶ stock language graph
                   final text ──▶ optional TTS ──▶ tagged WAV
```

The result is one pullable tag and one physical custom sidecar GGUF. It is not
one directly runnable heterogeneous GGUF: standard GGUF loaders require one
architecture-specific tensor inventory. The adapter resolves the custom layer,
creates filtered component views, and owns media routing and audio I/O.

## Design boundaries

### GGUF does not define the execution graph

GGUF can index all tensors and metadata in one file, but stock loaders validate
that inventory against one selected architecture. The suite therefore ships
the heterogeneous file as a custom manifest layer while retaining normal
Ollama model/projector layers for native execution.

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

Video generation is not part of the experiment. The versioned adapter and
pinned llama.cpp workers implement turn-based audio/image/video input, semantic
routing, and tagged TTS output. Stock Ollama does not natively parse those
media extensions; callers requiring them use the adapter endpoint.

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
hidden-state fusion is blocked, while a namespaced multi-graph GGUF is used as
the custom media layer of one logical Ollama tag. `--require-native` exits with
status 2 when tensor-level signatures do not match.

#### Build the six-view sidecar

Convert, quantize, and test each graph first, then pack the base model/projector,
comprehension model/projector, and TTS model/projector:

```bash
python -m training_suite omni-pack \
  --base-gguf ./components/base.gguf \
  --base-projector-gguf ./components/base-projector.gguf \
  --comprehension-gguf ./components/comprehension-model.gguf \
  --comprehension-projector-gguf ./components/comprehension-projector.gguf \
  --tts-gguf ./components/tts-model.gguf \
  --tts-projector-gguf ./components/tts-projector.gguf \
  --base-source org/base@revision \
  --comprehension-source org/omni@revision:q4_k_m \
  --tts-source org/tts@revision:q4_k_m \
  --out ./release/omni-sidecar.gguf

python -m training_suite omni-inspect ./release/omni-sidecar.gguf
```

The storage namespaces are unprefixed base, `b.p.*`, `a.c.m.*`, `a.c.p.*`,
`s.t.m.*`, and `s.t.p.*`. The command writes a pack report and custom-layer
descriptor. It does not write a misleading Modelfile: the heterogeneous file
must not be a stock `FROM` target.

Create/copy a normal Ollama tag from the verified base, then attach the sidecar:

```bash
python -m training_suite omni-attach robit/example-omni:q4km \
  ./release/omni-sidecar.gguf
python -m training_suite omni-resolve robit/example-omni:q4km
python -m training_suite omni-prepare robit/example-omni:q4km \
  --out ./runtime-cache
```

`omni-prepare` produces disposable component files for runtimes without
filtered mmap support. Delete that cache only after workers stop. The
comprehension graph must be self-contained, and TTS must be independently
text-conditioned; an Omni Talker coupled to a different Thinker is insufficient.

#### Audio wire contract

The Omni adapter accepts audio as a tagged message field analogous to
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

#### Reference runtime

The reference adapter exposes one `/api/chat` endpoint while coordinating a
pinned llama.cpp comprehension worker, stock Ollama, and a Qwen3-TTS worker.
It resolves all media weights from the installed tag; component URLs are
internal deployment details. See the [runtime guide](docs/omni-adapter/runtime.md)
and [examples](examples/omni_adapter/README.md).

For phone testing, the [authenticated Omni portal](examples/omni_portal/README.md)
adds hold-to-record WAV capture, adaptive VAD, environmental-audio analysis,
image/video/GIF uploads, silent-video handling, PDF/DOCX/text retrieval, a live
camera preview, bounded camera-call frames, safe Markdown with responsive GFM
pipe tables, viewport-bounded
token-pinned chat scrolling, native Ollama reasoning control, deterministic
Qwen3-TTS profiles, allowlisted Female/Male presets, request-local WAV voice
cloning, streamed PCM playback, and
continuously submitted barge-in voice-call turns. An explicit wrench toggle
beside reasoning adds eight server-pinned Ollama tool schemas for locally
browser-driven public web discovery/fetch,
attached-document search, time/capabilities, and temporary session recall. The
portal completes bounded multi-round tool chains before optional TTS and shows
a live collapsible execution trace with each assistant reply. Tools default off;
the harness uses no hosted search API or key. The low-latency worker stays
resident and emits two-frame, roughly 160 ms PCM windows. Distinct browser
sessions are queued through a bounded GPU lane with no shared conversation
history. Reload-safe IndexedDB state is scoped per browser session and expires
five minutes after page leave. Document excerpts use a bounded,
hashed lexical index isolated to the browser session, while content-redacted
timing journals expire five minutes after a page leaves and delete immediately
with the trash control. The active/queued user count is shown beside **ONLINE**.

Every comprehension call disables llama.cpp prompt-slot reuse so the current
audio/image/video bytes—not a prior media embedding—are evaluated. Camera
re-recording replaces the prior unsent camera clip, submitted video appears as
a looping thumbnail, and each new media turn excludes all earlier media bytes
while retaining prior text dialogue as explicitly non-current context. Call
prompting answers intent rather than parroting the transcript. A privacy-bounded
CPU/RAM/GPU/network/date-time snapshot is refreshed into trusted context every
turn. The broker-compliant CUDA-only supervisor publishes a
temporary Cloudflare HTTPS tunnel. See the [phone portal runbook](docs/omni-adapter/phone-portal.md)
for endpoints, lifecycle, isolation, diagnostics, and the complete verification
matrix.

#### Video comprehension

The planner detects the donor's `video-input` path and records a
`video_understanding` component. A native Omni runtime should decode the video,
preserve frame order and temporal metadata, and return semantic text to the
same Ollama language stage used by audio.

The adapter accepts tagged MP4/WebM/GIF, validates container signatures, bounds
sampling, normalizes GIF to MP4, passes video bytes through llama.cpp's
`input_video`, and optionally demuxes a separate 16 kHz audio part. A clip with
no audio track remains a valid visual-only turn. Production deployments must additionally
bound duration/resolution/decoder resources and should not claim sample-accurate
alignment from the reference demux path.

### 6. Publish the combined release

Push the repository documentation first. Publish and verify the sidecar GGUF,
model card, pack report, and hashes on Hugging Face. Then push the already
attached Ollama `q4km` and `latest` tags. A release is complete only after a
pull round trip preserves the custom layer digest and `ollama show` still
reports the standard capabilities.

The complete order, rollback, credential hygiene, and mandatory cleanup gates
are in the [build/release runbook](docs/omni-adapter/build-and-release.md).

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
| `omni-pack` | Pack six model/projector views into one custom GGUF sidecar |
| `omni-inspect` | Validate sidecar schema, metadata, and tensor namespaces |
| `omni-unpack` | Materialize one executable component view |
| `omni-attach` | Attach the sidecar as a custom layer on an existing Ollama tag |
| `omni-resolve` | Locate and validate that layer in a local Ollama manifest |
| `omni-prepare` | Build a disposable media-worker cache from an installed tag |
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
   capability, push repository docs, publish/verify Hugging Face artifacts,
   then push/pull/verify Ollama tags.
9. **Treat embedded components as independent graphs.** Audio/video
   comprehension and TTS have their own weights, revisions, loaders, health
   checks, memory budgets, and failure modes even though they share one logical
   tag and one custom sidecar. Record them in the bundle manifest.
10. **Clean up completed sessions.** Large temporary weights must not accumulate
    indefinitely.

## Artifact lifecycle and storage cleanup

Use one explicit output directory per run. Cleanup is permitted only after:

1. the final local model loads;
2. required capability and behavioral tests pass;
3. every requested Hugging Face and Ollama push succeeds;
4. Hugging Face files/model card and pulled Ollama sidecar digest are verified;
5. compact reproducibility metadata has been saved.

Then remove run-local downloaded safetensor shards, merged safetensors, LoRA
checkpoints already distilled into the verified deliverable, F16/BF16 GGUF
intermediates, partial downloads, redundant conversion outputs, and disposable
sidecar materializations. Retain Modelfiles, manifests, source revisions,
component digests, licenses, splice reports, evaluation reports, and any final
GGUF that is itself a required deliverable.

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
| `OMNI_COMPREHENSION_CONTEXT_TOKENS` | `65536` | Example Qwen3-Omni comprehension context limit |
| `OMNI_LANGUAGE_URL` | `http://127.0.0.1:11434` | Example adapter Ollama base URL |
| `OMNI_TTS_URL` | `http://127.0.0.1:8091/synthesize` | Example adapter TTS endpoint |
| `OMNI_OLLAMA_MODEL` | unset | Resolve the TTS sidecar from this installed model tag |
| `OMNI_COMPONENT_CACHE` | `training_suite/outputs/omni-cache` | Disposable materialized-view cache |
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
| `training_suite/models/single_gguf.py` | Six-view GGUF sidecar schema, packer, inspector, and materializer |
| `training_suite/models/ollama_sidecar.py` | Ollama custom-layer attach, resolve, and runtime-cache preparation |
| `training_suite/omni_runtime.py` | Audio-understanding → Ollama → TTS HTTP cascade |
| `docs/omni-adapter/` | Wire ABI, GGUF ABI, runtime patch guide, release runbook, schemas, and tests |
| `examples/omni_adapter/` | Runnable reference server plus Python and JavaScript clients |
| `examples/omni_portal/` | Authenticated mobile UI, smoke runner, lifecycle supervisor, and Cloudflare tunnel |
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
- [Omni phone portal](examples/omni_portal/README.md)
- [Omni portal tools and chaining](docs/omni-adapter/tools.md)

## Tests

```bash
training_suite/.venv/bin/python -m pytest -q tests
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
- Stock Ollama ignores the custom sidecar layer and does not gain audio/video/TTS
  fields. The adapter is required for those paths; stock execution covers the
  tag's standard text/vision/tools/thinking layers.
- The reference runtime implements all four turn-based routes. Persistent TTS,
  streaming, filtered mmap views, and sample-accurate video/audio alignment are
  follow-up production work.
- Model and dataset licenses vary. Review every source license and acceptable-use
  condition before training, merging, or publishing derived artifacts.

## License and attribution

Repository code and model/data artifacts may have different licenses. Qwen base
models, Ornith derivatives, datasets, donor projectors, and generated outputs
must retain their original notices and comply with their respective terms. Do
not infer that the repository's code license grants redistribution rights for
third-party weights or datasets.
