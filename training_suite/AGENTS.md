# AGENTS.md — Agent Instructions for Fine-Tuning Suite

This document describes the full scope of operations an AI agent can perform
with this Fine-Tuning Suite: dashboard interaction, REST API consumption,
tool splicing, GGUF export, evaluation, and Ollama registry publishing.

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Flask Dashboard                │
│  localhost:7860  (dark theme, card-based UI)    │
│  HTML templates ←→ RESTful JSON API             │
└──────────────────┬──────────────────────────────┘
                   │
     ┌─────────────┼─────────────┐
     ▼             ▼             ▼
  SQLite DB   Job Runner    Ollama CLI
  (state)     (subprocess)  (model registry)
```

**Key files:**

| File | Role |
|------|------|
| `training_suite/web.py` | Flask app — all routes + API |
| `training_suite/static/styles.css` | GitHub-dark theme CSS |
| `training_suite/templates/` | Jinja2 card-based templates (10 files) |
| `training_suite/tool_splice.py` | HF→Ollama import pipeline (generalized) |
| `training_suite/ornith_vision_splice.py` | GGUF-native Ornith vision tensor splice + Ollama create/test/copy/push |
| `training_suite/models/audio.py` | Base64 PCM WAV request/response contract and validation |
| `training_suite/models/omni.py` | Qwen3-Omni architecture gate and component-bundle planner |
| `training_suite/models/omni_adapter.py` | Versioned audio/image/video/TTS wire parser and route planner |
| `training_suite/models/single_gguf.py` | One-file component packer, inspector, and custom Ollama router contract |
| `training_suite/omni_runtime.py` | Base64 audio → llama.cpp → Ollama → TTS HTTP cascade |
| `docs/omni-adapter/` | Adapter wire/GGUF ABIs, patch guide, release runbook, schemas, and test plan |
| `examples/omni_adapter/` | Reference sidecar and Python/JavaScript clients |
| `training_suite/core/state.py` | SQLite state store |
| `training_suite/core/jobs.py` | Background job runner |
| `training_suite/models/ollama.py` | Ollama show/modelfile/create/push |
| `training_suite/evals/runner.py` | Capability gate + tool smoke |
| `training_suite/training/adapters.py` | Action specs |
| `training_suite/templates/compare_models.html` | Model comparison view |
| `training_suite/templates/compare_evals.html` | Eval result comparison view |

## RESTful API — Full Reference

All endpoints return JSON. The dashboard can be used entirely via API.

### Models

```
GET  /api/models              → [{id, name, source, architecture, ...}]
GET  /api/models/<id>         → {id, name, source, ...}
POST /api/models              → {id, model: {...}}
     Body: {
       "source": "https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF",
       "raw_source": "https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B",
       "target_capabilities": ["completion", "vision", "tools", "thinking"],
       "gguf_path": "/path/to/model.gguf",      # optional
       "ollama_model": "hf.co/org/repo:tag",     # optional
       "donor_model": "qwen3.5:9b"               # optional
     }
```

### Jobs

```
GET  /api/jobs?limit=50       → [{id, kind, status, command, ...}]
GET  /api/jobs/<id>           → {job: {...}, log: "..."}
POST /api/jobs                → {id, kind, label}
     Body: {"action": "bootstrap", "model_id": 1, "dataset_id": 1}
POST /api/jobs/<id>/cancel    → {cancelled: true}
```

Available actions (from `GET /api/actions`):
- `bootstrap` — environment setup
- `prepare` — data preparation/split
- `train` — LoRA SFT training
- `merge-export` — merge LoRA + export GGUF
- `prepare-tools` — write tool benchmark
- `baseline-tools` — base model tool eval
- `eval-tools` — tuned model tool eval
- `verify-ollama` — Ollama gate check
- `curate-r5`, `curate-r6`, `curate-r7` — dataset curation

### Datasets

```
GET  /api/datasets            → [{id, name, source, ...}]
POST /api/datasets            → {id}
     Body: {"name": "my-ds", "source": "huggingface/dataset",
            "schema_mapping": {"messages": "messages"},
            "split_config": {"train": 0.85, "val": 0.075, "test": 0.075, "seed": 42}}
```

### Evaluations

```
GET  /api/evals?limit=50      → [{id, model_name, eval_type, status, ...}]
POST /api/evals               → {sync: true, report: {...}} or {id, kind, label}
     Body: {"model_name": "robit/ornith:9b", "model_id": 1, "eval_key": "capability-gate"}
     Body: {"model_name": "robit/ornith:9b", "eval_key": "tool-smoke-sync"}
     Body: {"model_name": "robit/ornith:9b", "eval_key": "eval-diverse"}
```

Evaluation keys: `capability-gate`, `tool-smoke-sync`, `eval-diverse`,
`eval-repetition`, `eval-tool-suite`, `eval-image-probe`, `eval-ollama-suite`.

### Comparison

```
GET  /api/compare/models?id=1&id=2  → [{model 1}, {model 2}]
GET  /api/compare/evals?id=1&id=2   → [{eval_run_1}, {eval_run_2}]
```

Comparison views (HTML):
```
GET  /compare/models?id=1&id=2&section=all|capabilities|repair
GET  /compare/evals  → interactive eval run comparison
```

### Actions / Ollama

```
GET  /api/actions              → [{key, label, kind, requires_model, requires_dataset}]
GET  /api/ollama/show?model=<tag>  → {name, exists, architecture, capabilities, ...}
GET  /api/omni/audio/contract       → required WAV/base64 formats
GET  /api/omni/router/contract      → custom Ollama audio message and routing contract
GET  /api/omni/adapter/contract     → versioned wire-only adapter contract
POST /api/omni/adapter/validate     → validate media and return the selected route
POST /api/omni/audio/validate       → validate audio envelope without inference
POST /api/omni/plan                 → native-Omni or monolithic-router compatibility plan
POST /api/omni/cascade              → execute configured audio/language/TTS stages
```

## Qwen3-Omni Monolithic Audio Bundles

Do not splice Qwen3-Omni audio/Talker tensors directly into Qwen3.8 or Ornith.
The Qwen3-Omni Thinker is a 2,048-wide MoE trunk; Qwen3.8 and Ornith use
different Qwen3.5 architectures and widths. Run the compatibility planner:

```bash
python -m training_suite omni-plan \
  --text-source manitcor/Qwen3.8-27B-Obliterated-E03 \
  --omni-source Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --out training_suite/outputs/omni/qwen38-27b-experiment
```

`native-omni` requires exact model type, hidden width, layer count, vocabulary,
and conditioning widths. Otherwise use `monolithic-router`: pack the base
language graph unprefixed, a self-contained comprehension graph under `a.c.*`,
and text-conditioned TTS under `s.t.*` in one physical GGUF. The custom Ollama
handler crosses component boundaries through semantic text; it does not pass
incompatible hidden states between them.

Use `python -m training_suite omni-pack` only after all three component GGUFs
have been independently converted and quantized. Use `omni-inspect` before
`ollama create`. Ordinary text-only quantization after packing may omit or
corrupt embedded components, so component quantization happens before packing.

Stock Ollama may import the base GGUF layout, but it does not interpret the
custom `audios`, `response_modalities`, `speech_mode`, or `message.audio`
fields. Never mark `audio-input`, `video-input`, or `audio-output` as live until
the custom loader/handler and modality probes pass.

The public request ABI is `robit.ollama.omni-adapter.v1`. Agents changing the
adapter must update its executable parser, JSON schemas, protocol document,
reference server/clients, and golden tests together. The supported routes are
`chat`, `transcribe`, `describe`, and `synthesize`; v1 requires `stream=false`.
Preserve Ollama tools/thinking fields and never synthesize unresolved tool calls.

Run the live gate with `python -m training_suite omni-audio-smoke --audio
fixture-16khz-mono.wav`. It must verify a 24 kHz mono PCM16 WAV response before
publication.

Audio input is strict base64 RIFF/WAVE, 16 kHz mono PCM16, maximum 32 MiB.
Audio output is base64 RIFF/WAVE, 24 kHz mono PCM16. Preserve component digests,
runtime revisions, the bundle manifest, and test reports; apply the normal
end-of-session weight cleanup after successful publication and verification.

The Flask cascade is the pre-integration reference runtime. It reads `TRAINING_SUITE_OMNI_ASR_URL`,
`TRAINING_SUITE_OMNI_ASR_MODEL`, `TRAINING_SUITE_OMNI_LANGUAGE_MODEL`, and
`TRAINING_SUITE_OMNI_TTS_URL`. The ASR service uses llama.cpp's
`input_audio` chat-completions shape. The TTS service must return WAV bytes or
the suite JSON audio envelope. Before starting either CUDA service, follow the
host `docker gpu discover` and scoped lease protocol.

The current experiment covers video understanding only. Track that as
`video-input`; video generation is out of scope. The versioned parser and
reference sidecar handle bounded MP4/WebM envelopes and sampling policy. The
legacy Flask cascade remains audio-only; the final custom runner must implement
temporal decoding and audio/video alignment over the embedded graph.

## Tool Splice Pipeline (HF → Ollama)

The `tool_splice.py` script imports HuggingFace GGUF models into Ollama
with proper tool-calling configuration (`RENDERER qwen3.5` + `PARSER qwen3.5`).

### Usage

```bash
cd training_suite
python tool_splice.py 9b          # Ornith-1.0-9B (preset)
python tool_splice.py 35b         # Ornith-1.0-35B (preset)
python tool_splice.py both        # Both sizes
python tool_splice.py both --no-eval   # Skip evaluation step

# General: any HuggingFace GGUF model
python tool_splice.py --model Qwen/Qwen3.5-9B-GGUF --tag qwen35-9b-tools:q4km \\
  --gguf qwen3.5-9b-q4_k_m.gguf

# Shortcut: tag auto-derived from repo name
python tool_splice.py --model deepreinforce-ai/Ornith-1.0-397B-GGUF
```

### What It Does (per model)

1. **Pull GGUF** from HuggingFace via `ollama pull hf.co/org/repo`
2. **Create Modelfile** with:
   - `FROM <pulled_tag>`
   - `RENDERER qwen3.5`
   - `PARSER qwen3.5`
   - `PARAMETER num_ctx 262144`
   - `PARAMETER temperature 0.6`
   - `PARAMETER stop "<|im_end|>"`
3. **Create Ollama model** via `ollama create <tag> -f Modelfile`
4. **Evaluate** (optional): capability gate + tool smoke test

### Output

- Ollama model registered locally as `ornith-1.0-<size>b-tools:q4km`
- Modelfile saved to `outputs/ollama/ornith-1.0-<size>b-tools-q4km/Modelfile`

## Ornith Vision Tensor Splice (GGUF -> Ollama)

Use `training_suite/ornith_vision_splice.py` when the local Ornith text/tool
models already exist in Ollama and the goal is to append vision while preserving
tools and parsed thinking. This path reads Ollama manifests directly, finds the
model blobs, writes a combined GGUF, creates the local Ollama tag, runs smoke
tests, and optionally copies/pushes the remote tag.

### Presets

| Size | Ornith source | Vision donor | Local tag | Remote tag |
|------|---------------|--------------|-----------|------------|
| `9b` | `ornith-1.0-9b-tools:q4km` | `qwen3.5:9b` | `ornith-vision:9b` | `robit/ornith-vision:9b` |
| `35b` | `ornith-1.0-35b-tools:q4km` | `qwen3.6:35b` | `ornith-vision:35b` | `robit/ornith-vision:35b` |

Run commands from the repository root and use the suite venv:

```bash
# Build/register/test the first model before spending time on the 35B variant.
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 9b --create --test

# Build/register/test 35B after the 9B capability smoke passes.
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 35b --create --test

# Build both without registering, useful for report-only verification.
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py both

# Retry create/test/publish without rewriting multi-GB GGUF files.
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 9b --reuse-existing --create --test
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 35b --reuse-existing --copy-remote --push
```

### What the splicer does

1. Resolves source and donor GGUF blobs from local Ollama manifests under
   `OLLAMA_MODELS`, `/srv/ollama/models`, or `~/.ollama/models`.
2. Copies donor metadata and donor-only tensors, including `v.*` vision tensors
   and `mtp.*` tensors.
3. Replaces matching text tensors (`blk.*`, `token_embd.*`, `output.*`,
   `output_norm.*`) with Ornith tensors. The script aliases
   `.ssm_dt.bias` source tensors to donor `.ssm_dt` names.
4. Preserves packed quantized tensor bytes directly. For non-quantized tensors,
   it writes the GGUFReader storage array shape, not the logical tensor shape.
   Passing logical `raw_shape` for F16/F32 tensors will transpose metadata and
   create a model that Ollama can show but cannot load.
5. Allows same-element reshapes only for non-quantized tensors when a compatible
   donor uses a singleton dimension.
6. Patches `*.rope.dimension_sections` from 3 to 4 values when needed, sets
   `clip.has_vision_encoder=true`, and writes `general.name`,
   `general.basename`, and `general.finetune`.
7. Verifies every output tensor shape against the donor before returning a
   successful splice report.
8. Writes a Modelfile with `RENDERER qwen3.5`, `PARSER qwen3.5`,
   `REQUIRES 0.17.1`, 262k context, and the Qwen stop token.

### Outputs and validation

For each size, expect:

```text
training_suite/outputs/ornith_vision/<size>/ornith-vision-<size>.q4km.gguf
training_suite/outputs/ornith_vision/<size>/Modelfile
training_suite/outputs/ornith_vision/<size>/splice_report.json
training_suite/logs/ornith_vision_test_<size>.json
```

The 9B report should show `replaced=427`, `kept=456`, `vision_tensors=441`,
`mtp_tensors=15`, and `shape_mismatches_vs_donor=0`. The 35B report should show
`replaced=733`, `kept=461`, `vision_tensors=441`, `mtp_tensors=20`,
`reshaped=40`, and `shape_mismatches_vs_donor=0`.

The smoke test checks all three required additions:

```bash
training_suite/.venv/bin/python - <<'PY'
from training_suite.evals.runner import capability_gate, tool_smoke
print(capability_gate("ornith-vision:9b", target=["vision", "tools", "thinking"]))
print(tool_smoke("ornith-vision:9b"))
PY
```

The script also sends a rendered `42` image through `/api/chat`. Thinking models
can spend many tokens in the parsed `thinking` field before answer content, so
the vision smoke uses `num_predict=512`. If a manual image probe returns empty
content with a non-empty `thinking` field, increase `num_predict` before
assuming vision is broken.

### Publish

After local tests pass:

```bash
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 9b --reuse-existing --copy-remote --push
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 35b --reuse-existing --copy-remote --push
```

Equivalent manual commands:

```bash
ollama cp ornith-vision:9b robit/ornith-vision:9b
ollama push robit/ornith-vision:9b
ollama cp ornith-vision:35b robit/ornith-vision:35b
ollama push robit/ornith-vision:35b
```

### Troubleshooting

- Use `training_suite/.venv/bin/python`; the system `python` may not exist or
  may not have `gguf`, `httpx`, and `Pillow`.
- Keep outputs under `training_suite/outputs/ornith_vision/`; `/tmp` may be too
  small for 6-23 GB GGUF artifacts.
- If `/api/chat` returns HTTP 500 or Ollama says the model failed to load,
  inspect `ollama show <tag> --verbose`. Transposed non-quantized shapes such as
  `F16 [4304 1152]` for vision MLP weights mean an older/broken splicer wrote
  logical shapes incorrectly; regenerate with the current script.
- If `ollama push` fails, confirm `ollama signin` and registry permissions for
  the `robit/ornith-vision` namespace.
- Large pushes are slow on modest uplinks; the script allows up to three hours
  per `ollama push` so the 35B tag can finish without a false timeout.

### Re-quantization (if needed)

To convert from scratch using llama.cpp:

```bash
# 1. Clone llama.cpp
git clone --depth 1 https://github.com/ggerganov/llama.cpp vendor/llama.cpp

# 2. Convert HF safetensors → F16 GGUF
python vendor/llama.cpp/convert_hf_to_gguf.py \
  path/to/hf_model --outfile model.f16.gguf --outtype f16

# 3. Quantize to Q4_K_M
cmake -B vendor/llama.cpp/build vendor/llama.cpp -DLLAMA_CURL=OFF
cmake --build vendor/llama.cpp/build --target llama-quantize -j
vendor/llama.cpp/build/bin/llama-quantize model.f16.gguf model.q4km.gguf Q4_K_M

# 4. Create Ollama model
ollama create my-model:q4km -f Modelfile
# Modelfile contents:
# FROM ./model.q4km.gguf
# RENDERER qwen3.5
# PARSER qwen3.5
# PARAMETER num_ctx 262144
# PARAMETER temperature 0.6
# PARAMETER stop "<|im_end|>"
```

## Publishing to Ollama Registry

```bash
# 1. Copy to remote tag
ollama cp ornith-1.0-9b-tools:q4km robit/ornith:9b

# 2. Push
ollama push robit/ornith:9b
# Model available at https://ollama.com/robit/ornith:9b
```

You must be signed in (`ollama signin`) before pushing.

## Mandatory End-of-Session Storage Cleanup

Any agent that downloads or produces model weights must treat cleanup as the
final workflow phase. This prevents completed sessions from accumulating
duplicate safetensors, full-precision exports, quantizations, and Ollama
imports on the host.

Cleanup is allowed only after all of the following are true:

1. The final Ollama model was created successfully.
2. Required capability and behavioral tests passed, including live vision,
   tools, and thinking probes when those capabilities are expected.
3. `ollama push` completed successfully for every requested remote tag.
4. The public tag or remote registry manifest was fetched and verified.
5. The source repository, revision, quantization, Modelfile, license, and test
   results needed to reproduce the build were recorded.

After the gate passes:

- Remove every run-local `*.safetensors` file, including downloaded model
  shards, merged checkpoints, and LoRA adapters that have already been
  distilled into and verified in the published Ollama model.
- Remove run-local F16/BF16 GGUF intermediates, partial downloads, conversion
  shards, and redundant quantized GGUF copies. Retain a standalone final GGUF
  only when it is an explicit deliverable or is needed for another active run.
- Remove only cache entries and temporary directories created for the completed
  run. Hugging Face caches and donor/base weights may be shared by other work,
  so inspect references before deleting them.
- Keep Modelfiles, manifests or digests, splice reports, evaluation reports,
  logs, licenses, and compact metadata in the repository output structure.
- Never manually remove files from `/srv/ollama/models/blobs`, another
  `OLLAMA_MODELS` blob directory, or its manifest tree. Use
  `ollama rm <obsolete-local-tag>` for local tags that are no longer needed.
- Measure and report the run directory's size before and after cleanup so the
  reclaimed space is visible in the final handoff.

Use a unique, explicit directory for each run. Before any deletion, list the
exact candidate files with `find` and inspect the directory with `du -sh`.
Never run a recursive cleanup against the repository root, `outputs/`, a cache
root, `$HOME`, or a path assembled from an unset variable.

## Dashboard Styling

The dashboard uses a **GitHub-dark theme** (matching ollama.com/robit/ornith):

- Background: `#0d1117`, cards: `#161b22`, borders: `#30363d`
- Accent: `#58a6ff` (blue), green: `#3fb950`, red: `#f85149`
- Card-based layout with `.card` > `.card-header` > `.card-body`
- Color-coded badges: `.badge-green`, `.badge-blue`, `.badge-purple`, etc.
- Status pills: `.status-succeeded` (green), `.status-running` (yellow), `.status-failed` (red)
- Capability pills: `.cap-vision` (green), `.cap-tools` (blue), `.cap-thinking` (orange)
- Dark form elements, monospace code blocks, progress bars

### Template Structure

```
base.html → topbar (Dashboard, Intake, Datasets, Actions, Export, Evaluation, Compare)
  ├── dashboard.html       → models table + recent jobs + recent evals (3 cards)
  ├── job.html             → summary grid + live log with auto-refresh
  ├── model_detail.html    → summary grid + capabilities + repair plan + metadata
  ├── intake.html          → form with source, raw_source, gguf_path, ollama_model
  ├── datasets.html        → register form + registered table + curation recipes
  ├── actions.html         → action selector + jobs table
  ├── export.html          → Modelfile generator + create/push buttons
  ├── evaluation.html      → eval runner + eval runs history
  ├── compare_models.html  → multi-model side-by-side comparison
  └── compare_evals.html   → multi-eval result side-by-side comparison
```

## Complete End-to-End Workflow

### For an existing HF GGUF model (like Ornith):

```bash
# 1. Pull GGUF into Ollama
ollama pull hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest

# 2. Create Modelfile with tool calling
mkdir -p outputs/ollama/ornith-9b-tools
cat > outputs/ollama/ornith-9b-tools/Modelfile << 'EOF'
FROM hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF:latest
TEMPLATE {{ .Prompt }}
RENDERER qwen3.5
PARSER qwen3.5
PARAMETER num_ctx 262144
PARAMETER num_predict 16384
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER stop "<|im_end|>"
EOF

# 3. Create in Ollama
ollama create ornith-1.0-9b-tools:q4km -f outputs/ollama/ornith-9b-tools/Modelfile

# 4. Verify tool calling works
python -c "
from training_suite.evals.runner import tool_smoke, capability_gate
print('Gate:', capability_gate('ornith-1.0-9b-tools:q4km'))
print('Smoke:', tool_smoke('ornith-1.0-9b-tools:q4km'))
"

# 5. Publish
ollama cp ornith-1.0-9b-tools:q4km robit/ornith:9b
ollama push robit/ornith:9b
```

### Using the REST API (agent-friendly):

```bash
# Intake a model
curl -X POST http://localhost:7860/api/models \
  -H 'Content-Type: application/json' \
  -d '{"source": "https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B"}'

# List models
curl http://localhost:7860/api/models

# Start a job
curl -X POST http://localhost:7860/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{"action": "bootstrap"}'

# Check job progress (poll every 2s)
curl http://localhost:7860/api/jobs/1

# Run capability gate eval
curl -X POST http://localhost:7860/api/evals \
  -H 'Content-Type: application/json' \
  -d '{"model_name": "robit/ornith:9b", "eval_key": "capability-gate"}'
```

## Currently Published Models

| Tag | Size | Pull Command |
|-----|------|-------------|
| `robit/ornith:9b` | 5.6 GB | `ollama pull robit/ornith:9b` |
| `robit/ornith:35b` | 21 GB | `ollama pull robit/ornith:35b` |

Both have `tools`, `thinking`, `completion` capabilities with Q4_K_M quantization
and `RENDERER qwen3.5` for structured tool calls.

## Ornith Evaluation Results

### Diverse Stochastic Eval (38 tests, 9 categories)

| Model | Overall | Code | Conciseness | Inst. Follow | Reasoning | Format Bleed | System Prompt | Tangent Resist. |
|-------|---------|------|-------------|-------------|-----------|-------------|--------------|-----------------|
| Ornith 9B | **92.1%** (35/38) | 100% | 100% | 100% | 50% | 66.7% | 66.7% | 100% |
| Ornith 35B | **86.8%** (33/38) | 0% | 100% | 77.8% | 50% | 100% | 66.7% | 100% |

### Repetition Stress Test (15 tests, 5 categories)

| Model | Overall Loop Rate | Math | Open-ended | Ambiguous | Complex | Diversity |
|-------|-----------------|------|------------|-----------|---------|-----------|
| Ornith 9B | **33.3%** (5/15) | 100% (3/3) | 33.3% (1/3) | 33.3% (1/3) | 0% (0/3) | 0% (0/3) |
| Ornith 35B | **20.0%** (3/15) | 100% (3/3) | 0% (0/3) | 0% (0/3) | 0% (0/3) | 0% (0/3) |

**Key observations:**
- 9B leads on diverse eval (92.1% vs 86.8%), especially code and instruction following
- 35B leads on repetition resistance (80% clean vs 66.7%) and format bleed
- Both models struggle with math-heavy loop categories
- `pirate_speak` test passed by 35B but failed by 9B

### Eval Scripts

| Script | Tests | Description |
|--------|-------|-------------|
| `eval_diverse.py` | 38 | 9-category stochastic eval (code, reasoning, conciseness, etc.) |
| `eval_repetition.py` | 15 | N-gram repetition + cyclic pattern detection |
| `eval_via_ollama.py` | 200+ | GSM8K (100), MMLU (100), Vision probes (3) |
| `hard_tool_tests.py` | 22 | Adversarial tool-calling via Open Agents gateway |
| `probe_tools.py` | 1 | Tool smoke test (get_weather) |

## Dashboard Styling
