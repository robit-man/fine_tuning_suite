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
```

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
