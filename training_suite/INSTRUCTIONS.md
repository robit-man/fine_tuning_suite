# Complete Training & Capability Merge Instructions

Step-by-step guide for distilling reasoning into Qwen3.5-9B with full capability preservation (vision, tools, thinking).

## Prerequisites

- 3x NVIDIA A100 80GB (or equivalent; 1x works with longer training time)
- Python 3.12+, CUDA 12.x
- Ollama installed (`ollama serve` running)
- llama.cpp built with `llama-quantize` and `llama-export-lora` binaries
- HuggingFace `datasets` library

## Phase 1: Environment Setup

```bash
# Bootstrap creates .venv with all dependencies
python app.py bootstrap

# Verify llama.cpp tools
ls vendor/llama.cpp/build/bin/llama-export-lora
ls vendor/llama.cpp/build/bin/llama-quantize

# If missing, build llama.cpp:
cd vendor/llama.cpp
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --target llama-quantize llama-export-lora -j
cd ../..
```

## Phase 2: Data Curation

### Strategy

Training data must be DIVERSE to avoid mode collapse. The critical lesson from R3 (which failed) was that 93.8% math-only data destroys instruction following.

Proven mix ratios:
- 45-50% reasoning traces (Bespoke-Stratos, PrimeIntellect)
- 30-35% instruction following (Tulu-3, SlimOrca)
- 5% short Q&A
- 2-5% format-constrained examples (YES/NO, JSON, one-word)
- 1-2% conversational multi-turn
- 1% concise code

### R7 Approach (Additive — Recommended)

```bash
# This starts from R5's proven base and layers additional data on top
python curate_r7.py
```

Produces: `data/splits/r7_additive_{train,val,test}.jsonl`

### Key Datasets on HuggingFace

All datasets used in this pipeline with direct links:

| Dataset | HuggingFace Link | Load Command |
|---------|-----------------|--------------|
| Bespoke-Stratos-17k | [bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | `load_dataset("bespokelabs/Bespoke-Stratos-17k")` |
| Tulu 3 SFT Mixture | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | `load_dataset("allenai/tulu-3-sft-mixture", streaming=True)` |
| SlimOrca | [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | `load_dataset("Open-Orca/SlimOrca", streaming=True)` |
| PrimeIntellect SYNTHETIC-1 | [PrimeIntellect/SYNTHETIC-1-SFT-Data](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1-SFT-Data) | `load_dataset("PrimeIntellect/SYNTHETIC-1-SFT-Data", streaming=True)` |
| Mixture-of-Thoughts | [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) | `load_dataset("open-r1/Mixture-of-Thoughts", "all", streaming=True)` |
| OpenThoughts-114k | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | `load_dataset("open-thoughts/OpenThoughts-114k", streaming=True)` |
| Alpaca-cleaned | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) | `load_dataset("yahma/alpaca-cleaned")` |
| GSM8K (eval) | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `load_dataset("openai/gsm8k", "main", split="test")` |
| MMLU (eval) | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | `load_dataset("cais/mmlu", "all", split="test")` |

Additional datasets referenced in research but not directly used in training:

| Dataset | Link |
|---------|------|
| PrimeIntellect SYNTHETIC-2-SFT-verified | [PrimeIntellect/SYNTHETIC-2-SFT-verified](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified) |
| INTELLECT-MATH-SFT-Data | [PrimeIntellect/INTELLECT-MATH-SFT-Data](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-MATH-SFT-Data) |
| NuminaMath-CoT | [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| OpenHermes-2.5 | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |
| WildChat-1M | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) |
| INTELLECT-1 Collection | [PrimeIntellect/intellect-1-dataset](https://huggingface.co/collections/PrimeIntellect/intellect-1-dataset) |

Base model: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (Apache 2.0)

```python
from datasets import load_dataset

# Reasoning (primary)
stratos = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
pi_sft = load_dataset("PrimeIntellect/SYNTHETIC-1-SFT-Data", split="train", streaming=True)

# Instruction diversity
tulu3 = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
orca = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)

# Anti-repetition (filter traces with loops)
mot = load_dataset("open-r1/Mixture-of-Thoughts", "all", split="train", streaming=True)
```

### Critical: Repetition Filtering

Always filter training data for repetition loops:

```python
from collections import Counter
import re

def has_repetition(text, threshold=0.08):
    sentences = re.split(r'[.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) < 3:
        return False
    counts = Counter(sentences)
    repeat_ratio = sum(c - 1 for c in counts.values() if c >= 2) / len(sentences)
    return repeat_ratio > threshold
```

### Critical: Format-Constrained Examples

Include 50+ explicit format-constrained examples to prevent format bleeding:
- YES/NO responses
- One-word answers
- Just-the-number answers
- JSON format responses
- Comma-separated lists
- Numbered lists

Without these, the model will output markdown headers and analysis blocks for simple questions.

## Phase 3: Training

### Hyperparameters (Proven for Qwen3.5-9B)

```bash
DISTILL_TRAIN_FILE=r7_additive_train \
DISTILL_VAL_FILE=r7_additive_val \
DISTILL_OUTPUT_SUFFIX=.r7-additive \
DISTILL_LR=1e-4 \
DISTILL_LORA_R=32 \
DISTILL_LORA_ALPHA=64 \
DISTILL_EPOCHS=1 \
DISTILL_PATIENCE=3 \
CUDA_VISIBLE_DEVICES=0,1,2 \
python app.py train
```

### Key Config Notes

- **LoRA r=32, alpha=64**: Literature says r=64-128 ideal but r=32 fits in VRAM with 3x A100 and zombie processes. r=16 is too low (causes regressions).
- **LR=1e-4**: Standard for LoRA on Qwen-family models. 2e-4 caused catastrophic forgetting in R4.
- **1 epoch**: For 4000+ samples, 1 epoch is sufficient. The Tulu-3 paper confirms this.
- **Patience=3**: Early stopping on eval loss.
- **target_modules="all-linear"**: Auto-detects all linear layers (excludes embeddings/lm_head).
- **Completion-only loss masking**: Only assistant tokens contribute to loss.

### Training Output

Adapter saved to: `outputs/checkpoints/<name>/final_adapter/`

## Phase 4: Vision-Preserving GGUF Export

This is the critical capability-preservation step. The proven pipeline:

### Step 1: Filter LoRA Adapter

Remove `linear_attn` (Gated DeltaNet) tensors — llama.cpp can't handle v-head reorder on low-rank decomposed tensors.

```python
from safetensors.torch import load_file, save_file
import json, shutil
from pathlib import Path

adapter_dir = Path("outputs/checkpoints/<name>/final_adapter")
filtered_dir = Path("outputs/checkpoints/<name>/final_adapter_filtered")
filtered_dir.mkdir(exist_ok=True)

st = load_file(str(adapter_dir / "adapter_model.safetensors"))
keep = {k: v for k, v in st.items() if "linear_attn" not in k}
save_file(keep, str(filtered_dir / "adapter_model.safetensors"))

# Copy config and tokenizer files
cfg = json.load(open(adapter_dir / "adapter_config.json"))
json.dump(cfg, open(filtered_dir / "adapter_config.json", "w"), indent=2)
for f in adapter_dir.iterdir():
    if f.name not in ("adapter_model.safetensors", "adapter_config.json", "training_args.bin"):
        shutil.copy2(f, filtered_dir / f.name)
```

### Step 2: Convert LoRA to GGUF

```bash
python vendor/llama.cpp/convert_lora_to_gguf.py \
    outputs/checkpoints/<name>/final_adapter_filtered \
    --outfile outputs/gguf_adapter/lora.f16.gguf \
    --outtype f16
```

### Step 3: Merge into Base GGUF

Use a base that has correct `rope.dimension_sections` (4 elements). The R5 vision GGUF or any working vision GGUF with `[11,11,10,0]` works.

```bash
# BASE_GGUF must be the qwen3.5:9b Q4_K_M blob from Ollama
# (883 tensors with vision, OR a previously working vision GGUF)
BASE_GGUF="/srv/ollama/models/blobs/sha256-<hash>"

vendor/llama.cpp/build/bin/llama-export-lora \
    -m "$BASE_GGUF" \
    --lora outputs/gguf_adapter/lora.f16.gguf \
    -o outputs/gguf_merged/merged.f16.gguf
```

This produces an 883-tensor F16 GGUF with:
- 128 text tensors modified by LoRA
- 755 tensors (vision + MTP + other) byte-for-byte from base

### Step 4: Quantize

```bash
vendor/llama.cpp/build/bin/llama-quantize \
    outputs/gguf_merged/merged.f16.gguf \
    outputs/gguf_merged/merged.q4km.gguf \
    Q4_K_M
```

**If quantize fails with `rope.dimension_sections` error**: Use a base GGUF that already has 4-element rope sections (like the R5 vision GGUF) instead of the raw `qwen3.5:9b` blob.

### Step 5: Register with Ollama

```bash
cat > Modelfile << 'EOF'
FROM ./merged.q4km.gguf

RENDERER qwen3.5
PARSER qwen3.5

PARAMETER num_ctx 262144
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER stop "<|im_end|>"
EOF

ollama create my-model:q4km -f Modelfile
```

**Critical**: `RENDERER qwen3.5` and `PARSER qwen3.5` are REQUIRED for tool calling to work. Without them, Ollama reports "no tools capability".

### Step 6: Upload to Ollama

```bash
ollama cp my-model:q4km robit/my-model:q4km
ollama push robit/my-model:q4km
```

## Phase 5: Comprehensive Evaluation

### Diverse Stochastic Eval (38 tests, 9 categories)

```bash
python eval_diverse.py my-model:q4km --base qwen3.5:9b --out logs/eval.json
```

Tests: instruction following, conciseness, format compliance, conversational memory, system prompt adherence, tangent resistance, stochastic stability, code generation, reasoning.

### Repetition Stress Test (15 tests)

```bash
python eval_repetition.py my-model:q4km --out logs/repetition.json
```

Tests: multi-step math, open-ended analysis, ambiguous paradoxes, diversity challenges, complex optimization.

### Tool Calling Tests (12 adversarial)

```bash
python hard_tool_tests.py my-model:q4km
```

Tests: many-tool selection, nested objects, arrays, enums, chaining, G7 capitals.

### Vision Probes (3 rendered text)

```bash
python image_probe.py my-model:q4km
```

Tests: HELLO, 42, BANANA image recognition.

## Phase 6: End-of-Session Cleanup

Run this phase after every completed training, distillation, conversion, or
Ollama publishing session. Large intermediate files are disposable only after
the published model is proven recoverable.

### Cleanup Gate

Do not delete weight files until all of these checks pass:

1. `ollama create` succeeded for the final local tag.
2. The required evaluation suite passed. For multimodal agentic models, verify
   an actual image response, structured `tool_calls`, and a populated
   `message.thinking` field.
3. `ollama push` succeeded for each requested remote tag.
4. Fetch the public model page or remote registry manifest and confirm the
   expected tag, model layer, vision projector, renderer, and parser.
5. Record the source repository and revision, quantization, Modelfile, license,
   remote tag, layer digests, and evaluation results.

### Inventory Before Deletion

Keep each run's large artifacts in a unique directory. Replace
`RUN_NAME_HERE` with the completed run's literal directory name; do not leave
the placeholder unchanged.

```bash
cleanup_run_dir=$(realpath -- /srv/fine_tuning_suite/training_suite/outputs/sessions/RUN_NAME_HERE)

# Refuse broad or unresolved targets.
test -n "$cleanup_run_dir"
test "$cleanup_run_dir" != /srv/fine_tuning_suite/training_suite/outputs
test "$cleanup_run_dir" != /srv/fine_tuning_suite/training_suite/outputs/sessions
case "$cleanup_run_dir" in
  /srv/fine_tuning_suite/training_suite/outputs/sessions/*) ;;
  *) echo "Refusing cleanup outside the sessions directory" >&2; exit 1 ;;
esac
test -d "$cleanup_run_dir"

du -sh "$cleanup_run_dir"
find "$cleanup_run_dir" -type f \
  \( -name '*.safetensors' -o \
     -name '*.f16.gguf' -o \
     -name '*.bf16.gguf' -o \
     -name '*.partial' -o \
     -name '*.incomplete' \
  \) -print
```

Review that list before deleting anything. It must contain only artifacts from
the completed run.

### Remove Completed-Run Weight Artifacts

Once the cleanup gate and inventory review are complete:

```bash
# Safetensors are mandatory cleanup after their distilled/published Ollama
# replacement has been verified.
find "$cleanup_run_dir" -type f -name '*.safetensors' -delete

# Remove full-precision and interrupted conversion intermediates.
find "$cleanup_run_dir" -type f \
  \( -name '*.f16.gguf' -o \
     -name '*.bf16.gguf' -o \
     -name '*.partial' -o \
     -name '*.incomplete' \
  \) -delete

du -sh "$cleanup_run_dir"
```

Also remove redundant quantized GGUF copies when the final model is registered
in Ollama and published, unless a standalone GGUF is an explicit deliverable.
Delete only the exact, reviewed files for that run.

Use `ollama rm <obsolete-local-tag>` to remove unneeded local source, import,
or staging tags after their remote replacement is verified. Never manually
delete files from `/srv/ollama/models/blobs`, another `OLLAMA_MODELS` blob
directory, or the Ollama manifest tree.

Keep the Modelfile, source revision, license, remote tag and digests, evaluation
reports, logs, and other compact reproducibility metadata. Record the before
and after sizes in the session handoff.

## Lessons Learned (R3 through R7)

### What Destroys Models
- Training on >90% single-domain data (math only) causes mode collapse
- LoRA r=16 is too low for 9B models — causes catastrophic forgetting
- LR=2e-4 is too aggressive for LoRA on Qwen — use 1e-4
- Evaluating only at temperature=0 hides stochastic failures
- Replacing working data backbone instead of layering on top causes regression

### What Works
- Diverse training mix (50% reasoning + 35% instruction + 15% other)
- LoRA r=32 with alpha=64 (2x ratio)
- LR=1e-4 with cosine schedule
- 1 epoch for >1000 samples
- 75+ format-constrained examples prevent format bleeding
- Repetition filtering on all reasoning traces before training
- Additive data approach: keep what works, only add on top
- The llama-export-lora pipeline for vision preservation (byte-for-byte copy of base tensors)

### What Fixes Repetition
- Filter repetitive traces from training data (has_repetition > 0.08)
- Use Mixture-of-Thoughts / PrimeIntellect SYNTHETIC-1 (pre-filtered)
- Generate anti-loop corrected traces from model's own failures
- Variable-length reasoning: compress 50% of traces > 5000 chars

## Architecture Overview

```
Qwen3.5-9B (base)
    |
    v
LoRA SFT Training (r=32, target_modules=all-linear)
    |
    v
Filter linear_attn from adapter
    |
    v
convert_lora_to_gguf.py -> LoRA GGUF (256 tensors, ~116MB)
    |
    v
llama-export-lora -> Merge into base Q4_K_M (883 tensors, vision preserved)
    |
    v
llama-quantize -> Q4_K_M (~6.3GB)
    |
    v
ollama create (RENDERER qwen3.5 + PARSER qwen3.5)
    |
    v
Capabilities: Vision + Tools + Thinking + Instruction Following
```
