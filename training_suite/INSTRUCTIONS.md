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
