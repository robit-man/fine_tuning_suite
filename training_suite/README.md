# Qwen3.5-9B Reasoning Distillation — Fine-Tuning Suite

End-to-end pipeline for distilling reasoning capabilities into Qwen3.5-9B via LoRA SFT, with vision preservation, tool calling, and thinking mode support.

Produces Ollama-ready GGUF models with full capabilities: **vision, tools, thinking, instruction following**.

## Quick Start

```bash
# 1. Install dependencies
python app.py bootstrap

# 2. Curate training data (R7 — research-backed, additive approach)
python curate_r7.py

# 3. Train with LoRA SFT
DISTILL_TRAIN_FILE=r7_additive_train \
DISTILL_VAL_FILE=r7_additive_val \
DISTILL_OUTPUT_SUFFIX=.r7-additive \
DISTILL_LR=1e-4 DISTILL_LORA_R=32 DISTILL_LORA_ALPHA=64 \
DISTILL_EPOCHS=1 DISTILL_PATIENCE=3 \
python app.py train

# 4. Export to vision-capable GGUF (proven pipeline)
python splice_and_export.sh  # see INSTRUCTIONS.md for manual steps

# 5. Evaluate
python eval_diverse.py <model_name> --base qwen3.5:9b
python eval_repetition.py <model_name>
```

## Local Dashboard

```bash
# Run from the repository root
python3 -m venv .venv-dashboard
.venv-dashboard/bin/python -m pip install flask jinja2 werkzeug httpx
.venv-dashboard/bin/python -m training_suite db-init
.venv-dashboard/bin/python -m training_suite ornith-seed --donor-model qwen3.5:9b
.venv-dashboard/bin/python -m training_suite web --host 127.0.0.1 --port 7860
```

The dashboard exposes model intake, dataset registration, action jobs, Ollama
export, and evaluation gates over a shared SQLite state store in
`state/suite.sqlite3`. It does not download large model weights during intake;
it inspects Hugging Face metadata, local GGUF files, and local `ollama show`
output, then launches heavyweight work as monitored jobs.

## File Index

### Core Training Harness

| File | Purpose |
|------|---------|
| `app.py` | Main training harness — bootstrap, prepare data, train (DDP), eval, export |
| `requirements.txt` | Pinned Python dependencies |

### Data Curation

| File | Purpose |
|------|---------|
| `curate_r5.py` | R5 dataset: Bespoke-Stratos-17k + Tulu-3 + SlimOrca + format examples |
| `curate_r6.py` | R6 dataset: Mixture-of-Thoughts + OpenThoughts + anti-loop data |
| `curate_r7.py` | R7 dataset: R5 base + PrimeIntellect SYNTHETIC-1 + anti-loop (additive) |
| `generate_antiloop.py` | Generate anti-repetition training data from model's own failures |

### Evaluation

| File | Purpose |
|------|---------|
| `eval_diverse.py` | 38-test stochastic eval across 9 categories (instruction, format, conciseness, etc.) |
| `eval_repetition.py` | 15-test repetition stress test (math, open-ended, ambiguous, diversity) |
| `eval_via_ollama.py` | GSM8K + MMLU + vision probes via Ollama API |
| `hard_tool_tests.py` | 12 adversarial tool-calling stress tests |
| `image_probe.py` | Vision capability probing (rendered text: HELLO, 42, BANANA) |
| `probe_tools.py` | Tool-calling capability checker |

### GGUF Export & Vision Splice

| File | Purpose |
|------|---------|
| `splice_vision_v2.py` | Splice trained text weights into base multimodal HF model (preserves vision) |
| `splice_vision_r7.py` | R7-specific vision splice |
| `gguf_text_surgery.py` | GGUF-level text tensor substitution (swap text in combined vision GGUF) |
| `gguf_patch_rope_sections.py` | Patch rope.dimension_sections from 3 to 4 elements |
| `gguf_set_vision_flag.py` | Add clip.has_vision_encoder flag to GGUF |

## Instruction Document

See **[INSTRUCTIONS.md](INSTRUCTIONS.md)** for the complete step-by-step guide covering:

1. Environment setup
2. Dataset selection and curation
3. Training configuration and hyperparameters
4. Vision-preserving GGUF export pipeline
5. Comprehensive capability testing
6. Ollama registration and upload

## Proven Pipeline (R7)

The pipeline that produces a working vision+tools+thinking model:

```
Filter linear_attn from LoRA adapter
    -> convert_lora_to_gguf.py (LoRA GGUF)
    -> llama-export-lora (merge into base qwen3.5:9b Q4_K_M)
    -> llama-quantize (F16 -> Q4_K_M)
    -> ollama create (RENDERER qwen3.5 + PARSER qwen3.5)
```

This preserves all 883 tensors (427 text + 441 vision + 15 MTP) with vision byte-for-byte identical to base.

## Key Hyperparameters

| Parameter | R5 (research) | R7 (additive) |
|-----------|---------------|---------------|
| Base model | Qwen/Qwen3.5-9B | Qwen/Qwen3.5-9B |
| LoRA rank | 32 | 32 |
| LoRA alpha | 64 | 64 |
| Learning rate | 1e-4 | 1e-4 |
| Epochs | 1 | 1 |
| Batch size | 1 (grad_accum=8) | 1 (grad_accum=8) |
| Max seq length | 5120 | 5120 |
| Early stopping | patience=3 | patience=3 |

## Datasets (HuggingFace)

### Primary Training Data

| Dataset | Size | Use | Link |
|---------|------|-----|------|
| Bespoke-Stratos-17k | 17k | Reasoning distillation (DeepSeek-R1, Qwen-native) | [bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) |
| Tulu 3 SFT Mixture | 939k | Production instruction following diversity | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| SlimOrca | 518k | Curated GPT-4 instructions (Orca-style) | [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) |
| PrimeIntellect SYNTHETIC-1 | 894k | Verified math/code/STEM reasoning | [PrimeIntellect/SYNTHETIC-1-SFT-Data](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1-SFT-Data) |

### Anti-Repetition & Supplementary

| Dataset | Size | Use | Link |
|---------|------|-----|------|
| Mixture-of-Thoughts | 350k | Verified reasoning traces | [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) |
| OpenThoughts-114k | 114k | DeepSeek-R1 curated traces | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| Alpaca-cleaned | 52k | General instruction following | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) |

### Evaluation

| Dataset | Use | Link |
|---------|-----|------|
| GSM8K | Math reasoning holdout | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| MMLU | Multi-subject knowledge | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |

### Additional Referenced Datasets

| Dataset | Link |
|---------|------|
| PrimeIntellect SYNTHETIC-2-SFT-verified (105k) | [PrimeIntellect/SYNTHETIC-2-SFT-verified](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified) |
| INTELLECT-MATH-SFT-Data (733k) | [PrimeIntellect/INTELLECT-MATH-SFT-Data](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-MATH-SFT-Data) |
| NuminaMath-CoT (860k) | [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| OpenHermes-2.5 (1M) | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |
| WildChat-1M (838k) | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) |

### Base Model

[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) (Apache 2.0)

## Evaluation Results

| Model | Diverse Eval | Repetition | Vision | Tools | Thinking |
|-------|-------------|------------|--------|-------|----------|
| Base qwen3.5:9b | 79.0% | N/A | Yes | Yes | Yes |
| cudabenchmarktest (R3) | 73.7% | N/A | No | Yes | Yes |
| R5 (research) | 84.2% | 20% loops | No | Yes | Yes |
| R7 (additive) | 86.8% | N/A | Yes | Yes | Yes |

## Models on Ollama

- `robit/qwen3.5-9b-r7-research:q4km` — text model (tools + thinking)
- `robit/qwen3.5-9b-r7-research-vision:q4km` — full vision model (vision + tools + thinking)

## License

Training data licenses vary by source. Model weights are derived from Qwen3.5-9B (Apache 2.0).
