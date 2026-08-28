# Fine-Tuning Suite

End-to-end pipeline for LLM fine-tuning, tool splicing, GGUF export, and Ollama registry publishing.  
Supports Qwen-based models from 9B to 397B with **vision, tools, thinking, instruction following**.

Includes a full **dark-theme Flask dashboard** with **RESTful API** for agent/MCP toolkit integration,  
and a `tool_splice.py` pipeline for importing HuggingFace models → Ollama with tool-calling config.

## Quick Start — Training Pipeline

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

## Dashboard — Dark Theme + REST API

The Flask dashboard provides model intake, job monitoring, GGUF export, and evaluation gates.  
**Dark GitHub-themed UI** matching ollama.com/robit/ornith reference design.

```bash
cd training_suite
pip install flask jinja2 werkzeug httpx
python -m training_suite db-init
python -m training_suite web --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`.

### RESTful API (Agent/MCP Toolkit)

All endpoints return JSON. Full list in `AGENTS.md`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/models` | List models |
| `POST` | `/api/models` | Intake model |
| `GET` | `/api/models/<id>` | Model detail |
| `GET` | `/api/jobs` | List jobs |
| `POST` | `/api/jobs` | Start job |
| `GET` | `/api/jobs/<id>` | Job detail + log |
| `POST` | `/api/jobs/<id>/cancel` | Cancel job |
| `GET` | `/api/datasets` | List datasets |
| `POST` | `/api/datasets` | Register dataset |
| `GET` | `/api/evals` | List eval runs |
| `POST` | `/api/evals` | Run evaluation |
| `GET` | `/api/actions` | Available actions |

## Quick Start — Tool Splice & Ollama Upload

Import any HuggingFace GGUF model into Ollama with proper tool-calling config:

```bash
python tool_splice.py 9b     # Ornith-1.0-9B → Ollama with RENDERER/PARSER
python tool_splice.py 35b    # Ornith-1.0-35B → Ollama
python tool_splice.py both   # Both sizes
```

Add `--no-eval` to skip evaluation. See `AGENTS.md` for full agent instructions.

## Quick Start — Ornith Vision Tensor Splice

Append vision capability to the local Ornith tool/thinking GGUFs by using a compatible Qwen multimodal
GGUF as the skeleton and transplanting Ornith text tensors in-place:

```bash
# Build, register, and smoke-test the 9B vision variant first
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 9b --create --test

# Then build/register/test the 35B variant
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 35b --create --test

# Reuse an existing GGUF for create/test/push retries
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 9b --reuse-existing --create --test

# Publish targets after local tests pass
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 9b --reuse-existing --copy-remote --push
training_suite/.venv/bin/python training_suite/ornith_vision_splice.py 35b --reuse-existing --copy-remote --push
```

Preset inputs are `ornith-1.0-9b-tools:q4km` + `qwen3.5:9b` and
`ornith-1.0-35b-tools:q4km` + `qwen3.6:35b`. Outputs are written under
`training_suite/outputs/ornith_vision/<size>/` with a splice report and Modelfile.
The generated Ollama variants use `RENDERER qwen3.5` + `PARSER qwen3.5` so the
capability gate should report `vision`, `tools`, and `thinking`.

## End-of-Session Storage Cleanup

Every completed training, distillation, conversion, or publishing session must
end with a storage cleanup. Do not delete artifacts until the final Ollama tag
has passed its required capability tests, `ollama push` has succeeded, and the
remote tag or registry manifest has been verified.

After that gate passes, remove the completed run's local `*.safetensors`
checkpoints and adapters, downloaded Hugging Face weight shards, temporary
F16/BF16 GGUFs, and redundant conversion outputs. Keep the Modelfile, source
revision, evaluation reports, licenses, and other small reproducibility
metadata. Use `ollama rm <obsolete-local-tag>` for Ollama-managed models; never
delete files directly from the Ollama blob or manifest store.

Use a unique output directory per run and inspect its exact contents and size
before deletion. See [End-of-Session Cleanup](training_suite/INSTRUCTIONS.md#phase-6-end-of-session-cleanup)
for the full verification and cleanup checklist.

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
| `ornith_vision_splice.py` | GGUF-native Ornith text tensor transplant into Qwen vision donors; create/test/copy/push Ollama variants |
| `gguf_patch_rope_sections.py` | Patch rope.dimension_sections from 3 to 4 elements |
| `gguf_set_vision_flag.py` | Add clip.has_vision_encoder flag to GGUF |

## Instruction Document

See **[INSTRUCTIONS.md](training_suite/INSTRUCTIONS.md)** for the complete step-by-step guide covering:

1. Environment setup
2. Dataset selection and curation
3. Training configuration and hyperparameters
4. Vision-preserving GGUF export pipeline
5. Comprehensive capability testing
6. Ollama registration and upload
7. End-of-session storage cleanup

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

| Dataset | Size | Use | License | Link |
|---------|------|-----|---------|------|
| Bespoke-Stratos-17k | 17k | Reasoning distillation (DeepSeek-R1 traces, Qwen-native) | Apache 2.0 | [bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) |
| Tulu 3 SFT Mixture | 939k | Production instruction following diversity (18 sources) | ODC-BY 1.0 | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| SlimOrca | 518k | Curated GPT-4 instruction data (Orca-style) | MIT | [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) |
| PrimeIntellect SYNTHETIC-1 | 894k | Verified math/code/STEM reasoning (DeepSeek-R1, filtered) | Apache 2.0 | [PrimeIntellect/SYNTHETIC-1-SFT-Data](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1-SFT-Data) |

### Anti-Repetition & Supplementary

| Dataset | Size | Use | License | Link |
|---------|------|-----|---------|------|
| Mixture-of-Thoughts | 350k | Verified reasoning traces (math, code, science) | Apache 2.0 | [open-r1/Mixture-of-Thoughts](https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts) |
| OpenThoughts-114k | 114k | DeepSeek-R1 curated reasoning traces | Apache 2.0 | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| Alpaca-cleaned | 52k | General instruction following (used in R5) | CC-BY-4.0 | [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) |

### Evaluation Benchmarks

| Dataset | Use | Link |
|---------|-----|------|
| GSM8K | Math reasoning holdout (100 locked samples) | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) |
| MMLU | Multi-subject knowledge evaluation | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) |

### Other PrimeIntellect Datasets (Referenced)

| Dataset | Size | Use | Link |
|---------|------|-----|------|
| SYNTHETIC-2-SFT-verified | 105k | Highest-quality verified (DeepSeek-R1-0528) | [PrimeIntellect/SYNTHETIC-2-SFT-verified](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified) |
| INTELLECT-MATH-SFT-Data | 733k | Math-specific (QwQ-generated, NuminaMath) | [PrimeIntellect/INTELLECT-MATH-SFT-Data](https://huggingface.co/datasets/PrimeIntellect/INTELLECT-MATH-SFT-Data) |
| NuminaMath-CoT | 860k | Math with chain-of-thought (K12 to olympiad) | [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| OpenHermes-2.5 | 1M | General instruction mix (14 sources) | — | [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |
| WildChat-1M | 838k | Real ChatGPT user conversations | ODC-BY 1.0 | [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) |

### Base Model

| Model | Link |
|-------|------|
| Qwen3.5-9B | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) |

## Evaluation Results

| Model | Diverse Eval | Repetition | Vision | Tools | Thinking |
|-------|-------------|------------|--------|-------|----------|
| Base qwen3.5:9b | 79.0% | N/A | Yes | Yes | Yes |
| cudabenchmarktest (R3) | 73.7% | N/A | No | Yes | Yes |
| R5 (research) | 84.2% | 20% loops | No | Yes | Yes |
| R7 (additive) | 86.8% | N/A | Yes | Yes | Yes |

## Models on Ollama

### Ornith (deepreinforce-ai) — Tool-Spliced

| Tag | Size | Capabilities | Pull |
|-----|------|-------------|------|
| `robit/ornith:9b` | 5.6 GB | tools, thinking, completion | `ollama pull robit/ornith:9b` |
| `robit/ornith:35b` | 21 GB | tools, thinking, completion | `ollama pull robit/ornith:35b` |
| `robit/ornith-vision:9b` | 6.7 GB | vision, tools, thinking, completion | `ollama pull robit/ornith-vision:9b` |
| `robit/ornith-vision:35b` | 22.6 GB | vision, tools, thinking, completion | `ollama pull robit/ornith-vision:35b` |

Both use `RENDERER qwen3.5` + `PARSER qwen3.5` for structured tool calls,  
`num_ctx 262144`, temperature 0.6, top_p 0.95.

### Legacy Qwen Fine-Tunes

| Tag | Size | Capabilities |
|-----|------|-------------|
| `robit/qwen3.5-9b-r7-research:q4km` | 5.6 GB | tools, thinking |
| `robit/qwen3.5-9b-r7-research-vision:q4km` | 19 GB | vision, tools, thinking |

## License

Training data licenses vary by source. Model weights are derived from Qwen3.5-9B (Apache 2.0)  
and Ornith-1.0 (deepreinforce-ai).
