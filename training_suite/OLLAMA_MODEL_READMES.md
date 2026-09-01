# Ollama Model READMEs

All four model descriptions in one file. Copy each section to the corresponding model page on ollama.com.

The release-ready draft for `robit/qwen3.8-27b-e03-obliterated-omni` is
maintained at
[`docs/omni-adapter/model-page-template.md`](../docs/omni-adapter/model-page-template.md),
with exact provenance and local results in the linked release record. Replace
only the remote-publication placeholders after the pushed tags are verified.

---
---

# robit/qwen3.5-9b-r7-research:q4km

---

# Qwen3.5-9B R7 Research (Q4_K_M)

Fine-tuned Qwen3.5-9B with distilled reasoning from research-backed datasets. Trained via LoRA SFT with an additive data strategy that preserves base model capabilities while improving instruction following and reasoning.

## Capabilities

- **Thinking** — produces structured reasoning in `<think>` blocks
- **Tool calling** — structured `tool_calls` via Ollama `/api/chat` with `tools` parameter
- **Instruction following** — concise answers, format constraints, system prompt adherence

## Eval Results

| Benchmark | Score |
|-----------|-------|
| Diverse stochastic eval (38 tests, 9 categories) | **86.8%** |
| Base qwen3.5:9b on same eval | 79.0% |

## Training

- **Base model**: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Method**: LoRA SFT (r=32, alpha=64, LR=1e-4, 1 epoch)
- **Data**: Additive mix of 4043 samples from:
  - [bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) — DeepSeek-R1 reasoning traces
  - [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) — instruction diversity
  - [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) — curated GPT-4 instructions
  - [PrimeIntellect/SYNTHETIC-1-SFT-Data](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1-SFT-Data) — verified math/code/STEM
- **Training suite**: [robit-man/fine_tuning_suite](https://github.com/robit-man/fine_tuning_suite)

## Quickstart

```bash
ollama run robit/qwen3.5-9b-r7-research:q4km
```

## Parameters

- `RENDERER qwen3.5` + `PARSER qwen3.5` (enables tool calling)
- `temperature 0.6`, `top_p 0.95`
- `stop "<|im_end|>"`

## License

Derived from Qwen3.5-9B (Apache 2.0). Training data licenses vary by source.

---
---

# robit/qwen3.5-9b-r7-research-vision:q4km

---

# Qwen3.5-9B R7 Research Vision (Q4_K_M)

Fine-tuned Qwen3.5-9B with distilled reasoning and full vision support. 883 tensors (427 text + 441 vision + 15 MTP) — vision tower preserved byte-for-byte from base via `llama-export-lora` merge.

## Capabilities

- **Vision** — image understanding (reads text, describes scenes, answers visual questions)
- **Thinking** — structured reasoning in `<think>` blocks
- **Tool calling** — structured `tool_calls` via Ollama `/api/chat`
- **Instruction following** — concise answers, format constraints, system prompt adherence

## Eval Results

| Benchmark | Score |
|-----------|-------|
| Diverse stochastic eval (38 tests) | **86.8%** |
| Vision probe (rendered text) | **PASS** (reads "42" from image) |
| Tool calling | **PASS** (structured tool_calls) |
| Thinking | **PASS** (produces thinking field) |

## Training

- **Base model**: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Method**: LoRA SFT (r=32, alpha=64, LR=1e-4, 1 epoch), merged via `llama-export-lora` to preserve vision
- **Data**: Additive mix of 4043 samples from:
  - [bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) — DeepSeek-R1 reasoning traces
  - [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) — instruction diversity
  - [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) — curated GPT-4 instructions
  - [PrimeIntellect/SYNTHETIC-1-SFT-Data](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1-SFT-Data) — verified math/code/STEM
- **Vision preservation**: LoRA filtered (linear_attn removed) -> `convert_lora_to_gguf` -> `llama-export-lora` into base Q4_K_M GGUF. All 441 vision tensors + 15 MTP tensors unchanged.
- **Training suite**: [robit-man/fine_tuning_suite](https://github.com/robit-man/fine_tuning_suite)

## Quickstart

```bash
ollama run robit/qwen3.5-9b-r7-research-vision:q4km
```

### Image chat

```bash
IMG64=$(base64 -w0 path/to/image.jpg)
curl -s http://localhost:11434/api/chat \
  -d '{"model":"robit/qwen3.5-9b-r7-research-vision:q4km","messages":[{"role":"user","content":"Describe this image.","images":["'"$IMG64"'"]}]}'
```

## Parameters

- `RENDERER qwen3.5` + `PARSER qwen3.5` (enables tool calling + vision)
- `num_ctx 262144` (max context)
- `temperature 0.6`, `top_p 0.95`
- `stop "<|im_end|>"`

## License

Derived from Qwen3.5-9B (Apache 2.0). Training data licenses vary by source.

---
---

# robit/qwen3.5-9b-r5-research:q4km

---

# Qwen3.5-9B R5 Research (Q4_K_M)

Fine-tuned Qwen3.5-9B with distilled reasoning from research-backed datasets. R5 was the first round to use production-quality data sources (Bespoke-Stratos, Tulu-3, SlimOrca) and achieved 84.2% on diverse eval — surpassing the base model. Superseded by R7 (86.8%).

## Capabilities

- **Thinking** — produces structured reasoning in `<think>` blocks
- **Tool calling** — structured `tool_calls` via Ollama `/api/chat`
- **Instruction following** — concise answers, format constraints, system prompt adherence

## Eval Results

| Benchmark | Score |
|-----------|-------|
| Diverse stochastic eval (38 tests) | **84.2%** |
| Base qwen3.5:9b on same eval | 79.0% |

## Training

- **Base model**: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Method**: LoRA SFT (r=32, alpha=64, LR=1e-4, 1 epoch)
- **Data**: 4122 samples from:
  - [bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) — DeepSeek-R1 reasoning traces
  - [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) — instruction diversity
  - [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) — curated GPT-4 instructions
- **Training suite**: [robit-man/fine_tuning_suite](https://github.com/robit-man/fine_tuning_suite)

## Quickstart

```bash
ollama run robit/qwen3.5-9b-r5-research:q4km
```

## Parameters

- `RENDERER qwen3.5` + `PARSER qwen3.5`
- `temperature 0.6`, `top_p 0.95`
- `stop "<|im_end|>"`

## Note

R5 is superseded by [robit/qwen3.5-9b-r7-research:q4km](https://ollama.com/robit/qwen3.5-9b-r7-research:q4km) which adds PrimeIntellect data and scores 86.8%.

## License

Derived from Qwen3.5-9B (Apache 2.0). Training data licenses vary by source.

---
---

# robit/qwen3.5-9b-r5-vision:q4km

---

# Qwen3.5-9B R5 Vision (Q4_K_M)

Fine-tuned Qwen3.5-9B with distilled reasoning and full vision support. 883 tensors — vision tower preserved byte-for-byte from base. R5 was the first vision-capable distilled model. Superseded by R7 vision (86.8% eval + PrimeIntellect data).

## Capabilities

- **Vision** — image understanding (reads text, describes scenes, answers visual questions)
- **Thinking** — structured reasoning in `<think>` blocks
- **Tool calling** — structured `tool_calls` via Ollama `/api/chat`
- **Instruction following** — concise answers, format constraints

## Eval Results

| Benchmark | Score |
|-----------|-------|
| Diverse stochastic eval (38 tests) | **84.2%** |
| Vision probe (rendered text) | **PASS** |
| Tool calling | **PASS** |

## Training

- **Base model**: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- **Method**: LoRA SFT (r=32, alpha=64, LR=1e-4, 1 epoch), merged via `llama-export-lora`
- **Data**: 4122 samples from:
  - [bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) — DeepSeek-R1 reasoning traces
  - [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) — instruction diversity
  - [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) — curated GPT-4 instructions
- **Training suite**: [robit-man/fine_tuning_suite](https://github.com/robit-man/fine_tuning_suite)

## Quickstart

```bash
ollama run robit/qwen3.5-9b-r5-vision:q4km
```

### Image chat

```bash
IMG64=$(base64 -w0 path/to/image.jpg)
curl -s http://localhost:11434/api/chat \
  -d '{"model":"robit/qwen3.5-9b-r5-vision:q4km","messages":[{"role":"user","content":"Describe this image.","images":["'"$IMG64"'"]}]}'
```

## Parameters

- `RENDERER qwen3.5` + `PARSER qwen3.5`
- `num_ctx 131072`
- `temperature 0.6`, `top_p 0.95`
- `stop "<|im_end|>"`

## Note

R5 Vision is superseded by [robit/qwen3.5-9b-r7-research-vision:q4km](https://ollama.com/robit/qwen3.5-9b-r7-research-vision:q4km) which adds PrimeIntellect data and scores 86.8%.

## License

Derived from Qwen3.5-9B (Apache 2.0). Training data licenses vary by source.
