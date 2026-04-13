# Model Specification Cards

<table>
<tr>
<td width="50%" valign="top">

<div align="center">

### 🧠 R7 Research
**`robit/qwen3.5-9b-r7-research:q4km`**

</div>

<table width="100%">
<tr><td colspan="2" style="border-bottom: 8px solid black;"><b>Model Facts</b></td></tr>
<tr><td><b>Base</b></td><td>Qwen3.5-9B</td></tr>
<tr><td><b>Quant</b></td><td>Q4_K_M</td></tr>
<tr><td><b>Size</b></td><td>5.6 GB</td></tr>
<tr><td><b>Tensors</b></td><td>427</td></tr>
<tr><td><b>Context</b></td><td>262,144</td></tr>
<tr><td colspan="2" style="border-bottom: 4px solid black;"><b>Diverse Eval</b> &nbsp; <code>86.8%</code></td></tr>
<tr><td colspan="2"><b>Capabilities</b></td></tr>
<tr><td>✅ Thinking</td><td><code>&lt;think&gt;</code> blocks</td></tr>
<tr><td>✅ Tool Calling</td><td>Structured <code>tool_calls</code></td></tr>
<tr><td>✅ Instructions</td><td>Format, JSON, Y/N</td></tr>
<tr><td>❌ Vision</td><td>Text only</td></tr>
<tr><td colspan="2" style="border-bottom: 4px solid black;"><b>Training</b></td></tr>
<tr><td>LoRA rank</td><td>r=32, α=64</td></tr>
<tr><td>LR</td><td>1e-4 cosine</td></tr>
<tr><td>Data</td><td>4,043 samples</td></tr>
<tr><td>Epochs</td><td>1</td></tr>
<tr><td colspan="2" style="border-bottom: 2px solid black;"></td></tr>
<tr><td><b>Ollama</b></td><td><a href="https://ollama.com/robit/qwen3.5-9b-r7-research">ollama.com</a></td></tr>
<tr><td><b>HF FP16</b></td><td><a href="https://huggingface.co/cudabenchmarktest/qwen3.5-9b-r7-research">huggingface.co</a></td></tr>
</table>

</td>
<td width="50%" valign="top">

<div align="center">

### 👁️ R7 Research Vision
**`robit/qwen3.5-9b-r7-research-vision:q4km`**

</div>

<table width="100%">
<tr><td colspan="2" style="border-bottom: 8px solid black;"><b>Model Facts</b></td></tr>
<tr><td><b>Base</b></td><td>Qwen3.5-9B</td></tr>
<tr><td><b>Quant</b></td><td>Q4_K_M</td></tr>
<tr><td><b>Size</b></td><td>6.3 GB</td></tr>
<tr><td><b>Tensors</b></td><td>883 (427T + 441V + 15M)</td></tr>
<tr><td><b>Context</b></td><td>262,144</td></tr>
<tr><td colspan="2" style="border-bottom: 4px solid black;"><b>Diverse Eval</b> &nbsp; <code>86.8%</code></td></tr>
<tr><td colspan="2"><b>Capabilities</b></td></tr>
<tr><td>✅ Thinking</td><td><code>&lt;think&gt;</code> blocks</td></tr>
<tr><td>✅ Tool Calling</td><td>Structured <code>tool_calls</code></td></tr>
<tr><td>✅ Instructions</td><td>Format, JSON, Y/N</td></tr>
<tr><td>✅ Vision</td><td>Image understanding</td></tr>
<tr><td colspan="2" style="border-bottom: 4px solid black;"><b>Training</b></td></tr>
<tr><td>LoRA rank</td><td>r=32, α=64</td></tr>
<tr><td>LR</td><td>1e-4 cosine</td></tr>
<tr><td>Data</td><td>4,043 samples</td></tr>
<tr><td>Vision</td><td>Byte-for-byte from base</td></tr>
<tr><td colspan="2" style="border-bottom: 2px solid black;"></td></tr>
<tr><td><b>Ollama</b></td><td><a href="https://ollama.com/robit/qwen3.5-9b-r7-research-vision">ollama.com</a></td></tr>
<tr><td><b>HF FP16</b></td><td><a href="https://huggingface.co/cudabenchmarktest/qwen3.5-9b-r7-research-vision">huggingface.co</a></td></tr>
</table>

</td>
</tr>
<tr>
<td width="50%" valign="top">

<div align="center">

### 📘 R5 Research
**`robit/qwen3.5-9b-r5-research:q4km`**

</div>

<table width="100%">
<tr><td colspan="2" style="border-bottom: 8px solid black;"><b>Model Facts</b></td></tr>
<tr><td><b>Base</b></td><td>Qwen3.5-9B</td></tr>
<tr><td><b>Quant</b></td><td>Q4_K_M</td></tr>
<tr><td><b>Size</b></td><td>5.6 GB</td></tr>
<tr><td><b>Tensors</b></td><td>427</td></tr>
<tr><td><b>Context</b></td><td>131,072</td></tr>
<tr><td colspan="2" style="border-bottom: 4px solid black;"><b>Diverse Eval</b> &nbsp; <code>84.2%</code></td></tr>
<tr><td colspan="2"><b>Capabilities</b></td></tr>
<tr><td>✅ Thinking</td><td><code>&lt;think&gt;</code> blocks</td></tr>
<tr><td>✅ Tool Calling</td><td>Structured <code>tool_calls</code></td></tr>
<tr><td>✅ Instructions</td><td>Format, JSON, Y/N</td></tr>
<tr><td>❌ Vision</td><td>Text only</td></tr>
<tr><td colspan="2" style="border-bottom: 4px solid black;"><b>Training</b></td></tr>
<tr><td>LoRA rank</td><td>r=32, α=64</td></tr>
<tr><td>LR</td><td>1e-4 cosine</td></tr>
<tr><td>Data</td><td>4,122 samples</td></tr>
<tr><td>Epochs</td><td>1</td></tr>
<tr><td colspan="2" style="border-bottom: 2px solid black;"></td></tr>
<tr><td><b>Ollama</b></td><td><a href="https://ollama.com/robit/qwen3.5-9b-r5-research">ollama.com</a></td></tr>
<tr><td><b>HF GGUF</b></td><td><a href="https://huggingface.co/cudabenchmarktest/qwen3.5-9b-r5-research-GGUF">huggingface.co</a></td></tr>
<tr><td colspan="2"><i>Superseded by R7</i></td></tr>
</table>

</td>
<td width="50%" valign="top">

<div align="center">

### 👁️ R5 Vision
**`robit/qwen3.5-9b-r5-vision:q4km`**

</div>

<table width="100%">
<tr><td colspan="2" style="border-bottom: 8px solid black;"><b>Model Facts</b></td></tr>
<tr><td><b>Base</b></td><td>Qwen3.5-9B</td></tr>
<tr><td><b>Quant</b></td><td>Q4_K_M</td></tr>
<tr><td><b>Size</b></td><td>6.3 GB</td></tr>
<tr><td><b>Tensors</b></td><td>883 (427T + 441V + 15M)</td></tr>
<tr><td><b>Context</b></td><td>131,072</td></tr>
<tr><td colspan="2" style="border-bottom: 4px solid black;"><b>Diverse Eval</b> &nbsp; <code>84.2%</code></td></tr>
<tr><td colspan="2"><b>Capabilities</b></td></tr>
<tr><td>✅ Thinking</td><td><code>&lt;think&gt;</code> blocks</td></tr>
<tr><td>✅ Tool Calling</td><td>Structured <code>tool_calls</code></td></tr>
<tr><td>✅ Instructions</td><td>Format, JSON, Y/N</td></tr>
<tr><td>✅ Vision</td><td>Image understanding</td></tr>
<tr><td colspan="2" style="border-bottom: 4px solid black;"><b>Training</b></td></tr>
<tr><td>LoRA rank</td><td>r=32, α=64</td></tr>
<tr><td>LR</td><td>1e-4 cosine</td></tr>
<tr><td>Data</td><td>4,122 samples</td></tr>
<tr><td>Vision</td><td>Byte-for-byte from base</td></tr>
<tr><td colspan="2" style="border-bottom: 2px solid black;"></td></tr>
<tr><td><b>Ollama</b></td><td><a href="https://ollama.com/robit/qwen3.5-9b-r5-vision">ollama.com</a></td></tr>
<tr><td colspan="2"><i>Superseded by R7 Vision</i></td></tr>
</table>

</td>
</tr>
</table>

---

<details>
<summary><b>Tensor Breakdown Legend</b></summary>

| Code | Meaning | Count |
|------|---------|-------|
| **T** | Text (language model blocks, embeddings, norms) | 427 |
| **V** | Vision (patch embed, transformer blocks, merger) | 441 |
| **M** | MTP (multi-token prediction heads) | 15 |

- **Text-only models** (427 tensors): Language model only, no image input
- **Vision models** (883 tensors): Full multimodal — text + vision encoder + MTP, created via `llama-export-lora` merge preserving base vision byte-for-byte

</details>

<details>
<summary><b>Quantization Reference</b></summary>

| Format | Bits/Weight | Size (9B) | Quality | Speed |
|--------|------------|-----------|---------|-------|
| FP16 | 16.0 | ~18 GB | Baseline | Slow |
| Q8_0 | 8.0 | ~9.5 GB | Near-lossless | Medium |
| **Q4_K_M** | **4.8** | **~5.6 GB** | **Good** | **Fast** |
| Q4_0 | 4.0 | ~5.0 GB | Acceptable | Fastest |

All published models use **Q4_K_M** — best balance of quality and speed for the 9B parameter class. FP16 checkpoints available on HuggingFace for custom quantization.

</details>

<details>
<summary><b>Capability Definitions</b></summary>

| Capability | What It Means |
|------------|---------------|
| **Thinking** | Model produces `<think>...</think>` reasoning blocks before answering. Ollama exposes this as a separate `thinking` field in the API response. |
| **Tool Calling** | When given a `tools` array in the API request, the model returns structured `tool_calls` with function name and arguments instead of text. Requires `RENDERER qwen3.5` + `PARSER qwen3.5` in Modelfile. |
| **Instructions** | Model follows format constraints: YES/NO only, one-word answers, JSON output, numbered lists, comma-separated, system prompt roles. |
| **Vision** | Model accepts `images` (base64) in chat messages and can describe, read text from, and answer questions about images. Requires the 883-tensor GGUF with vision encoder. |

</details>

<details>
<summary><b>Training Data Sources</b></summary>

| Dataset | Samples Used | Purpose |
|---------|-------------|---------|
| [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) | 2,000 | DeepSeek-R1 reasoning traces |
| [Tulu 3 SFT Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | 1,358 | Instruction diversity |
| [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | 451 | Curated GPT-4 instructions |
| [PrimeIntellect SYNTHETIC-1](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-1-SFT-Data) | 312 | Verified math/code/STEM (R7 only) |
| Static examples | ~100 | Format constraints, conversations, code |

Full pipeline: [robit-man/fine_tuning_suite](https://github.com/robit-man/fine_tuning_suite)

</details>
