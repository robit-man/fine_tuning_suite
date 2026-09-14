# Ornith 1.5 Omni third-party model notices

The two Ornith 1.5 Omni packages contain independently executable GGUF views
from the projects below. Each view remains attributable to its authors and is
subject to its source license.

| Packaged view | Source | License |
|---|---|---|
| Stock language/vision base | [`ornith-ai/Ornith-1.5-9B`](https://huggingface.co/ornith-ai/Ornith-1.5-9B) at `489cb97981b8654bcfcf30ce1f94ed1b62e07b53` | MIT |
| Obliterated language/vision base | [`OBLITERATUS/Ornith-1.5-9B-OBLITERATED`](https://huggingface.co/OBLITERATUS/Ornith-1.5-9B-OBLITERATED) at `82df4702cc5e716cf7573a8942fee28ae8f65088` | MIT |
| Audio/image/video comprehension | [`ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF`](https://huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF) at `6e35a28f4a19b18730f8949b0c579c6429649ab8` | Apache-2.0 upstream |
| Text-to-speech | [`ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`](https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF) at `ca27d74bc954b73dadab5b71ca265d87fc861a7c` | Apache-2.0 upstream |
| GGUF runtime/conversion | [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) | MIT |

The source model pages and license files are authoritative. Immutable
revisions, component byte sizes, tensor counts, and SHA-256 digests are in the
two `ornith15-*-sidecar-manifest.json` files. The package changes container
layout and tensor namespacing; it does not claim authorship of the weights.

Ornith and Qwen are project names associated with their respective owners.
These packages are independent releases and are not official Ornith, Qwen,
llama.cpp, Ollama, OBLITERATUS, or ggml-org releases. No endorsement is
implied.
