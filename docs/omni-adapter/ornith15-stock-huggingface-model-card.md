---
license: other
license_name: Mixed MIT and Apache-2.0 component licenses
base_model:
  - ornith-ai/Ornith-1.5-9B
  - Qwen/Qwen3-Omni-30B-A3B-Instruct
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base
language:
  - en
tags:
  - gguf
  - ollama
  - ornith
  - qwen3-omni
  - multimodal
  - audio
  - video
  - automatic-speech-recognition
  - text-to-speech
  - tool-use
  - custom-code
---

# Ornith-1.5-9B-Omni (Q4_K_M)

This repository supplies the custom six-view GGUF sidecar for the logical
Ollama model `robit/ornith-1.5-omni:q4km`. Its base reasoning model is the
stock [`ornith-ai/Ornith-1.5-9B`](https://huggingface.co/ornith-ai/Ornith-1.5-9B),
not the OBLITERATUS derivative.

The release combines:

- stock Ornith 1.5 9B text generation, native image vision, parsed optional
  thinking, and structured tools;
- Qwen3-Omni audio, image, and sampled-video comprehension;
- Qwen3-TTS text-conditioned 24 kHz speech generation.

These are independently executable graphs connected by a documented semantic
adapter. This is not a claim that incompatible hidden states were tensor-spliced.

## Published locations

- Ollama: [`robit/ornith-1.5-omni:q4km`](https://ollama.com/robit/ornith-1.5-omni:q4km)
- Ollama default alias: [`robit/ornith-1.5-omni:latest`](https://ollama.com/robit/ornith-1.5-omni)
- Runtime: [`robit-man/qwen-omni-adapters`](https://github.com/robit-man/qwen-omni-adapters)
- Build/release source: [`robit-man/fine_tuning_suite`](https://github.com/robit-man/fine_tuning_suite)

## Important: sidecar, not a standalone graph

`ornith-1.5-9b-omni-q4km.gguf` is one valid GGUF v3 container containing six
namespaced model/projector views. It is not a standard single-architecture
`FROM` target for stock Ollama.

The Ollama tag retains normal stock model/projector layers and adds the sidecar
as `application/vnd.robit.ollama.omni.bundle.v1+gguf`. Stock Ollama executes
text, native image vision, tools, and thinking. The Robit adapter resolves the
same installed tag and runs the audio/video/TTS views with the pinned
llama.cpp runtime.

## Install and run

```bash
ollama pull robit/ornith-1.5-omni:q4km
git clone https://github.com/robit-man/qwen-omni-adapters.git
cd qwen-omni-adapters

OMNI_MODEL=robit/ornith-1.5-omni:q4km \
OMNI_LANGUAGE_MODEL=robit/ornith-1.5:9b \
./deploy.sh
```

The deployment validates the sidecar, materializes disposable media views,
starts CUDA workers under the host GPU broker, exercises local smoke gates,
and prints the authenticated portal URL. Ordinary Ollama clients may use the
same tag directly for text/image/tools/thinking.

Advanced users can inspect or extract the Hugging Face sidecar with:

```bash
python -m training_suite omni-inspect ./ornith-1.5-9b-omni-q4km.gguf
python -m training_suite omni-attach your-exact-stock-ornith-tag:q4km \
  ./ornith-1.5-9b-omni-q4km.gguf
python -m training_suite omni-prepare your-exact-stock-ornith-tag:q4km \
  --out ./runtime-cache
```

The target must contain the exact base and projector digests below. Attaching
the sidecar to an arbitrary model does not make that model Omni.

## Artifact inventory

| Item | Value |
|---|---|
| File | `ornith-1.5-9b-omni-q4km.gguf` |
| Size | 27,915,305,504 bytes |
| SHA-256 | `1302c601bbfa81d0efa80167beecaac14b6ecbbada377ff814ce1da4d22e0f6d` |
| Tensors | 2,889 |
| Base model | 9.0B, Q4_K_M, 262,144-token context |
| Artifact schema | `robit.ollama-monolithic-omni.v3` |
| Wire schema | `robit.ollama.omni-adapter.v1` |
| llama.cpp | `458681e1d5d4a29a1463c4732e03226cf384b997` |

The base namespace is unprefixed; `b.p.*` is its projector; `a.c.m.*` and
`a.c.p.*` are comprehension; `s.t.m.*` and `s.t.p.*` are TTS. The exact
machine-readable inventory is published as
`ornith15-stock-sidecar-manifest.json`.

## Capability boundary

| Capability | Executor |
|---|---|
| Text, tools, optional thinking | stock Ornith through Ollama |
| Native image understanding | stock Ornith projector through Ollama |
| ASR and environmental sound interpretation | Qwen3-Omni through adapter |
| Sampled video/GIF understanding | Qwen3-Omni through adapter |
| Spoken replies and voice references | Qwen3-TTS through adapter |

The model-level generation setting uses Ollama's `num_predict=-1`, removing
the source tag's inherited 16,384-token output cap. Clients may still set a
per-request output limit.

## Limitations and safety

- Stock Ollama does not execute audio/video/TTS from the custom layer; use the
  adapter for those routes.
- The bridge is semantic text, not a learned dense connector.
- Video support is comprehension only, not video generation.
- Media, transcripts, OCR, captions, and tool results are untrusted input.
- Public deployments need authentication, rate limits, isolated tool
  execution, bounded decoding, monitoring, and appropriate output controls.

See the included release record, sidecar manifest, validation report, and
third-party notices for reproducibility and attribution.
