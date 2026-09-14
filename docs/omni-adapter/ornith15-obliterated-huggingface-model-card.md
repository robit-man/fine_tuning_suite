---
license: mit
base_model:
  - OBLITERATUS/Ornith-1.5-9B-OBLITERATED
  - Qwen/Qwen3-Omni-30B-A3B-Instruct
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base
language:
  - en
tags:
  - gguf
  - ollama
  - ornith
  - obliterated
  - qwen3-omni
  - multimodal
  - audio
  - video
  - automatic-speech-recognition
  - text-to-speech
  - tool-use
  - custom-code
---

# Ornith-1.5-9B-Obliterated-Omni (Q4_K_M)

This repository supplies the custom six-view GGUF sidecar for the logical
Ollama model `robit/ornith-1.5-obliterated-omni:q4km`. Its base reasoning model
is [`OBLITERATUS/Ornith-1.5-9B-OBLITERATED`](https://huggingface.co/OBLITERATUS/Ornith-1.5-9B-OBLITERATED),
not stock Ornith.

The release combines:

- obliterated Ornith 1.5 9B text generation, native image vision, parsed
  optional thinking, and structured tools;
- Qwen3-Omni audio, image, and sampled-video comprehension;
- Qwen3-TTS text-conditioned 24 kHz speech generation.

These are independently executable graphs connected by a documented semantic
adapter. This is not a claim that incompatible hidden states were tensor-spliced.

## Published locations

- Ollama: [`robit/ornith-1.5-obliterated-omni:q4km`](https://ollama.com/robit/ornith-1.5-obliterated-omni:q4km)
- Ollama default alias: [`robit/ornith-1.5-obliterated-omni:latest`](https://ollama.com/robit/ornith-1.5-obliterated-omni)
- Runtime: [`robit-man/qwen-omni-adapters`](https://github.com/robit-man/qwen-omni-adapters)
- Build/release source: [`robit-man/fine_tuning_suite`](https://github.com/robit-man/fine_tuning_suite)

## Important: sidecar, not a standalone graph

`ornith-1.5-9b-obliterated-omni-q4km.gguf` is one valid GGUF v3 container
containing six namespaced model/projector views. It is not a standard
single-architecture `FROM` target for stock Ollama.

The Ollama tag retains normal OBLITERATUS model/projector layers and adds the
sidecar as `application/vnd.robit.ollama.omni.bundle.v1+gguf`. Stock Ollama
executes text, native image vision, tools, and thinking. The Robit adapter
resolves the same installed tag and runs audio/video/TTS with the pinned
llama.cpp runtime.

## Install and run

```bash
ollama pull robit/ornith-1.5-obliterated-omni:q4km
git clone https://github.com/robit-man/qwen-omni-adapters.git
cd qwen-omni-adapters

OMNI_MODEL=robit/ornith-1.5-obliterated-omni:q4km \
OMNI_LANGUAGE_MODEL=robit/ornith-1.5-obliterated:9b \
./deploy.sh
```

Ordinary Ollama clients may use the same tag directly for its standard
text/image/tools/thinking features. Audio/video input and audio output require
the adapter.

Advanced sidecar workflow:

```bash
python -m training_suite omni-inspect \
  ./ornith-1.5-9b-obliterated-omni-q4km.gguf
python -m training_suite omni-attach your-exact-obliterated-tag:q4km \
  ./ornith-1.5-9b-obliterated-omni-q4km.gguf
python -m training_suite omni-prepare your-exact-obliterated-tag:q4km \
  --out ./runtime-cache
```

## Artifact inventory

| Item | Value |
|---|---|
| File | `ornith-1.5-9b-obliterated-omni-q4km.gguf` |
| Size | 28,066,286,368 bytes |
| SHA-256 | `22a38a92994dd9b21cee273c7fbbe5b21e57498bc6f56e1d6646211fc641b55e` |
| Tensors | 2,904 |
| Base model | 9.2B, Q4_K_M, 262,144-token context |
| Artifact schema | `robit.ollama-monolithic-omni.v3` |
| Wire schema | `robit.ollama.omni-adapter.v1` |
| llama.cpp | `458681e1d5d4a29a1463c4732e03226cf384b997` |

The exact machine-readable inventory is published as
`ornith15-obliterated-sidecar-manifest.json`.

## Capability boundary

| Capability | Executor |
|---|---|
| Text, tools, optional thinking | obliterated Ornith through Ollama |
| Native image understanding | OBLITERATUS projector through Ollama |
| ASR and environmental sound interpretation | Qwen3-Omni through adapter |
| Sampled video/GIF understanding | Qwen3-Omni through adapter |
| Spoken replies and voice references | Qwen3-TTS through adapter |

The model-level generation setting uses Ollama's `num_predict=-1`, removing
the source tag's inherited 16,384-token output cap. Clients may still set a
per-request output limit.

## Limitations and safety

- Stock Ollama does not execute audio/video/TTS from the custom layer.
- The bridge is semantic text, not a learned dense connector.
- Video support is comprehension only, not video generation.
- The base deliberately reduces refusal behavior. It is not a safety-aligned
  substitute for upstream Ornith and requires application-level safeguards.
- Media, transcripts, OCR, captions, and tool results are untrusted input.
- Public deployments need authentication, rate limits, sandboxed tools,
  bounded decoding, monitoring, and appropriate output controls.

See the included release record, sidecar manifest, validation report, and
third-party notices for reproducibility and attribution.
