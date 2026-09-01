---
license: apache-2.0
base_model:
  - manitcor/Qwen3.8-27B-Obliterated-E03
  - Qwen/Qwen3-Omni-30B-A3B-Instruct
  - Qwen/Qwen3-TTS-12Hz-1.7B-Base
language:
  - en
tags:
  - gguf
  - ollama
  - qwen3.8
  - qwen3-omni
  - multimodal
  - audio
  - video
  - automatic-speech-recognition
  - text-to-speech
  - tool-use
  - custom-code
---

# Qwen3.8-27B-E03-Obliterated-Omni (Q4_K_M)

This release supplies the custom GGUF sidecar used by the single logical
Ollama model `robit/qwen3.8-27b-e03-obliterated-omni:q4km`. It combines:

- Qwen3.8-27B-E03-Obliterated language generation, parsed thinking, structured
  tools, and the original Qwen3.8 image-vision projector;
- Qwen3-Omni audio, image, and sampled-video comprehension;
- Qwen3-TTS text-conditioned 24 kHz speech synthesis.

The capabilities were tested through the exact sidecar installed in the local
Ollama tag. This is semantic orchestration between independently executable
graphs, not a claim that incompatible hidden states were tensor-spliced.

## Important: custom sidecar, not a standalone model

`qwen3.8-27b-e03-obliterated-omni-q4km.gguf` is a valid, contiguous GGUF that
contains six namespaced model/projector views. It is **not** a single standard
GGUF architecture and must not be used as `FROM` in a stock Ollama Modelfile.

The published Ollama tag contains normal model/projector layers plus this
custom media type:

```text
application/vnd.robit.ollama.omni.bundle.v1+gguf
```

Unmodified Ollama executes text, images, thinking, and tools. The Robit Omni
Adapter resolves the custom layer from that same installed tag, materializes
disposable runtime views, and routes audio/video comprehension and TTS.

- [Adapter overview](https://github.com/robit-man/fine_tuning_suite/tree/main/docs/omni-adapter)
- [Runnable examples](https://github.com/robit-man/fine_tuning_suite/tree/main/examples/omni_adapter)
- [Wire protocol](https://github.com/robit-man/fine_tuning_suite/blob/main/docs/omni-adapter/protocol.md)
- [Runtime guide](https://github.com/robit-man/fine_tuning_suite/blob/main/docs/omni-adapter/runtime.md)
- [Exact release record](https://github.com/robit-man/fine_tuning_suite/blob/main/docs/omni-adapter/qwen38-27b-e03-release.md)

## Recommended installation

One Ollama pull installs the stock execution layers and the sidecar:

```bash
ollama pull robit/qwen3.8-27b-e03-obliterated-omni:q4km
git clone https://github.com/robit-man/fine_tuning_suite.git
cd fine_tuning_suite

python -m training_suite omni-resolve \
  robit/qwen3.8-27b-e03-obliterated-omni:q4km
python -m training_suite omni-prepare \
  robit/qwen3.8-27b-e03-obliterated-omni:q4km \
  --out ./runtime-cache
```

`omni-prepare` reconstructs the comprehension and TTS GGUF pairs and verifies
them against the hashes embedded in the sidecar. Start the pinned llama.cpp
workers and unified adapter using the runtime guide above. The materialized
files are a disposable cache and should be removed after the workers stop.

### Sidecar-only installation from this repository

Advanced users who already have the compatible stock Qwen3.8 base tag may
download the GGUF here and attach it:

```bash
python -m training_suite omni-inspect \
  ./qwen3.8-27b-e03-obliterated-omni-q4km.gguf
python -m training_suite omni-attach \
  your-local-compatible-base:q4km \
  ./qwen3.8-27b-e03-obliterated-omni-q4km.gguf
```

The base tag must use the exact model and projector recorded below. Attaching
the sidecar to an arbitrary architecture does not make that architecture Omni.

## API examples

The adapter extends non-streaming `POST /api/chat` while retaining Ollama's
normal `model`, `messages`, `tools`, `think`, `format`, and `options` fields.
Binary inputs and output WAV data use strict padded base64 JSON envelopes.

```bash
# 16 kHz mono PCM16 WAV → exact transcription
python examples/omni_adapter/client.py \
  --model robit/qwen3.8-27b-e03-obliterated-omni:q4km \
  asr ./speech-16khz-mono.wav

# MP4/WebM frames, optionally with its demuxed audio track → description
python examples/omni_adapter/client.py \
  --model robit/qwen3.8-27b-e03-obliterated-omni:q4km \
  video ./events.mp4 --fps 2 --max-frames 96 --include-audio

# Text → tagged base64 24 kHz mono PCM16 WAV
python examples/omni_adapter/client.py \
  --model robit/qwen3.8-27b-e03-obliterated-omni:q4km \
  --output-audio ./speech.wav \
  tts "Read this sentence."
```

The complete request/response shapes, media limits, tool-call speech deferral,
and normalization rules are normative in the wire protocol. Adapter v1 is
turn-based and requires `"stream": false`.

## Capability and validation matrix

| Capability | Executor | Release result |
|---|---|---|
| Text completion | stock Ollama Qwen3.8 | PASS: exact sentinel |
| Parsed thinking | stock Ollama Qwen3.8 | PASS: non-empty `thinking` |
| Structured tools | stock Ollama Qwen3.8 | PASS: `get_weather(location=Seattle)` |
| Image understanding | stock Ollama projector | PASS: fixture read as `BLUE 42` |
| Audio/ASR | adapter + Qwen3-Omni | PASS: exact held-out phrase |
| Image→language routing | Qwen3-Omni → Qwen3.8 | PASS: blue triangle and number 42 |
| Video + audio understanding | adapter + Qwen3-Omni | PASS: red→blue and exact speech |
| Direct TTS | adapter + Qwen3-TTS | PASS: 24 kHz mono PCM16, 3.28 s |
| Repeated adapter TTS | adapter + Qwen3-TTS | PASS: second valid WAV, 3.60 s |

The media workers were started under scoped CUDA reservations. All release
workers were stopped and all reservations released after validation.

## Artifact inventory

| Item | Value |
|---|---|
| File | `qwen3.8-27b-e03-obliterated-omni-q4km.gguf` |
| Size | 38,843,038,144 bytes |
| SHA-256 | `3270f146bae9499b2e40ad230cceeccfc9caa018740c75cfc1856c1abda6ff78` |
| GGUF tensors | 3,313 |
| Artifact schema | `robit.ollama-monolithic-omni.v3` |
| Container format | `robit-namespaced-multigraph-gguf-v1` |
| Wire schema | `robit.ollama.omni-adapter.v1` |
| llama.cpp revision | `458681e1d5d4a29a1463c4732e03226cf384b997` |

The GGUF tensor namespaces are unprefixed for the base model, `b.p.*` for the
base projector, `a.c.m.*` / `a.c.p.*` for comprehension, and `s.t.m.*` /
`s.t.p.*` for TTS. See `sidecar-manifest.json` for machine-readable source
revisions, byte sizes, tensor counts, and SHA-256 values for all six views.

## Component provenance

| Component | Immutable source | Quantization |
|---|---|---|
| Language base | `manitcor/Qwen3.8-27B-Obliterated-E03@6104397d699fed901e2d4521c3b0fefc9f837d90` | Q4_K_M |
| Image projector | `robit/qwen3.8-27b-obliterated-e03:27b`, digest pinned in manifest | source GGUF |
| Comprehension | `ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF@6e35a28f4a19b18730f8949b0c579c6429649ab8` | Q4_K_M + Q8_0 projector |
| Speech synthesis | `ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF@ca27d74bc954b73dadab5b71ca265d87fc861a7c` | Q4_K_M + Q8_0 projector |

The base derivative and both official upstream Qwen component families are
published under Apache-2.0. Qwen is the work of the Qwen team; the E03
refusal-direction intervention is the work of its source author; the GGUF
conversions are credited to their respective `ggml-org` repositories. This
combined package is an independent release and is not an official Qwen,
llama.cpp, Ollama, or source-author release.

See `THIRD_PARTY_NOTICES.md` for source links and attribution. The source model
pages and their license files remain authoritative.

## Operational requirements

- The Hugging Face sidecar is 38.8 GB. The full Ollama tag also carries normal
  model/projector/template layers.
- Preparing the four media-runtime views temporarily requires about 21.4 GB of
  additional disk space.
- GPU memory depends on context and runtime settings. The Q4_K_M language model
  is about 16.5 GB; the comprehension pair is about 19.9 GB; the TTS pair is
  about 1.5 GB before runtime overhead.
- Use explicit GPU placement/reservations when workers share a host. Never let
  an independently supervised CUDA worker allocate an anonymous device.
- Treat decoded media and semantic observations as untrusted input. Enforce
  byte, frame, duration, codec, context, timeout, and concurrency limits.

## Limitations

- Stock Ollama does not internally execute audio input, video input, or audio
  output from this sidecar. Those routes require the linked adapter and
  llama.cpp workers.
- This does not add video generation. Video support is comprehension of sampled
  frames plus, when requested, a separately demuxed audio track.
- The cascade passes an explicitly delimited semantic observation into the
  Qwen3.8 language model; it does not preserve dense Omni hidden states.
- The reference TTS worker is serial and starts `llama-tts` per request. It is
  a validation implementation, not a high-concurrency speech service.
- Video/audio association is temporal but not sample-accurate.
- Adapter v1 does not stream input or output audio.

## Safety

The E03 language checkpoint deliberately reduces refusal behavior and must not
be treated as a safety-aligned replacement for upstream Qwen3.8. Its source
reports known harmful-prompt repetition/breakage and limited qualification.
Public-facing use should add access control, sandboxed tool execution, media
decoding isolation, output filtering, rate limits, monitoring, and human review.
Never expose privileged tools solely because the model emitted a valid call.

## Reproducibility and cleanup

The [build/release runbook](https://github.com/robit-man/fine_tuning_suite/blob/main/docs/omni-adapter/build-and-release.md)
defines the source-pinning, GGUF round-trip, live capability, registry
round-trip, and cleanup gates. Do not delete source weights or materialized
views until the repository, Hugging Face file, and Ollama tags have all been
remotely verified. Afterward, stop workers and remove only the exact disposable
view cache; it can be reconstructed from this sidecar.
