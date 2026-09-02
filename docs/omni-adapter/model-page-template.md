# Model Card Template: Qwen3.8 27B E03 Obliterated Omni

Replace bracketed release-state values only after their verification gates
pass. Never describe the sidecar GGUF as a directly loadable stock model.

# Qwen3.8-27B-E03-Obliterated-Omni (Q4_K_M)

One logical Ollama model combining Qwen3.8 text reasoning, thinking, structured
tools, and native image vision with Qwen3-Omni audio/video comprehension and
Qwen3-TTS speech output.

## Important runtime boundary

The Ollama tag contains normal stock model/projector layers plus one custom
namespaced GGUF sidecar layer. Unmodified Ollama executes text, image vision,
thinking, and tools. Audio input, video input, and TTS require the Robit Omni
Adapter, which resolves and executes component views from the same installed
tag.

The Hugging Face GGUF is the custom sidecar payload, not a standalone
single-architecture model for `ollama create FROM`.

- Adapter docs: <https://github.com/robit-man/fine_tuning_suite/tree/main/docs/omni-adapter>
- Examples: <https://github.com/robit-man/fine_tuning_suite/tree/main/examples/omni_adapter>
- Request schema: <https://github.com/robit-man/fine_tuning_suite/blob/main/docs/omni-adapter/schema/request-v1.schema.json>

## Pull

```bash
ollama pull robit/qwen3.8-27b-e03-obliterated-omni:q4km
```

## Capability matrix

| Capability | Executor | Result/report |
|---|---|---|
| Text completion | stock Ollama Qwen3.8 | `[PASS]` |
| Thinking | stock Ollama Qwen3.8 | `[PASS]` |
| Structured tools | stock Ollama Qwen3.8 | `[PASS]` |
| Image understanding | stock Ollama projector | `[PASS]` |
| Audio/ASR | adapter + Qwen3-Omni sidecar | `[PASS]` |
| Video understanding | adapter + Qwen3-Omni sidecar | `[PASS]` |
| TTS | adapter + Qwen3-TTS sidecar | `[PASS]` |

Video generation and streaming audio are not included.

## Prepare adapter views

```bash
python -m training_suite omni-resolve \
  robit/qwen3.8-27b-e03-obliterated-omni:q4km
python -m training_suite omni-prepare \
  robit/qwen3.8-27b-e03-obliterated-omni:q4km \
  --out ./runtime-cache
```

Follow the linked runtime guide to start the pinned llama.cpp comprehension and
TTS workers plus the unified adapter endpoint.

## Examples

```bash
# ASR
python examples/omni_adapter/client.py \
  --model robit/qwen3.8-27b-e03-obliterated-omni:q4km \
  asr ./speech-16khz-mono.wav

# Video plus its audio track
python examples/omni_adapter/client.py \
  --model robit/qwen3.8-27b-e03-obliterated-omni:q4km \
  video ./events.mp4 --fps 2 --max-frames 96 --include-audio

# Direct TTS
python examples/omni_adapter/client.py \
  --model robit/qwen3.8-27b-e03-obliterated-omni:q4km \
  --output-audio ./speech.wav \
  tts "Read this sentence."
```

Adapter v1 accepts 16 kHz mono PCM16 WAV, JPEG/PNG/WebP, and bounded
MP4/WebM/GIF. Output audio is tagged base64 RIFF/WAVE, PCM16, mono, 24 kHz under
`message.audio`. Adapter v1 is turn-based and requires `stream:false`.

## Artifact layout

The sidecar uses schema `robit.ollama-monolithic-omni.v3`:

- unprefixed: base model;
- `b.p.*`: base projector;
- `a.c.m.*` / `a.c.p.*`: comprehension model/projector;
- `s.t.m.*` / `s.t.p.*`: TTS model/codec projector.

This is a multi-graph semantic router, not a hidden-state tensor graft.

## Provenance

| Component | Source/revision | Quantization | SHA-256 |
|---|---|---|---|
| Base | `[repo@revision]` | Q4_K_M | `[digest]` |
| Base projector | `[source]` | `[type]` | `[digest]` |
| Comprehension | `[repo@revision:file]` | Q4_K_M | `[digest]` |
| Comprehension projector | `[repo@revision:file]` | Q8_0 | `[digest]` |
| TTS | `[repo@revision:file]` | Q4_K_M | `[digest]` |
| TTS projector | `[repo@revision:file]` | Q8_0 | `[digest]` |
| Sidecar GGUF | `[filename]` | mixed | `[digest]` |

## Limitations

- Media routes require the adapter; stock Ollama silently ignores the custom
  layer and does not parse `audios`, `videos`, or `message.audio`.
- Comprehension crosses into Qwen3.8 as untrusted semantic text, so dense
  cross-modal information is not preserved.
- The reference TTS wrapper is serial and keeps one matching voice profile
  resident; production deployments should size concurrency explicitly.
- Current video-audio handling demuxes the audio track separately and does not
  promise sample-accurate alignment.
- Only capabilities demonstrated by the linked release record are claimed.
- Review all component licenses and usage terms before redistribution or
  commercial use.

## Reproducibility

Built with [Fine-Tuning Suite](https://github.com/robit-man/fine_tuning_suite).
See the [build and release runbook](https://github.com/robit-man/fine_tuning_suite/blob/main/docs/omni-adapter/build-and-release.md)
and `[release-record-url]` for exact revisions, hashes, validation results, and
cleanup evidence.
