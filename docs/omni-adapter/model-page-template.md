# Ollama Model Page Template: Qwen3.8 27B E03 Obliterated Omni

> Release template. Replace every `<...>` value from the verified release
> report. Do not publish the capability list until all gates in
> [testing.md](testing.md) pass against the pushed tag.

# Qwen3.8-27B-E03-Obliterated-Omni (`<quantization>`)

Qwen3.8-27B-E03-Obliterated combined with independently executable
audio/image/video comprehension and text-to-speech graphs in one physical GGUF.
The base language model retains thinking and structured tool behavior. Media is
handled through the versioned Robit Omni adapter.

## Runtime requirement

This model uses `robit.ollama.omni-adapter.v1` and requires the custom Ollama
runtime build `<runtime-version>` at commit `<runtime-commit>`. Unmodified
Ollama may import or execute only the base language view and does not understand
the added audio/video/TTS request fields.

- Adapter documentation:
  https://github.com/robit-man/fine_tuning_suite/tree/main/docs/omni-adapter
- Python, JavaScript, and server examples:
  https://github.com/robit-man/fine_tuning_suite/tree/main/examples/omni_adapter
- Request schema:
  https://github.com/robit-man/fine_tuning_suite/blob/main/docs/omni-adapter/schema/request-v1.schema.json

## Verified capabilities

Publish only the rows that passed the release report `<report-url>`.

| Capability | Result | Test fixture/report |
|---|---|---|
| Text completion | `<PASS>` | `<link>` |
| Thinking | `<PASS>` | `<link>` |
| Structured tool calls | `<PASS>` | `<link>` |
| Image comprehension | `<PASS>` | `<link>` |
| ASR/audio comprehension | `<PASS>` | `<link>` |
| Video comprehension | `<PASS>` | `<link>` |
| Direct TTS | `<PASS>` | `<link>` |
| Audio → reasoning → TTS | `<PASS>` | `<link>` |

Video generation is not included.

## Pull

```bash
ollama pull robit/qwen3.8-27b-e03-obliterated-omni:<tag>
```

## Text, thinking, and tools

Use Ollama's normal `/api/chat` fields. Set `think:true` (or a supported effort)
and provide normal Ollama `tools`. Media additions are optional, so text-only
clients retain the standard request shape.

## ASR

```bash
python examples/omni_adapter/client.py \
  --model robit/qwen3.8-27b-e03-obliterated-omni:<tag> \
  asr ./speech-16khz-mono.wav
```

## Video comprehension

```bash
python examples/omni_adapter/client.py \
  --model robit/qwen3.8-27b-e03-obliterated-omni:<tag> \
  video ./events.mp4 --fps 2 --max-frames 96 --include-audio
```

## Direct TTS

```bash
python examples/omni_adapter/client.py \
  --model robit/qwen3.8-27b-e03-obliterated-omni:<tag> \
  --output-audio ./speech.wav \
  tts "Read this sentence."
```

## Combined audio-in/audio-out chat

```bash
python examples/omni_adapter/client.py \
  --model robit/qwen3.8-27b-e03-obliterated-omni:<tag> \
  --output-audio ./answer.wav \
  chat --audio ./question.wav --speak \
  --prompt "Answer the recorded question."
```

Adapter v1 accepts 16 kHz mono PCM16 WAV audio, JPEG/PNG/WebP images, and bounded
MP4/WebM video. Speech output is base64 24 kHz mono PCM16 WAV under
`message.audio`. It is a turn-based API and requires `stream:false`.

## Artifact layout

One GGUF contains:

- unprefixed Qwen3.8 language tensors;
- `a.c.*` comprehension tensors;
- `s.t.*` text-conditioned speech tensors.

This is a multi-graph runtime with a semantic-text bridge, not an unsupported
hidden-state tensor splice.

## Provenance

| Component | Source and immutable revision | Quantization | SHA-256 |
|---|---|---|---|
| Language | `manitcor/Qwen3.8-27B-Obliterated-E03@<revision>` | `<type>` | `<digest>` |
| Comprehension | `<repo@revision>` | `<type>` | `<digest>` |
| TTS | `<repo@revision>` | `<type>` | `<digest>` |
| Combined GGUF | `<release>` | mixed/component-specific | `<digest>` |

## Limitations and safety

- Requires the custom runtime version above.
- Media observations are treated as untrusted evidence before language/tool
  routing.
- TTS is deferred when the assistant returns unresolved tool calls.
- Accuracy, latency, VRAM, context, languages, voices, and maximum media duration
  are exactly those measured in `<release-report>`; do not infer upstream model
  claims that were not retested after conversion and quantization.
- Review all component licenses and terms before redistribution or commercial
  use.

## Build and test

Built with [Fine-Tuning Suite](https://github.com/robit-man/fine_tuning_suite).
The reproducible commands, runtime patch requirements, test matrix, and cleanup
procedure are in the
[build/release runbook](https://github.com/robit-man/fine_tuning_suite/blob/main/docs/omni-adapter/build-and-release.md).
