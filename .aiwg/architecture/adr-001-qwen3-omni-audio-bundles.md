# ADR-001: Represent Omni Audio as a GGUF Bundle and Execution Contract

## Status

Superseded by ADR-002 for artifact layout. The architecture-compatibility
analysis and semantic-text routing boundary remain applicable.

## Context

Qwen3-Omni audio is a graph of coupled components, not a capability flag that
can be added to an arbitrary language GGUF. The official Instruct checkpoint
contains a 2,048-wide MoE Thinker, audio and vision encoders, a 1,024-wide
Talker conditioned on Thinker hidden states, and a 16-codebook code2wav decoder.
The existing Qwen3.8 obliterated trunk is a 5,120-wide, 64-layer Qwen3.5 dense
model. Ornith 1.5 9B is a 4,096-wide, 32-layer Qwen3.5 model. Neither is tensor-
or hidden-state-compatible with the Omni Thinker.

Current llama.cpp `libmtmd` supports experimental Qwen3-Omni audio and image
input. Its generated-audio API is presently Qwen3-TTS-specific. Ollama 0.32.x
can package the language model and a projector but does not expose a generated
audio response or schedule the Omni Thinker → Talker → code2wav graph.

## Decision

The suite will produce an **Omni bundle**, not falsely claim one self-contained
GGUF:

1. `model.gguf` — language/Thinker model where supported by GGUF runtimes.
2. `mmproj.gguf` — audio/vision preprocessing and projection.
3. Talker weights — runtime-native speech code generator; not presently an
   Ollama-loadable GGUF layer.
4. code2wav weights — runtime-native speech-code decoder/vocoder; not presently
   an Ollama-loadable GGUF layer.
5. `omni_bundle.json` — component graph, compatibility evidence, runtime gates,
   and audio wire contract.

Native text-tensor substitution is allowed only when model type, hidden width,
layer count, vocabulary, and multimodal conditioning widths match. Otherwise,
the suite selects a cascade:

```text
base64 WAV → Omni/ASR audio understanding → Qwen3.8 or Ornith in Ollama
           → independently conditioned Qwen3-TTS → base64 WAV
```

Phase-1 input is RIFF/WAVE, mono PCM16 at 16 kHz. Output is RIFF/WAVE, mono
PCM16 at 24 kHz. Both are carried as strict base64 in JSON. Audio is a byte
stream, not a bitmap; the transport resembles Ollama image input only because
both use base64.

Video understanding reuses the Omni visual/frame path. Video generation is out
of scope.

## Consequences

### Positive

- Prevents loadable-looking but invalid cross-architecture tensor splices.
- Keeps Qwen3.8/Ornith tools, thinking, and language behavior intact.
- Makes runtime limitations machine-readable and testable.
- Permits migration to native Ollama audio when its API and loader support the
  full component graph.

### Negative

- Voice-to-voice inference requires an orchestration sidecar today.
- Cascade mode loses the native Omni Talker's direct hidden-state conditioning.
- A single Ollama registry tag cannot yet represent the complete audio-output
  execution graph.

## Migration

1. Generate plans with `python -m training_suite omni-plan`.
2. Validate audio requests through `/api/omni/audio/validate`.
3. Add llama.cpp audio-understanding and independent TTS process adapters.
4. Gate each runtime with transcript, semantic-audio, waveform, tools, thinking,
   and vision tests.
5. Replace the sidecar only after Ollama exposes native audio input/output and a
   manifest format for all required components.

## Rollback

Omni support is additive. Remove the bundle/API modules and continue using the
existing vision/tool pipeline; existing database rows and Ollama models are
unchanged.
