# ADR-002: Monolithic GGUF with custom Ollama audio routing

- Status: Accepted for experiment
- Date: 2026-08-31
- Supersedes: ADR-001 artifact-layout decision; ADR-001 compatibility analysis remains valid

## Context

The required deliverable is one physical GGUF that Ollama can import with one
`FROM` reference. The artifact must retain a Qwen3.8 or Ornith language model,
understand audio and video through Qwen3-Omni-derived components, and optionally
generate TTS audio. A custom Ollama-compatible request handler and runner are
allowed.

The component architectures are not hidden-state compatible. Qwen3-Omni audio,
vision, Thinker, Talker, and codec tensors cannot be copied directly into a
Qwen3.8 or Ornith language graph and expected to function. The single-file
requirement is therefore an artifact constraint, not a requirement that all
weights form one transformer graph.

Current Ollama create/compatibility code already recognizes embedded tensor
prefixes including `a.` and `s.` as compatibility tensors that must be
preserved. Its llama.cpp compatibility layer also has a precedent for hiding
embedded modality tensors from the text loader and constructing filtered views
of a monolithic GGUF.

## Decision

Build a monolithic GGUF with three executable graph namespaces:

| Namespace | Role |
|---|---|
| unprefixed | Qwen3.8 or Ornith language, reasoning, thinking, and tools |
| `a.c.*` | Self-contained audio/video comprehension graph |
| `s.t.*` | Independently text-conditioned TTS graph |

The base GGUF remains the primary `general.architecture`. Every embedded
component's original metadata is copied under a component metadata namespace.
The custom runner reconstructs a component view by stripping its tensor and
metadata prefixes.

The file carries `robit.audio_bundle.*` metadata containing the schema,
component digests, routing contract, and source provenance. The suite's
`omni-pack` command writes the file and `omni-inspect` validates it.

## Runtime contract

The versioned wire schema is `robit.ollama.omni-adapter.v1`. The artifact schema
remains `robit.ollama-monolithic-audio.v1`; storage and API compatibility are
versioned independently. The complete normative contract is maintained in
`docs/omni-adapter/protocol.md`.

The custom Ollama `/api/chat` extension accepts audio in a message field modeled
after image input:

```json
{
  "model": "robit/combined:latest",
  "messages": [{
    "role": "user",
    "content": "What happened in this recording?",
    "audios": [{
      "mime_type": "audio/wav",
      "encoding": "base64",
      "sample_rate_hz": 16000,
      "channels": 1,
      "sample_width_bits": 16,
      "data": "<base64 RIFF/WAVE bytes>"
    }]
  }],
  "omni": {
    "schema": "robit.ollama.omni-adapter.v1",
    "task": "chat"
  },
  "response_modalities": ["text", "audio"],
  "speech_mode": "auto",
  "stream": false
}
```

When speech is selected, the response places a tagged waveform inside the
assistant message:

```json
{
  "message": {
    "role": "assistant",
    "content": "The speaker asked for a weather update.",
    "audio": {
      "mime_type": "audio/wav",
      "encoding": "base64",
      "sample_rate_hz": 24000,
      "channels": 1,
      "sample_width_bits": 16,
      "data": "<base64 RIFF/WAVE bytes>"
    }
  }
}
```

JSON cannot contain raw binary bytes, so turn-based responses use base64.
Streaming may use tagged base64 PCM chunks in Ollama's NDJSON stream or a
separate binary transport negotiated by the custom client.

## Routing

1. Text-only input goes directly to the base language graph.
2. Audio/video input goes to the comprehension graph.
3. The comprehension graph returns semantic text, which is passed to the base
   language graph with the user's instruction.
4. The language graph returns normal content, thinking, and tool calls.
5. `speech_mode` and `response_modalities` determine whether the final text is
   passed to the TTS graph.
6. The response always retains text; generated audio is an additional tagged
   field.

An automatic model router may choose speech only when the client leaves the
decision open. Explicit `always` and `never` requests take precedence.

## Why the comprehension component is self-contained

Without a trained bridge, an audio/vision encoder from Qwen3-Omni cannot feed a
Qwen3.8 or Ornith hidden space. The initial single-file implementation embeds a
self-contained comprehension graph and crosses the boundary through semantic
text. A later learned adapter may replace that graph if it passes multimodal
alignment evaluations.

Likewise, the original Qwen3-Omni Talker is conditioned on its matching Thinker.
The initial output component must therefore be a text-conditioned TTS model,
not an unmodified Talker attached to Qwen3.8/Ornith hidden states.

## Required Ollama changes

The custom build must:

1. recognize the `robit.audio_bundle.schema` marker;
2. preserve `a.c.*` and `s.t.*` tensors during create/quantize/copy/push;
3. exclude embedded component tensors from the base text loader;
4. expose filtered metadata/tensor views for comprehension and TTS loaders;
5. account for each component in memory scheduling and load them lazily;
6. accept the `audios`, `response_modalities`, and `speech_mode` API fields;
7. return tagged audio and stream it without corrupting the normal Ollama
   response contract;
8. keep tool calls and thinking fields intact;
9. validate input size, WAV format, cancellation, backpressure, and timeouts.
10. accept bounded `videos` envelopes, normalize frames in temporal order, and
    preserve optional video-audio alignment;
11. implement `chat`, `transcribe`, `describe`, and `synthesize` routes under the
    versioned wire schema.

## Consequences

- The registry contains one model blob and one model tag.
- The file can be very large because it contains multiple complete graphs.
- Stock Ollama may import and run the base language tensors, but audio/video/TTS
  behavior requires the custom handler and loader.
- One physical file does not imply all components must remain resident in GPU
  memory simultaneously.
- Component quantization and conversion must happen before packing; ordinary
  text-only quantizers must not be run blindly after packing.
- Source component revisions and digests remain necessary even though the
  published artifact is monolithic.

## Verification gates

- `omni-inspect` finds non-empty base, `a.c.*`, and `s.t.*` tensor sets.
- `ollama create` preserves the output blob and the base text model loads.
- Text, tools, thinking, and existing vision behavior do not regress.
- Audio comprehension passes known speech and non-speech fixtures.
- Video comprehension preserves frame order and answers temporal questions.
- TTS output is valid 24 kHz mono PCM16 WAV.
- `speech_mode=never`, `always`, and `auto` route correctly.
- Copy/push/pull produces an identical bundle digest or an explicitly recorded
  registry-layer digest.

## Implementation references

- `docs/omni-adapter/` — wire ABI, GGUF ABI, runtime patch guide, build/release
  runbook, schemas, model-page template, and test plan;
- `training_suite/models/omni_adapter.py` — executable v1 parser and route
  planner;
- `examples/omni_adapter/` — HTTP reference adapter and clients.
