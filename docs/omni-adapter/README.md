# Ollama Omni Adapter

The Ollama Omni Adapter is the runtime and wire specification for serving one
Ollama model tag backed by one physical GGUF that contains:

```text
unprefixed tensors   Qwen3.8 or Ornith language/reasoning/tools graph
a.c.* tensors        self-contained audio/image/video comprehension graph
s.t.* tensors        independently text-conditioned speech synthesis graph
```

The file is monolithic; execution is not. A custom Ollama build must expose
filtered views of the three tensor namespaces and route semantic text between
the graphs. This prevents unsafe hidden-state splicing between architectures
whose widths, vocabularies, layer counts, or conditioning contracts differ.

## Status

| Deliverable | Status |
|---|---|
| One-file GGUF packer and inspector | Implemented and unit tested |
| Versioned request/response parser | Implemented as `robit.ollama.omni-adapter.v1` |
| Audio, image, and video envelope validation | Implemented |
| ASR, media-description, chat, and TTS routes | Specified and exercised by the reference server |
| Python and JavaScript clients | Implemented under `examples/omni_adapter/` |
| HTTP sidecar reference adapter | Implemented for development and integration tests |
| Patched Ollama in-process loader/executor | Design specified; implementation pending in an Ollama fork |
| Production Qwen3-Omni comprehension GGUF converter | Pending |
| Production Qwen3-TTS GGUF converter/executor | Pending |

The sidecar proves the protocol and component boundaries. It does not make
stock Ollama execute the embedded `a.c.*` or `s.t.*` tensors.

## Documentation map

- [Wire protocol](protocol.md) — request fields, media envelopes, tasks,
  responses, errors, and compatibility rules.
- [GGUF ABI](gguf-abi.md) — tensor namespaces, metadata keys, filtered model
  views, quantization order, and validation.
- [Runtime and Ollama patch guide](runtime.md) — router state machine, required
  Ollama changes, component interfaces, video normalization, and lifecycle.
- [Build and release runbook](build-and-release.md) — source selection,
  conversion, packing, Ollama creation, publication, rollback, and cleanup.
- [Test plan](testing.md) — unit, component, integration, live capability, and
  release gates.
- [Ollama model-page template](model-page-template.md) — release description,
  runtime warning, examples, provenance, and verified-capability table.
- [Runnable examples](../../examples/omni_adapter/README.md) — sidecar,
  Python client, JavaScript client, and curl requests.
- [Request JSON Schema](schema/request-v1.schema.json) and
  [response JSON Schema](schema/response-v1.schema.json).

## Contract in one request

```json
{
  "model": "robit/qwen3.8-omni:latest",
  "messages": [{
    "role": "user",
    "content": "Answer the question in the recording and speak the answer.",
    "audios": [{
      "mime_type": "audio/wav",
      "encoding": "base64",
      "data": "<base64 16 kHz mono PCM16 RIFF/WAVE>"
    }]
  }],
  "omni": {
    "schema": "robit.ollama.omni-adapter.v1",
    "task": "chat"
  },
  "response_modalities": ["text", "audio"],
  "speech_mode": "always",
  "think": true,
  "stream": false
}
```

The response remains Ollama-shaped. The extension adds `message.audio` and an
`adapter` trace:

```json
{
  "model": "robit/qwen3.8-omni:latest",
  "message": {
    "role": "assistant",
    "content": "The answer is ...",
    "thinking": "...",
    "audio": {
      "type": "audio",
      "mime_type": "audio/wav",
      "encoding": "base64",
      "sample_rate_hz": 24000,
      "channels": 1,
      "sample_width_bits": 16,
      "data": "<base64 RIFF/WAVE>"
    }
  },
  "adapter": {
    "schema": "robit.ollama.omni-adapter.v1",
    "task": "chat",
    "route": ["comprehension", "language", "tts"]
  },
  "done": true
}
```

JSON transports waveform bytes as base64. This is a byte-preserving transport
encoding, not a bitmap conversion. A future streaming revision may carry
binary audio frames, but adapter v1 intentionally requires `stream: false`.

## Reference implementation

Validate the protocol without inference:

```bash
curl -s http://127.0.0.1:7860/api/omni/adapter/contract
curl -s http://127.0.0.1:7860/api/omni/adapter/validate \
  -H 'content-type: application/json' \
  --data-binary @request.json
```

Run the sidecar example after configuring comprehension and TTS component
servers:

```bash
training_suite/.venv/bin/python examples/omni_adapter/server.py
training_suite/.venv/bin/python examples/omni_adapter/client.py \
  --model robit/qwen3.8-omni:latest \
  asr ./speech-16khz-mono.wav
```

See the [examples guide](../../examples/omni_adapter/README.md) for all routes.

## Compatibility position

The adapter extends Ollama's native `/api/chat` shape rather than replacing it:

- normal `model`, `messages`, `tools`, `think`, `format`, `options`,
  `keep_alive`, and log-probability fields pass through to the language graph;
- native `message.images` remains accepted;
- `message.audios`, `message.videos`, `omni`, `response_modalities`, `speech_mode`,
  and `speech` are additive fields handled by the custom adapter;
- `thinking` and `tool_calls` are preserved in the response;
- speech generation is deferred when unresolved tool calls are returned.

Official Ollama currently documents images, tools, and thinking on `/api/chat`,
but not these audio/video/TTS extensions. See the
[chat API](https://docs.ollama.com/api/chat),
[model import guide](https://docs.ollama.com/import), and
[Modelfile reference](https://docs.ollama.com/modelfile). The additions in this
repository therefore require the documented custom runner or the sidecar.

Qwen3-Omni's official implementation processes interleaved text, images,
audio, and video and can produce 24 kHz audio. Its serving limitations and
`use_audio_in_video` behavior are documented in the
[Qwen3-Omni repository](https://github.com/QwenLM/Qwen3-Omni). The independent
speech component in this design follows the text-conditioned
[Qwen3-TTS project](https://github.com/QwenLM/Qwen3-TTS), because an Omni Talker
conditioned on a different Thinker cannot safely consume Qwen3.8/Ornith hidden
states.
