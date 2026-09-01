# Ollama Omni Adapter

The Robit Omni Adapter adds turn-based audio input, video comprehension, and
speech output to an otherwise normal Ollama model tag. The first target is:

```text
robit/qwen3.8-27b-e03-obliterated-omni:q4km
```

Callers use one model name and one `/api/chat`-shaped endpoint. Stock Ollama
continues to execute Qwen3.8 completion, native image vision, parsed thinking,
and structured tools. The adapter resolves the same tag's custom media layer
for Qwen3-Omni comprehension and Qwen3-TTS synthesis.

## What “one model” means

An Ollama model is an OCI-style manifest, not necessarily one filesystem blob.
This design deliberately uses one logical model tag with these layers:

```text
Ollama manifest: robit/qwen3.8-27b-e03-obliterated-omni:q4km
  ├── standard Ollama model layer       Qwen3.8 language/reasoning/tools
  ├── standard Ollama projector layer   native Qwen3.8 image vision
  ├── template, parameters, and license
  └── Robit Omni sidecar layer           one namespaced multi-graph GGUF
       ├── unprefixed                     byte-identical base model view
       ├── b.p.*                          byte-identical base projector view
       ├── a.c.m.* + a.c.p.*             Qwen3-Omni comprehension pair
       └── s.t.m.* + s.t.p.*             Qwen3-TTS model/codec pair
```

The custom layer media type is:

```text
application/vnd.robit.ollama.omni.bundle.v1+gguf
```

This gives users one pull, one tag, and one public request contract. It does
not mean that unmodified Ollama executes every graph internally. Unknown OCI
layers are ignored by stock Ollama; the reference adapter finds the layer in
the local manifest, validates it, materializes disposable component views,
and runs those views with the pinned llama.cpp multimedia tools.

## Why the layer boundary is necessary

GGUF has one architecture metadata set and one contiguous tensor inventory.
Stock Ollama/llama.cpp validates that inventory against the selected graph.
Combining unrelated Qwen3.8, Qwen3-Omni, and Qwen3-TTS tensors as one directly
loaded model therefore fails required-tensor accounting. Appending unindexed
bytes is also unsuitable because Ollama normalizes the imported model blob,
while large opaque metadata is loaded eagerly and wastes tens of gigabytes of
RAM.

The sidecar remains one valid GGUF container and retains all six reproducible
views, but it is attached as a custom manifest layer rather than used as the
stock model layer. This is the only tested arrangement that preserves native
Ollama behavior and ships all media weights under the same tag without unsafe
hidden-state splicing.

## Capability ownership

| Capability | Executor | Stock Ollama alone? |
|---|---|---|
| Text completion | Qwen3.8 standard model layer | Yes |
| Thinking | Qwen3.8 renderer/parser | Yes |
| Structured tools | Qwen3.8 renderer/parser | Yes |
| Image understanding | standard Qwen3.8 projector | Yes |
| Audio understanding / ASR | Qwen3-Omni sidecar views through adapter | No |
| Video understanding | Qwen3-Omni sidecar views through adapter | No |
| TTS | Qwen3-TTS sidecar views through adapter | No |
| Video generation | not included | No |

The media boundary between comprehension and Qwen3.8 is semantic text. Media
output is treated as untrusted evidence and never as a system instruction.
TTS is independently text-conditioned; the Qwen3-Omni Talker cannot consume a
different Thinker's hidden states without a trained bridge.

## Status

| Deliverable | Status |
|---|---|
| Six-view GGUF v3 packer, inspector, and materializer | Implemented and tested |
| Ollama sidecar attach, resolve, and cache preparation | Implemented and tested |
| Wire schema `robit.ollama.omni-adapter.v1` | Implemented and tested |
| Audio/image/video parsing and bounded validation | Implemented and tested |
| ASR, describe, chat, and synthesize routes | Implemented in reference adapter |
| Python and JavaScript clients | Implemented |
| Qwen3-Omni audio/image/video inference | Live-tested with pinned llama.cpp |
| Qwen3-TTS 24 kHz PCM16 output | Live-tested with pinned llama.cpp |
| Native in-process audio/video/TTS in upstream Ollama | Not available |
| Streaming audio/video ABI | Not included in adapter v1 |

## Request and response

```json
{
  "model": "robit/qwen3.8-27b-e03-obliterated-omni:q4km",
  "messages": [{
    "role": "user",
    "content": "Answer the recording and speak the answer.",
    "audios": [{
      "mime_type": "audio/wav",
      "encoding": "base64",
      "data": "<base64 16 kHz mono PCM16 WAV>"
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

The response remains Ollama-shaped. Speech is added under `message.audio` as a
tagged base64 RIFF/WAVE envelope: PCM16, mono, 24 kHz. Base64 is a transport
encoding for bytes, not a bitmap. Adapter v1 requires `stream:false`.

## Installed-tag workflow

Inspect and prepare the media views directly from the one installed tag:

```bash
python -m training_suite omni-resolve \
  robit/qwen3.8-27b-e03-obliterated-omni:q4km

python -m training_suite omni-prepare \
  robit/qwen3.8-27b-e03-obliterated-omni:q4km \
  --out ./runtime-cache
```

`runtime-cache` is derived data. Delete it after the runtime stops; the next
session can reconstruct it from the tag's sidecar layer.

## Documentation map

- [Wire protocol](protocol.md)
- [GGUF and Ollama layer ABI](gguf-abi.md)
- [Runtime guide](runtime.md)
- [Build and release runbook](build-and-release.md)
- [Test plan](testing.md)
- [First release record](qwen38-27b-e03-release.md)
- [Hugging Face model card](huggingface-model-card.md)
- [Machine-readable first-release manifest](sidecar-manifest.json)
- [Machine-readable validation report](validation-report.json)
- [Ollama custom-layer descriptor](ollama-sidecar-layer.json)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Model-page template](model-page-template.md)
- [Runnable examples](../../examples/omni_adapter/README.md)
- [Request schema](schema/request-v1.schema.json) and
  [response schema](schema/response-v1.schema.json)

Official Ollama supports images, tools, and thinking but does not currently
define these audio/video/TTS fields. Clients requiring media must call the
adapter endpoint described here; ordinary Ollama clients can use the same tag
for its native capabilities.
