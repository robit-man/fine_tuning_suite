# Architecture Impact: Qwen3-Omni Audio Experiment

## Change Summary

Add compatibility-gated planning, a namespaced multi-graph GGUF sidecar,
custom Ollama manifest-layer handling, and byte-level media contracts while
retaining Qwen3.8/Ornith as the stock-runnable language graph.

## Component Impact

| Component | Change | Compatibility |
|---|---|---|
| Model intake | Detect `audio-input` and `audio-output` from HF/GGUF metadata | Additive |
| GGUF inspection | Detect audio encoder and Talker/code2wav pairs | Additive |
| Bundle planner | Generate native-Omni or monolithic-router plans | New component |
| Sidecar packer | Preserve six base/projector/comprehension/TTS views under isolated namespaces | New component |
| Ollama packaging | Attach one custom GGUF layer to a normal runnable manifest; resolve and materialize views | Extended |
| REST API | Adds versioned audio/image/video/TTS contracts, validation, plan, and reference routing endpoints | Additive |
| Adapter examples | Adds reference sidecar plus Python and JavaScript clients | New component |
| Dashboard | Displays audio-input/audio-output capability pills | Additive |
| Evaluation | Unit gates for WAV transport and architecture compatibility | Additive |

## Principal Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Invalid tensor graft | Critical | Exact architecture and dimension gate |
| Ollama appears audio-capable but drops data | High | Runtime support recorded separately from artifact metadata |
| Unbounded base64 payload | High | Strict decode and 32 MiB input limit |
| Wrong sampling format | Medium | Require 16 kHz mono PCM16 WAV input |
| TTS output is not parseable | Medium | Require 24 kHz mono PCM16 WAV response envelope |
| Disk growth from multi-component weights | High | Existing post-verification session cleanup policy applies |
| Text-only quantizer drops embedded tensors | High | Quantize components before packing; inspect final namespaces |
| Registry strips unknown custom layer | Critical | Push/pull round-trip and digest verification before release |
| Media prompt injection reaches tools | High | Delimit semantic observations as untrusted evidence; preserve tool schemas separately |
| Video decoder resource exhaustion | High | Sandboxed decoder plus size, FPS, frame, pixel, duration, and timeout limits |

## Phase Gate

Construction of large artifacts is allowed only after a plan reports either
`native-omni` or `ready-for-monolithic-pack`. Publication remains blocked until
every component digest is recorded, live audio input/output, video/vision,
tools, and thinking pass from the installed sidecar, and both Hugging Face and
Ollama registry artifacts are verified remotely.
