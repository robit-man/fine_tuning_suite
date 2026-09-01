# Omni Adapter Test Plan

A release is valid only when the repository, container, Ollama manifest,
individual graphs, combined routes, registry copies, and cleanup gates pass.

## Test levels

| Level | Purpose |
|---|---|
| Unit/schema | Validate envelopes, routes, GGUF views, sidecar manifests, and response encoding |
| Artifact | Verify hashes, tensor inventories, metadata, and round-trip materialization |
| Component | Execute base, projector, comprehension, and TTS pairs independently |
| Integration | Exercise the public adapter across real component workers |
| Regression | Confirm stock Qwen3.8 text/vision/tools/thinking remain unchanged |
| Publication | Verify Hugging Face files and Ollama push/pull layer survival |

## Repository gate

```bash
python -m pytest -q tests
python -m compileall -q training_suite examples tests
ruff check training_suite examples tests
git diff --check
```

The unit suite covers strict audio/video/image decoding, item and size limits,
all four routes, Ollama field preservation, speech/tool deferral, TTS envelope
validation, six-view GGUF packing/materialization, and custom Ollama layer
attach/resolve/cache preparation.

## Artifact gate

For the release sidecar:

1. `omni-inspect` reports schema `robit.ollama-monolithic-omni.v3`.
2. Exactly six expected view counts are nonzero.
3. Sidecar size and SHA-256 match the release record.
4. Every materialized view matches its pinned input SHA-256 and tensor count.
5. The installed Ollama manifest has normal model/projector layers and exactly
   one Robit sidecar layer with the same digest.
6. `ollama show` loads the standard graph without seeing sidecar tensors.

The sidecar itself is not sent to stock Ollama as a model layer. A successful
GGUF parse is not evidence that heterogeneous graphs can be executed together.

## Fixtures

Use compact redistributable fixtures with documented expected assertions:

| Fixture | Assertion |
|---|---|
| 16 kHz mono PCM16 speech | exact or thresholded transcript |
| image with color/shape/text | all expected visible facts |
| silent temporal video | event order preserved |
| video with speech/sound | visual order plus audio assertion |
| direct TTS sentence | positive-duration 24 kHz mono PCM16 WAV |

Record fixture digest, size, duration/dimensions, codec, license, and expected
result. Do not commit large or restricted media.

## Component gates

### Stock Ollama base

- coherent completion;
- `think=true` returns a separately parsed thinking field;
- a tool request returns the expected structured function and arguments;
- a native image fixture is understood correctly;
- architecture, context, quantization, projector, and capabilities match the
  source base tag.

### Qwen3-Omni comprehension

- exact clean-speech transcription;
- image description/OCR assertions;
- temporal video order;
- video audio used when requested;
- video audio ignored when disabled;
- malformed media fails without a crash or unbounded allocation.

### Qwen3-TTS

- RIFF/WAVE container, PCM16, mono, 24 kHz;
- positive and bounded duration;
- at least two repeated requests succeed;
- empty/excessive text and unsupported options have defined errors;
- selected languages/voices are only advertised after their own tests.

## Route matrix

| Request | Expected route | Critical result |
|---|---|---|
| text chat | `language` | stock response; no media worker required |
| image/audio/video chat | `comprehension → language` | observation used as untrusted evidence |
| media chat with speech | `comprehension → language → tts` | text plus valid tagged WAV |
| ASR | `comprehension` | direct transcript, no language paraphrase |
| video describe | `comprehension` | temporal and audio policy honored |
| direct TTS | `tts` | supplied text synthesized |
| media tool request | `comprehension → language` | tool call preserved; TTS deferred |
| tool-result follow-up | `language → tts` | final answer spoken, not tool JSON |

## Negative and resilience tests

- missing model/messages/user message;
- unknown explicit schema or task;
- `stream:true` under adapter v1;
- invalid or oversized base64;
- MIME/container mismatch;
- compressed or wrong-rate WAV;
- unsupported image/video container;
- invalid sampling FPS/frame count;
- ffmpeg, comprehension, language, and TTS timeout;
- prompt injection in OCR/transcript/subtitles;
- unresolved tools with speech requested;
- insufficient memory, cancellation, and repeated worker restart;
- adapter media fields sent to an ordinary model without a sidecar.

Errors must be typed and must not echo raw media, secrets, thinking, or tool
arguments unnecessarily.

## Publication gate

- [ ] source revisions, licenses, sizes, and hashes recorded;
- [ ] llama.cpp and repository commits recorded;
- [ ] unit, compile, lint, and diff checks pass;
- [ ] six-view artifact round trip passes;
- [ ] stock text/vision/thinking/tools pass;
- [ ] ASR, image, video, and video-audio tests pass;
- [ ] direct and repeated TTS pass;
- [ ] adapter output envelope validates;
- [ ] repository documentation is pushed;
- [ ] Hugging Face GGUF/model card/report are remotely verified;
- [ ] Ollama `q4km` and `latest` are remotely verified;
- [ ] pulled Ollama tag retains the custom sidecar digest;
- [ ] temporary CUDA services are stopped and leases released;
- [ ] cleanup is measured and performed only after every preceding gate.

If any modality fails, remove that claim or block the release. An accepted
request, tensor prefix, local model creation, or successful upload command is
not enough on its own.
