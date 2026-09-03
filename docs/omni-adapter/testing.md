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
PYTHONPATH=. python -m pytest -q tests
python -m compileall -q training_suite examples tests
ruff check training_suite examples tests
git diff --check
```

The unit suite covers strict audio/video/image/GIF decoding, silent-video
handling, item and size limits, all four routes, Ollama field preservation,
speech/tool deferral, long-form TTS block assembly, TTS envelope validation,
six-view GGUF packing/materialization, and custom Ollama layer
attach/resolve/cache preparation. Portal tests additionally cover native
reasoning-off routing, adaptive call VAD, smooth message handling, concurrent
session and document-index isolation, bounded PDF/text ingestion,
content-redacted diagnostic expiry/deletion, streamed PCM relay, public-web
egress rejection, session memory/web-index isolation, and structured
multi-round tool execution. The tool gate proves both synchronous and NDJSON
chains continue past the former 50-call boundary, stops exact duplicate
no-progress loops, and verifies isolated session-scoped sub-agent delegation.
It also includes local-browser result and
redirect parsing, provider-challenge fail-closed behavior, network-free session
recall, search→fetch, textual-call compatibility, live bounded receipts, and
memory-write→memory-search→final chains while ensuring tools remain off without
explicit client opt-in and unresolved calls never reach TTS.
The browser-cache harness covers restore, five-minute logical expiry, media
preview retention, and explicit clear. Environment tests assert bounded output
and the omission of IP/MAC data. The persistent-TTS harness proves that two
prompts reuse one process while returning independent framed PCM sequences.
The real-component gate also synthesizes distinct A, B, A sentinel prompts and
transcribes their waveforms; the recognized order must remain A, B, A.

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
| animated GIF | normalized temporal order preserved |
| video with speech/sound | visual order plus audio assertion |
| direct TTS sentence | positive-duration 24 kHz mono PCM16 WAV |
| long TTS paragraph | multiple blocks, one start event, contiguous sequence |
| PDF/text document | bounded extraction, relevant retrieval, session isolation |

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
- long text continues across the per-generation frame limit and yields a complete replay WAV;
- the first audio event is emitted only with the first real PCM window;
- at least two repeated requests succeed;
- matching-profile repeated requests retain the same resident worker PID;
- distinct A, B, A prompts transcribe back in A, B, A order with no one-turn lag;
- the default stream window is two codec frames and disables proxy buffering;
- `verify_pcm_stream.py` passes on raw server PCM, with no deterministic prefix
  or high median discontinuity at decoder-window boundaries;
- empty/excessive text and unsupported options have defined errors;
- selected languages/voices are only advertised after their own tests.

## Route matrix

| Request | Expected route | Critical result |
|---|---|---|
| text chat | `language` | stock response; no media worker required |
| image/audio/video chat | `comprehension → language` | observation used as untrusted evidence |
| environmental audio chat | `comprehension → language` | transcript and non-speech acoustic evidence remain separate |
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
- adapter media fields sent to an ordinary model without a sidecar;
- sequential distinct clips with llama.cpp prompt caching accidentally enabled;
- quiet/click/steady-noise VAD fixtures causing remote ASR requests;
- rapid call segments creating parallel inference requests instead of one bounded consolidated pending turn;
- two concurrent browser sessions receiving one another's marker or journal;
- trash followed by a late aborted-request event recreating diagnostic data.
- cache restoration re-submitting display-only media to inference;
- live-call prompting that merely echoes or paraphrases the user by default;
- runtime snapshots exposing hostnames, addresses, processes, or credentials.

Errors must be typed and must not echo raw media, secrets, thinking, or tool
arguments unnecessarily.

## Publication gate

- [ ] source revisions, licenses, sizes, and hashes recorded;
- [ ] llama.cpp and repository commits recorded;
- [ ] unit, compile, lint, and diff checks pass;
- [ ] six-view artifact round trip passes;
- [ ] stock text/vision/thinking/tools pass;
- [ ] ASR, image, video, and video-audio tests pass;
- [ ] sequential red → blue → red video isolation passes with
      `cache_prompt=false` and no stale media embedding;
- [ ] direct, repeated, and A → B → A prompt/audio-alignment TTS pass;
- [ ] phone VAD, concurrent-session isolation, and diagnostic TTL/delete
      harnesses pass;
- [ ] IndexedDB reload/expiry/clear, environment privacy, and persistent-TTS
      reuse harnesses pass;
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
