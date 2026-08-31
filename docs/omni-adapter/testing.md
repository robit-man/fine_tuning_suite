# Omni Adapter Test Plan

The combined model is releasable only when storage, protocol, every component,
the full route, and the original language capabilities pass independently.

## Test levels

| Level | Purpose | Hardware |
|---|---|---|
| Unit | Validate envelopes, routing, GGUF namespaces, and response encoding | CPU |
| Schema | Keep examples and API objects compatible with adapter v1 | CPU |
| Component | Prove base, comprehension, and TTS graphs independently | CPU/GPU as required |
| Integration | Prove ASR, video, language, tools, thinking, and TTS routing | GPU/runtime |
| Regression | Compare candidate against the verified base and component baselines | GPU/runtime |
| Release | Verify local tag, pushed tag, manifests, digests, and documentation | Production-like |

## Repository tests

```bash
uv run --with pytest --with httpx --with flask --with numpy --with gguf \
  python -m pytest -q

python3 -m compileall -q training_suite examples tests
git diff --check
```

The unit suite covers:

- 16 kHz PCM WAV validation and 24 kHz response encoding;
- MP4/WebM and JPEG/PNG/WebP signature checks;
- decoded-size, item-count, schema, task, and sampling validation;
- `chat`, `transcribe`, `describe`, and `synthesize` route selection;
- preservation of tools and thinking through the reference server;
- speech generation and WAV validation;
- real tiny-GGUF packing with all three tensor namespaces;
- Flask contract and validation endpoints.

## Required fixtures

Keep compact, redistributable fixtures with licenses and expected results:

| Fixture | Required property |
|---|---|
| `silence-1s.wav` | 16 kHz mono PCM16 silence |
| `speech-en.wav` | Clean English speech with known transcript |
| `speech-multilingual.wav` | Supported non-English speech with known transcript |
| `non-speech.wav` | Distinct sound events such as bell, footsteps, and music |
| `image-text.png` | Legible text plus objects for OCR/description |
| `temporal-events.mp4` | At least three events whose order matters |
| `av-alignment.mp4` | Visible event and aligned audio cue/speech |
| `direct-tts.txt` | Punctuation, digits, abbreviations, and multilingual text |

Do not commit large or restricted fixtures. Record SHA-256, duration,
dimensions, codec, license, and expected semantic assertions.

## Component gates

### Base graph

- text completion is coherent;
- `think=true` yields correctly parsed `message.thinking` where supported;
- structured tool call has the expected name and arguments;
- follow-up tool result produces a final answer;
- context size and tokenizer behavior match the pre-bundle baseline;
- native image understanding remains unchanged if included.

### Comprehension graph

- clean-speech word error rate is within the selected baseline threshold;
- non-speech caption identifies expected events without inventing speech;
- image description/OCR covers required assertions;
- video response preserves event order;
- audio-video response uses both streams when requested;
- `include_audio_from_video=false` prevents the audio track from affecting the
  result;
- malformed media fails without crashing or allocating unbounded memory.

### TTS graph

- output is complete RIFF/WAVE, 24 kHz, mono, PCM16;
- output duration is positive and bounded;
- transcript intelligibility meets the selected ASR round-trip threshold;
- requested supported voice is stable across repeated calls;
- unsupported voice/language/style returns a defined error or documented
  fallback;
- empty input and excessive text are rejected cleanly.

## Route matrix

| Request | Expected route | Critical assertions |
|---|---|---|
| text chat | `language` | No media graph load; tools/thinking unchanged |
| audio chat, text response | `comprehension → language` | Semantic observation used; no audio output |
| audio chat, speech response | `comprehension → language → tts` | Text plus valid WAV |
| ASR | `comprehension` | Transcript is not paraphrased by language graph |
| video describe | `comprehension` | Temporal ordering and frame/audio policy |
| multimodal tool request | `comprehension → language` | Tool call preserved; TTS deferred |
| tool-result follow-up with speech | `language → tts` | Final answer spoken, not tool JSON |
| direct TTS | `tts` | Input text returned and synthesized exactly |

## Negative tests

At minimum:

- missing model/messages/user message;
- unknown explicit schema;
- `stream:true` under v1;
- invalid/whitespace/non-padded base64;
- spoofed MIME versus container signature;
- compressed or wrong-rate WAV;
- unsupported image/video container or codec;
- per-item and aggregate limit violations;
- FPS, frame-count, duration, resolution, and decompression-bomb limits;
- comprehension timeout/cancellation;
- TTS invalid JSON or malformed WAV;
- insufficient GPU memory and component eviction under load;
- adapter fields submitted to an ordinary model;
- prompt injection present in OCR, subtitles, or transcripts;
- unresolved tool calls with speech requested.

Errors must not echo media or secrets.

## One-GGUF conformance

For the production artifact, verify:

1. Exactly one model GGUF is referenced by the Modelfile.
2. Its digest equals the release manifest.
3. `omni-inspect` finds nonzero base, comprehension, and TTS tensor counts.
4. Each filtered view matches its pre-pack component tensor inventory.
5. Stock base tensors do not expose `a.c.*` or `s.t.*` to the text loader.
6. All contexts map the same physical file; no undisclosed weight sidecar is
   required.
7. Text-only loading does not allocate comprehension/TTS execution state.
8. Lazy component load and eviction do not corrupt later base requests.

## Performance and reliability

Record cold and warm measurements for:

- component load time;
- time to first transcript/description;
- language time to first token and total tokens per second;
- TTS time to first waveform and real-time factor;
- peak CPU RAM and VRAM by component;
- video decode and preprocessing time;
- maximum stable concurrent audio, video, and text requests;
- cancellation latency and memory reclamation;
- repeated load/evict cycles.

Set release thresholds from a measured baseline. Do not publish guessed latency
or VRAM figures.

## Publication gate

All boxes are required for a capability-complete tag:

- [ ] component sources, revisions, licenses, and digests recorded;
- [ ] custom Ollama fork commit and binary digest recorded;
- [ ] pack report and final GGUF SHA-256 recorded;
- [ ] unit/schema suite green;
- [ ] base text/tools/thinking regressions green;
- [ ] audio comprehension/ASR gates green;
- [ ] image and video comprehension gates green;
- [ ] TTS format and intelligibility gates green;
- [ ] combined media → language → TTS route green;
- [ ] tool-call speech deferral green;
- [ ] malformed media and resource-limit gates green;
- [ ] local Ollama tag verified;
- [ ] remote tag push completed and remote manifest/digest verified;
- [ ] model page links to protocol, examples, runtime requirement, limitations,
  provenance, licenses, and test report;
- [ ] run-local safetensors and conversion intermediates cleaned only after all
  preceding checks.

If one modality fails, do not advertise that modality. A successful
`ollama create`, visible tensor prefix, or accepted request is not evidence of
correct inference.
