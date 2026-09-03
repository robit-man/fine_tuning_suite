# Omni Adapter v1 Wire Protocol

`robit.ollama.omni-adapter.v1` is an additive extension to Ollama's non-streaming
`POST /api/chat` request. It defines binary media envelopes, task routing, and
speech output while preserving normal language, thinking, and tool behavior.

## Normative language

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY describe requirements for
interoperable implementations. The executable authority for validation is
`training_suite.models.omni_adapter`; the JSON schemas provide a portable
structural description.

## Endpoint and transport

- Endpoint: `POST /api/chat`
- Content type: `application/json`
- Request body: UTF-8 JSON object
- Streaming: v1 MUST use `"stream": false`
- Inline binary: standard padded base64 without whitespace
- Maximum decoded payloads: 32 MiB per audio item, 20 MiB per image, and
  256 MiB per video
- Maximum items per request: 8 audio, 16 image, and 4 video items

Deployments MAY set lower limits. They MUST return a clear 4xx error instead of
silently truncating media.

## Request fields

### Preserved Ollama fields

The adapter passes these fields to the language stage when `omni.task=chat`:

| Field | Behavior |
|---|---|
| `model` | Selects the combined Ollama tag and is required |
| `messages` | Preserves roles, text, tool-call history, and tool results |
| `tools` | Passed unchanged to the language stage |
| `think` | Passed unchanged; `thinking` remains separate in the response |
| `format` | Passed unchanged |
| `options` | Passed unchanged |
| `keep_alive` | Passed unchanged |
| `logprobs`, `top_logprobs` | Passed unchanged when supported by the runtime |

An implementation MUST NOT include base64 media in the language prompt. It
passes media only to the comprehension graph and injects its semantic result as
clearly delimited, untrusted observation text.

For `chat`, the comprehension graph is a perception encoder, not a second
assistant. It MUST NOT answer the user. Audio-bearing observations SHOULD place
verbatim speech inside `<speech_transcript>` and non-speech acoustic evidence
inside `<audio_observation>`. Acoustic evidence includes environmental events,
ambience, music, speaker activity, temporal changes, and uncertainty without
duplicating transcript text. Visual evidence belongs in `<visual_observation>`,
with no conversational reply outside those tags. A client may attribute only
the extracted transcript—not either observation—to the user's chat role.

The runtime MUST withhold conversational message text from the comprehension
graph. That text is an instruction to the language model, while the media graph
receives only media plus a modality-specific extraction instruction. This
separation prevents prompts such as “reply naturally” from producing a second,
misrouted assistant answer during perception.

### Adapter fields

| Field | Required | Meaning |
|---|---:|---|
| `omni.schema` | Recommended | Must be `robit.ollama.omni-adapter.v1` when supplied |
| `omni.task` | No | `chat`, `transcribe`, `describe`, or `synthesize`; default `chat` |
| `omni.include_audio_from_video` | No | Whether video decoding exposes its audio track; default `true` |
| `omni.require_speech` | No | When `true`, a comprehension result without a tagged speech transcript finalizes before language/TTS; default `false` |
| `response_modalities` | No | Non-empty subset of `text`, `audio`; default `text` |
| `speech_mode` | No | `auto`, `always`, or `never`; default `auto` |
| `speech` | No | Backend-specific voice, language, cloning, sampling, and style hints |

`speech_mode` has the following precedence:

1. `always` requests TTS.
2. `never` disables TTS.
3. `auto` requests TTS only when `response_modalities` contains `audio`.

`omni.task=synthesize` always executes TTS. `transcribe` and `describe` bypass
the language stage; they return comprehension text directly and may synthesize
that text when `speech_mode` requests audio.
`require_speech` is intended for VAD-driven live-call turns: environmental
evidence is still returned in the observation/final adapter metadata, but a
sound-only capture cannot consume the language or TTS stages. Leave it false
for explicit environmental-audio questions.

The reference Qwen3-TTS Base worker supports `language`, trusted server-local
`speaker_file`, request-local base64 WAV `speaker_audio`, `temperature`,
`top_k`, `top_p`, `seed`, and `max_frames`. `speaker_audio` is bounded to 10
MiB and 0.5–30 seconds and is removed after its generation. Other fields are
backend capabilities, not portable guarantees. In particular, its current
llama.cpp API has no separate natural-language style-instruction channel.
The reference adapter treats `max_frames` as a per-block ceiling: long replies
are split at natural text boundaries, PCM sequence numbering remains continuous,
and the complete final WAV concatenates every block. The adapter trace reports
the number as `tts_blocks`.

## Message media fields

Media is attached to a message alongside Ollama's string `content`:

```json
{
  "role": "user",
  "content": "What changed after the speaker entered the room?",
  "audios": [],
  "images": [],
  "videos": []
}
```

The adapter accepts media on earlier messages so history can be reconstructed,
but `transcribe`, `describe`, and `synthesize` validate their required input on
the last user message.

### Audio envelope

Audio input MUST be a complete uncompressed RIFF/WAVE containing 16 kHz, mono,
signed 16-bit PCM samples:

```json
{
  "mime_type": "audio/wav",
  "encoding": "base64",
  "data": "UklGR..."
}
```

The validator checks the RIFF/WAVE header, PCM compression mode, sample rate,
channel count, sample width, decoded size, and base64 syntax. Clients SHOULD
resample and downmix before encoding:

```bash
ffmpeg -i input.m4a -ac 1 -ar 16000 -c:a pcm_s16le speech-16khz-mono.wav
```

### Image envelope

Native Ollama bare-base64 image strings remain valid. A structured envelope is
preferred when building generic clients:

```json
{
  "mime_type": "image/png",
  "encoding": "base64",
  "data": "iVBOR..."
}
```

Supported MIME types are `image/jpeg`, `image/png`, and `image/webp`. The
declared type MUST match the decoded container signature.

### Video envelope

```json
{
  "mime_type": "video/mp4",
  "encoding": "base64",
  "data": "AAAAIGZ0eXA...",
  "sampling": {
    "fps": 1.0,
    "max_frames": 96,
    "include_audio": true
  }
}
```

Supported inputs are MP4, WebM, and animated GIF. The reference runtime
normalizes GIF to a bounded, silent MP4 before comprehension. MP4/WebM clips
with no audio stream are valid visual-only inputs; optional audio demux MUST
return no audio part rather than failing the video turn. Container support does
not imply every codec is decodable; the runtime SHOULD use FFmpeg or another
sandboxed decoder and return a media error for unsupported streams.

Sampling fields:

- `fps`: requested sample rate greater than 0 and at most 30;
- `max_frames`: hard cap from 1 through 1024;
- `include_audio`: whether to demux the audio track for temporal audio-visual
  comprehension.

The runtime MUST preserve frame order and SHOULD record source timestamps. If
it clips or subsamples due to context limits, the response adapter trace SHOULD
report the actual frame count and covered time range.

Documents are intentionally not part of adapter v1. A portal or application
MAY extract and retrieve document text before forwarding a normal text message,
but it MUST remove raw document envelopes, mark retrieved excerpts as untrusted
data, bound the inserted context, and isolate any retrieval index per user
session. This keeps the portable adapter concerned with model-native media
while allowing PDF/DOCX/text workflows in the reference portal.

## Tasks and routes

### `chat`

`chat` is the combined-agent route:

```text
audio/image/video → comprehension → semantic observation
                                            ↓
normal messages/tools/think → language model → content/tool_calls/thinking
                                            ↓
                                   optional text-conditioned TTS
```

Text-only chat skips comprehension. Speech is not generated while the response
contains unresolved `tool_calls`; the client executes the tools, appends tool
results, and submits the next turn before TTS.

The authenticated reference portal can perform that client role when the user
enables its wrench toggle (`portal_auto_tools=true`). It pins its own 20-tool
schema array, executes no more than 50 bounded rounds/calls per turn, emits running/completed
tool events on the portal NDJSON extension, and returns bounded arguments and
result evidence under `response.portal`. The toggle defaults off. This flag and
trace are portal extensions, not fields in portable adapter v1.

### `transcribe`

Requires audio on the last user message. It executes only the comprehension
graph with a transcription instruction and returns its text as
`message.content`. This route is the ASR example and avoids a second language
model paraphrasing the transcript.

### `describe`

Requires at least one audio, image, or video item on the last user message. It
executes only comprehension and returns a temporal/semantic description.

### `synthesize`

Requires text and no input media on the last user message. It bypasses language
generation and sends that text directly to the TTS graph. The same text is
returned in `message.content` and as the optional `transcript` associated with
`message.audio`.

## Response

The adapter preserves the normal Ollama response object. It adds an `audio`
field to the assistant message when TTS runs:

```json
{
  "type": "audio",
  "mime_type": "audio/wav",
  "encoding": "base64",
  "container": "wav",
  "codec": "pcm_s16le",
  "sample_rate_hz": 24000,
  "channels": 1,
  "sample_width_bits": 16,
  "frames": 24000,
  "duration_ms": 1000,
  "decoded_bytes": 48044,
  "data": "UklGR...",
  "transcript": "Text synthesized into this waveform."
}
```

The adapter also adds a compact trace:

```json
{
  "adapter": {
    "schema": "robit.ollama.omni-adapter.v1",
    "task": "chat",
    "route": ["comprehension", "language", "tts"],
    "input_modalities": ["text", "audio"],
    "evidence_provenance": {
      "current_media_modalities": ["audio"],
      "current_visual_input": false,
      "tool_data_is_visual_input": false,
      "prior_dialogue_is_current_observation": false
    },
    "speech_synthesized": true
  }
}
```

The provenance booleans are authoritative origin metadata. In particular,
audio-only requests remain `current_visual_input=false` even if a perception
backend emits visual-looking prose. Tool results and prior dialogue can inform
an answer but cannot be represented as something currently seen.

Implementations SHOULD omit internal chain-of-thought from logs and traces.
Normal Ollama `message.thinking` behavior is controlled by the request and the
runtime; the adapter does not merge it into spoken content.

## Errors

Recommended status mapping:

| Status | Condition |
|---:|---|
| 400 | Invalid schema, malformed base64, wrong media format, unsupported task, or `stream:true` |
| 413 | Deployment-specific request or decoded-media limit exceeded |
| 422 | Container is valid but its codec or stream layout cannot be normalized |
| 500 | Adapter invariant or component loader failure |
| 502 | Component execution returned an invalid response |
| 503 | Required component is unavailable or not loaded |
| 507 | Insufficient CPU/GPU memory to load the requested graph |

Errors are JSON objects containing at least `error` and `schema`. They MUST NOT
echo base64 payloads, decoded media, model prompts, or secrets.

## Security requirements

- Validate decoded size before inference and enforce aggregate request limits.
- Treat media metadata, OCR, transcripts, and captions as untrusted input.
- Isolate media decoding; disable network resolution unless explicitly needed.
- Do not accept arbitrary local file paths from remote callers.
- Bound frame count, duration, dimensions, channels, and decompression ratios.
- Redact input media and generated speech from default logs.
- Authenticate and rate-limit the adapter when it is exposed beyond loopback.
- Preserve tool schemas but do not let semantic observations create or mutate
  tool definitions.

## Versioning

The schema identifier is the compatibility boundary. Additive response fields
may be introduced within v1. Any change to media encoding, task semantics,
streaming framing, required fields, or tensor-view selection requires a new
schema identifier. Runtimes MUST reject unknown explicit schema values.
