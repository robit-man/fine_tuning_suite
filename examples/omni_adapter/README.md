# Omni Adapter Examples

These examples expose one Ollama-shaped endpoint for a logical model tag while
routing audio/video/TTS through component views stored in that tag's custom
GGUF sidecar layer.

Files:

- `server.py` — request parser and comprehension→Ollama→TTS router;
- `tts_server.py` — serial reference wrapper for llama.cpp `llama-tts`;
- `client.py` — Python CLI for ASR, video, direct TTS, and combined chat;
- `javascript_client.mjs` — dependency-free Node audio-chat example.

The example is a conformance/development runtime. It is not a claim that stock
Ollama itself accepts audio/video fields or produces waveform bytes.

## 1. Pull and prepare the model

```bash
ollama pull robit/qwen3.8-27b-e03-obliterated-omni:q4km

MODEL=robit/qwen3.8-27b-e03-obliterated-omni:q4km
CACHE=/srv/omni-runtime/qwen38-q4km

python -m training_suite omni-resolve "$MODEL"
python -m training_suite omni-prepare "$MODEL" --out "$CACHE"
```

The cache is disposable. It can always be reconstructed from the installed
sidecar and should be removed after every worker using it stops.

## 2. Start Qwen3-Omni comprehension

Build the pinned llama.cpp tools and start `llama-server` with the extracted
pair:

```bash
./training_suite/vendor/llama.cpp/build/bin/llama-server \
  -m "$CACHE/comprehension-model.gguf" \
  --mmproj "$CACHE/comprehension-projector.gguf" \
  --host 127.0.0.1 --port 8901 \
  --jinja -ngl 99 -c 65536
```

On a broker-managed CUDA host, do not run that command anonymously. First run
`docker gpu discover`, then wrap it with the scoped `docker gpu run` protocol
from `/usr/local/share/ollama-unify/AGENTS.md`.

The current llama.cpp request translations are:

- audio → `{"type":"input_audio","input_audio":{"data":"<raw base64>"}}`;
- image → `image_url` data URI;
- video → `{"type":"input_video","input_video":{"data":"<raw base64>"}}`.

For video with sound, `server.py` also demuxes the first audio stream using
ffmpeg and submits 16 kHz mono PCM16 WAV as a separate audio part.
The comprehension GGUF declares a 65,536-token context. The reference runtime
clamps sampling to 32 frames and 2 fps, then retries a context-overflow response
with progressively smaller frame caps. `OMNI_COMPREHENSION_CONTEXT_TOKENS` and
`OMNI_COMPREHENSION_MAX_OUTPUT_TOKENS` tell the adapter the deployed limits.

## 3. Start TTS

The wrapper can resolve the installed model itself:

```bash
export OMNI_OLLAMA_MODEL="$MODEL"
export OMNI_COMPONENT_CACHE="$CACHE"
export LLAMA_TTS_BIN=./training_suite/vendor/llama.cpp/build/bin/llama-tts
export OMNI_TTS_PORT=8892
python examples/omni_adapter/tts_server.py
```

The wrapper is serial and reloads the model on each request. In a manually
scoped CUDA deployment, pass `OLLAMA_UNIFY_GPU_LEASE`, `OMNI_TTS_GPU_UUID`, and
`CUDA_VISIBLE_DEVICES` with the exact reserved UUID. The wrapper then calls
broker `prepare`, verifies that `llama-tts` is resident on that GPU, and calls
`ready` for each generation. `OMNI_TTS_GPU_LAYERS=-1` enables full offload;
`0` remains available only for explicit CPU development tests.
`OMNI_TTS_BROKER_TRANSITION_TIMEOUT_S` defaults to 330 seconds so an unrelated
in-flight Ollama request delays synthesis rather than forcing an anonymous CUDA
fallback. A timed-out transition attempts to restore the stable `ready` state.

### Qwen3-TTS voice controls

The packaged TTS component is `Qwen3-TTS-12Hz-1.7B-Base-GGUF`. The wrapper
accepts these `speech`/`POST /synthesize` fields:

| Field | Meaning |
|---|---|
| `language` | Qwen3-TTS language code; portal default `en` |
| `speaker_file` | Server-local WAV/MP3 reference for voice cloning |
| `speaker_audio` | Base64 WAV envelope for a request-local speaker clone (0.5–30 seconds, 10 MiB maximum) |
| `temperature` | Semantic-token sampling temperature, default `0.7` |
| `top_k` | Semantic-token top-k, default `40` |
| `top_p` | Semantic-token nucleus threshold, default `0.9` |
| `seed` | Fixed integer for reproducibility; default `42`, `-1` is random |
| `max_frames` | Maximum generated codec frames, capped by the server |

The installed llama.cpp Qwen3-TTS Base path exposes speaker-embedding cloning
but not the official Python runtime's `ref_audio + ref_text` in-context clone
path or a distinct natural-language style instruction. `speaker_audio` is
materialized in a private per-generation temporary directory and removed after
generation. The generic adapter schema retains `voice` and `style` for other
backends; this reference worker does not claim to honor them. The separate
VoiceDesign and CustomVoice checkpoints described in the
[official Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS) are not
silently emulated by this Base GGUF.

### Experimental PCM stream

The vendored private-fork `llama-tts` can expose each completed, stateful
code2wav decoder window before generation ends:

```bash
./training_suite/prepare_llama_cpp.sh

./training_suite/vendor/llama.cpp/build/bin/llama-tts \
  -m tts-model.gguf --mmproj tts-projector.gguf \
  --prompt "Read this sentence." --output final.wav \
  --tts-stream --tts-stream-frames 72 > speech.s16le
```

The tracked bootstrap checks out the verified llama.cpp commit, applies
`training_suite/patches/llama.cpp-qwen3tts-pcm-stream.patch` idempotently, and
builds CUDA-enabled `llama-server` and `llama-tts`. Set
`LLAMA_CPP_BUILD_JOBS` or `LLAMA_CPP_BUILD_DIR` when needed; an existing source
checkout must already be at the pinned commit.

In this mode stdout contains only headerless 24 kHz, mono, signed PCM16
little-endian bytes; llama.cpp diagnostics remain on stderr. `final.wav` is
still written and validated normally. The worker exposes the same stream over
`POST /synthesize/stream` with `Content-Type:
audio/pcm;rate=24000;channels=1;format=s16le` plus `X-Audio-Codec`,
`X-Audio-Sample-Rate`, `X-Audio-Channels`, `X-Audio-Stream-Version`, and
`X-Audio-Stream-Frames` headers. The JSON body is the same as `/synthesize`
and may override `stream_frames`; `OMNI_TTS_STREAM_FRAMES` sets the server
default.

The interactive default is four codec frames, about 320 ms for the packaged
12 Hz model. Values from 1 through 72 are accepted; larger windows improve
aggregate decoder throughput, while smaller windows reduce time to first PCM
at the cost of more decoder invocations. A shorter utterance flushes once at
end of speech. These are real state-carrying code2wav calls, not chunks cut
from a finished WAV. HTTP byte-chunk boundaries are not semantic decoder
boundaries, so clients must buffer incomplete 16-bit samples.

This route is experimental. If generation fails after response headers, the
PCM stream terminates early; the final WAV is still checked server-side when
generation succeeds. The reference worker also reloads the model per request,
so its startup latency remains unsuitable for hard realtime or full duplex.
A resident libmtmd worker can retain the same PCM contract.

## 4. Start the unified adapter

```bash
export OMNI_COMPREHENSION_URL=http://127.0.0.1:8901/v1/chat/completions
export OMNI_COMPREHENSION_MODEL=local-qwen3-omni
export OMNI_COMPREHENSION_CONTEXT_TOKENS=65536
export OMNI_LANGUAGE_URL=http://127.0.0.1:11434
export OMNI_LANGUAGE_MODEL=robit/qwen3.8-27b-obliterated-e03:27b
export OMNI_TTS_URL=http://127.0.0.1:8892/synthesize
export OMNI_ADAPTER_PORT=8910
python examples/omni_adapter/server.py
```

`OMNI_LANGUAGE_MODEL` is optional. It is appropriate when the combined tag and
core tag reference identical normal model/projector blobs: requests and
responses remain pinned to the logical combined tag, while Ollama loads the
core tag without double-counting the custom sidecar for GPU placement.

Health and contract:

```bash
curl -fsS http://127.0.0.1:8910/healthz
curl -fsS http://127.0.0.1:8910/api/omni/adapter/contract
```

### Experimental streamed response route

The portable `robit.ollama.omni-adapter.v1` route remains `POST /api/chat` with
`"stream": false`. This reference server additionally exposes
`POST /api/chat/stream` for the phone harness. Send the same request with
`"stream": true`; the response is `application/x-ndjson` with these events:

| Event | Payload | Timing |
|---|---|---|
| `stage` | `stage: comprehension\|language\|tts` | Before a blocking stage |
| `observation` | untrusted semantic media evidence | After comprehension |
| `delta` | Ollama `message.content` and/or `message.thinking` delta | During language generation |
| `final` | complete normal adapter response | After optional TTS |
| `error` | safe error text and schema | If a stage fails after headers |

The final response is authoritative. `adapter.text_streamed` and
`adapter.audio_streamed` describe the actual transport. TTS-capable streamed
turns add `audio_start`, base64 `audio_delta` PCM16 chunks, and `audio_end`
before the final event. The final message still contains a complete WAV for
replay and compatibility. Clients must not interpret NDJSON or HTTP transport
chunk boundaries as codec-frame boundaries.

For media `chat`, comprehension is perception-only: it cannot answer the user.
Its output uses `<speech_transcript>`, `<audio_observation>`, and
`<visual_observation>` evidence tags. `audio_observation` carries non-speech
events, ambience, music, speaker activity, temporal changes, and uncertainty;
it does not duplicate the transcript.
Conversational message text is withheld from the comprehension graph and is
sent only to the language model; the media graph receives a modality-specific
extraction instruction instead.
The `observation` event additionally exposes `transcript` only when a tagged
speech transcript was produced; the authoritative final response mirrors that
value as `adapter.input_transcript`. Clients must never display the raw semantic
observation as a user-authored chat message.
The event and final adapter trace similarly expose tagged acoustic evidence as
`audio_observation` and `adapter.audio_observation`. Audio-only `transcribe`
remains the fast ASR route; attach audio with a text question to run combined
speech and environmental analysis through the language model.

## Python client

Global options must precede the subcommand.

```bash
# ASR
python examples/omni_adapter/client.py \
  --endpoint http://127.0.0.1:8910/api/chat \
  --model "$MODEL" \
  asr ./speech-16khz-mono.wav

# Video comprehension, including its audio track
python examples/omni_adapter/client.py \
  --endpoint http://127.0.0.1:8910/api/chat \
  --model "$MODEL" \
  video ./events.mp4 --fps 1 --max-frames 32 --include-audio

# Direct TTS
python examples/omni_adapter/client.py \
  --endpoint http://127.0.0.1:8910/api/chat \
  --model "$MODEL" \
  --output-audio ./speech.wav \
  tts "Read this sentence."

# Media → Qwen3.8 reasoning → speech
python examples/omni_adapter/client.py \
  --endpoint http://127.0.0.1:8910/api/chat \
  --model "$MODEL" \
  --output-audio ./answer.wav \
  chat --audio ./question.wav --speak \
  --prompt "Answer the recorded question."
```

The client writes decoded audio to `--output-audio` and redacts the large
base64 value from normal stdout. Use `--print-audio-base64` only when the raw
JSON transport value is specifically needed.

## Request shape

```json
{
  "model": "robit/qwen3.8-27b-e03-obliterated-omni:q4km",
  "messages": [{
    "role": "user",
    "content": "What did I say?",
    "audios": [{
      "mime_type": "audio/wav",
      "encoding": "base64",
      "data": "<16 kHz mono PCM16 WAV>"
    }]
  }],
  "omni": {
    "schema": "robit.ollama.omni-adapter.v1",
    "task": "chat"
  },
  "response_modalities": ["text", "audio"],
  "speech_mode": "always",
  "speech": {
    "language": "en",
    "speaker_file": "/srv/voices/reference.wav",
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "seed": 42
  },
  "think": true,
  "stream": false
}
```

Image envelopes use `message.images`; video envelopes use `message.videos`
with `mime_type`, `encoding`, `data`, and optional `sampling` containing `fps`,
`max_frames`, and `include_audio`.

Output speech appears at `message.audio`:

```json
{
  "type": "audio",
  "mime_type": "audio/wav",
  "encoding": "base64",
  "sample_rate_hz": 24000,
  "channels": 1,
  "sample_width_bits": 16,
  "data": "<base64 RIFF/WAVE>"
}
```

## Tools and thinking

For `task=chat`, normal `tools` and `think` are forwarded to stock Ollama. The
adapter does not add a reasoning system instruction or `/no_think` prompt
suffix: `think:false` and `think:true` remain native Ollama booleans. It
preserves native `message.thinking`, separates malformed fallback
`<think>...</think>` blocks even when tags cross stream chunks, and removes any
such reasoning from visible/TTS text when `think:false`. This sanitation is a
fail-closed response boundary, not a substitute for native mode selection.
Structured `tool_calls` remain unchanged. If unresolved tool calls are present,
it does not synthesize their JSON; speech can resume after the client submits
tool results and receives final assistant text.

Direct `transcribe`/`describe` routes return comprehension output without a
second Qwen3.8 pass. Direct `synthesize` returns the input text plus audio.

## JavaScript

```bash
OMNI_ADAPTER_URL=http://127.0.0.1:8910/api/chat \
OMNI_MODEL="$MODEL" \
node examples/omni_adapter/javascript_client.mjs \
  ./speech-16khz-mono.wav ./answer.wav
```

## Shutdown

Stop the adapter, TTS wrapper, and comprehension worker; verify broker leases
are released; then delete only the explicit cache path. Do not manually delete
the installed Ollama blob or manifest. Use `ollama rm` only when intentionally
uninstalling the model.
