# Omni Adapter Examples

These examples implement the public `robit.ollama.omni-adapter.v1` request and
response format intended for the combined Qwen3.8/Ornith Omni model.

- `server.py` is a readable HTTP sidecar that routes comprehension, Ollama
  language generation, and TTS while the in-process custom Ollama runner is
  being developed.
- `client.py` sends ASR, direct TTS, video-description, or combined chat
  requests and writes returned speech to WAV.
- `javascript_client.mjs` demonstrates an audio-in/audio-out chat with Node.js
  built-ins.

The sidecar and final custom runner use the same API contract. The sidecar uses
separate HTTP component servers and therefore does not prove that a monolithic
GGUF executes inside stock Ollama.

## Install

From the repository root:

```bash
python3 training_suite/app.py bootstrap
```

Or create a small example environment containing Flask and HTTPX. The examples
must be launched from the repository root so the `training_suite` package is on
the Python import path.

## Configure the reference server

```bash
export OMNI_COMPREHENSION_URL=http://127.0.0.1:8901/v1/chat/completions
export OMNI_COMPREHENSION_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
export OMNI_LANGUAGE_URL=http://127.0.0.1:11434
export OMNI_TTS_URL=http://127.0.0.1:8091/synthesize
export OMNI_ADAPTER_PORT=11435
```

The comprehension endpoint receives OpenAI-style multimodal content parts:

- `audio_url.url`: base64 WAV data URI;
- `image_url.url`: base64 JPEG/PNG/WebP data URI;
- `video_url.url`: base64 MP4/WebM data URI plus optional `sampling`;
- `mm_processor_kwargs.use_audio_in_video`: requested video-audio policy.

Qwen3-Omni's official vLLM Serve example documents image and audio URL parts,
while its native Transformers/vLLM processor path documents video input and
`use_audio_in_video`. A selected server may require a small translation at
`build_comprehension_payload()` for video. Do not assume every generic
OpenAI-compatible server accepts `video_url` or processor kwargs.

The TTS endpoint receives:

```json
{
  "text": "Text to synthesize",
  "output": {
    "mime_type": "audio/wav",
    "container": "wav",
    "codec": "pcm_s16le",
    "sample_rate_hz": 24000,
    "channels": 1,
    "sample_width_bits": 16
  },
  "voice": "optional voice identifier"
}
```

It must return raw `audio/wav` bytes or the JSON audio envelope documented in
the [wire protocol](../../docs/omni-adapter/protocol.md).

Before starting any CUDA component service on the managed host, run
`docker gpu discover` and use the scoped reservation protocol in
`/usr/local/share/ollama-unify/AGENTS.md`.

Start the adapter:

```bash
training_suite/.venv/bin/python examples/omni_adapter/server.py
curl -s http://127.0.0.1:11435/healthz
```

## Normalize audio

Adapter v1 accepts 16 kHz mono PCM16 WAV:

```bash
ffmpeg -i recording.m4a -ac 1 -ar 16000 -c:a pcm_s16le speech.wav
```

## Python examples

### ASR without language-model paraphrasing

```bash
training_suite/.venv/bin/python examples/omni_adapter/client.py \
  --model robit/qwen3.8-omni:latest \
  asr ./speech.wav
```

This selects `omni.task=transcribe` and executes only comprehension.

### Direct TTS

```bash
training_suite/.venv/bin/python examples/omni_adapter/client.py \
  --model robit/qwen3.8-omni:latest \
  --voice speaker-1 \
  --output-audio ./direct.wav \
  tts "Read this sentence exactly as written."
```

This selects `omni.task=synthesize`, bypasses the language model, and writes the
returned 24 kHz PCM16 WAV.

### Video comprehension

```bash
training_suite/.venv/bin/python examples/omni_adapter/client.py \
  --model robit/qwen3.8-omni:latest \
  video ./events.mp4 \
  --fps 2 \
  --max-frames 96 \
  --include-audio \
  --prompt "Describe the events and quote any spoken question."
```

Use `--no-include-audio` to test vision-only temporal comprehension.

### Combined audio question, tools/thinking-capable language, and TTS

```bash
training_suite/.venv/bin/python examples/omni_adapter/client.py \
  --model robit/qwen3.8-omni:latest \
  --output-audio ./answer.wav \
  chat \
  --audio ./question.wav \
  --prompt "Answer the recorded question concisely." \
  --speak
```

The client defaults to `think=true`. The adapter preserves normal Ollama tool
definitions if a calling application includes them. When a response contains
unresolved `tool_calls`, TTS is deferred until the client sends tool results and
receives final assistant text.

### Mixed image, video, and audio

```bash
training_suite/.venv/bin/python examples/omni_adapter/client.py \
  --model robit/qwen3.8-omni:latest \
  chat \
  --image ./reference.png \
  --video ./events.mp4 \
  --audio ./question.wav \
  --prompt "Relate the recording and reference image to the video timeline."
```

## JavaScript example

Node.js 18 or newer provides `fetch`:

```bash
OMNI_MODEL=robit/qwen3.8-omni:latest \
OMNI_ADAPTER_URL=http://127.0.0.1:11435/api/chat \
node examples/omni_adapter/javascript_client.mjs ./question.wav ./answer.wav
```

## Curl example

Create request JSON without putting binary data on the shell command line:

```bash
python3 - <<'PY' > /tmp/omni-request.json
import base64
import json
from pathlib import Path

audio = base64.b64encode(Path("speech.wav").read_bytes()).decode("ascii")
print(json.dumps({
    "model": "robit/qwen3.8-omni:latest",
    "messages": [{
        "role": "user",
        "content": "Transcribe this recording.",
        "audios": [{
            "mime_type": "audio/wav",
            "encoding": "base64",
            "data": audio,
        }],
    }],
    "omni": {
        "schema": "robit.ollama.omni-adapter.v1",
        "task": "transcribe",
    },
    "response_modalities": ["text"],
    "speech_mode": "never",
    "stream": False,
}))
PY

curl -s http://127.0.0.1:11435/api/chat \
  -H 'content-type: application/json' \
  --data-binary @/tmp/omni-request.json
```

The `/tmp` request contains the media and should be deleted when the test is
complete.

## Validate without component servers

The Fine-Tuning Suite exposes the same parser without running inference:

```bash
training_suite/.venv/bin/python -m training_suite web --port 7860

curl -s http://127.0.0.1:7860/api/omni/adapter/validate \
  -H 'content-type: application/json' \
  --data-binary @/tmp/omni-request.json
```

The response shows input modalities, normalized media metadata, and the selected
route but never returns the base64 payload.

## Adapting a component server

If a backend expects another multimodal syntax, change only
`build_comprehension_payload()` in `server.py`. Keep the public adapter request
stable. Common translations include:

- `audio_url` to `input_audio`;
- `video_url` to a server-side uploaded object identifier;
- video to sampled `image_url` parts plus a separately demuxed `audio_url`;
- backend-specific frame-rate or `use_audio_in_video` processor options.

The final in-process runner replaces these HTTP translations with direct tensor
views from the one GGUF, but it must produce the same response shape.

## Limits

- Adapter v1 is turn-based and requires `stream:false`.
- The example server is for trusted local development; add authentication,
  request limits, decoder isolation, and production WSGI serving before remote
  exposure.
- Base64 increases request size. Production proxies must configure body limits
  above the chosen decoded-media limits while still enforcing bounded input.
- The example does not download or convert any model weights.
- Stock Ollama does not understand `audios`, `videos`, `omni`, or
  `message.audio`; use the sidecar or the documented custom build.
