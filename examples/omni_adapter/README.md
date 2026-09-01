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
  --jinja -ngl 99 -c 8192
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
| `temperature` | Semantic-token sampling temperature, default `0.7` |
| `top_k` | Semantic-token top-k, default `40` |
| `top_p` | Semantic-token nucleus threshold, default `0.9` |
| `seed` | Fixed integer for reproducibility; default `42`, `-1` is random |
| `max_frames` | Maximum generated codec frames, capped by the server |

The installed llama.cpp Qwen3-TTS Base path exposes audio-reference cloning but
does not expose a distinct natural-language style instruction. The generic
adapter schema retains `voice` and `style` for other backends; this reference
worker does not claim to honor them. For a stable, specified voice, use a clean
speaker reference plus a fixed seed and conservative sampling.

## 4. Start the unified adapter

```bash
export OMNI_COMPREHENSION_URL=http://127.0.0.1:8901/v1/chat/completions
export OMNI_COMPREHENSION_MODEL=local-qwen3-omni
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
  video ./events.mp4 --fps 2 --max-frames 96 --include-audio

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
adapter preserves `message.thinking` and structured `tool_calls`. If unresolved
tool calls are present, it does not synthesize their JSON; speech can resume
after the client submits tool results and receives final assistant text.

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
