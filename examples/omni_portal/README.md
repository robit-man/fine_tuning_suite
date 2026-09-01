# Robit Omni Phone Portal

This example is a phone-first web client for
`robit/qwen3.8-27b-e03-obliterated-omni:q4km`. It exposes the existing Omni
adapter through one authenticated HTTPS origin and keeps Ollama,
Qwen3-Omni comprehension, and Qwen3-TTS bound to loopback.

The interface borrows the visual language of the
[NOCLIP documentation](https://noclip.org/docs): black grid background,
compact monospace type, yellow accents, thin borders, fixed runtime status,
and a responsive control rail. It does not copy NOCLIP assets or documentation
content.

## Features

| Portal control | Adapter route |
|---|---|
| Text chat | Qwen3.8 language |
| Microphone or WAV chat | Qwen3-Omni comprehension → Qwen3.8 |
| Transcribe | Qwen3-Omni comprehension only |
| Camera/image describe | Qwen3-Omni comprehension only |
| MP4/WebM describe | ordered video frames plus optional demuxed audio |
| Speak | Qwen3-TTS direct synthesis |
| Speak output | final Qwen3.8 text → Qwen3-TTS |
| Thinking | preserved `message.thinking`, shown in a collapsed panel |
| Safe tools | allow-listed tool execution followed by a second model turn |

Microphone capture is encoded in the browser as a complete 16 kHz mono PCM16
WAV. Generated speech is returned as tagged base64 24 kHz mono PCM16 WAV and
rendered with native phone playback controls. This is turn-based media upload,
not realtime audio streaming; adapter v1 requires `stream:false`.

## One-command deployment

From the repository root:

```bash
examples/omni_portal/start.sh --daemon
```

The command:

1. verifies the installed Ollama tag and sidecar;
2. reconstructs the four disposable media-runtime views when missing;
3. runs `docker gpu discover` and selects an unclaimed broker-approved GPU;
4. starts comprehension under a scoped `docker gpu run` lease, or in automatic
   mode falls back to CPU isolation if another broker transition blocks CUDA;
5. starts CPU-only TTS, the unified adapter, and authenticated portal;
6. runs a local status and exact-text smoke gate;
7. starts a Cloudflare Quick Tunnel and prints an HTTPS URL containing the
   access token in its URL fragment.

The URL has this form:

```text
https://random-words.trycloudflare.com/#access=HIGH_ENTROPY_TOKEN
```

Fragments are not sent in HTTP requests or referrer headers. The browser keeps
the token in session storage and sends it as an `Authorization: Bearer` header
only to same-origin `/api/*` routes. Do not publish the complete access URL.

Manage the deployment:

```bash
examples/omni_portal/start.sh --status
examples/omni_portal/start.sh --stop
```

Runtime state and logs default to
`training_suite/outputs/omni_portal_runtime`. The supervisor owns all child
PIDs. On stop it drains the tunnel, portal, adapter, TTS wrapper, and
comprehension worker in that order. It then removes only a component cache
carrying its ownership marker. Set `OMNI_KEEP_CACHE=1` to keep the reconstructed
views for a near-term restart.

## Why TTS defaults to CPU

The current `llama-tts` reference executable is single-shot and allocates its
model on each request. A broker-managed service may not claim stable GPU
readiness and then grow VRAM without `prepare`. The bootstrap therefore sets
`CUDA_VISIBLE_DEVICES` empty and `--gpu-layers 0` for TTS. The persistent
comprehension graph prefers a scoped 30 GiB CUDA reservation. In `auto` mode it
uses `CUDA_VISIBLE_DEVICES` empty and `-ngl 0` if the broker refuses a new
transition; Ollama remains on broker-owned lanes in either case.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OMNI_MODEL` | release `q4km` tag | Pinned portal model |
| `OMNI_PORTAL_RUNTIME_ROOT` | `training_suite/outputs/omni_portal_runtime` | State/cache/log root |
| `OMNI_COMPREHENSION_GPU_UUID` | broker-selected | Explicit approved GPU override |
| `OMNI_COMPREHENSION_VRAM_MIB` | `30000` | Scoped reservation |
| `OMNI_COMPREHENSION_MODE` | `auto` | `auto`, strict `cuda`, or CPU-only `cpu` |
| `OMNI_PORTAL_TOKEN` | generated | At least 24 characters |
| `OMNI_KEEP_CACHE` | `0` | Keep materialized views after stop |
| `OMNI_PORTAL_MAX_BODY_BYTES` | 96 MiB | Same-origin JSON request cap |

Ports `8901`, `8892`, `8910`, and `8920` are loopback-only. The Cloudflare
metrics endpoint defaults to loopback port `49312`.

## Full smoke test

The startup gate deliberately performs only health checks and a text sentinel
so publication is not delayed by several serial media generations. Run every
route against a live deployment with:

```bash
TOKEN_FILE=training_suite/outputs/omni_portal_runtime/state/access-token.txt

.venv-omni/bin/python examples/omni_portal/smoke.py \
  --endpoint http://127.0.0.1:8920 \
  --token-file "$TOKEN_FILE" \
  --model robit/qwen3.8-27b-e03-obliterated-omni:q4km \
  --text --tool --tts \
  --audio ./speech-16khz-mono.wav \
  --image ./image.png \
  --video ./video.mp4
```

The smoke runner verifies non-empty media responses, exact text sentinel,
allow-listed tool completion, and the 24 kHz mono PCM16 TTS contract. It never
prints media base64 or the access token.

## Security boundary

- Only the portal is tunneled; Ollama and all workers remain on loopback.
- Every inference/status API requires a constant-time bearer-token match.
- The model tag is fixed server-side and streaming requests are rejected.
- Requests are limited to one in-flight inference and 96 MiB encoded JSON.
- The browser caps decoded image, video, and audio sizes below adapter limits.
- CSP, frame denial, no-referrer, no-store, and same-origin camera/microphone
  policies are applied.
- Tool execution is limited to `get_current_time` and
  `get_portal_capabilities`; unknown names return an error result and cannot
  execute programs, access files, or make network requests.
- Media observations remain untrusted evidence at the adapter boundary.

Cloudflare Quick Tunnels are temporary development endpoints, not durable
production ingress. Stop the portal after the phone test. For a persistent
deployment, replace the quick tunnel with a named tunnel plus Cloudflare
Access while retaining the portal bearer token.
