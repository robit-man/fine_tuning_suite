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

## Interface and routing

The phone UI is deliberately a single chat surface. Press and hold the
microphone icon to record; the live waveform disappears on release and the
resulting 16 kHz WAV appears as a playable attachment before it is sent. The
paperclip accepts WAV, JPEG, PNG, WebP, MP4, and WebM. The speaker icon is the
only output toggle: gray requests text only, while yellow requests both text
and synthesized audio.

| Composer action | Adapter route |
|---|---|
| Text chat | Qwen3.8 language |
| Audio attachment with no prompt | Qwen3-Omni direct transcription |
| Audio attachment with a prompt | Qwen3-Omni comprehension → Qwen3.8 |
| Image/video with no prompt | Qwen3-Omni direct description |
| Image/video with a prompt | Qwen3-Omni comprehension → Qwen3.8 |
| Speaker icon enabled | final text → Qwen3-TTS → 24 kHz WAV |

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
4. acquires one scoped 45 GiB lease and starts comprehension on that exact UUID;
5. starts broker-coordinated CUDA TTS, the unified adapter, and portal;
6. runs local status, exact-text, and GPU TTS smoke gates;
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

## CUDA-only media policy

The phone deployment has no CPU inference fallback. Persistent comprehension
uses `-ngl 99` on the exact GPU UUID assigned by a manual scoped broker lease.
TTS uses `--gpu-layers -1` on the same UUID. Because the current `llama-tts`
binary is single-shot, the wrapper calls broker `prepare` before every load,
waits until the TTS PID is visible as resident on that UUID, and then calls
`ready`. A failed reservation or residency check aborts deployment or the
request instead of silently running inference on CPU. Ollama language requests
continue through broker-owned GPU lanes.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `OMNI_MODEL` | release `q4km` tag | Pinned portal model |
| `OMNI_PORTAL_RUNTIME_ROOT` | `training_suite/outputs/omni_portal_runtime` | State/cache/log root |
| `OMNI_COMPREHENSION_GPU_UUID` | broker-selected | Explicit approved GPU override |
| `OMNI_COMPREHENSION_VRAM_MIB` | `45000` | Shared comprehension/TTS scoped reservation |
| `OMNI_PORTAL_TOKEN` | generated | At least 24 characters |
| `OMNI_KEEP_CACHE` | `0` | Keep materialized views after stop |
| `OMNI_PORTAL_MAX_BODY_BYTES` | 96 MiB | Same-origin JSON request cap |

Ports `8901`, `8892`, `8910`, and `8920` are loopback-only. The Cloudflare
metrics endpoint defaults to loopback port `49312`.

## Full smoke test

The startup gate performs health checks, a text sentinel, and a GPU TTS
generation before publishing ingress. Run every media route against a live
deployment with:

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
- Media inference has no CPU fallback; CUDA residency is verified before the
  scoped lease is marked ready.
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
