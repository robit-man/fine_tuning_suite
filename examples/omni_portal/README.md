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
and synthesized audio. Assistant text renders a safe Markdown subset including
headings, emphasis, lists, block quotes, links, and fenced code. Sending a turn
clears the composer and attachments immediately.

The brain icon controls the real Ollama `think` request field. Gray disables
reasoning output; violet enables it. When enabled, streamed reasoning is shown
in a collapsed Reasoning disclosure beneath the answer instead of being mixed
into the visible response.

The camera icon in the upper-right control group opens the device camera and microphone with a live, muted
in-interface preview. Tap it again—or press Send—to stop and attach the bounded
recording as MP4 or WebM, including its audio track when the browser provides
one. Recordings use a constrained video bitrate to remain below the portal
upload limit. Starting Call while the camera is active pauses the attachment
recording and sends the current visual frame with each detected speech turn;
this provides bounded live visual conversation without presenting an unbounded
video stream to the model context.

The phone icon at the upper right starts hands-free voice mode. Browser-side
voice activity detection groups microphone samples into utterances after 700
ms of silence, submits each complete 16 kHz WAV through Qwen3-Omni and Qwen3.8,
streams the response text as Ollama produces it, then plays the Qwen3-TTS reply.
The microphone remains active during inference and playback. Sustained speech
stops current playback, records the interruption, and queues the new turn as
soon as the in-flight generation releases the serial inference lane. Echo
cancellation and a higher interruption threshold reduce self-triggering. The
live waveform and status line show whether the portal is listening,
understanding, preparing speech, speaking, or handling an interruption.
Tap the phone icon again to stop capture, abort an in-flight request, and stop
playback.

| Composer action | Adapter route |
|---|---|
| Text chat | Qwen3.8 language |
| Audio attachment with no prompt | Qwen3-Omni direct transcription |
| Audio attachment with a prompt | Qwen3-Omni comprehension → Qwen3.8 |
| Image/video with no prompt | Qwen3-Omni direct description |
| Image/video with a prompt | Qwen3-Omni comprehension → Qwen3.8 |
| Device camera capture | live local preview → MP4/WebM turn → video comprehension |
| Camera + phone icons | current visual frame + repeated speech turns → spoken replies |
| Speaker icon enabled | final text → streamed Qwen3-TTS PCM → replayable 24 kHz WAV |
| Brain icon enabled | Ollama `think:true` → collapsible streamed reasoning |
| Phone icon enabled | repeated/barge-in audio → comprehension → streamed Qwen3.8 text → TTS |

Microphone capture is encoded in the browser as a complete 16 kHz mono PCM16
WAV. `/api/chat/stream` is a portal extension that relays NDJSON stage events,
Ollama text/reasoning deltas, and one authoritative final response. It does not
change the portable adapter v1 contract, whose `/api/chat` route still requires
`stream:false`. Generated speech arrives as base64-tagged PCM16 deltas and is
scheduled directly into the browser's unlocked Web Audio context. The final
event also carries the complete tagged 24 kHz mono PCM16 WAV for replay and
adapter compatibility. The browser remains receptive to barge-in throughout
generation and scheduled playback. The viewport disables focus and pinch zoom
for a stable app-like mobile layout.

Video is sampled at 24 frames by the phone and clamped to at most 32 frames and
2 fps by the adapter. The comprehension GGUF declares a 65,536-token context,
and the deployment now starts `llama-server` at that native limit instead of
the former 8,192-token test setting. If a multimodal prompt still exceeds the
available context, the adapter retries only the comprehension stage with
progressively smaller frame caps (24/16/8/4/1 as applicable). A recorded video
turn therefore retains temporal comprehension whenever it fits; camera-call
mode uses one current frame per speech turn for lower latency.

## Voice configuration and cloning

The TTS weights are Qwen3-TTS 12 Hz 1.7B Base, not LuxTTS. Tap the waveform
button in the portal header to open the request-local voice panel. It provides:

- a voice-clone toggle;
- phone recording or WAV upload for a clean 3–10 second reference;
- in-browser reference playback and removal;
- language, temperature, top-p, top-k, seed, and maximum-frame controls.

The reference is sent as a bounded base64 WAV envelope. The portal validates
its container, duration (0.5–30 seconds), and 10 MiB decoded limit, then the TTS
worker writes it into a per-generation temporary directory for `llama-tts`.
The file is deleted as soon as that generation finishes. The browser cannot
select a server path. Only clone a voice you own or have permission to use.

Edit
[`voice-profile.json`](voice-profile.json) before startup to pin the server-side
defaults or a trusted server-local reference used by manual spoken replies and
call mode:

```json
{
  "schema": "robit.omni.voice-profile.v1",
  "name": "studio-voice",
  "language": "en",
  "speaker_file": "voices/studio-reference.wav",
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.9,
  "seed": 42,
  "max_frames": 512
}
```

The portable contract is
[`voice-profile-v1.schema.json`](../../docs/omni-adapter/schema/voice-profile-v1.schema.json).

`speaker_file` may be an absolute path or a path relative to the profile.
Qwen3-TTS accepts WAV or MP3 server references; the browser path deliberately
accepts WAV only. Use a clean, single-speaker clip without music or
reverberation. A fixed non-negative `seed` makes repeated turns reproducible.
`seed: -1` deliberately restores randomized voices and prosody. Lower
temperature/top-p/top-k generally improves consistency; higher values add
variation and can reintroduce timbre drift. Supported language codes are `zh`,
`en`, `de`, `it`, `pt`, `es`, `ja`, `ko`, `fr`, and `ru`.

The current llama.cpp interface for the Base checkpoint extracts a speaker
embedding from reference audio (`--tts-speaker-file`). It does not yet expose
the official Python stack's higher-fidelity `ref_audio + ref_text` in-context
clone path. It also has no separate natural-language style-instruction input.
Do not prepend style directions to spoken text: the model may read them aloud.
The official [Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS)
documents VoiceDesign and CustomVoice capabilities, but those are separate
checkpoint/runtime paths; controls for them are not presented as if they were
present in this Base GGUF. Use a dedicated reference clip for the intended
timbre, accent, and delivery today.

Select another profile without editing the default:

```bash
OMNI_VOICE_PROFILE=/srv/voices/production.json \
  examples/omni_portal/start.sh --daemon
```

The portal validates the profile at startup, rejects direct client `speech`
paths, and applies only the bounded `portal_voice` controls from its own UI.
`/api/status` reports safe defaults and whether a server reference is active
without exposing its filesystem path.

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
| `OMNI_LANGUAGE_MODEL` | E03 core `27b` tag | Equivalent base/projector tag used to avoid loading the custom sidecar twice |
| `OMNI_PORTAL_RUNTIME_ROOT` | `training_suite/outputs/omni_portal_runtime` | State/cache/log root |
| `OMNI_COMPREHENSION_GPU_UUID` | broker-selected | Explicit approved GPU override |
| `OMNI_COMPREHENSION_VRAM_MIB` | `45000` | Shared comprehension/TTS scoped reservation |
| `OMNI_COMPREHENSION_CONTEXT_TOKENS` | `65536` | Native comprehension worker context; propagated to the adapter |
| `OMNI_PORTAL_TOKEN` | generated | At least 24 characters |
| `OMNI_VOICE_PROFILE` | `examples/omni_portal/voice-profile.json` | Validated server-side Qwen3-TTS profile |
| `OMNI_TTS_STREAM_FRAMES` | `12` | Codec frames per live PCM decode window; 12 is about one second of audio |
| `OMNI_TTS_BROKER_TRANSITION_TIMEOUT_S` | `330` | Maximum wait for scoped prepare/ready transitions |
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
allow-listed tool completion, the direct ASR-to-TTS path when `--audio` and
`--tts` are combined, and the 24 kHz mono PCM16 TTS contract. It never prints
media base64 or the access token.

## Security boundary

- Only the portal is tunneled; Ollama and all workers remain on loopback.
- Every inference/status API requires a constant-time bearer-token match.
- The model tag is fixed server-side. Portable `/api/chat` rejects streaming;
  authenticated `/api/chat/stream` relays the bounded portal NDJSON extension.
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
