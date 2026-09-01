# Phone Portal Deployment Runbook

The Robit Omni Phone Portal provides a temporary HTTPS endpoint for testing the
published model from iOS or Android. Its source and complete usage guide live
in [`examples/omni_portal`](../../examples/omni_portal/README.md).

## Deployment topology

```text
phone browser
  │ HTTPS + fragment-delivered bearer token
  ▼
Cloudflare Quick Tunnel
  │ loopback origin
  ▼
Omni phone portal :8920
  │ authenticated, fixed-model proxy
  ▼
Omni adapter :8910
  ├── Qwen3-Omni comprehension :8901  (broker-scoped CUDA, persistent)
  ├── Ollama language :11434           (broker-owned lanes)
  └── Qwen3-TTS :8892                  (broker-coordinated CUDA, single-shot)
```

The portal remains externally pinned to the combined Omni tag. Its language
stage uses the equivalent E03 core Ollama tag through `OMNI_LANGUAGE_MODEL` so
the broker sizes only the base/projector layers; startup verifies that both tags
reference the same standard blobs. This avoids counting the combined tag's custom 38.8 GB
media sidecar a second time after those media views are already loaded by the
scoped workers.

No component port listens on a public interface. The public URL is a temporary
capability URL; possession of its fragment grants portal access for that
session.

## Readiness gate

Before the tunnel is published, the supervisor requires:

- valid sidecar resolution for the requested Ollama tag;
- four complete materialized component views;
- successful CUDA broker discovery, scoped lease acquisition, and verified
  comprehension residency on the assigned UUID;
- HTTP 2xx health from comprehension, TTS, adapter, portal, and Ollama;
- exact `PORTAL TEXT OK` language sentinel and valid GPU-generated TTS WAV
  through the authenticated portal.

Any failed gate terminates the children, releases the scoped lease, and removes
only the portal-owned cache. The tunnel is never started after a failed local
gate.

## Start, inspect, and stop

```bash
examples/omni_portal/start.sh --daemon
examples/omni_portal/start.sh --status
tail -f training_suite/outputs/omni_portal_runtime/logs/supervisor.log
examples/omni_portal/start.sh --stop
```

The printed URL must be opened as-is so its `#access=...` fragment reaches the
browser. Microphone permission requires the HTTPS endpoint. Hold the microphone
icon while speaking; release it to create a playable WAV attachment, then send.
The speaker icon switches between text-only and text-plus-TTS replies. The
waveform button opens Qwen3-TTS voice cloning and sampling controls. The phone
icon starts automatic voice turns: local voice activity detection waits for a
silence boundary, submits the recording, streams language deltas, and schedules
live 24 kHz PCM synthesis chunks. The microphone remains active for barge-in;
sustained speech cancels playback and queues the interruption. Portable adapter
v1 remains `stream:false`; `/api/chat/stream` is an authenticated portal
extension with one authoritative final adapter response.

The camera icon captures device video and microphone audio with a live preview
inside the composer. Tap again or send to finalize a 30-second-maximum MP4/WebM
attachment. The model receives the complete bounded turn; adapter v1 does not
continuously ingest an open camera stream.

Assistant responses render a DOM-built safe Markdown subset. The composer clears
as soon as send accepts a request, its focus border remains neutral, and the
locked mobile viewport prevents focus/pinch zoom. Page content is not
selectable and does not expose iOS touch callouts; text entry fields retain
normal cursor and editing behavior.

Reasoning defaults off. The portal always sends a boolean Ollama `think` field:
`false` until the brain control is explicitly enabled, then `true`. Native
`message.thinking` deltas and fallback `<think>` blocks are separated into the
collapsed reasoning view only while enabled. Disabled reasoning is suppressed
before display and speech synthesis.

## Voice stability

The stack's speech weights are Qwen3-TTS 12 Hz 1.7B Base. Without a speaker
reference, `seed: -1` allows voice/timbre changes between requests. The portal's
default profile pins seed `42` and the checked-in, metadata-free
`examples/omni_portal/voices/default_voice.wav` reference. To override it,
record or upload a clean WAV in the header voice panel, set another trusted
server-local `speaker_file` in `examples/omni_portal/voice-profile.json`, or
point `OMNI_VOICE_PROFILE` to another profile. Browser references are
validated, materialized only for the generation, and deleted afterward; the
browser cannot choose a server path.
See the portal README for sampling fields and the distinction between Base
speaker-embedding cloning and separate VoiceDesign/CustomVoice checkpoints.

## Verification matrix

| Check | Expected result |
|---|---|
| `/healthz` through tunnel | HTTP 200 without internal details |
| `/api/status` without bearer | HTTP 401 |
| `/api/status` with bearer | all four stages `ok=true` |
| Text | exact sentinel |
| Microphone/WAV | non-empty transcription or chat response |
| Device camera | live preview, bounded MP4/WebM attachment, video comprehension |
| Speaker + microphone | transcription followed by valid spoken audio |
| Call control | silence-delimited audio turn followed by automatic playback |
| Voice clone | recorded/uploaded WAV changes speaker timbre and remains replayable |
| Image | non-empty visual description |
| Video with audio | ordered visual description plus spoken content |
| TTS | valid 24 kHz mono PCM16 WAV playable on phone |
| CUDA scope | comprehension and each TTS process resident on reserved UUID |

Use `examples/omni_portal/smoke.py` for the machine-verifiable form of these
checks. Browser microphone/camera permission and phone speaker output require a
manual device check.

## Rollback

For this temporary deployment, rollback is shutdown:

```bash
examples/omni_portal/start.sh --stop
```

This removes external ingress first, then stops local HTTP services and the
broker-scoped worker. The Ollama model/tag and sidecar blob are retained. The
portal-owned materialized cache is removed unless `OMNI_KEEP_CACHE=1` was set.
