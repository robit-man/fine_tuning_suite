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
  └── Qwen3-TTS :8892                  (broker-scoped CUDA, persistent)
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
An unprompted audio clip runs the full comprehension-to-language route: the
tagged ASR transcript replaces the temporary text in your bubble. Additional
environmental audio appears under a **Sounds heard** disclosure using the same
visual pattern as reasoning. With no intelligible speech, the acoustic scene is
shown directly at half brightness instead of being presented as a transcript.
The intelligent reply remains in the assistant bubble. Use the adapter's
explicit `omni.task="transcribe"` outside the composer when a
transcription-only response is required.
The speaker icon switches between text-only and text-plus-TTS replies. The
waveform button opens Qwen3-TTS voice cloning and sampling controls. The phone
icon starts automatic voice turns. Local VAD calibrates ambient noise for 650
ms, requires 120 ms of sustained activity above a sensitive adaptive threshold, and
closes after 750 ms of silence. Quiet, clicks, and elevated steady noise do not
call remote ASR. The green waveform is translucent while inactive and opaque
only during confirmed activity. The microphone remains active while previous
turns are queued or understanding; each later confirmed segment is submitted
immediately instead of overwriting a single pending clip. During playback,
sustained speech uses a stricter threshold and cancels the current audio without
stopping capture. Portable adapter v1 remains `stream:false`;
`/api/chat/stream` is an authenticated portal extension with stage, text,
thinking, PCM, and one authoritative final adapter event.

Qwen3-TTS uses two codec frames per state-carrying decoder window by default,
approximately 160 ms for the packaged 12 Hz model. Its persistent CUDA worker
loads the model/projector/speaker once and resets generation state between
requests. The patched code2wav graph is sized to the actual frame count; it must
never persist state after fixed-window rear padding. The raw PCM continuity gate
must pass before evaluating browser smoothing. The browser schedules the first
received PCM with an 80 ms playout
lead, keeps a 3 ms late-arrival floor, and uses a guarded 3 ms crossfade for
sufficiently large contiguous buffers. The final response retains a complete
replayable WAV assembled from the original PCM. `stage=tts` is a
preparing state; `audio_start` is emitted only with the first actual PCM chunk,
so “streaming” never describes a request that is still in model prefill.
The 512-frame ceiling applies to each synthesis block. Longer replies are split
at natural text boundaries, retain one continuous PCM sequence across blocks,
and finish as a single assembled replay WAV instead of stopping near 40 seconds.
There is no aggregate TTS block ceiling, and the portal does not impose a
`num_predict` override on either typed chat or live-call language generation.

The camera icon captures device video and microphone audio with a live preview
inside the composer. Tap again or send to finalize a 30-second-maximum MP4/WebM
attachment. The model receives the complete bounded turn; adapter v1 does not
continuously ingest an open camera stream.

Silent MP4/WebM is valid and follows the visual-only route; absence of an audio
stream no longer fails FFmpeg demux. Animated GIF is normalized to bounded MP4
and remains visible as a looping preview. The paperclip also accepts multiple
PDF, DOCX, and UTF-8 text/code documents. The portal extracts them with strict
size/page/archive limits, creates a deterministic hashed lexical chunk index
isolated to the Secure browser session, and adds only the most relevant bounded
excerpts to each language turn as untrusted data. Trash clears both diagnostics
and the document index; idle indexes expire after five minutes.

Re-recording replaces the previous unsent camera capture. The exact submitted
blob remains visible as a muted looping thumbnail in the user turn. Each media
submission sends only its current media bytes. Prior text dialogue remains as
conversation context, but an explicit boundary says it is not current visual
evidence; prior media bytes are never replayed. The adapter also sets
`cache_prompt:false` on the llama.cpp comprehension request. This is mandatory:
multimodal prompt-slot reuse can otherwise answer a new clip from the previous
clip's decoded embeddings.

Assistant responses render a DOM-built safe Markdown subset. GFM pipe tables
support header delimiters, escaped pipes, inline formatting, and alignment
markers; wide tables scroll horizontally inside the message on narrow screens.
Completed assistant messages include a bottom-right copy control that copies the
original Markdown source. Scrolling upward pauses token auto-follow and reveals a
floating down-arrow above the composer; activating it returns to the latest turn
and resumes automatic following.
The composer clears as soon as send accepts a request, its focus border remains
neutral, and the locked mobile viewport prevents focus/pinch zoom. Page content
is not selectable and does not expose iOS touch callouts; text entry fields
retain normal cursor and editing behavior.

The live-call instruction answers the user's intent directly and forbids
echoing, transcription, or paraphrase unless requested. Both audio-only and
camera-call turns include the prior bounded text dialogue. A camera turn adds
only the newest frame and labels older scene descriptions as historical.

Same-origin IndexedDB persists rendered messages, text context, drafts, reply
audio, pending attachments, and bounded media previews across reload. Its key
contains a one-way scope derived from the Secure cookie. Active pages renew a
five-minute lease; page leave starts expiry, and trash deletes browser cache,
document index, and diagnostics immediately. Restored media is display-only and
never enters a new request automatically.

Each request prepends a fresh trusted runtime snapshot with date/time,
OS/architecture, CPU/load, RAM, bounded interface counters, and NVIDIA
utilization. Hostnames, addresses, routes, sockets, processes, credentials, and
session content are excluded.

Reasoning defaults off. The portal always sends a boolean Ollama `think` field:
`false` until the brain control is explicitly enabled, then `true`. Native
`message.thinking` deltas and fallback `<think>` blocks are separated into the
collapsed reasoning view only while enabled. No reasoning-control system prompt
or `/no_think` suffix is injected. Tagged-output sanitation is only a
fail-closed display/TTS guard.

## Multi-user isolation and diagnostics

The default portal admits four simultaneous HTTP requests and serializes them
through one GPU inference lane. The number beside **ONLINE** counts distinct
browser sessions with active or queued work. Request bodies, streams, media,
voice overrides, tool rounds, and responses remain request-local. Conversation
history exists only in each session-scoped IndexedDB record; the server has no
shared conversation or media history to bleed into another user. Records expire
five minutes after page leave and trash removes them immediately. The optional in-memory document
index is keyed by the hashed opaque session cookie, has independent bounds and
expiry, and is never searched across sessions.

A Secure, HttpOnly, SameSite=Strict cookie provides an opaque partition key for
the aggregate count and ephemeral diagnostics. `/api/activity` exposes only
aggregate queue counts. `GET /api/diagnostics` returns only the caller's timing
journal; `POST /api/diagnostics` accepts the browser's content-redacted stream
milestones; `DELETE /api/diagnostics` deletes only that journal. All require the
same bearer token as inference.

Diagnostic events may contain a request ID, modality flags, status,
queue/transport time, first text, TTS-stage, first PCM, and completion timing.
They never contain prompt/reply text, transcripts, thinking, audio, frames,
waveforms, tokens, IP addresses, or user-agent strings. Journals use hashed
filenames under `OMNI_PORTAL_SESSION_LOG_DIR`, expire five minutes after the
last heartbeat, and are purged at portal startup/shutdown. The trash control
aborts active page work, stops playback/call capture, clears browser history,
and deletes its journal immediately. Late events from the aborted request are
rejected; a later new request creates a fresh journal.

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
| VAD sensitivity and rejection | silence/click/steady-noise fixtures create zero requests; quiet and normal sustained speech create one per segment |
| Voice clone | recorded/uploaded WAV changes speaker timbre and remains replayable |
| Image | non-empty visual description |
| Video with audio | ordered visual description plus spoken content |
| Silent video | ordered visual description without an audio extraction error |
| Animated GIF | looping chat preview plus bounded temporal description |
| PDF/DOCX/text | relevant extracted chunks answer the query; another session retrieves none |
| Sequential video isolation | red → blue → red clips describe red → blue → red; `cache_prompt=false` |
| TTS | valid 24 kHz mono PCM16 WAV playable on phone |
| TTS first PCM | two-frame window, resident worker reuse, and browser first-audio milestone recorded |
| Concurrent sessions | two users show active/queued counts and receive only their own marker |
| Diagnostic lifecycle | journals are session-isolated; trash deletes immediately; idle data expires in 300 s |
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
Per-session diagnostic JSON is removed on shutdown regardless of component
cache retention.
