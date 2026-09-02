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
paperclip accepts WAV, JPEG, PNG, WebP, animated GIF, MP4, WebM, PDF, DOCX, and
bounded UTF-8 text/code files. Multiple files can be attached together. The speaker icon is the
only output toggle: gray requests text only, while yellow requests both text
and synthesized audio. Assistant text renders a safe Markdown subset including
headings, emphasis, lists, block quotes, links, and fenced code. Sending a turn
clears the composer and attachments immediately.

An audio attachment sent without typed text is a conversational turn, not a
direct-ASR result. The portal briefly shows `Audio clip`, replaces that text in
the originating user bubble with the adapter's tagged verbatim
`input_transcript`, and places any additional non-speech acoustic evidence in a
**Sounds heard** disclosure styled like the reasoning trace. If no intelligible
speech exists, the acoustic observation becomes the bubble's half-brightness
primary text instead of being mislabelled as ASR. Both evidence channels pass
through Qwen3.8. Clients that need transcription without a language reply can
still call the adapter explicitly with `omni.task="transcribe"`.

The brain icon controls the real Ollama `think` request field. Gray is the
default and sends the boolean `think:false`; violet explicitly sends
`think:true`. No system instruction, prompt suffix, or synthetic `/no_think`
message controls this mode. When enabled, streamed reasoning is shown in a
collapsed Reasoning disclosure beneath the answer instead of being mixed into
the visible response. The adapter preserves Ollama's native
`message.thinking` channel and also separates malformed fallback
`<think>...</think>` output, including tags split across stream chunks. The
latter is a fail-closed output guard, not the reasoning control. Reasoning is
never sent to TTS.

The camera icon in the upper-right control group opens the device camera and microphone with a live, muted
in-interface preview. Tap it again—or press Send—to stop and attach the bounded
recording as MP4 or WebM, including its audio track when the browser provides
one. Recordings use a constrained video bitrate to remain below the portal
upload limit. Starting Call while the camera is active pauses the attachment
recording and sends the current visual frame with each detected speech turn;
this provides bounded live visual conversation without presenting an unbounded
video stream to the model context.

The phone icon at the upper right starts hands-free voice mode. Browser-side
voice activity detection first calibrates ambient noise for 650 ms, requires
120 ms of sustained activity above a more sensitive adaptive threshold, and closes an
utterance after 750 ms of silence. It submits a 16 kHz WAV only after that
confirmed utterance; quiet, transient clicks, and elevated steady room noise do
not call remote ASR. The waveform border, line, and label remain translucent
while inactive and become opaque only while VAD is active. Confirmed speech is
sent through Qwen3-Omni and Qwen3.8, response text is relayed as Ollama produces
it, and Qwen3-TTS speech is streamed back.
The microphone remains active during inference and playback. Every confirmed
speech segment is submitted immediately as its own request-local turn, even
while earlier turns are queued or understanding; a single pending slot cannot
overwrite continued speech. Sustained speech during playback stops the current
audio while capture continues. Echo cancellation and a higher playback-time
interruption threshold reduce self-triggering. The live waveform and status
line show the number of processing turns while explicitly inviting the user to
keep speaking. Tap the phone icon again to stop capture, abort every outstanding
turn, and stop playback.

Call turns include a dedicated conversational contract: answer the speaker's
intent directly, do not echo/transcribe/paraphrase unless explicitly asked,
and use the preceding text dialogue for continuity. Camera-call turns send only
the newest frame as current visual evidence while retaining prior text as
conversation—not as proof that an older scene remains visible.

| Composer action | Adapter route |
|---|---|
| Text chat | Qwen3.8 language |
| Audio attachment with no prompt | Qwen3-Omni transcript/acoustic evidence → Qwen3.8 reply |
| Audio attachment with a prompt | Qwen3-Omni comprehension → Qwen3.8 |
| Image/video with no prompt | Qwen3-Omni evidence → context-aware Qwen3.8 conversation |
| Image/video with a prompt | Qwen3-Omni comprehension → Qwen3.8 |
| Silent video or animated GIF | bounded visual-only comprehension |
| PDF/DOCX/text or code | extraction → session-isolated retrieval → Qwen3.8 |
| Documents plus media | retrieved excerpts + current media observation → Qwen3.8 |
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
for a stable app-like mobile layout. Conversation text and decorative content
also disable touch/mouse selection and iOS callouts; normal editing remains
enabled in the composer and voice configuration fields.

The TTS stage status means the speech request has started. It changes to
streaming only after the adapter receives the first actual PCM bytes; an
`audio_start` event is never emitted merely because a TTS HTTP request was
opened. This makes the UI and timing journal distinguish model/prefill latency
from network or browser playback buffering.

The patched Qwen3-TTS worker is warmed once and remains CUDA-resident across
default-profile requests. Prompts travel over a bounded framed stdin/stdout
protocol, so repeated turns avoid reloading the backbone and projector. Live
decoding defaults to two codec frames (about 160 ms) per PCM window, and the
browser starts with an 80 ms playout lead. A guarded 8 ms crossfade smooths
large contiguous buffer boundaries; a late buffer instead receives a short
fade-in. A changed voice profile safely replaces the resident worker; inline
uploaded speaker audio uses the isolated single-shot fallback and then rewarms
the default profile.

The 512-frame voice setting is a per-generation ceiling (roughly 42.7 seconds
at 12 Hz). Replies that would exceed it are split at natural sentence/word
boundaries, streamed as one continuous PCM sequence with the same pinned voice
configuration, and assembled into one complete replay WAV. This prevents the
former approximately 40-second truncation; the final adapter trace exposes the
number of generated `tts_blocks`.

The viewport is a fixed three-row grid, so the conversation—not the page—is the
scroll container. New user and assistant messages pin it to the newest turn;
streaming token updates use immediate scrolling so repeated deltas cannot keep
restarting a smooth animation behind the response. The small number beside
**ONLINE** reports distinct browser sessions with
an active or queued inference request; it is an aggregate only and is never
used as conversation state.

Conversation history, rendered messages, reply audio, drafts, and bounded media
previews are stored in same-origin IndexedDB under a one-way scope derived from
the Secure session cookie. Reloading the same page restores them without ever
re-sending old media bytes to inference. The active page refreshes a five-minute
lease; leaving marks the cache for expiry after five minutes, and the next
access removes expired data. The trash control deletes that browser cache
immediately together with the server-side document index and diagnostics.

Every turn also receives a fresh trusted runtime snapshot containing local and
UTC date/time, OS/architecture, CPU/load, RAM, bounded network-interface
counters, and NVIDIA VRAM/utilization/temperature/power. It deliberately omits
hostnames, IP/MAC addresses, routes, sockets, process data, credentials, and
session content, and instructs the model not to volunteer irrelevant telemetry.

Video is sampled at 24 frames by the phone and clamped to at most 32 frames and
2 fps by the adapter. The comprehension GGUF declares a 65,536-token context,
and the deployment now starts `llama-server` at that native limit instead of
the former 8,192-token test setting. If a multimodal prompt still exceeds the
available context, the adapter retries only the comprehension stage with
progressively smaller frame caps (24/16/8/4/1 as applicable). A recorded video
turn therefore retains temporal comprehension whenever it fits; camera-call
mode uses one current frame per speech turn for lower latency.

MP4/WebM clips without an audio track are accepted as visual-only media. The
adapter probes for an audio stream before demux and simply omits the audio part
when none exists. Animated GIF is shown in a loop in the originating user turn
and normalized to a maximum 30-second, frame-bounded MP4 for the same temporal
comprehension path.

PDF, DOCX, and UTF-8 text/code uploads are handled by the portal rather than the
portable adapter ABI. PDF extraction is capped at 200 pages, DOCX ZIP expansion
is bounded, binary/unsupported text is rejected, and extracted text is chunked
into a 384-dimension deterministic hashed lexical index. Retrieval is capped at
eight chunks/12,000 characters per turn and injected as explicitly untrusted
document data. Raw files and extracted chunks remain only in memory, are keyed
by the opaque Secure browser-session cookie, clear with the trash button, and
expire five minutes after activity stops. This is retrieval over extracted text,
not a claim that PDF pixels or arbitrary office formats are natively understood
by the GGUF.

Every comprehension request explicitly sets llama.cpp `cache_prompt:false`.
Prompt-slot reuse is safe for ordinary token prefixes but the pinned
multimodal worker can otherwise retain a previous clip's decoded embeddings.
The regression gate submits red → blue → red videos and requires red → blue →
red answers. Do not remove this setting as a performance optimization.

## Voice configuration and cloning

The TTS weights are Qwen3-TTS 12 Hz 1.7B Base, not LuxTTS. Tap the waveform
button in the portal header to open the request-local voice panel. It provides:

- a button-style voice-clone toggle with no visible checkbox;
- an allowlisted preset selector with Female as the default and Male as the
  secondary reference;
- phone recording or WAV upload for a clean 3–10 second reference;
- in-browser reference playback and removal;
- language, temperature, top-p, top-k, seed, and maximum-frame controls.

The reference is sent as a bounded base64 WAV envelope. The portal validates
its container, duration (0.5–30 seconds), and 10 MiB decoded limit, then the TTS
worker writes it into a per-generation temporary directory for `llama-tts`.
The file is deleted as soon as that generation finishes. The browser cannot
select a server path. Only clone a voice you own or have permission to use.

The repository ships two metadata-free, 16 kHz mono PCM16 references.
[`voices/female_voice.wav`](voices/female_voice.wav) is the active Female
default, converted from the repository owner's `entering_phase_nine.ogg`.
[`voices/default_voice.wav`](voices/default_voice.wav) is the 13.009-second Male
secondary preset retained from the prior deployment. Preset IDs are resolved
only against the server profile allowlist; the browser never supplies a server
path. Recording or uploading a request-local WAV temporarily overrides the
selected preset.

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
  "presets": [
    {"id": "studio", "label": "Studio", "speaker_file": "voices/studio-reference.wav", "default": true}
  ],
  "temperature": 0.7,
  "top_k": 40,
  "top_p": 0.9,
  "seed": 42,
  "max_frames": 512
}
```

The portable contract is
[`voice-profile-v1.schema.json`](../../docs/omni-adapter/schema/voice-profile-v1.schema.json).

`speaker_file` and each preset `speaker_file` may be an absolute path or a path
relative to the profile. A preset profile requires exactly one default.
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
TTS uses `--gpu-layers -1` on the same UUID. The wrapper calls broker `prepare`,
loads the patched persistent `llama-tts` worker, waits for both CUDA residency
and its explicit ready frame, and then calls `ready`. Default-profile requests
reuse that resident graph. A failed reservation or residency check aborts
deployment or the request instead of silently running inference on CPU. Ollama
language requests continue through broker-owned GPU lanes.

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
| `OMNI_TTS_STREAM_FRAMES` | `2` | Codec frames per live PCM decode window; two is about 160 ms and balances first-audio latency with smooth playback |
| `OMNI_TTS_PERSISTENT` | `1` | Keep the default-profile Qwen3-TTS graph resident between turns |
| `OMNI_TTS_WARM_SPEAKER_FILE` | bundled default voice | Speaker reference loaded by the resident worker |
| `OMNI_TTS_BROKER_TRANSITION_TIMEOUT_S` | `330` | Maximum wait for scoped prepare/ready transitions |
| `OMNI_KEEP_CACHE` | `0` | Keep materialized views after stop |
| `OMNI_PORTAL_MAX_BODY_BYTES` | 96 MiB | Same-origin JSON request cap |
| `OMNI_PORTAL_INFERENCE_SLOTS` | `1` | Simultaneous upstream inference lanes; keep at one for the shared single-lane GPU stack |
| `OMNI_PORTAL_MAX_INFLIGHT_REQUESTS` | `4` | Active plus queued portal requests before a bounded 503 response |
| `OMNI_PORTAL_SESSION_LOG_DIR` | runtime `session-logs` | Content-redacted, per-session timing journals |
| `OMNI_PORTAL_SESSION_LOG_TTL_S` | `300` | Inactive-session diagnostic retention; five minutes by default |

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
- The default admits four simultaneous portal requests and serializes them
  through one GPU inference lane. Queue tickets, request payloads, upstream
  responses, streaming iterators, voice settings, and tool rounds are
  request-local.
- Conversation history is browser-session-local. Same-origin IndexedDB restores
  that session after reload and expires it five minutes after page leave; it is
  never shared server state or addressable by another session. Old cached media
  is display-only and is never replayed into a later inference request.
  Its only content-bearing server session state is the bounded in-memory
  document chunk index, keyed by a hash of the opaque session cookie and never
  addressable across sessions.
- A random, Secure, HttpOnly, SameSite=Strict cookie partitions the aggregate
  activity count and ephemeral diagnostic journal. It is never supplied to a
  model or used to recover conversation context. `/api/activity` exposes only
  aggregate counts and requires the same bearer token as inference.
- Per-session diagnostics contain request IDs, modality flags, queue/transport
  timings, status codes, and browser milestones such as first text and first
  PCM. They never contain prompts, replies, transcripts, thinking, media,
  waveforms, bearer tokens, IP addresses, or user-agent strings. Journals are
  stored under hashed filenames, are inaccessible across browser sessions, and
  are purged five minutes after the session heartbeat stops.
- The trash button aborts the page's active request/call, stops playback,
  clears and deletes the IndexedDB browser session, and deletes that session's diagnostic
  journal and document index immediately. A late completion from the aborted request cannot
  recreate the deleted journal; a genuinely new request starts a new one.
- Encoded JSON is limited to 96 MiB.
- The browser caps decoded image, video, and audio sizes below adapter limits.
- CSP, frame denial, no-referrer, no-store, and same-origin camera/microphone
  policies are applied.
- Tool execution is limited to `get_current_time` and
  `get_portal_capabilities`; unknown names return an error result and cannot
  execute programs, access files, or make network requests.
- Media observations remain untrusted evidence at the adapter boundary.
- During audio-only sends and calls, only the adapter's tagged
  `input_transcript` is rendered in the user bubble. Raw perception output is
  never attributed to the user, and the pending assistant card remains hidden
  until an assistant delta is available.
- Re-recording with the camera replaces the prior unsent camera clip, so only
  the latest captured segment enters a request. The exact submitted clip is
  retained as a muted looping video thumbnail on its user message; separately
  uploaded video attachments remain additive.
- Every new media submission starts an isolated model context. Earlier media
  bytes and their generated descriptions are excluded, while the successful
  newest media turn becomes the context for subsequent text-only follow-ups.
  Visual-call turns likewise use only the current frame; audio-only calls keep
  their conversational history.
- Every chat audio attachment performs combined ASR and environmental sound
  analysis. Speech is exposed separately from non-speech events, ambience,
  music, speaker activity, and temporal changes, so acoustic observations reach
  the language model without appearing as user-authored transcript text. For an
  unprompted clip, the tagged speech transcript replaces the temporary user
  placeholder and the language-model answer remains a separate assistant turn.

Cloudflare Quick Tunnels are temporary development endpoints, not durable
production ingress. Stop the portal after the phone test. For a persistent
deployment, replace the quick tunnel with a named tunnel plus Cloudflare
Access while retaining the portal bearer token.
