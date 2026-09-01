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
The speaker icon switches between text-only and text-plus-TTS replies. If phone
autoplay is blocked after a long inference, use the audio player's play control.

## Verification matrix

| Check | Expected result |
|---|---|
| `/healthz` through tunnel | HTTP 200 without internal details |
| `/api/status` without bearer | HTTP 401 |
| `/api/status` with bearer | all four stages `ok=true` |
| Text | exact sentinel |
| Microphone/WAV | non-empty transcription or chat response |
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
