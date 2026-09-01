"""Smoke-test an authenticated Robit Omni phone portal."""

from __future__ import annotations

import argparse
import base64
import io
import json
import wave
from pathlib import Path
from typing import Any

import httpx

SCHEMA = "robit.ollama.omni-adapter.v1"

SAFE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Return the current portal-host date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def envelope(path: Path, mime_type: str) -> dict[str, Any]:
    return {"mime_type": mime_type, "encoding": "base64", "data": encoded(path)}


def base_request(model: str, task: str, message: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [message],
        "omni": {"schema": SCHEMA, "task": task},
        "response_modalities": ["text"],
        "speech_mode": "never",
        "think": task == "chat",
        "stream": False,
        "options": {"num_predict": 512},
    }


def call(client: httpx.Client, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(endpoint + "/api/chat", json=payload)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("message"), dict):
        raise TypeError("portal returned no message object")
    return data


def stream_call(
    client: httpx.Client, endpoint: str, payload: dict[str, Any]
) -> tuple[dict[str, Any], bytes]:
    request = dict(payload)
    request["stream"] = True
    events: list[dict[str, Any]] = []
    pcm = bytearray()
    with client.stream(
        "POST", endpoint + "/api/chat/stream", json=request
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "error":
                raise RuntimeError(str(event.get("error") or "stream failed"))
            if event.get("type") == "audio_delta":
                pcm.extend(
                    base64.b64decode(
                        str((event.get("audio") or {}).get("data") or ""),
                        validate=True,
                    )
                )
            events.append(event)
    finals = [event.get("response") for event in events if event.get("type") == "final"]
    if len(finals) != 1 or not isinstance(finals[0], dict):
        raise RuntimeError("portal stream returned no authoritative final response")
    return finals[0], bytes(pcm)


def validate_wav(audio: dict[str, Any]) -> None:
    raw = base64.b64decode(str(audio.get("data") or ""), validate=True)
    with wave.open(io.BytesIO(raw), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getnframes() > 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--tts", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--tool", action="store_true")
    args = parser.parse_args()

    token = args.token_file.read_text().strip()
    client = httpx.Client(
        headers={"Authorization": f"Bearer {token}"},
        timeout=1200,
    )
    endpoint = args.endpoint.rstrip("/")
    status = client.get(endpoint + "/api/status")
    status.raise_for_status()
    status_data = status.json()
    if not status_data.get("ok"):
        raise RuntimeError(f"portal status gate failed: {status_data}")

    checks: dict[str, Any] = {"status": "pass"}
    if args.text:
        payload = base_request(
            args.model,
            "chat",
            {"role": "user", "content": "Reply with exactly: PORTAL TEXT OK"},
        )
        result = call(client, endpoint, payload)
        content = str(result["message"].get("content") or "")
        if "PORTAL TEXT OK" not in content:
            raise RuntimeError(f"text sentinel missing: {content!r}")
        checks["text"] = "pass"

    if args.audio:
        payload = base_request(
            args.model,
            "transcribe",
            {
                "role": "user",
                "content": "Transcribe faithfully.",
                "audios": [envelope(args.audio, "audio/wav")],
            },
        )
        result = call(client, endpoint, payload)
        if not str(result["message"].get("content") or "").strip():
            raise RuntimeError("ASR returned empty text")
        checks["audio"] = "pass"
        if args.tts:
            payload["response_modalities"] = ["text", "audio"]
            payload["speech_mode"] = "always"
            result = call(client, endpoint, payload)
            validate_wav(result["message"].get("audio") or {})
            if (result.get("adapter") or {}).get("route") != [
                "comprehension",
                "tts",
            ]:
                raise RuntimeError(
                    f"ASR-to-TTS route was not direct: {result.get('adapter')!r}"
                )
            checks["audio_tts"] = "pass"

    if args.image:
        mime = {".png": "image/png", ".webp": "image/webp"}.get(
            args.image.suffix.lower(), "image/jpeg"
        )
        payload = base_request(
            args.model,
            "describe",
            {
                "role": "user",
                "content": "Describe the image and preserve visible text.",
                "images": [envelope(args.image, mime)],
            },
        )
        result = call(client, endpoint, payload)
        if not str(result["message"].get("content") or "").strip():
            raise RuntimeError("image description returned empty text")
        checks["image"] = "pass"

    if args.video:
        mime = "video/webm" if args.video.suffix.lower() == ".webm" else "video/mp4"
        video = envelope(args.video, mime)
        video["sampling"] = {"fps": 1, "max_frames": 24, "include_audio": True}
        payload = base_request(
            args.model,
            "describe",
            {
                "role": "user",
                "content": "Describe temporal order and spoken content.",
                "videos": [video],
            },
        )
        result = call(client, endpoint, payload)
        if not str(result["message"].get("content") or "").strip():
            raise RuntimeError("video description returned empty text")
        checks["video"] = "pass"

    if args.tts:
        payload = base_request(
            args.model,
            "synthesize",
            {"role": "user", "content": "Robit Omni portal speech is online."},
        )
        payload["response_modalities"] = ["text", "audio"]
        payload["speech_mode"] = "always"
        if args.stream:
            result, pcm = stream_call(client, endpoint, payload)
            if not pcm or len(pcm) % 2:
                raise RuntimeError("streamed TTS returned invalid PCM16 bytes")
            adapter = result.get("adapter") or {}
            if adapter.get("audio_streamed") is not True:
                raise RuntimeError(f"adapter did not confirm audio streaming: {adapter!r}")
            checks["tts_stream"] = "pass"
        else:
            result = call(client, endpoint, payload)
        validate_wav(result["message"].get("audio") or {})
        checks["tts"] = "pass"

    if args.tool:
        payload = base_request(
            args.model,
            "chat",
            {
                "role": "user",
                "content": "Use get_current_time, then report the returned date and time.",
            },
        )
        payload["tools"] = SAFE_TOOLS
        payload["portal_auto_tools"] = True
        result = call(client, endpoint, payload)
        executed = (result.get("portal") or {}).get("safe_tools_executed") or []
        if not executed or executed[0].get("name") != "get_current_time":
            raise RuntimeError(f"safe tool was not executed: {executed!r}")
        checks["tool"] = "pass"

    print(json.dumps({"ok": True, "checks": checks}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
