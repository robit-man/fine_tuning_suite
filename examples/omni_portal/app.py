"""Authenticated, phone-first web portal for the Robit Omni adapter.

The portal is intentionally a narrow same-origin proxy. It never exposes the
component workers or Ollama directly, pins requests to one published model,
and executes only two read-only demonstration tools from an explicit allowlist.
"""

from __future__ import annotations

import base64
import copy
import hmac
import json
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training_suite.models.audio import AudioContractError, decode_wav_payload

ADAPTER_SCHEMA = "robit.ollama.omni-adapter.v1"
DEFAULT_MODEL = "robit/qwen3.8-27b-e03-obliterated-omni:q4km"
MAX_TOOL_ROUNDS = 2
VOICE_PROFILE_SCHEMA = "robit.omni.voice-profile.v1"
QWEN3_TTS_LANGUAGES = {"zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"}
VOICE_SPEECH_FIELDS = {
    "language",
    "speaker_file",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "max_frames",
}
VOICE_CLIENT_FIELDS = {
    "clone_enabled",
    "speaker_audio",
    "language",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "max_frames",
}
MAX_SPEAKER_REFERENCE_BYTES = 10 * 1024 * 1024

SAFE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": (
                "Return the current date, local time, and UTC offset from the "
                "portal host. This is a read-only test tool."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portal_capabilities",
            "description": (
                "Return the media and model capabilities exposed by this portal. "
                "This is a read-only test tool."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def load_voice_profile(path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not read voice profile {path}: {exc}") from exc
    if not isinstance(profile, dict):
        raise TypeError("voice profile must be a JSON object")
    if profile.get("schema") != VOICE_PROFILE_SCHEMA:
        raise RuntimeError(f"voice profile schema must be {VOICE_PROFILE_SCHEMA}")
    unknown = set(profile) - VOICE_SPEECH_FIELDS - {"schema", "name"}
    if unknown:
        raise RuntimeError(f"unknown voice profile fields: {sorted(unknown)}")
    language = str(profile.get("language") or "en").strip()
    if language not in QWEN3_TTS_LANGUAGES:
        supported = ", ".join(sorted(QWEN3_TTS_LANGUAGES))
        raise RuntimeError(f"voice profile language must be one of: {supported}")
    profile["language"] = language
    speaker = str(profile.get("speaker_file") or "").strip()
    if speaker:
        speaker_path = Path(speaker).expanduser()
        if not speaker_path.is_absolute():
            speaker_path = path.parent / speaker_path
        speaker_path = speaker_path.resolve()
        if not speaker_path.is_file():
            raise RuntimeError(f"voice profile speaker file does not exist: {speaker_path}")
        if speaker_path.suffix.lower() not in {".wav", ".mp3"}:
            raise RuntimeError("voice profile speaker file must be WAV or MP3")
        profile["speaker_file"] = str(speaker_path)
    else:
        profile.pop("speaker_file", None)
    numeric_ranges = {
        "temperature": (0.0, 2.0),
        "top_k": (0, 1000),
        "top_p": (0.0, 1.0),
        "max_frames": (1, 2048),
    }
    for key, (minimum, maximum) in numeric_ranges.items():
        if key not in profile:
            continue
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"voice profile {key} must be numeric")
        if not minimum <= value <= maximum:
            raise RuntimeError(
                f"voice profile {key} must be between {minimum} and {maximum}"
            )
    if "seed" in profile and (
        isinstance(profile["seed"], bool) or not isinstance(profile["seed"], int)
    ):
        raise RuntimeError("voice profile seed must be an integer")
    if not -1 <= int(profile.get("seed", 42)) <= 2_147_483_647:
        raise RuntimeError("voice profile seed must be -1 or a 32-bit non-negative integer")
    return profile


@dataclass(frozen=True)
class PortalConfig:
    adapter_url: str
    adapter_health_url: str
    comprehension_health_url: str
    tts_health_url: str
    ollama_health_url: str
    model: str
    access_token: str
    voice_profile: Mapping[str, Any] = field(default_factory=dict)
    timeout_s: float = 1200
    max_body_bytes: int = 96 * 1024 * 1024

    @classmethod
    def from_environment(cls) -> PortalConfig:
        adapter_url = os.environ.get(
            "OMNI_ADAPTER_URL", "http://127.0.0.1:8910/api/chat"
        ).strip()
        access_token = os.environ.get("OMNI_PORTAL_TOKEN", "").strip()
        if len(access_token) < 24:
            raise RuntimeError("OMNI_PORTAL_TOKEN must contain at least 24 characters")
        profile_path = Path(
            os.environ.get(
                "OMNI_VOICE_PROFILE",
                str(Path(__file__).resolve().parent / "voice-profile.json"),
            )
        ).expanduser()
        return cls(
            adapter_url=adapter_url,
            adapter_health_url=os.environ.get(
                "OMNI_ADAPTER_HEALTH_URL", "http://127.0.0.1:8910/healthz"
            ).strip(),
            comprehension_health_url=os.environ.get(
                "OMNI_COMPREHENSION_HEALTH_URL", "http://127.0.0.1:8901/health"
            ).strip(),
            tts_health_url=os.environ.get(
                "OMNI_TTS_HEALTH_URL", "http://127.0.0.1:8892/healthz"
            ).strip(),
            ollama_health_url=os.environ.get(
                "OMNI_OLLAMA_HEALTH_URL", "http://127.0.0.1:11434/api/tags"
            ).strip(),
            model=os.environ.get("OMNI_MODEL", DEFAULT_MODEL).strip(),
            access_token=access_token,
            voice_profile=load_voice_profile(profile_path),
            timeout_s=float(os.environ.get("OMNI_PORTAL_TIMEOUT_S", "1200")),
            max_body_bytes=int(
                os.environ.get("OMNI_PORTAL_MAX_BODY_BYTES", str(96 * 1024 * 1024))
            ),
        )


class PortalError(RuntimeError):
    """A safe, user-visible portal failure."""


class PortalRequestError(ValueError):
    """A safe client request validation failure."""


def _voice_override(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise PortalRequestError("portal_voice must be an object")
    unknown = set(raw) - VOICE_CLIENT_FIELDS
    if unknown:
        raise PortalRequestError(f"unknown portal_voice fields: {sorted(unknown)}")

    clone_enabled = raw.get("clone_enabled", False)
    if not isinstance(clone_enabled, bool):
        raise PortalRequestError("portal_voice.clone_enabled must be a boolean")
    result: dict[str, Any] = {"clone_enabled": clone_enabled}

    if "language" in raw:
        language = str(raw["language"] or "").strip()
        if language not in QWEN3_TTS_LANGUAGES:
            supported = ", ".join(sorted(QWEN3_TTS_LANGUAGES))
            raise PortalRequestError(
                f"portal_voice.language must be one of: {supported}"
            )
        result["language"] = language

    numeric_ranges = {
        "temperature": (0.0, 2.0, float),
        "top_k": (0, 1000, int),
        "top_p": (0.0, 1.0, float),
        "max_frames": (1, 2048, int),
    }
    for key, (minimum, maximum, expected_type) in numeric_ranges.items():
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PortalRequestError(f"portal_voice.{key} must be numeric")
        if expected_type is int and not isinstance(value, int):
            raise PortalRequestError(f"portal_voice.{key} must be an integer")
        if not minimum <= value <= maximum:
            raise PortalRequestError(
                f"portal_voice.{key} must be between {minimum} and {maximum}"
            )
        result[key] = expected_type(value)

    if "seed" in raw:
        seed = raw["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PortalRequestError("portal_voice.seed must be an integer")
        if not -1 <= seed <= 2_147_483_647:
            raise PortalRequestError(
                "portal_voice.seed must be -1 or a 32-bit non-negative integer"
            )
        result["seed"] = seed

    reference = raw.get("speaker_audio")
    if reference is not None:
        if not clone_enabled:
            raise PortalRequestError(
                "portal_voice.speaker_audio requires clone_enabled=true"
            )
        try:
            decoded = decode_wav_payload(
                reference, max_bytes=MAX_SPEAKER_REFERENCE_BYTES
            )
        except AudioContractError as exc:
            raise PortalRequestError(f"invalid voice reference: {exc}") from exc
        if decoded.duration_ms < 500:
            raise PortalRequestError("voice reference must be at least 0.5 seconds")
        if decoded.duration_ms > 30_000:
            raise PortalRequestError("voice reference must be no longer than 30 seconds")
        result["speaker_audio"] = {
            "mime_type": "audio/wav",
            "encoding": "base64",
            "data": base64.b64encode(decoded.data).decode("ascii"),
        }
    return result


def _json_object(response: httpx.Response, stage: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise PortalError(f"{stage} did not return JSON") from exc
    if not isinstance(data, dict):
        raise PortalError(f"{stage} returned a non-object JSON response")
    return data


def _tool_arguments(call: Mapping[str, Any]) -> Mapping[str, Any]:
    function = call.get("function")
    if not isinstance(function, Mapping):
        return {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, Mapping):
        return arguments
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, Mapping) else {}
    return {}


def _execute_safe_tool(call: Mapping[str, Any]) -> tuple[str, str]:
    function = call.get("function")
    name = str(function.get("name") or "") if isinstance(function, Mapping) else ""
    _tool_arguments(call)  # Parse and reject malformed shapes without using them.
    if name == "get_current_time":
        now = datetime.now().astimezone()
        result = {
            "date": now.date().isoformat(),
            "time": now.isoformat(timespec="seconds"),
            "utc_offset": now.strftime("%z"),
            "timezone": str(now.tzinfo),
        }
    elif name == "get_portal_capabilities":
        result = {
            "input": ["text", "microphone", "wav", "image", "video"],
            "output": ["text", "thinking", "tool_calls", "audio/wav"],
            "tasks": ["chat", "transcribe", "describe", "synthesize"],
            "audio_input": "16 kHz mono PCM16 WAV",
            "audio_output": "24 kHz mono PCM16 WAV",
            "schema": ADAPTER_SCHEMA,
        }
    else:
        result = {
            "error": "tool_not_allowed",
            "allowed": [
                "get_current_time",
                "get_portal_capabilities",
            ],
        }
    return name or "unknown", json.dumps(result, separators=(",", ":"))


def _without_media(messages: list[Any]) -> list[Any]:
    cleaned = copy.deepcopy(messages)
    for message in cleaned:
        if isinstance(message, dict):
            message.pop("audios", None)
            message.pop("images", None)
            message.pop("videos", None)
    return cleaned


def _tool_followup(
    payload: Mapping[str, Any], response: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    message = response.get("message")
    if not isinstance(message, Mapping):
        return None, []
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return None, []

    followup = copy.deepcopy(dict(payload))
    messages = _without_media(list(followup.get("messages") or []))
    assistant = {
        key: copy.deepcopy(value)
        for key, value in message.items()
        if key in {"role", "content", "thinking", "tool_calls"}
    }
    assistant["role"] = "assistant"
    messages.append(assistant)
    executed: list[dict[str, str]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        name, content = _execute_safe_tool(call)
        messages.append({"role": "tool", "tool_name": name, "content": content})
        executed.append({"name": name, "result": content})
    followup["messages"] = messages
    return followup, executed


def _probe(client: httpx.Client, url: str) -> dict[str, Any]:
    try:
        response = client.get(url, timeout=5)
        return {"ok": response.status_code < 400, "status": response.status_code}
    except httpx.HTTPError:
        return {"ok": False, "status": None}


def create_app(
    config: PortalConfig | None = None,
    client: httpx.Client | None = None,
) -> Flask:
    root = Path(__file__).resolve().parent
    app = Flask(
        __name__,
        static_folder=str(root / "static"),
        static_url_path="/assets",
        template_folder=str(root / "templates"),
    )
    runtime = config or PortalConfig.from_environment()
    session = client or httpx.Client(timeout=runtime.timeout_s)
    inference_gate = threading.BoundedSemaphore(1)
    app.config["MAX_CONTENT_LENGTH"] = runtime.max_body_bytes

    def authorized() -> bool:
        value = request.headers.get("Authorization", "")
        supplied = value[7:].strip() if value.lower().startswith("bearer ") else ""
        return bool(supplied) and hmac.compare_digest(supplied, runtime.access_token)

    def apply_voice_profile(payload: dict[str, Any]) -> None:
        client_voice = _voice_override(payload.pop("portal_voice", None))
        speech = {
            key: copy.deepcopy(value)
            for key, value in runtime.voice_profile.items()
            if key in VOICE_SPEECH_FIELDS
        }
        clone_enabled = client_voice.pop("clone_enabled", None)
        speaker_audio = client_voice.pop("speaker_audio", None)
        if clone_enabled is False:
            speech.pop("speaker_file", None)
        elif clone_enabled is True:
            if speaker_audio is None and not speech.get("speaker_file"):
                raise PortalRequestError(
                    "voice clone is enabled but no reference audio is configured"
                )
            if speaker_audio is not None:
                speech.pop("speaker_file", None)
                speech["speaker_audio"] = speaker_audio
        speech.update(client_voice)
        payload.pop("speech", None)
        if speech:
            payload["speech"] = speech

    def apply_reasoning_mode(payload: dict[str, Any]) -> None:
        think = payload.get("think", False)
        if not isinstance(think, bool):
            raise PortalRequestError("portal think must be a boolean")
        payload["think"] = think

    @app.after_request
    def secure_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; "
            "media-src 'self' data: blob:; connect-src 'self'; "
            "script-src 'self'; style-src 'self'; object-src 'none'; "
            "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        )
        response.headers["Permissions-Policy"] = (
            "camera=(self), microphone=(self), geolocation=()"
        )
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "request exceeds the portal upload limit"}), 413

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            model=runtime.model,
            max_upload_mib=runtime.max_body_bytes // (1024 * 1024),
        )

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "service": "robit-omni-portal"})

    @app.get("/api/status")
    def status():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        stages = {
            "adapter": _probe(session, runtime.adapter_health_url),
            "comprehension": _probe(session, runtime.comprehension_health_url),
            "tts": _probe(session, runtime.tts_health_url),
            "ollama": _probe(session, runtime.ollama_health_url),
        }
        return jsonify(
            {
                "ok": all(item["ok"] for item in stages.values()),
                "model": runtime.model,
                "schema": ADAPTER_SCHEMA,
                "stages": stages,
                "safe_tools": SAFE_TOOLS,
                "voice_profile": {
                    "name": str(runtime.voice_profile.get("name") or "default"),
                    "language": str(runtime.voice_profile.get("language") or "en"),
                    "speaker_reference": bool(
                        runtime.voice_profile.get("speaker_file")
                    ),
                    "temperature": float(
                        runtime.voice_profile.get("temperature", 0.7)
                    ),
                    "top_k": int(runtime.voice_profile.get("top_k", 40)),
                    "top_p": float(runtime.voice_profile.get("top_p", 0.9)),
                    "seed": int(runtime.voice_profile.get("seed", 42)),
                    "max_frames": int(
                        runtime.voice_profile.get("max_frames", 512)
                    ),
                    "clone_mode": "speaker_embedding",
                    "client_reference_wav": True,
                },
                "streaming": {
                    "text": True,
                    "audio": True,
                    "audio_transport": "pcm_s16le_deltas_with_final_wav",
                    "barge_in": True,
                },
            }
        )

    @app.post("/api/chat")
    def chat():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        if not inference_gate.acquire(blocking=False):
            return jsonify({"error": "another inference is in progress"}), 429
        try:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return jsonify({"error": "request body must be a JSON object"}), 400
            auto_tools = payload.pop("portal_auto_tools", False) is True
            if payload.get("model") != runtime.model:
                return jsonify({"error": "portal model tag is fixed"}), 400
            if payload.get("stream") is not False:
                return jsonify({"error": "portal requires stream=false"}), 400
            apply_reasoning_mode(payload)
            apply_voice_profile(payload)

            executed: list[dict[str, str]] = []
            current_payload: dict[str, Any] = payload
            for _round in range(MAX_TOOL_ROUNDS + 1):
                upstream = session.post(runtime.adapter_url, json=current_payload)
                data = _json_object(upstream, "adapter")
                if upstream.status_code >= 400:
                    return jsonify(data), upstream.status_code
                if not auto_tools:
                    break
                followup, round_tools = _tool_followup(current_payload, data)
                if followup is None:
                    break
                executed.extend(round_tools)
                current_payload = followup
            else:
                raise PortalError("safe tool loop exceeded its round limit")

            data["portal"] = {
                "schema": "robit.omni-phone-portal.v1",
                "safe_tools_executed": executed,
            }
            return jsonify(data)
        except PortalRequestError as exc:
            return jsonify({"error": str(exc)}), 400
        except (httpx.HTTPError, PortalError) as exc:
            return jsonify({"error": str(exc)}), 502
        finally:
            inference_gate.release()

    @app.post("/api/chat/stream")
    def chat_stream():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "request body must be a JSON object"}), 400
        if payload.pop("portal_auto_tools", False) is True:
            return jsonify(
                {"error": "automatic tool rounds are unavailable while streaming"}
            ), 400
        if payload.get("model") != runtime.model:
            return jsonify({"error": "portal model tag is fixed"}), 400
        if payload.get("stream") is not True:
            return jsonify({"error": "stream endpoint requires stream=true"}), 400
        try:
            apply_reasoning_mode(payload)
            apply_voice_profile(payload)
        except PortalRequestError as exc:
            return jsonify({"error": str(exc)}), 400
        if not inference_gate.acquire(blocking=False):
            return jsonify({"error": "another inference is in progress"}), 429

        try:
            upstream_request = session.build_request(
                "POST", runtime.adapter_url.rstrip("/") + "/stream", json=payload
            )
            upstream = session.send(upstream_request, stream=True)
        except httpx.HTTPError as exc:
            inference_gate.release()
            return jsonify({"error": str(exc)}), 502

        def relay():
            try:
                yield from upstream.iter_bytes()
            finally:
                upstream.close()
                inference_gate.release()

        return Response(
            stream_with_context(relay()),
            status=upstream.status_code,
            content_type=upstream.headers.get(
                "content-type", "application/x-ndjson; charset=utf-8"
            ),
            headers={"X-Accel-Buffering": "no"},
        )

    return app


if __name__ == "__main__":
    create_app().run(
        host=os.environ.get("OMNI_PORTAL_HOST", "127.0.0.1"),
        port=int(os.environ.get("OMNI_PORTAL_PORT", "8920")),
        debug=False,
        threaded=True,
    )
