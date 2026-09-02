"""Authenticated, phone-first web portal for the Robit Omni adapter.

The portal is intentionally a narrow same-origin proxy. It never exposes the
component workers or Ollama directly, pins requests to one published model,
and executes only two read-only demonstration tools from an explicit allowlist.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import json
import math
import os
import secrets
import sys
import threading
import time
from collections import deque
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
    make_response,
    render_template,
    request,
    stream_with_context,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.omni_portal.documents import DocumentError, SessionDocumentStore
from examples.omni_portal.environment import runtime_environment_system_message
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
SESSION_COOKIE_NAME = "omni_portal_session"
DIAGNOSTIC_TTL_SECONDS = 5 * 60
DIAGNOSTIC_NUMERIC_FIELDS = {
    "queue_wait_ms",
    "upstream_headers_ms",
    "first_upstream_byte_ms",
    "response_headers_ms",
    "first_event_ms",
    "first_text_ms",
    "tts_stage_ms",
    "audio_start_ms",
    "first_audio_delta_ms",
    "complete_ms",
    "total_ms",
    "status",
}
DIAGNOSTIC_BOOLEAN_FIELDS = {
    "audio_requested",
    "thinking_requested",
    "has_audio_input",
    "has_image_input",
    "has_video_input",
    "has_document_input",
}
DIAGNOSTIC_STRING_FIELDS = {"request_id", "task", "outcome"}

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
    inference_slots: int = 1
    max_inflight_requests: int = 4
    session_log_dir: Path | None = None
    session_log_ttl_s: float = DIAGNOSTIC_TTL_SECONDS

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
            inference_slots=max(
                1, int(os.environ.get("OMNI_PORTAL_INFERENCE_SLOTS", "1"))
            ),
            max_inflight_requests=max(
                1, int(os.environ.get("OMNI_PORTAL_MAX_INFLIGHT_REQUESTS", "4"))
            ),
            session_log_dir=Path(
                os.environ.get(
                    "OMNI_PORTAL_SESSION_LOG_DIR",
                    "training_suite/outputs/omni_portal_runtime/session-logs",
                )
            ).expanduser(),
            session_log_ttl_s=max(
                1.0,
                float(
                    os.environ.get(
                        "OMNI_PORTAL_SESSION_LOG_TTL_S",
                        str(DIAGNOSTIC_TTL_SECONDS),
                    )
                ),
            ),
        )


class PortalError(RuntimeError):
    """A safe, user-visible portal failure."""


class PortalRequestError(ValueError):
    """A safe client request validation failure."""


@dataclass
class _InferenceTicket:
    session_id: str
    released: bool = False


class _InferenceQueue:
    """Bounded, session-counted queue around isolated upstream requests."""

    def __init__(self, slots: int, max_inflight: int) -> None:
        self.slots = max(1, slots)
        self.max_inflight = max(self.slots, max_inflight)
        self._gate = threading.BoundedSemaphore(self.slots)
        self._lock = threading.Lock()
        self._inflight = 0
        self._active = 0
        self._sessions: dict[str, int] = {}

    def acquire(self, session_id: str, timeout_s: float) -> _InferenceTicket | None:
        with self._lock:
            if self._inflight >= self.max_inflight:
                return None
            self._inflight += 1
            self._sessions[session_id] = self._sessions.get(session_id, 0) + 1
        if not self._gate.acquire(timeout=timeout_s):
            self._remove_inflight(session_id)
            return None
        with self._lock:
            self._active += 1
        return _InferenceTicket(session_id=session_id)

    def _remove_inflight(self, session_id: str) -> None:
        with self._lock:
            self._inflight -= 1
            remaining = self._sessions.get(session_id, 1) - 1
            if remaining > 0:
                self._sessions[session_id] = remaining
            else:
                self._sessions.pop(session_id, None)

    def release(self, ticket: _InferenceTicket) -> None:
        with self._lock:
            if ticket.released:
                return
            ticket.released = True
            self._active -= 1
        self._gate.release()
        self._remove_inflight(ticket.session_id)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "users": len(self._sessions),
                "inflight": self._inflight,
                "active": self._active,
                "queued": self._inflight - self._active,
                "slots": self.slots,
                "limit": self.max_inflight,
            }


@dataclass
class _DiagnosticSession:
    last_seen: float
    last_seen_at: str
    events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=240))
    request_ids: set[str] = field(default_factory=set)
    timer: threading.Timer | None = None


class _SessionDiagnostics:
    """Short-lived, content-redacted diagnostic journals keyed by session."""

    def __init__(self, directory: Path | None, ttl_s: float) -> None:
        self.directory = directory.resolve() if directory is not None else None
        self.ttl_s = max(0.05, ttl_s)
        self._lock = threading.Lock()
        self._sessions: dict[str, _DiagnosticSession] = {}
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            for stale in self.directory.glob("*.json"):
                stale.unlink(missing_ok=True)
            for stale in self.directory.glob("*.tmp"):
                stale.unlink(missing_ok=True)

    @staticmethod
    def _key(session_id: str) -> str:
        return hashlib.sha256(session_id.encode("utf-8")).hexdigest()

    @staticmethod
    def _now_text() -> str:
        return datetime.now().astimezone().isoformat(timespec="milliseconds")

    def _schedule_locked(self, key: str, record: _DiagnosticSession) -> None:
        if record.timer is not None:
            return
        timer = threading.Timer(self.ttl_s, self._expire, args=(key,))
        timer.daemon = True
        record.timer = timer
        timer.start()

    def _expire(self, key: str) -> None:
        path: Path | None = None
        with self._lock:
            record = self._sessions.get(key)
            if record is None:
                return
            record.timer = None
            remaining = self.ttl_s - (time.monotonic() - record.last_seen)
            if remaining > 0:
                timer = threading.Timer(remaining, self._expire, args=(key,))
                timer.daemon = True
                record.timer = timer
                timer.start()
                return
            self._sessions.pop(key, None)
            if self.directory is not None:
                path = self.directory / f"{key}.json"
        if path is not None:
            path.unlink(missing_ok=True)

    def _persist_locked(self, key: str, record: _DiagnosticSession) -> None:
        if self.directory is None:
            return
        path = self.directory / f"{key}.json"
        temporary = self.directory / f"{key}.tmp"
        payload = {
            "schema": "robit.omni.session-diagnostics.v1",
            "session": key[:16],
            "last_seen_at": record.last_seen_at,
            "ttl_seconds": self.ttl_s,
            "events": list(record.events),
        }
        temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        temporary.replace(path)

    def open_session(self, session_id: str) -> None:
        key = self._key(session_id)
        with self._lock:
            now = time.monotonic()
            record = self._sessions.get(key)
            if record is None:
                record = _DiagnosticSession(now, self._now_text())
                self._sessions[key] = record
                self._schedule_locked(key, record)
            else:
                record.last_seen = now
                record.last_seen_at = self._now_text()

    def touch(self, session_id: str) -> None:
        key = self._key(session_id)
        with self._lock:
            record = self._sessions.get(key)
            if record is None:
                return
            record.last_seen = time.monotonic()
            record.last_seen_at = self._now_text()

    def begin_request(
        self, session_id: str, request_id: str, fields: Mapping[str, Any]
    ) -> None:
        self.open_session(session_id)
        key = self._key(session_id)
        with self._lock:
            record = self._sessions[key]
            record.request_ids.add(request_id)
            self._append_locked(
                key,
                record,
                "request_received",
                {"request_id": request_id, **dict(fields)},
            )

    def record(
        self,
        session_id: str,
        event: str,
        fields: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> bool:
        key = self._key(session_id)
        with self._lock:
            record = self._sessions.get(key)
            if record is None:
                return False
            if request_id is not None and request_id not in record.request_ids:
                return False
            record.last_seen = time.monotonic()
            record.last_seen_at = self._now_text()
            self._append_locked(key, record, event, fields or {})
            return True

    def _append_locked(
        self,
        key: str,
        record: _DiagnosticSession,
        event: str,
        fields: Mapping[str, Any],
    ) -> None:
        item = {"at": self._now_text(), "event": event}
        item.update(_diagnostic_fields(fields))
        record.events.append(item)
        self._persist_locked(key, record)

    def snapshot(self, session_id: str) -> dict[str, Any]:
        key = self._key(session_id)
        with self._lock:
            record = self._sessions.get(key)
            if record is None:
                return {"ttl_seconds": self.ttl_s, "events": []}
            record.last_seen = time.monotonic()
            record.last_seen_at = self._now_text()
            return {
                "ttl_seconds": self.ttl_s,
                "events": copy.deepcopy(list(record.events)),
            }

    def clear(self, session_id: str) -> None:
        key = self._key(session_id)
        with self._lock:
            record = self._sessions.pop(key, None)
            if record is not None and record.timer is not None:
                record.timer.cancel()
        if self.directory is not None:
            (self.directory / f"{key}.json").unlink(missing_ok=True)
            (self.directory / f"{key}.tmp").unlink(missing_ok=True)


def _diagnostic_fields(raw: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in DIAGNOSTIC_NUMERIC_FIELDS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0 or numeric > 86_400_000:
            continue
        result[key] = int(numeric) if key == "status" else round(numeric, 2)
    for key in DIAGNOSTIC_BOOLEAN_FIELDS:
        value = raw.get(key)
        if isinstance(value, bool):
            result[key] = value
    for key in DIAGNOSTIC_STRING_FIELDS:
        value = raw.get(key)
        if not isinstance(value, str):
            continue
        value = value.strip()
        if value and len(value) <= 64 and all(
            character.isalnum() or character in {"-", "_", "."}
            for character in value
        ):
            result[key] = value
    return result


def _request_diagnostic_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    omni = payload.get("omni")
    task = str(omni.get("task") or "chat") if isinstance(omni, Mapping) else "chat"
    messages = payload.get("messages")
    message_items = messages if isinstance(messages, list) else []
    modalities = payload.get("response_modalities")
    output_items = modalities if isinstance(modalities, list) else []
    return {
        "task": task,
        "audio_requested": "audio" in output_items,
        "thinking_requested": payload.get("think") is True,
        "has_audio_input": any(
            isinstance(message, Mapping) and bool(message.get("audios"))
            for message in message_items
        ),
        "has_image_input": any(
            isinstance(message, Mapping) and bool(message.get("images"))
            for message in message_items
        ),
        "has_video_input": any(
            isinstance(message, Mapping) and bool(message.get("videos"))
            for message in message_items
        ),
        "has_document_input": any(
            isinstance(message, Mapping) and bool(message.get("documents"))
            for message in message_items
        ),
    }


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
            "input": [
                "text",
                "microphone",
                "wav",
                "image",
                "video",
                "gif",
                "pdf",
                "docx",
                "utf-8 text/code",
            ],
            "output": ["text", "thinking", "tool_calls", "audio/wav"],
            "tasks": ["chat", "transcribe", "describe", "synthesize"],
            "audio_input": "16 kHz mono PCM16 WAV",
            "audio_understanding": [
                "verbatim speech transcription",
                "environmental sound and acoustic scene analysis",
            ],
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
            message.pop("documents", None)
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
    inference_queue = _InferenceQueue(
        slots=runtime.inference_slots,
        max_inflight=runtime.max_inflight_requests,
    )
    diagnostics = _SessionDiagnostics(
        directory=runtime.session_log_dir,
        ttl_s=runtime.session_log_ttl_s,
    )
    documents = SessionDocumentStore(ttl_s=runtime.session_log_ttl_s)
    app.config["MAX_CONTENT_LENGTH"] = runtime.max_body_bytes

    def authorized() -> bool:
        value = request.headers.get("Authorization", "")
        supplied = value[7:].strip() if value.lower().startswith("bearer ") else ""
        return bool(supplied) and hmac.compare_digest(supplied, runtime.access_token)

    def request_session_id() -> str:
        supplied = str(request.cookies.get(SESSION_COOKIE_NAME) or "").strip()
        if 16 <= len(supplied) <= 128 and all(
            character.isalnum() or character in {"-", "_"} for character in supplied
        ):
            return supplied
        return secrets.token_urlsafe(24)

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

    def apply_runtime_environment(payload: dict[str, Any]) -> None:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise PortalRequestError("messages must be a non-empty array")
        messages.insert(0, runtime_environment_system_message())

    def apply_document_context(
        payload: dict[str, Any], session_id: str
    ) -> list[dict[str, Any]]:
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise PortalRequestError("messages must be a non-empty array")
        last_user: dict[str, Any] | None = None
        uploads: list[Any] = []
        for raw_message in messages:
            if not isinstance(raw_message, dict):
                continue
            raw_documents = raw_message.pop("documents", [])
            if raw_documents:
                if raw_message.get("role") != "user":
                    raise PortalRequestError("documents are accepted only on user messages")
                if not isinstance(raw_documents, list):
                    raise PortalRequestError("message.documents must be an array")
                uploads.extend(raw_documents)
            if raw_message.get("role") == "user":
                last_user = raw_message
        if last_user is None:
            raise PortalRequestError("messages must contain a user message")
        query = str(last_user.get("content") or "").strip()
        try:
            context, accepted = documents.prepare(session_id, uploads, query)
        except DocumentError as exc:
            raise PortalRequestError(str(exc)) from exc
        if context:
            last_user["content"] = (
                f"{context}\n\n<user_request>\n{query}\n</user_request>"
            )
        return accepted

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
        if (
            request.path.startswith("/api/")
            or request.path.startswith("/assets/")
            or request.path == "/"
        ):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "request exceeds the portal upload limit"}), 413

    @app.get("/")
    def index():
        browser_session = request_session_id()
        response = make_response(
            render_template(
                "index.html",
                model=runtime.model,
                max_upload_mib=runtime.max_body_bytes // (1024 * 1024),
                session_scope=hashlib.sha256(
                    f"robit-omni-browser-cache:{browser_session}".encode()
                ).hexdigest(),
            )
        )
        response.set_cookie(
            SESSION_COOKIE_NAME,
            browser_session,
            max_age=24 * 60 * 60,
            secure=True,
            httponly=True,
            samesite="Strict",
            path="/",
        )
        return response

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "service": "robit-omni-portal"})

    @app.get("/api/status")
    def status():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        session_id = request_session_id()
        diagnostics.touch(session_id)
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
                "documents": {
                    "supported": ["pdf", "docx", "utf-8 text/code"],
                    "retrieval": "session-isolated hashed lexical embeddings",
                    **documents.stats(session_id),
                },
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
                "audio_understanding": {
                    "speech_transcription": True,
                    "environmental_sound_analysis": True,
                    "evidence_field": "adapter.audio_observation",
                },
                "runtime_environment": {
                    "refreshed_each_turn": True,
                    "includes": ["date/time", "CPU/load", "RAM", "NVIDIA GPUs", "network counters"],
                    "excludes": ["hostnames", "addresses", "processes", "credentials", "session content"],
                },
                "requests": inference_queue.snapshot(),
            }
        )

    @app.get("/api/activity")
    def activity():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        diagnostics.touch(request_session_id())
        return jsonify(inference_queue.snapshot())

    @app.route("/api/diagnostics", methods=["GET", "POST", "DELETE"])
    def session_diagnostics():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        session_id = request_session_id()
        if request.method == "DELETE":
            diagnostics.clear(session_id)
            documents.clear(session_id)
            return Response(status=204)
        if request.method == "GET":
            return jsonify(diagnostics.snapshot(session_id))
        payload = request.get_json(silent=True)
        if not isinstance(payload, Mapping):
            return jsonify({"error": "diagnostic event must be an object"}), 400
        event = str(payload.get("event") or "").strip()
        if event not in {"client_stream_timing", "page_leave"}:
            return jsonify({"error": "unsupported diagnostic event"}), 400
        request_id = str(payload.get("request_id") or "").strip() or None
        accepted = diagnostics.record(
            session_id,
            event,
            payload,
            request_id=request_id,
        )
        return jsonify({"accepted": accepted})

    @app.post("/api/chat")
    def chat():
        if not authorized():
            return jsonify({"error": "unauthorized"}), 401
        ticket: _InferenceTicket | None = None
        session_id = request_session_id()
        request_id = secrets.token_urlsafe(9)
        started = time.monotonic()
        request_logged = False
        outcome_status = 500
        try:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return jsonify({"error": "request body must be a JSON object"}), 400
            auto_tools = payload.pop("portal_auto_tools", False) is True
            if payload.get("model") != runtime.model:
                return jsonify({"error": "portal model tag is fixed"}), 400
            if payload.get("stream") is not False:
                return jsonify({"error": "portal requires stream=false"}), 400
            diagnostic_fields = _request_diagnostic_fields(payload)
            apply_reasoning_mode(payload)
            apply_voice_profile(payload)
            apply_runtime_environment(payload)
            accepted_documents = apply_document_context(payload, session_id)
            diagnostics.begin_request(
                session_id,
                request_id,
                diagnostic_fields,
            )
            request_logged = True
            queue_started = time.monotonic()
            ticket = inference_queue.acquire(session_id, runtime.timeout_s)
            if ticket is None:
                outcome_status = 503
                return jsonify({"error": "inference queue is full or timed out"}), 503
            diagnostics.record(
                session_id,
                "queue_acquired",
                {
                    "request_id": request_id,
                    "queue_wait_ms": (time.monotonic() - queue_started) * 1000,
                },
                request_id=request_id,
            )

            executed: list[dict[str, str]] = []
            current_payload: dict[str, Any] = payload
            for _round in range(MAX_TOOL_ROUNDS + 1):
                upstream = session.post(runtime.adapter_url, json=current_payload)
                data = _json_object(upstream, "adapter")
                if upstream.status_code >= 400:
                    outcome_status = upstream.status_code
                    response = jsonify(data)
                    response.headers["X-Omni-Request-ID"] = request_id
                    return response, upstream.status_code
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
                "documents_indexed": accepted_documents,
            }
            outcome_status = 200
            response = jsonify(data)
            response.headers["X-Omni-Request-ID"] = request_id
            return response
        except PortalRequestError as exc:
            outcome_status = 400
            return jsonify({"error": str(exc)}), 400
        except (httpx.HTTPError, PortalError) as exc:
            outcome_status = 502
            return jsonify({"error": str(exc)}), 502
        finally:
            if ticket is not None:
                inference_queue.release(ticket)
            if request_logged:
                diagnostics.record(
                    session_id,
                    "request_complete",
                    {
                        "request_id": request_id,
                        "status": outcome_status,
                        "total_ms": (time.monotonic() - started) * 1000,
                    },
                    request_id=request_id,
                )

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
        session_id = request_session_id()
        diagnostic_fields = _request_diagnostic_fields(payload)
        try:
            apply_reasoning_mode(payload)
            apply_voice_profile(payload)
            apply_runtime_environment(payload)
            apply_document_context(payload, session_id)
        except PortalRequestError as exc:
            return jsonify({"error": str(exc)}), 400
        request_id = secrets.token_urlsafe(9)
        started = time.monotonic()
        diagnostics.begin_request(
            session_id,
            request_id,
            diagnostic_fields,
        )
        queue_started = time.monotonic()
        ticket = inference_queue.acquire(session_id, runtime.timeout_s)
        if ticket is None:
            diagnostics.record(
                session_id,
                "request_complete",
                {
                    "request_id": request_id,
                    "status": 503,
                    "total_ms": (time.monotonic() - started) * 1000,
                },
                request_id=request_id,
            )
            return jsonify({"error": "inference queue is full or timed out"}), 503
        diagnostics.record(
            session_id,
            "queue_acquired",
            {
                "request_id": request_id,
                "queue_wait_ms": (time.monotonic() - queue_started) * 1000,
            },
            request_id=request_id,
        )

        try:
            upstream_request = session.build_request(
                "POST", runtime.adapter_url.rstrip("/") + "/stream", json=payload
            )
            upstream = session.send(upstream_request, stream=True)
            diagnostics.record(
                session_id,
                "upstream_headers",
                {
                    "request_id": request_id,
                    "status": upstream.status_code,
                    "upstream_headers_ms": (time.monotonic() - started) * 1000,
                },
                request_id=request_id,
            )
        except httpx.HTTPError as exc:
            inference_queue.release(ticket)
            diagnostics.record(
                session_id,
                "request_complete",
                {
                    "request_id": request_id,
                    "status": 502,
                    "outcome": "upstream_error",
                    "total_ms": (time.monotonic() - started) * 1000,
                },
                request_id=request_id,
            )
            return jsonify({"error": str(exc)}), 502

        def relay():
            first_byte = True
            try:
                for chunk in upstream.iter_bytes():
                    if first_byte:
                        first_byte = False
                        diagnostics.record(
                            session_id,
                            "first_upstream_byte",
                            {
                                "request_id": request_id,
                                "first_upstream_byte_ms": (
                                    time.monotonic() - started
                                )
                                * 1000,
                            },
                            request_id=request_id,
                        )
                    yield chunk
            finally:
                upstream.close()
                inference_queue.release(ticket)
                diagnostics.record(
                    session_id,
                    "request_complete",
                    {
                        "request_id": request_id,
                        "status": upstream.status_code,
                        "total_ms": (time.monotonic() - started) * 1000,
                    },
                    request_id=request_id,
                )

        return Response(
            stream_with_context(relay()),
            status=upstream.status_code,
            content_type=upstream.headers.get(
                "content-type", "application/x-ndjson; charset=utf-8"
            ),
            headers={
                "X-Accel-Buffering": "no",
                "X-Omni-Request-ID": request_id,
            },
        )

    return app


if __name__ == "__main__":
    create_app().run(
        host=os.environ.get("OMNI_PORTAL_HOST", "127.0.0.1"),
        port=int(os.environ.get("OMNI_PORTAL_PORT", "8920")),
        debug=False,
        threaded=True,
    )
