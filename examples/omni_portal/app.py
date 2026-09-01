"""Authenticated, phone-first web portal for the Robit Omni adapter.

The portal is intentionally a narrow same-origin proxy. It never exposes the
component workers or Ollama directly, pins requests to one published model,
and executes only two read-only demonstration tools from an explicit allowlist.
"""

from __future__ import annotations

import copy
import hmac
import json
import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from flask import Flask, jsonify, render_template, request

ADAPTER_SCHEMA = "robit.ollama.omni-adapter.v1"
DEFAULT_MODEL = "robit/qwen3.8-27b-e03-obliterated-omni:q4km"
MAX_TOOL_ROUNDS = 2

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


@dataclass(frozen=True)
class PortalConfig:
    adapter_url: str
    adapter_health_url: str
    comprehension_health_url: str
    tts_health_url: str
    ollama_health_url: str
    model: str
    access_token: str
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
            timeout_s=float(os.environ.get("OMNI_PORTAL_TIMEOUT_S", "1200")),
            max_body_bytes=int(
                os.environ.get("OMNI_PORTAL_MAX_BODY_BYTES", str(96 * 1024 * 1024))
            ),
        )


class PortalError(RuntimeError):
    """A safe, user-visible portal failure."""


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
        except (httpx.HTTPError, PortalError) as exc:
            return jsonify({"error": str(exc)}), 502
        finally:
            inference_gate.release()

    return app


if __name__ == "__main__":
    create_app().run(
        host=os.environ.get("OMNI_PORTAL_HOST", "127.0.0.1"),
        port=int(os.environ.get("OMNI_PORTAL_PORT", "8920")),
        debug=False,
        threaded=True,
    )

