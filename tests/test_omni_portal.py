from __future__ import annotations

import json
from pathlib import Path

import httpx

from examples.omni_portal.app import (
    DEFAULT_MODEL,
    PortalConfig,
    create_app,
    load_voice_profile,
)

TOKEN = "portal-test-token-with-more-than-24-characters"


def _config(**overrides) -> PortalConfig:
    values = {
        "adapter_url": "http://adapter/api/chat",
        "adapter_health_url": "http://adapter/healthz",
        "comprehension_health_url": "http://comprehension/health",
        "tts_health_url": "http://tts/healthz",
        "ollama_health_url": "http://ollama/api/tags",
        "model": DEFAULT_MODEL,
        "access_token": TOKEN,
        "timeout_s": 30,
        "max_body_bytes": 1024 * 1024,
    }
    values.update(overrides)
    return PortalConfig(**values)


def _request(**overrides):
    body = {
        "model": DEFAULT_MODEL,
        "messages": [{"role": "user", "content": "Hello"}],
        "omni": {"schema": "robit.ollama.omni-adapter.v1", "task": "chat"},
        "response_modalities": ["text"],
        "speech_mode": "never",
        "think": True,
        "stream": False,
    }
    body.update(overrides)
    return body


def test_portal_index_has_mobile_security_headers_and_no_token() -> None:
    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200))))
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert b"Omni Chat" in response.data
    assert b"ROBIT" not in response.data
    assert b'id="waveform-canvas"' in response.data
    assert b'id="speak-toggle"' in response.data
    assert b'id="call-button"' in response.data
    assert b'id="camera-button"' in response.data
    assert b'id="camera-video"' in response.data
    assert b'aria-pressed="false"' in response.data
    assert b'maximum-scale=1' in response.data
    assert b'user-scalable=no' in response.data
    assert TOKEN.encode() not in response.data
    assert "microphone=(self)" in response.headers["Permissions-Policy"]
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_portal_assets_include_markdown_call_flow_and_neutral_composer() -> None:
    javascript = Path("examples/omni_portal/static/portal.js").read_text()
    css = Path("examples/omni_portal/static/portal.css").read_text()

    assert "function renderMarkdown" in javascript
    assert "function startCall" in javascript
    assert "function submitCallUtterance" in javascript
    assert "function startCameraCapture" in javascript
    assert "function stopCameraCapture" in javascript
    assert 'elements.prompt.value = ""' in javascript
    assert ".composer textarea:focus" in css
    assert "box-shadow: none" in css


def test_portal_api_requires_bearer_token() -> None:
    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200))))
    client = app.test_client()

    assert client.get("/api/status").status_code == 401
    assert client.post("/api/chat", json=_request()).status_code == 401


def test_portal_status_probes_all_internal_stages() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(200, json={"ok": True})

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().get(
        "/api/status", headers={"Authorization": f"Bearer {TOKEN}"}
    )

    assert response.status_code == 200
    assert response.json["ok"] is True
    assert set(seen) == {"adapter", "comprehension", "tts", "ollama"}
    assert response.json["model"] == DEFAULT_MODEL


def test_portal_pins_model_and_proxies_normal_response() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": DEFAULT_MODEL,
                "message": {"role": "assistant", "content": "Hello back."},
                "adapter": {"route": ["language"]},
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    headers = {"Authorization": f"Bearer {TOKEN}"}

    bad = client.post("/api/chat", headers=headers, json=_request(model="other"))
    good = client.post("/api/chat", headers=headers, json=_request())

    assert bad.status_code == 400
    assert good.status_code == 200
    assert good.json["message"]["content"] == "Hello back."
    assert good.json["portal"]["safe_tools_executed"] == []
    assert seen[0]["model"] == DEFAULT_MODEL


def test_voice_profile_resolves_relative_speaker_and_validates_language(
    tmp_path,
) -> None:
    speaker = tmp_path / "reference.wav"
    speaker.write_bytes(b"RIFF")
    profile_path = tmp_path / "voice.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "robit.omni.voice-profile.v1",
                "name": "studio",
                "language": "en",
                "speaker_file": "reference.wav",
                "temperature": 0.5,
                "seed": 7,
            }
        )
    )

    profile = load_voice_profile(profile_path)

    assert profile["speaker_file"] == str(speaker.resolve())
    assert profile["seed"] == 7

    profile_path.write_text(
        json.dumps(
            {
                "schema": "robit.omni.voice-profile.v1",
                "language": "unsupported",
            }
        )
    )
    try:
        load_voice_profile(profile_path)
    except RuntimeError as exc:
        assert "language must be one of" in str(exc)
    else:
        raise AssertionError("unsupported TTS language was accepted")


def test_portal_enforces_server_voice_profile() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "Hello."}},
        )

    profile = {
        "name": "fixed-voice",
        "language": "en",
        "speaker_file": "/srv/voices/fixed.wav",
        "temperature": 0.4,
        "top_k": 20,
        "top_p": 0.8,
        "seed": 42,
        "max_frames": 512,
    }
    app = create_app(
        _config(voice_profile=profile),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(speech={"speaker_file": "/tmp/client-choice.wav", "seed": -1}),
    )

    assert response.status_code == 200
    assert seen[0]["speech"] == {
        key: value for key, value in profile.items() if key != "name"
    }


def test_portal_executes_only_allowlisted_tool_and_strips_media_on_followup() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "thinking": "tool needed",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_current_time",
                                    "arguments": {},
                                },
                            }
                        ],
                    },
                    "adapter": {"route": ["comprehension", "language"]},
                },
            )
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "It is now test time."},
                "adapter": {"route": ["language"]},
            },
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    body = _request(
        messages=[
            {
                "role": "user",
                "content": "Use the clock.",
                "images": [{"mime_type": "image/png", "data": "unused-in-mock"}],
            }
        ],
        portal_auto_tools=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=body,
    )

    assert response.status_code == 200
    assert len(requests) == 2
    assert "portal_auto_tools" not in requests[0]
    assert "images" not in requests[1]["messages"][0]
    tool_result = requests[1]["messages"][-1]
    assert tool_result["role"] == "tool"
    assert tool_result["tool_name"] == "get_current_time"
    assert response.json["portal"]["safe_tools_executed"][0]["name"] == "get_current_time"


def test_portal_rejects_streaming_before_proxy() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(stream=True),
    )

    assert response.status_code == 400
    assert calls == 0
