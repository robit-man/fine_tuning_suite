from __future__ import annotations

import base64
import io
import json
import wave
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
    assert b'id="think-toggle"' in response.data
    assert b'id="call-button"' in response.data
    assert b'id="camera-button"' in response.data
    assert b'id="camera-video"' in response.data
    assert b'id="voice-button"' in response.data
    assert b'id="voice-clone-enabled"' in response.data
    assert b'id="voice-reference-input"' in response.data
    assert b'href="/assets/favicon.svg"' in response.data
    assert response.data.index(b'id="camera-button"') < response.data.index(
        b'id="call-button"'
    )
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
    assert "function streamChat" in javascript
    assert "CALL_BARGE_THRESHOLD" in javascript
    assert "think: wantsThinking" in javascript
    assert "think: showThinking" in javascript
    assert "built.wantsThinking" in javascript
    assert "max_frames: 24" in javascript
    assert "function voicePayload" in javascript
    assert "function startVoiceReferenceRecording" in javascript
    assert 'elements.prompt.value = ""' in javascript
    assert ".composer textarea:focus" in css
    assert "box-shadow: none" in css
    assert "user-select: none" in css
    assert "-webkit-touch-callout: none" in css
    assert "event.transcript" in javascript
    assert "(data.adapter || {}).input_transcript" in javascript
    assert '(data.adapter || {}).observation || "Voice message"' not in javascript
    assert "assistant.node.hidden = true" in javascript
    assert ".message[hidden]" in css
    assert 'addFile(file, "video", "camera")' in javascript
    assert 'value.source !== "camera"' in javascript
    assert "function loopingVideo" in javascript
    assert "video.loop = true" in javascript
    assert "media: sentMedia" in javascript
    assert ".message-video-preview" in css
    assert "const hasMedia = state.attachments.length > 0" in javascript
    assert 'task === "chat" && !hasMedia' in javascript
    assert "if (built.hasMedia) state.history = []" in javascript
    assert "const callMessages = frame" in javascript
    assert "if (frame) state.history = []" in javascript


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
    assert response.json["voice_profile"]["clone_mode"] == "speaker_embedding"
    assert response.json["audio_understanding"] == {
        "speech_transcription": True,
        "environmental_sound_analysis": True,
        "evidence_field": "adapter.audio_observation",
    }
    assert response.json["voice_profile"]["client_reference_wav"] is True


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


def test_portal_defaults_reasoning_off_and_requires_boolean() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "Answer."}}
        )

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()
    headers = {"Authorization": f"Bearer {TOKEN}"}
    without_think = _request()
    without_think.pop("think")

    defaulted = client.post("/api/chat", headers=headers, json=without_think)
    invalid = client.post("/api/chat", headers=headers, json=_request(think="yes"))

    assert defaulted.status_code == 200
    assert seen[0]["think"] is False
    assert invalid.status_code == 400


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


def test_bundled_default_voice_is_metadata_free_pcm() -> None:
    profile_path = Path("examples/omni_portal/voice-profile.json")
    profile = load_voice_profile(profile_path)
    voice_path = Path(profile["speaker_file"])
    raw = voice_path.read_bytes()

    assert voice_path.name == "default_voice.wav"
    assert raw[:4] == b"RIFF"
    assert raw[8:12] == b"WAVE"
    assert raw[12:16] == b"fmt "
    assert raw[36:40] == b"data"
    with wave.open(str(voice_path), "rb") as wav:
        assert wav.getcomptype() == "NONE"
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert 500 <= round(wav.getnframes() * 1000 / wav.getframerate()) <= 30000


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


def test_portal_accepts_safe_client_voice_clone_and_controls() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "Hello."}}
        )

    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    reference = {
        "mime_type": "audio/wav",
        "encoding": "base64",
        "data": base64.b64encode(wav_bytes.getvalue()).decode("ascii"),
    }
    app = create_app(
        _config(voice_profile={"name": "default", "language": "en", "seed": 42}),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            portal_voice={
                "clone_enabled": True,
                "speaker_audio": reference,
                "language": "ja",
                "temperature": 0.55,
                "top_k": 24,
                "top_p": 0.8,
                "seed": 7,
                "max_frames": 384,
            },
            speech={"speaker_file": "/tmp/untrusted.wav"},
        ),
    )

    assert response.status_code == 200
    assert seen[0]["speech"]["language"] == "ja"
    assert seen[0]["speech"]["temperature"] == 0.55
    assert seen[0]["speech"]["seed"] == 7
    assert "speaker_file" not in seen[0]["speech"]
    assert seen[0]["speech"]["speaker_audio"]["data"] == reference["data"]
    assert "portal_voice" not in seen[0]


def test_portal_rejects_invalid_voice_clone_before_proxy() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    response = app.test_client().post(
        "/api/chat",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(
            portal_voice={
                "clone_enabled": True,
                "speaker_audio": {
                    "mime_type": "audio/wav",
                    "encoding": "base64",
                    "data": base64.b64encode(b"not a wav").decode("ascii"),
                },
            }
        ),
    )

    assert response.status_code == 400
    assert "invalid voice reference" in response.json["error"]
    assert calls == 0


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


def test_portal_stream_route_pins_profile_and_relays_ndjson() -> None:
    seen = []
    wire = (
        b'{"type":"delta","message":{"content":"Hi"}}\n'
        b'{"type":"final","response":{"message":{"content":"Hi"}}}\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        return httpx.Response(
            200,
            content=wire,
            headers={"content-type": "application/x-ndjson"},
        )

    profile = {"name": "fixed", "language": "en", "seed": 42}
    app = create_app(
        _config(voice_profile=profile),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    response = app.test_client().post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(stream=True, speech={"seed": -1}),
    )

    assert response.status_code == 200
    assert response.data == wire
    assert seen[0][0] == "/api/chat/stream"
    assert seen[0][1]["stream"] is True
    assert seen[0][1]["think"] is True
    assert seen[0][1]["speech"] == {"language": "en", "seed": 42}


def test_portal_stream_route_requires_auth_and_disables_auto_tools() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    app = create_app(_config(), httpx.Client(transport=httpx.MockTransport(handler)))
    client = app.test_client()

    assert client.post("/api/chat/stream", json=_request(stream=True)).status_code == 401
    rejected = client.post(
        "/api/chat/stream",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json=_request(stream=True, portal_auto_tools=True),
    )
    assert rejected.status_code == 400
    assert calls == 0
