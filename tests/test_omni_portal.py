from __future__ import annotations

import json

import httpx

from examples.omni_portal.app import DEFAULT_MODEL, PortalConfig, create_app

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
    assert b"Multimodal field console" in response.data
    assert TOKEN.encode() not in response.data
    assert "microphone=(self)" in response.headers["Permissions-Policy"]
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Referrer-Policy"] == "no-referrer"


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
