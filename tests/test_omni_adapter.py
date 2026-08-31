from __future__ import annotations

import base64
import io
import json
import wave
from pathlib import Path

import httpx
import pytest

from examples.omni_adapter.server import Config, build_comprehension_payload, execute
from training_suite.models.omni_adapter import (
    ADAPTER_SCHEMA,
    OmniAdapterError,
    adapter_contract,
    parse_adapter_request,
)


def _wav(sample_rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * 160)
    return out.getvalue()


def _encoded(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _mp4() -> bytes:
    return b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"


def _base_request(**overrides):
    request = {
        "model": "robit/qwen3.8-omni:latest",
        "messages": [{"role": "user", "content": "What happened?"}],
        "omni": {"schema": ADAPTER_SCHEMA, "task": "chat"},
        "response_modalities": ["text"],
        "speech_mode": "auto",
        "think": True,
        "stream": False,
    }
    request.update(overrides)
    return request


def test_adapter_contract_separates_wire_schema_from_bundle_schema() -> None:
    contract = adapter_contract()

    assert contract["schema"] == ADAPTER_SCHEMA
    assert contract["transport"]["streaming_v1"] is False
    assert contract["compatibility"]["message_extensions"] == ["audios", "videos"]
    assert contract["media"]["video"]["max_items"] == 4


def test_adapter_json_schemas_are_valid_json_and_use_v1_identifier() -> None:
    schema_dir = Path("docs/omni-adapter/schema")
    request_schema = json.loads((schema_dir / "request-v1.schema.json").read_text())
    response_schema = json.loads((schema_dir / "response-v1.schema.json").read_text())

    assert (
        request_schema["properties"]["omni"]["properties"]["schema"]["const"]
        == ADAPTER_SCHEMA
    )
    assert (
        response_schema["properties"]["adapter"]["properties"]["schema"]["const"]
        == ADAPTER_SCHEMA
    )


def test_chat_request_routes_video_and_audio_through_all_three_stages() -> None:
    request = _base_request(
        messages=[
            {
                "role": "user",
                "content": "What was said and shown?",
                "audios": [
                    {
                        "mime_type": "audio/wav",
                        "encoding": "base64",
                        "data": _encoded(_wav(16000)),
                    }
                ],
                "videos": [
                    {
                        "mime_type": "video/mp4",
                        "encoding": "base64",
                        "data": _encoded(_mp4()),
                        "sampling": {"fps": 2, "max_frames": 64, "include_audio": True},
                    }
                ],
            }
        ],
        response_modalities=["text", "audio"],
    )

    parsed = parse_adapter_request(request)

    assert parsed.route == ("comprehension", "language", "tts")
    assert parsed.input_modalities == ("text", "audio", "video")
    assert parsed.media[1].options["max_frames"] == 64
    assert parsed.passthrough["think"] is True


def test_stock_ollama_bare_base64_image_is_detected_by_signature() -> None:
    png = b"\x89PNG\r\n\x1a\n" + b"example"
    request = _base_request(
        messages=[
            {
                "role": "user",
                "content": "Describe this image.",
                "images": [_encoded(png)],
            }
        ]
    )

    parsed = parse_adapter_request(request)

    assert parsed.media[0].mime_type == "image/png"
    assert parsed.route == ("comprehension", "language")


@pytest.mark.parametrize(
    ("task", "message", "route"),
    [
        (
            "transcribe",
            {
                "role": "user",
                "content": "Transcribe.",
                "audios": [{"data": _encoded(_wav(16000))}],
            },
            ("comprehension",),
        ),
        (
            "describe",
            {
                "role": "user",
                "content": "Describe.",
                "videos": [{"data": _encoded(_mp4()), "mime_type": "video/mp4"}],
            },
            ("comprehension",),
        ),
        (
            "synthesize",
            {"role": "user", "content": "Read this exactly."},
            ("tts",),
        ),
    ],
)
def test_direct_tasks_select_one_component(task, message, route) -> None:
    request = _base_request(
        messages=[message],
        omni={"schema": ADAPTER_SCHEMA, "task": task},
    )

    assert parse_adapter_request(request).route == route


def test_adapter_rejects_streaming_and_spoofed_video_mime() -> None:
    with pytest.raises(OmniAdapterError, match="stream=false"):
        parse_adapter_request(_base_request(stream=True))

    request = _base_request(
        messages=[
            {
                "role": "user",
                "content": "Describe.",
                "videos": [
                    {
                        "mime_type": "video/webm",
                        "data": _encoded(_mp4()),
                    }
                ],
            }
        ]
    )
    with pytest.raises(OmniAdapterError, match="does not match"):
        parse_adapter_request(request)


def test_reference_server_preserves_tools_thinking_and_adds_audio() -> None:
    output_wav = _wav(24000)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        body = json.loads(request.content)
        if request.url.host == "comprehension":
            part = body["messages"][-1]["content"][0]
            assert part["type"] == "audio_url"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "the user said hello"}}]},
            )
        if request.url.host == "language":
            assert body["tools"][0]["function"]["name"] == "clock"
            assert "untrusted semantic output" in body["messages"][-1]["content"]
            return httpx.Response(
                200,
                json={
                    "model": body["model"],
                    "message": {
                        "role": "assistant",
                        "content": "Hello back.",
                        "thinking": "brief thought",
                    },
                    "done": True,
                },
            )
        if request.url.host == "tts":
            assert body["text"] == "Hello back."
            assert body["voice"] == "speaker-1"
            return httpx.Response(
                200, content=output_wav, headers={"content-type": "audio/wav"}
            )
        return httpx.Response(404)

    request = _base_request(
        messages=[
            {
                "role": "user",
                "content": "Reply to this recording.",
                "audios": [{"data": _encoded(_wav(16000))}],
            }
        ],
        response_modalities=["text", "audio"],
        speech={"voice": "speaker-1"},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "clock",
                    "description": "get time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    parsed = parse_adapter_request(request)
    config = Config(
        comprehension_url="http://comprehension/v1/chat/completions",
        comprehension_model="qwen3-omni",
        language_url="http://language",
        tts_url="http://tts/synthesize",
        timeout_s=30,
    )

    result = execute(
        parsed,
        config,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert seen == ["comprehension", "language", "tts"]
    assert result["message"]["thinking"] == "brief thought"
    assert base64.b64decode(result["message"]["audio"]["data"]) == output_wav
    assert result["adapter"]["route"] == ["comprehension", "language", "tts"]


def test_comprehension_payload_tags_video_for_qwen_style_server() -> None:
    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Describe.",
                    "videos": [{"mime_type": "video/mp4", "data": _encoded(_mp4())}],
                }
            ],
            omni={"task": "describe", "include_audio_from_video": False},
        )
    )
    payload = build_comprehension_payload(
        parsed,
        Config("http://comp", "omni", "http://ollama", "http://tts", 30),
    )

    assert payload["messages"][-1]["content"][0]["type"] == "video_url"
    assert payload["mm_processor_kwargs"]["use_audio_in_video"] is False
