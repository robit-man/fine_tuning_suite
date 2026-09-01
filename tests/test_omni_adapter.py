from __future__ import annotations

import base64
import io
import json
import os
import wave
from pathlib import Path

import httpx
import pytest

from examples.omni_adapter.client import printable_response
from examples.omni_adapter.server import (
    Config,
    build_comprehension_payload,
    execute,
    execute_stream,
)
from examples.omni_adapter.tts_server import Config as TTSConfig
from examples.omni_adapter.tts_server import TTSError, _command, _synthesis_spec
from examples.omni_adapter.tts_server import create_app as create_tts_app
from training_suite.models.audio import decode_wav_payload
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


def _tts_config(tmp_path: Path, **overrides) -> TTSConfig:
    values = {
        "binary": tmp_path / "llama-tts",
        "model": tmp_path / "model.gguf",
        "projector": tmp_path / "projector.gguf",
        "gpu_layers": 0,
    }
    values.update(overrides)
    return TTSConfig(**values)


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


def test_tts_stream_window_validation_and_cli_arguments(tmp_path: Path) -> None:
    config = _tts_config(tmp_path)
    spec = _synthesis_spec(config, {"text": "Hello", "stream_frames": 12})
    command = _command(config, spec, tmp_path / "speech.wav", stream=True)

    assert command[-3:] == ["--tts-stream", "--tts-stream-frames", "12"]
    with pytest.raises(TTSError, match="between 1 and 72"):
        _synthesis_spec(config, {"text": "Hello", "stream_frames": 0})
    with pytest.raises(TTSError, match="between 1 and 72"):
        _synthesis_spec(config, {"text": "Hello", "stream_frames": 73})


def test_tts_accepts_bounded_wav_speaker_envelope(tmp_path: Path) -> None:
    reference = io.BytesIO()
    with wave.open(reference, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000)
    envelope = {
        "mime_type": "audio/wav",
        "encoding": "base64",
        "data": _encoded(reference.getvalue()),
    }

    spec = _synthesis_spec(
        _tts_config(tmp_path), {"text": "Clone this voice.", "speaker_audio": envelope}
    )

    assert spec.speaker == ""
    assert spec.speaker_audio == reference.getvalue()
    with pytest.raises(TTSError, match="mutually exclusive"):
        _synthesis_spec(
            _tts_config(tmp_path),
            {
                "text": "No ambiguity.",
                "speaker_file": "/trusted/reference.wav",
                "speaker_audio": envelope,
            },
        )


def test_tts_http_stream_is_header_tagged_incremental_pcm(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "fake-llama-tts"
    binary.write_text(
        """#!/usr/bin/env python3
import sys
import wave

args = sys.argv[1:]
output = args[args.index("--output") + 1]
with wave.open(output, "wb") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(24000)
    wav.writeframes(bytes([1, 2, 3, 4]))
sys.stdout.buffer.write(bytes([1, 2]))
sys.stdout.buffer.flush()
sys.stdout.buffer.write(bytes([3, 4]))
sys.stdout.buffer.flush()
"""
    )
    os.chmod(binary, 0o755)
    config = _tts_config(tmp_path, binary=binary, stream_frames=12)
    response = create_tts_app(config).test_client().post(
        "/synthesize/stream",
        json={"text": "Hello"},
    )

    assert response.status_code == 200
    assert response.data == bytes([1, 2, 3, 4])
    assert response.headers["X-Audio-Codec"] == "pcm_s16le"
    assert response.headers["X-Audio-Sample-Rate"] == "24000"
    assert response.headers["X-Audio-Channels"] == "1"
    assert response.headers["X-Audio-Stream-Frames"] == "12"
    assert response.headers["Cache-Control"] == "no-store"


def test_example_client_redacts_audio_base64_by_default() -> None:
    response = {"message": {"audio": {"data": "YWJj", "decoded_bytes": 3}}}

    printable = printable_response(response, include_audio_base64=False)

    assert printable["message"]["audio"]["data"] == "<base64 omitted; 4 characters>"
    assert response["message"]["audio"]["data"] == "YWJj"
    assert printable_response(response, include_audio_base64=True) is response


def test_adapter_json_schemas_are_valid_json_and_use_v1_identifier() -> None:
    schema_dir = Path("docs/omni-adapter/schema")
    request_schema = json.loads((schema_dir / "request-v1.schema.json").read_text())
    response_schema = json.loads((schema_dir / "response-v1.schema.json").read_text())
    voice_schema = json.loads(
        (schema_dir / "voice-profile-v1.schema.json").read_text()
    )

    assert (
        request_schema["properties"]["omni"]["properties"]["schema"]["const"]
        == ADAPTER_SCHEMA
    )
    assert (
        response_schema["properties"]["adapter"]["properties"]["schema"]["const"]
        == ADAPTER_SCHEMA
    )
    assert (
        voice_schema["properties"]["schema"]["const"]
        == "robit.omni.voice-profile.v1"
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


def test_transcribe_with_spoken_output_routes_directly_to_tts() -> None:
    request = _base_request(
        messages=[
            {
                "role": "user",
                "content": "Transcribe.",
                "audios": [{"data": _encoded(_wav(16000))}],
            }
        ],
        omni={"schema": ADAPTER_SCHEMA, "task": "transcribe"},
        response_modalities=["text", "audio"],
        speech_mode="always",
    )

    assert parse_adapter_request(request).route == ("comprehension", "tts")


def test_transcribe_with_spoken_output_executes_tts() -> None:
    output_wav = _wav(24000)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "comprehension":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello from audio"}}]},
            )
        if request.url.host == "tts":
            assert json.loads(request.content)["text"] == "hello from audio"
            return httpx.Response(
                200, content=output_wav, headers={"content-type": "audio/wav"}
            )
        return httpx.Response(404)

    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Transcribe.",
                    "audios": [{"data": _encoded(_wav(16000))}],
                }
            ],
            omni={"schema": ADAPTER_SCHEMA, "task": "transcribe"},
            response_modalities=["text", "audio"],
            speech_mode="always",
        )
    )
    config = Config(
        "http://comprehension/v1/chat/completions",
        "qwen3-omni",
        "http://language",
        "http://tts/synthesize",
        30,
    )

    result = execute(
        parsed,
        config,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert seen == ["comprehension", "tts"]
    assert result["message"]["content"] == "hello from audio"
    assert result["adapter"]["speech_synthesized"] is True
    assert base64.b64decode(result["message"]["audio"]["data"]) == output_wav


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
            assert part["type"] == "input_audio"
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


def test_language_backend_override_preserves_logical_model_identity() -> None:
    logical_model = "robit/combined-omni:q4km"
    core_model = "robit/core-language:27b"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == core_model
        return httpx.Response(
            200,
            json={
                "model": core_model,
                "message": {"role": "assistant", "content": "Hello."},
                "done": True,
            },
        )

    parsed = parse_adapter_request(_base_request(model=logical_model))
    config = Config(
        "http://comprehension/v1/chat/completions",
        "qwen3-omni",
        "http://language",
        "http://tts/synthesize",
        30,
        language_model=core_model,
    )

    result = execute(
        parsed,
        config,
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert result["model"] == logical_model
    assert result["adapter"]["language_backend_model"] == core_model


def test_comprehension_payload_tags_video_for_qwen_style_server() -> None:
    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Describe.",
                    "videos": [
                        {
                            "mime_type": "video/mp4",
                            "data": _encoded(_mp4()),
                            "sampling": {"fps": 8, "max_frames": 96},
                        }
                    ],
                }
            ],
            omni={"task": "describe", "include_audio_from_video": False},
        )
    )
    payload = build_comprehension_payload(
        parsed,
        Config("http://comp", "omni", "http://ollama", "http://tts", 30),
    )

    assert payload["messages"][-1]["content"][0]["type"] == "input_video"
    assert payload["messages"][-1]["content"][0]["sampling"] == {
        "fps": 2.0,
        "max_frames": 32,
    }
    assert payload["mm_processor_kwargs"]["use_audio_in_video"] is False
    assert payload["max_tokens"] == 2048


def test_video_context_overflow_retries_with_lower_frame_cap() -> None:
    seen_caps = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        cap = body["messages"][-1]["content"][0]["sampling"]["max_frames"]
        seen_caps.append(cap)
        if len(seen_caps) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "type": "exceed_context_size_error",
                        "message": "request exceeds the available context size",
                    }
                },
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "video understood"}}]}
        )

    parsed = parse_adapter_request(
        _base_request(
            messages=[
                {
                    "role": "user",
                    "content": "Describe.",
                    "videos": [
                        {
                            "mime_type": "video/mp4",
                            "data": _encoded(_mp4()),
                            "sampling": {"fps": 1, "max_frames": 24},
                        }
                    ],
                }
            ],
            omni={"task": "describe", "include_audio_from_video": False},
        )
    )
    result = execute(
        parsed,
        Config("http://comp", "omni", "http://ollama", "http://tts", 30),
        httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert seen_caps == [24, 16]
    assert result["message"]["content"] == "video understood"


def test_reference_server_streams_thinking_and_text_then_final_response() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=(
                b'{"message":{"role":"assistant","thinking":"brief "},"done":false}\n'
                b'{"message":{"role":"assistant","content":"Hello "},"done":false}\n'
                b'{"message":{"role":"assistant","content":"back."},"done":true}\n'
            ),
            headers={"content-type": "application/x-ndjson"},
        )

    parsed = parse_adapter_request(_base_request())
    config = Config(
        "http://comprehension/v1/chat/completions",
        "qwen3-omni",
        "http://language",
        "http://tts/synthesize",
        30,
    )

    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            config,
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    assert seen[0]["stream"] is True
    assert seen[0]["think"] is True
    assert [event["type"] for event in events] == [
        "stage",
        "delta",
        "delta",
        "delta",
        "final",
    ]
    final = events[-1]["response"]
    assert final["message"]["content"] == "Hello back."
    assert final["message"]["thinking"] == "brief "
    assert final["adapter"]["text_streamed"] is True
    assert final["adapter"]["audio_streamed"] is False


def test_reference_server_streams_pcm_and_keeps_final_wav_envelope() -> None:
    pcm = b"\x01\x00\x02\x00\x03\x00"
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, json.loads(request.content)))
        if request.url.host == "language":
            return httpx.Response(
                200,
                content=(
                    b'{"message":{"role":"assistant","content":"Speak."},'
                    b'"done":true}\n'
                ),
            )
        if request.url.host == "tts":
            return httpx.Response(
                200,
                content=pcm,
                headers={"x-audio-codec": "pcm_s16le"},
            )
        return httpx.Response(404)

    parsed = parse_adapter_request(
        _base_request(
            response_modalities=["text", "audio"],
            speech_mode="always",
        )
    )
    config = Config(
        "http://comprehension/v1/chat/completions",
        "qwen3-omni",
        "http://language",
        "http://tts/synthesize",
        30,
    )

    events = [
        json.loads(chunk)
        for chunk in execute_stream(
            parsed,
            config,
            httpx.Client(transport=httpx.MockTransport(handler)),
        )
    ]

    assert [path for path, _body in seen] == ["/api/chat", "/synthesize/stream"]
    assert [event["type"] for event in events] == [
        "stage",
        "delta",
        "stage",
        "audio_start",
        "audio_delta",
        "audio_end",
        "final",
    ]
    streamed = base64.b64decode(events[4]["audio"]["data"])
    assert streamed == pcm
    final = events[-1]["response"]
    decoded = decode_wav_payload(final["message"]["audio"])
    assert decoded.sample_rate_hz == 24000
    assert decoded.frames == 3
    assert final["adapter"]["audio_streamed"] is True
    assert final["adapter"]["route"] == ["language", "tts"]
