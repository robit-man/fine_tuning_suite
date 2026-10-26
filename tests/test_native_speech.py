from __future__ import annotations

import io
import json
import struct
import wave
from pathlib import Path

import httpx
import pytest

from training_suite.evals.speech import ollama_speech_smoke
from training_suite.models.ollama import parse_ollama_show


def test_native_speech_client_matches_official_api_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/ollama-speech-0.40.11.json").read_text())
    seen = []

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["speech"]})
        seen.append(request)
        body = json.loads(request.content)
        assert request.url.path == fixture["endpoint"]
        assert set(body) <= set(fixture["request_fields"])
        assert set(fixture["required_fields"]) <= set(body)
        assert body["voice"] == "reference-fit"
        assert body["response_format"] in fixture["response_formats"]
        return httpx.Response(200, content=_wav(), headers={"Content-Type": "audio/wav"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = ollama_speech_smoke("phonon-clone", voice="reference-fit", client=client)
    assert report["ok"]
    assert len(seen) == 1
    assert report["output"]["sample_rate_hz"] == fixture["sample_rate_hz"]


def _wav(*, streamed=False, rate=24000, channels=1):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x10\x00" * 2400 * channels)
    data = bytearray(output.getvalue())
    if streamed:
        struct.pack_into("<I", data, 4, 0xFFFFFFFF)
        struct.pack_into("<I", data, 40, 0xFFFFFFFF)
    return bytes(data)


def test_native_gate_rejects_adapter_only_capability_without_running_inference():
    requests = []

    def handler(request):
        requests.append(request.url.path)
        return httpx.Response(
            200, json={"capabilities": ["completion", "audio-output"]}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = ollama_speech_smoke("omni:latest", client=client)
    assert not report["ok"]
    assert "does not advertise native speech" in report["error"]
    assert requests == ["/api/show"]


@pytest.mark.parametrize("streamed", [False, True])
def test_native_wav_counts_received_frames_and_saves_binary(tmp_path, streamed):
    audio = _wav(streamed=streamed)

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(
                200, json={"capabilities": ["speech"], "voices": ["default"]}
            )
        assert request.url.path == "/v1/audio/speech"
        return httpx.Response(200, content=audio, headers={"Content-Type": "audio/wav"})

    output = tmp_path / "native.wav"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = ollama_speech_smoke("native:latest", client=client, audio_out=output)
    assert report["ok"]
    assert report["output"]["frames"] == 2400
    assert report["output"]["duration_ms"] == 100
    assert output.read_bytes() == audio
    assert report["spoken_content_verified"] is False
    assert report["voice_similarity_verified"] is False


@pytest.mark.parametrize(
    "audio,mime,fmt,error",
    [
        (_wav(rate=16000), "audio/wav", "wav", "24 kHz mono PCM16"),
        (_wav(channels=2), "audio/wav", "wav", "24 kHz mono PCM16"),
        (b'{"audio":{"data":"base64"}}', "application/json", "wav", "binary WAV"),
        (b"\x01", "audio/pcm", "pcm", "incomplete"),
        (b"\x00\x00" * 20, "audio/pcm", "pcm", "silence"),
    ],
)
def test_native_gate_rejects_bad_audio(audio, mime, fmt, error):
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["speech"]})
        return httpx.Response(200, content=audio, headers={"Content-Type": mime})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = ollama_speech_smoke("native", client=client, response_format=fmt)
    assert report["ok"] is False
    assert error in report["error"]


def test_native_pcm_and_unsupported_voice_error():
    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["speech"]})
        return httpx.Response(
            200, content=b"\x10\x00" * 100, headers={"Content-Type": "audio/pcm"}
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = ollama_speech_smoke("native", client=client, response_format="pcm")
    assert report["ok"] is True
    assert report["output"]["frames"] == 100

    def unsupported(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["speech"]})
        return httpx.Response(400, json={"error": {"message": "unsupported voice"}})

    with httpx.Client(transport=httpx.MockTransport(unsupported)) as client:
        report = ollama_speech_smoke("native", client=client, voice="unknown")
    assert report["ok"] is False
    assert report["status_code"] == 400
    assert "unsupported voice" in report["error"]


def test_show_voices_are_separate_from_capabilities():
    parsed = parse_ollama_show("""
  Model
    architecture    phonon
  Capabilities
    speech
  Voices
    default
    Zoey
    Easton
  Parameters
    temperature    0.7
""")
    assert parsed.capabilities == ["speech"]
    assert parsed.voices == ["default", "Zoey", "Easton"]


def test_matching_text_interfaces_do_not_make_omni_speech_runnable_in_stock_ollama():
    from training_suite.models.omni import plan_omni_bundle

    config = {
        "model_type": "qwen3",
        "hidden_size": 2048,
        "vocab_size": 100,
        "num_hidden_layers": 8,
    }
    plan = plan_omni_bundle(
        text_config=config, omni_config=config, text_source="text", omni_source="omni"
    )
    native = plan["native_speech_delivery"]
    assert native["ready"] is False
    assert native["execution_owner"] == "ollama"
    assert native["official_binary_unmodified"] is True
    assert native["assessed_ollama_version"] == "0.40.11"
    assert native["external_tts_service_satisfies_requirement"] is False
    assert native["custom_runner_satisfies_requirement"] is False
    assert native["direct_weights_conversion_available"] is False
    assert native["status"] == "blocked-unsupported-qwen-speech-graph"
    assert native["distillation"]["preserves_original_omni_weights"] is False
    assert "unproven" in native["distillation"]["status"]


def test_missing_native_qwen_speech_cannot_be_repaired_by_packaging_or_adapter():
    from training_suite.models.intake import build_repair_plan

    plan = build_repair_plan(
        detected=["completion", "audio-output"],
        target=["completion", "audio-output", "speech", "tools"],
        source_architecture="qwen3_omni_moe",
        gguf_has_vision=True,
        raw_signature={},
        gguf_signature={},
        raw_source="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        source="omni.gguf",
    )
    assert plan["mode"] == "blocked"
    assert any("No direct weights-only speech conversion" in b for b in plan["blockers"])


@pytest.mark.parametrize(
    "architecture", ["phonon", "gradium_phonon_tts", "GradiumPhononTTSForConditionalGeneration"]
)
def test_supported_speech_graph_still_needs_native_verification(architecture):
    from training_suite.models.intake import build_repair_plan

    plan = build_repair_plan(
        detected=[],
        target=["speech"],
        source_architecture=architecture,
        gguf_has_vision=False,
        raw_signature={},
        gguf_signature={},
        raw_source="gradium/phonon-31m-en",
        source="checkpoint",
    )
    assert plan["mode"] == "native-speech-verification"
    assert not plan["blockers"]
