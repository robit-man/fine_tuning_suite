from __future__ import annotations

import base64
import io
import json
import wave

import httpx
import pytest

from training_suite.evals.runner import omni_audio_smoke
from training_suite.models.audio import (
    AudioContractError,
    encode_audio_response,
    validate_audio_input,
)
from training_suite.models.omni import (
    architecture_signature,
    assess_native_omni_splice,
    plan_omni_bundle,
    write_omni_bundle,
)
from training_suite.models.single_gguf import (
    BUNDLE_SCHEMA,
    audio_router_contract,
    inspect_monolithic_gguf,
    materialize_component_view,
    monolithic_bundle_manifest,
    pack_monolithic_gguf,
)
from training_suite.omni_runtime import OmniRuntimeConfig, run_http_cascade

QWEN38_CONFIG = {
    "architectures": ["Qwen3_5ForCausalLM"],
    "model_type": "qwen3_5_text",
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "vocab_size": 248320,
}

ORNITH_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 248320,
    },
    "vision_config": {"out_hidden_size": 4096},
}

OMNI_CONFIG = {
    "architectures": ["Qwen3OmniMoeForConditionalGeneration"],
    "model_type": "qwen3_omni_moe",
    "thinker_config": {
        "text_config": {
            "model_type": "qwen3_omni_moe_text",
            "hidden_size": 2048,
            "num_hidden_layers": 48,
            "vocab_size": 152064,
            "num_experts": 128,
            "num_experts_per_tok": 8,
        },
        "audio_config": {"output_dim": 2048, "sampling_rate": 16000},
        "vision_config": {"out_hidden_size": 2048},
        "video_token_id": 151656,
    },
    "talker_config": {
        "thinker_hidden_size": 2048,
        "num_code_groups": 16,
        "text_config": {"hidden_size": 1024, "num_hidden_layers": 20},
    },
    "code2wav_config": {"codebook_size": 2048, "num_quantizers": 16},
}


def _wav(sample_rate: int, *, frames: int = 160) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)
    return out.getvalue()


def test_audio_input_contract_accepts_base64_pcm16_wav() -> None:
    raw = _wav(16000)
    audio = validate_audio_input(
        {
            "mime_type": "audio/wav",
            "encoding": "base64",
            "data": base64.b64encode(raw).decode("ascii"),
        }
    )

    assert audio.sample_rate_hz == 16000
    assert audio.channels == 1
    assert audio.sample_width_bits == 16
    assert audio.data == raw


def test_audio_input_contract_rejects_wrong_sample_rate() -> None:
    encoded = base64.b64encode(_wav(44100)).decode("ascii")

    with pytest.raises(AudioContractError, match="expected 16000"):
        validate_audio_input(encoded)


def test_audio_output_is_json_safe_base64_wav() -> None:
    raw = _wav(24000)
    response = encode_audio_response(raw, transcript="hello")

    assert response["mime_type"] == "audio/wav"
    assert response["sample_rate_hz"] == 24000
    assert response["transcript"] == "hello"
    assert base64.b64decode(response["data"]) == raw


def test_monolithic_router_contract_tags_audio_like_a_binary_message_part() -> None:
    contract = audio_router_contract()

    assert contract["artifact"]["bundle_schema"] == BUNDLE_SCHEMA
    assert contract["schema"] == "robit.ollama.omni-adapter.v1"
    assert "audios" in contract["compatibility"]["message_extensions"]
    assert contract["media"]["audio"]["encoding"] == "base64"
    assert contract["response"]["message"]["audio"]["sample_rate_hz"] == 24000


def test_monolithic_manifest_is_one_physical_gguf_with_indexed_views() -> None:
    manifest = monolithic_bundle_manifest(
        base_source="qwen3.8",
        comprehension_source="qwen3-omni",
        tts_source="qwen3-tts",
        base_architecture="qwen3_5_text",
    )

    assert manifest["physical_bundle_artifacts"] == 1
    assert manifest["runtime"]["custom_media_handler_required"] is True
    assert [component["tensor_prefix"] for component in manifest["components"]] == [
        "a.c.m.",
        "s.t.m.",
    ]


def test_monolithic_packer_writes_three_tensor_namespaces(tmp_path) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")

    def make(path, value: float) -> None:
        writer = gguf.GGUFWriter(str(path), arch="llama")
        writer.add_key_value("general.name", path.stem, gguf.GGUFValueType.STRING)
        writer.add_key_value("llama.block_count", 1, gguf.GGUFValueType.UINT32)
        writer.add_tensor(
            "blk.0.test.weight",
            np.asarray([[value, value + 1]], dtype=np.float32),
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    base = tmp_path / "base.gguf"
    comprehension = tmp_path / "comprehension.gguf"
    tts = tmp_path / "tts.gguf"
    output = tmp_path / "model.gguf"
    make(base, 1)
    make(comprehension, 2)
    make(tts, 3)

    report = pack_monolithic_gguf(
        base_gguf=base,
        comprehension_gguf=comprehension,
        tts_gguf=tts,
        out_gguf=output,
    )
    inspection = inspect_monolithic_gguf(output)

    assert report["inspection"]["valid"] is True
    assert inspection["tensor_counts"] == {
        "base": 1,
        "comprehension": 1,
        "tts": 1,
    }
    reader = gguf.GGUFReader(str(output))
    assert len(reader.tensors) == 3
    assert inspection["tensor_count"] == 3
    assert [tensor.name for tensor in reader.tensors] == [
        "blk.0.test.weight",
        "a.c.m.blk.0.test.weight",
        "s.t.m.blk.0.test.weight",
    ]


def test_monolithic_packer_round_trips_all_six_executable_views(tmp_path) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")

    def make(path, name: str, value: float) -> None:
        writer = gguf.GGUFWriter(str(path), arch="llama")
        writer.add_key_value("general.name", name, gguf.GGUFValueType.STRING)
        writer.add_key_value("llama.block_count", 1, gguf.GGUFValueType.UINT32)
        writer.add_tensor(
            f"{name}.weight",
            np.asarray([[value, value + 1]], dtype=np.float32),
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    paths = {}
    for index, name in enumerate(
        (
            "base",
            "base_projector",
            "comprehension_model",
            "comprehension_projector",
            "tts_model",
            "tts_projector",
        ),
        start=1,
    ):
        paths[name] = tmp_path / f"{name}.gguf"
        make(paths[name], name, float(index))

    bundle = tmp_path / "omni.gguf"
    pack_monolithic_gguf(
        base_gguf=paths["base"],
        base_projector_gguf=paths["base_projector"],
        comprehension_gguf=paths["comprehension_model"],
        comprehension_projector_gguf=paths["comprehension_projector"],
        tts_gguf=paths["tts_model"],
        tts_projector_gguf=paths["tts_projector"],
        out_gguf=bundle,
    )
    inspection = inspect_monolithic_gguf(bundle)

    assert inspection["valid"] is True
    assert inspection["view_tensor_counts"] == {
        "base": 1,
        "base_projector": 1,
        "comprehension_model": 1,
        "comprehension_projector": 1,
        "tts_model": 1,
        "tts_projector": 1,
    }
    for name in inspection["view_tensor_counts"]:
        output = tmp_path / "materialized" / f"{name}.gguf"
        report = materialize_component_view(
            bundle_gguf=bundle,
            view=name,
            out_gguf=output,
        )
        reader = gguf.GGUFReader(str(output))
        assert report["tensor_count"] == 1
        assert [tensor.name for tensor in reader.tensors] == [f"{name}.weight"]
        assert reader.fields["general.name"].contents() == name
        assert output.read_bytes() == paths[name].read_bytes()


def test_qwen38_and_ornith_are_not_native_omni_splice_compatible() -> None:
    qwen = assess_native_omni_splice(QWEN38_CONFIG, OMNI_CONFIG)
    ornith = assess_native_omni_splice(ORNITH_CONFIG, OMNI_CONFIG)

    assert qwen["compatible"] is False
    assert ornith["compatible"] is False
    assert any("hidden_size" in item for item in qwen["mismatches"])
    assert any("model_type" in item for item in ornith["mismatches"])


def test_full_omni_config_has_all_component_signatures() -> None:
    signature = architecture_signature(OMNI_CONFIG)

    assert signature.model_type == "qwen3_omni_moe_text"
    assert signature.hidden_size == 2048
    assert signature.has_audio_encoder is True
    assert signature.has_vision_encoder is True
    assert signature.has_talker is True
    assert signature.has_code2wav is True
    assert signature.has_video_input is True
    assert signature.num_experts == 128
    assert signature.num_experts_per_token == 8
    assert signature.talker_hidden_size == 1024
    assert signature.num_code_groups == 16


def test_qwen38_plan_selects_monolithic_router_without_claiming_native_fusion() -> None:
    plan = plan_omni_bundle(
        text_config=QWEN38_CONFIG,
        omni_config=OMNI_CONFIG,
        text_source="manitcor/Qwen3.8-27B-Obliterated-E03",
        target_tag="robit/qwen3.8-27b-omni-experiment:latest",
    )

    assert plan["mode"] == "monolithic-router"
    assert plan["status"] == "ready-for-monolithic-pack"
    assert plan["artifact_policy"]["one_logical_ollama_tag"] is True
    assert plan["artifact_policy"]["single_custom_sidecar_gguf"] is True
    assert plan["artifact_policy"]["stock_ollama_direct_sidecar_import"] is False
    assert plan["artifact_policy"]["custom_media_adapter_required"] is True
    assert plan["runtime_support"]["ollama_0_32"]["generated_audio_response"] is False
    assert plan["deployment_profile"]["talker"]["num_code_groups"] == 16
    assert plan["runtime_support"]["vllm_omni"]["qwen3_omni_talker_output"] is True
    assert set(plan["components"]) == {
        "audio_understanding",
        "video_understanding",
        "language_model",
        "speech_synthesis",
    }
    assert "video-output" not in plan["requested_capabilities"]
    assert plan["video_policy"]["scope"].startswith("video comprehension")


def test_native_omni_bundle_plan_and_files(tmp_path) -> None:
    gguf = tmp_path / "model.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    gguf.write_bytes(b"GGUF-test")
    mmproj.write_bytes(b"GGUF-projector-test")
    plan = plan_omni_bundle(
        text_config=OMNI_CONFIG,
        omni_config=OMNI_CONFIG,
        text_source="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    )
    outputs = write_omni_bundle(
        tmp_path / "bundle",
        plan,
        text_gguf=str(gguf),
        mmproj_gguf=str(mmproj),
    )

    assert plan["mode"] == "native-omni"
    manifest = json.loads((tmp_path / "bundle" / "omni_bundle.json").read_text())
    assert manifest["component_files"]["language_model"]["exists"] is True
    assert "manifest" in outputs
    assert "FROM " + str(gguf.resolve()) in (tmp_path / "bundle" / "Modelfile").read_text()
    assert "FROM " + str(mmproj.resolve()) in (tmp_path / "bundle" / "Modelfile").read_text()


def test_router_plan_modelfile_does_not_attach_incompatible_omni_projector(tmp_path) -> None:
    gguf = tmp_path / "qwen38.gguf"
    mmproj = tmp_path / "omni-mmproj.gguf"
    gguf.write_bytes(b"GGUF-test")
    mmproj.write_bytes(b"GGUF-projector-test")
    plan = plan_omni_bundle(
        text_config=QWEN38_CONFIG,
        omni_config=OMNI_CONFIG,
        text_source="qwen3.8",
    )

    write_omni_bundle(
        tmp_path / "bundle",
        plan,
        text_gguf=str(gguf),
        mmproj_gguf=str(mmproj),
    )
    text = (tmp_path / "bundle" / "Modelfile").read_text()

    assert "FROM " + str(gguf.resolve()) in text
    assert "FROM " + str(mmproj.resolve()) not in text


def test_http_cascade_transports_audio_and_returns_base64_wav() -> None:
    output_wav = _wav(24000)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "asr":
            body = json.loads(request.content)
            audio_part = body["messages"][0]["content"][0]
            assert audio_part["type"] == "input_audio"
            assert base64.b64decode(audio_part["input_audio"]["data"])[:4] == b"RIFF"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "hello from audio"}}]},
            )
        if request.url.host == "ollama":
            body = json.loads(request.content)
            assert "hello from audio" in body["messages"][0]["content"]
            return httpx.Response(
                200,
                json={"message": {"content": "spoken answer", "thinking": "brief"}},
            )
        if request.url.host == "tts":
            body = json.loads(request.content)
            assert body["text"] == "spoken answer"
            return httpx.Response(200, content=output_wav, headers={"content-type": "audio/wav"})
        return httpx.Response(404)

    input_audio = base64.b64encode(_wav(16000)).decode("ascii")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = run_http_cascade(
        audio_payload=input_audio,
        prompt="Reply naturally.",
        config=OmniRuntimeConfig(
            asr_url="http://asr/v1/chat/completions",
            asr_model="qwen3-omni",
            language_model="qwen3.8",
            tts_url="http://tts/synthesize",
            ollama_url="http://ollama",
        ),
        client=client,
    )

    assert report["audio_understanding"]["content"] == "hello from audio"
    assert report["message"]["content"] == "spoken answer"
    assert report["message"]["audio"]["sample_rate_hz"] == 24000
    assert base64.b64decode(report["audio"]["data"]) == output_wav


def test_http_cascade_can_route_to_text_without_tts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "asr":
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "heard text"}}]},
            )
        if request.url.host == "ollama":
            return httpx.Response(200, json={"message": {"content": "text only"}})
        raise AssertionError("TTS must not be called for a text-only route")

    report = run_http_cascade(
        audio_payload=base64.b64encode(_wav(16000)).decode("ascii"),
        prompt="Reply in text.",
        config=OmniRuntimeConfig(
            asr_url="http://asr/v1/chat/completions",
            asr_model="qwen3-omni",
            language_model="qwen3.8",
            tts_url="",
            ollama_url="http://ollama",
        ),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        synthesize=False,
    )

    assert report["message"]["content"] == "text only"
    assert "audio" not in report["message"]
    assert report["routing"]["speech_synthesized"] is False


def test_live_audio_smoke_validates_returned_waveform(tmp_path, monkeypatch) -> None:
    fixture = tmp_path / "fixture.wav"
    fixture.write_bytes(_wav(16000))
    output = encode_audio_response(_wav(24000), transcript="answer")

    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            json={
                "audio": output,
                "audio_understanding": {"content": "heard"},
                "message": {"content": "answer"},
            },
        )

    monkeypatch.setattr("training_suite.evals.runner.httpx.post", fake_post)
    report = omni_audio_smoke(fixture)

    assert report["ok"] is True
    assert report["input"]["sample_rate_hz"] == 16000
    assert report["output"]["sample_rate_hz"] == 24000
