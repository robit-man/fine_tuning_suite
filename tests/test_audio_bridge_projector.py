from __future__ import annotations

import json

import pytest

from training_suite.models.audio_bridge_projector import (
    AUDIO_BRIDGE_PROJECTOR_SCHEMA,
    AudioBridgeProjectorError,
    build_audio_bridge_projector,
)


def _write_projector(path, *, audio: bool) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    writer = gguf.GGUFWriter(str(path), arch="clip")
    writer.add_key_value("general.name", path.stem, gguf.GGUFValueType.STRING)
    writer.add_key_value("clip.use_gelu", True, gguf.GGUFValueType.BOOL)
    if audio:
        writer.add_key_value("clip.has_vision_encoder", True, gguf.GGUFValueType.BOOL)
        writer.add_key_value("clip.has_audio_encoder", True, gguf.GGUFValueType.BOOL)
        writer.add_key_value("clip.audio.projection_dim", 4, gguf.GGUFValueType.UINT32)
        writer.add_key_value("clip.audio.embedding_length", 3, gguf.GGUFValueType.UINT32)
        writer.add_key_value("clip.audio.projector_type", "qwen3a", gguf.GGUFValueType.STRING)
        writer.add_tensor("a.test.weight", np.ones((3, 3), dtype=np.float32))
        writer.add_tensor("mm.a.mlp.1.weight", np.ones((3, 3), dtype=np.float32))
        writer.add_tensor("mm.a.mlp.2.weight", np.ones((4, 3), dtype=np.float32))
        writer.add_tensor("mm.a.mlp.2.bias", np.ones(4, dtype=np.float32))
        writer.add_tensor("v.donor.weight", np.ones((2, 2), dtype=np.float32))
    else:
        writer.add_key_value("clip.has_vision_encoder", True, gguf.GGUFValueType.BOOL)
        writer.add_key_value("clip.vision.projection_dim", 6, gguf.GGUFValueType.UINT32)
        writer.add_key_value("clip.projector_type", "qwen3vl_merger", gguf.GGUFValueType.STRING)
        writer.add_tensor("v.target.weight", np.ones((2, 2), dtype=np.float32))
        writer.add_tensor("mm.2.weight", np.ones((6, 2), dtype=np.float32))
        writer.add_tensor("mm.2.bias", np.ones(6, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_checkpoint(path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from training_suite.training.omni_encoder_bridge import (
        Qwen3AFinalProjectorBridge,
        Qwen3AFinalProjectorConfig,
        save_bridge_checkpoint,
    )

    bridge = Qwen3AFinalProjectorBridge(
        Qwen3AFinalProjectorConfig(source_width=3, target_width=6)
    )
    with torch.no_grad():
        bridge.projection.weight.fill_(2.0)
        bridge.projection.bias.fill_(3.0)
    save_bridge_checkpoint(path, bridge)


def test_assembly_keeps_target_vision_and_replaces_audio_output(tmp_path) -> None:
    gguf = pytest.importorskip("gguf")
    base = tmp_path / "base-mmproj.gguf"
    donor = tmp_path / "omni-mmproj.gguf"
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "bridge-mmproj.gguf"
    _write_projector(base, audio=False)
    _write_projector(donor, audio=True)
    _write_checkpoint(checkpoint)

    report = build_audio_bridge_projector(
        base_projector_gguf=base,
        omni_projector_gguf=donor,
        bridge_checkpoint=checkpoint,
        out_gguf=output,
        target_name="Test Audio Bridge",
    )

    reader = gguf.GGUFReader(str(output), "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    metadata = {key: field.contents() for key, field in reader.fields.items()}
    assert report["schema"] == AUDIO_BRIDGE_PROJECTOR_SCHEMA
    assert report["omni_audio_donor"]["thinker_included"] is False
    assert "v.target.weight" in tensors
    assert "v.donor.weight" not in tensors
    assert tuple(tensors["mm.a.mlp.2.weight"].shape) == (3, 6)
    assert metadata["clip.vision.projector_type"] == "qwen3vl_merger"
    assert metadata["clip.audio.projector_type"] == "qwen3a"
    assert metadata["clip.has_audio_encoder"] is True
    assert json.loads((output.with_suffix(".gguf.report.json")).read_text())["size_bytes"] > 0


def test_assembly_rejects_a_non_deployable_checkpoint(tmp_path) -> None:
    pytest.importorskip("safetensors")
    base = tmp_path / "base.gguf"
    donor = tmp_path / "donor.gguf"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _write_projector(base, audio=False)
    _write_projector(donor, audio=True)
    (checkpoint / "bridge_config.json").write_text(
        json.dumps({"bridge": {"architecture": "experimental"}}),
        encoding="utf-8",
    )
    (checkpoint / "bridge.safetensors").write_bytes(b"not used")

    with pytest.raises(AudioBridgeProjectorError, match="not a deployable"):
        build_audio_bridge_projector(
            base_projector_gguf=base,
            omni_projector_gguf=donor,
            bridge_checkpoint=checkpoint,
            out_gguf=tmp_path / "out.gguf",
            target_name="bad",
        )
