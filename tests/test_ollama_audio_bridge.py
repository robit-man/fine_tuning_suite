from __future__ import annotations

import hashlib
import json

import pytest

from training_suite.models.ollama_audio_bridge import (
    OLLAMA_AUDIO_BRIDGE_SCHEMA,
    OLLAMA_PROJECTOR_MEDIA_TYPE,
    create_audio_bridge_tag,
)
from training_suite.models.ollama_sidecar import (
    OMNI_LAYER_MEDIA_TYPE,
    manifest_path,
    prepare_ollama_sidecar,
)
from training_suite.models.single_gguf import pack_audio_bridge_sidecar


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gguf(path, architecture: str, tensor: str) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    writer = gguf.GGUFWriter(str(path), arch=architecture)
    writer.add_tensor(tensor, np.asarray([[1.0, 2.0]], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_create_audio_bridge_tag_replaces_projector_without_mutating_source(tmp_path) -> None:
    models = tmp_path / "models"
    blobs = models / "blobs"
    blobs.mkdir(parents=True)
    model_blob = blobs / "sha256-model"
    old_projector = blobs / "sha256-old-projector"
    model_blob.write_bytes(b"model")
    old_projector.write_bytes(b"projector")
    source = "robit/source:q4km"
    source_path = manifest_path(source, models)
    source_path.parent.mkdir(parents=True)
    source_manifest = {
        "schemaVersion": 2,
        "layers": [
            {
                "mediaType": "application/vnd.ollama.image.model",
                "digest": "sha256:model",
                "size": 5,
            },
            {
                "mediaType": OLLAMA_PROJECTOR_MEDIA_TYPE,
                "digest": "sha256:old-projector",
                "size": 9,
            },
        ],
    }
    source_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    combined = tmp_path / "combined.gguf"
    tts = tmp_path / "tts.gguf"
    tts_projector = tmp_path / "tts-projector.gguf"
    sidecar = tmp_path / "sidecar.gguf"
    _gguf(combined, "clip", "v.weight")
    _gguf(tts, "llama", "blk.0.weight")
    _gguf(tts_projector, "clip", "codec.weight")
    pack_audio_bridge_sidecar(
        tts_gguf=tts,
        tts_projector_gguf=tts_projector,
        out_gguf=sidecar,
        base_source="source",
        combined_projector_source="combined",
    )

    report = create_audio_bridge_tag(
        source_model=source,
        target_model="robit/target-audio-bridge:q4km",
        combined_projector_gguf=combined,
        tts_sidecar_gguf=sidecar,
        models_dir=models,
    )

    target_path = manifest_path("robit/target-audio-bridge:q4km", models)
    target = json.loads(target_path.read_text(encoding="utf-8"))
    assert report["schema"] == OLLAMA_AUDIO_BRIDGE_SCHEMA
    assert json.loads(source_path.read_text(encoding="utf-8")) == source_manifest
    assert [layer["mediaType"] for layer in target["layers"]] == [
        "application/vnd.ollama.image.model",
        OLLAMA_PROJECTOR_MEDIA_TYPE,
        OMNI_LAYER_MEDIA_TYPE,
    ]
    assert target["layers"][1]["digest"] == f"sha256:{_digest(combined)}"
    assert target["layers"][2]["digest"] == f"sha256:{_digest(sidecar)}"
    assert report["language_trunk_copies"] == 1
    assert report["omni_thinker_included"] is False

    prepared = prepare_ollama_sidecar(
        model="robit/target-audio-bridge:q4km",
        output_dir=tmp_path / "runtime-cache",
        models_dir=models,
    )
    assert prepared["profile"] == "trained-audio-bridge"
    assert set(prepared["views"]) == {"tts_model", "tts_projector"}
