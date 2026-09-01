from __future__ import annotations

import json
from pathlib import Path

import pytest

from training_suite.models.ollama_sidecar import (
    OMNI_LAYER_MEDIA_TYPE,
    attach_ollama_sidecar,
    manifest_path,
    prepare_ollama_sidecar,
    resolve_ollama_sidecar,
)
from training_suite.models.single_gguf import pack_monolithic_gguf


def _make_gguf(path: Path, name: str, value: float) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    writer = gguf.GGUFWriter(str(path), arch="llama")
    writer.add_key_value("general.name", name, gguf.GGUFValueType.STRING)
    writer.add_key_value("llama.block_count", 1, gguf.GGUFValueType.UINT32)
    writer.add_tensor(f"{name}.weight", np.asarray([[value, value + 1]], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_attach_and_resolve_omni_sidecar_layer(tmp_path: Path) -> None:
    base = tmp_path / "base.gguf"
    comprehension = tmp_path / "comprehension.gguf"
    tts = tmp_path / "tts.gguf"
    bundle = tmp_path / "bundle.gguf"
    _make_gguf(base, "base", 1)
    _make_gguf(comprehension, "comprehension", 2)
    _make_gguf(tts, "tts", 3)
    pack_monolithic_gguf(
        base_gguf=base,
        comprehension_gguf=comprehension,
        tts_gguf=tts,
        out_gguf=bundle,
    )

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    path = manifest_path("robit/test:q4km", models_dir)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 2,
                "layers": [
                    {
                        "mediaType": "application/vnd.ollama.image.model",
                        "digest": "sha256:base",
                        "size": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    attached = attach_ollama_sidecar(
        model="robit/test:q4km",
        bundle_gguf=bundle,
        models_dir=models_dir,
    )
    resolved = resolve_ollama_sidecar(model="robit/test:q4km", models_dir=models_dir)

    assert attached["layer"]["mediaType"] == OMNI_LAYER_MEDIA_TYPE
    assert resolved["bundle"] == str(
        models_dir / "blobs" / attached["layer"]["digest"].replace(":", "-", 1)
    )
    assert resolved["inspection"]["view_tensor_counts"]["base"] == 1
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert [layer["mediaType"] for layer in updated["layers"]] == [
        "application/vnd.ollama.image.model",
        OMNI_LAYER_MEDIA_TYPE,
    ]

    prepared = prepare_ollama_sidecar(
        model="robit/test:q4km",
        output_dir=tmp_path / "runtime-cache",
        views=("comprehension_model", "tts_model"),
        models_dir=models_dir,
    )
    assert prepared["disposable_cache"] is True
    assert set(prepared["views"]) == {"comprehension_model", "tts_model"}
    assert Path(prepared["views"]["tts_model"]["output"]).is_file()
