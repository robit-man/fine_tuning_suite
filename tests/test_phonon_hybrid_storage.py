import hashlib
import json

import pytest
import torch
from safetensors.torch import save_file

from training_suite.models import phonon_hybrid_storage as storage


@pytest.fixture
def installed_model(tmp_path, monkeypatch):
    store = tmp_path / "models"
    (store / "blobs").mkdir(parents=True)

    def blob(data):
        checksum = "sha256:" + hashlib.sha256(data).hexdigest()
        path = store / "blobs" / checksum.replace(":", "-")
        path.write_bytes(data)
        return checksum, path

    weights = tmp_path / "weights.safetensors"
    save_file({"voices.female": torch.ones(1, 256, 1)}, str(weights))
    tensor_digest, tensor = blob(weights.read_bytes())
    config_digest, _ = blob(b'{"baked_voices": ["female"]}')
    manifest = {
        "runner": "phonon_hybrid",
        "layers": [
            {"mediaType": "application/vnd.ollama.image.tensor", "digest": tensor_digest},
            {"mediaType": "application/vnd.ollama.image.tensor", "digest": tensor_digest},
            {"mediaType": "application/vnd.ollama.image.json", "digest": config_digest, "name": "config.json"},
        ],
    }
    child_digest, _ = blob(json.dumps(manifest).encode())
    index_digest, index = blob(json.dumps({"manifests": [{"runner": "phonon_hybrid", "digest": child_digest}]}).encode())
    tag = store / "manifests-v2/ollama.com/example/voice/latest"
    tag.parent.mkdir(parents=True)
    tag.symlink_to(index)
    captured = {}

    def from_state(directory, config, weights):
        captured.update(config=config, weights=weights, directory=directory)
        assert (directory / "config.json").read_text()
        return captured

    monkeypatch.setattr(storage.PhononHybrid, "from_state", from_state)
    return store, tensor, tag, captured, manifest, index_digest


def test_installed_manifest_list_loads_verified_tensors_and_discards_metadata(installed_model):
    store, _, _, captured, _, _ = installed_model
    result = storage.load_ollama_model(store, "example/voice")
    assert result is captured
    assert result["config"]["baked_voices"] == ["female"]
    assert set(result["weights"]) == {"voices.female"}
    assert torch.equal(result["weights"]["voices.female"], torch.ones(1, 256, 1))
    assert not result["directory"].exists()


def test_corrupted_tensor_is_rejected_before_model_construction(installed_model):
    store, tensor, _, captured, _, _ = installed_model
    tensor.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        storage.load_ollama_model(store, "example/voice")
    assert not captured


def test_metadata_path_cannot_escape_export_directory(installed_model):
    store, _, tag, captured, manifest, _ = installed_model
    manifest["layers"][-1]["name"] = "../../outside.json"
    tag.unlink()
    tag.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="metadata path"):
        storage.load_ollama_model(store, "example/voice")
    assert not captured


def test_manifest_symlink_cannot_escape_model_store(installed_model, tmp_path):
    store, _, tag, captured, manifest, _ = installed_model
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(manifest))
    tag.unlink()
    tag.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        storage.load_ollama_model(store, "example/voice")
    assert not captured


@pytest.mark.parametrize("tag", ["../../model", "a/b/c:tag", "a:b:c", "a/b:"])
def test_invalid_model_paths_are_rejected(tmp_path, tag):
    with pytest.raises(ValueError):
        storage.load_ollama_model(tmp_path, tag)


def test_descriptor_rejects_duplicate_named_tensors(tmp_path):
    weights = tmp_path / "weights.safetensors"
    save_file({"voices.female": torch.ones(1, 256, 1)}, str(weights))
    blob = {"path": str(weights), "digest": "sha256:" + hashlib.sha256(weights.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="duplicate tensor"):
        storage.load_descriptor({"directory": str(tmp_path), "tensor_blobs": [blob, blob]})
