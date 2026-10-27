"""Read verified hybrid tensors from an Ollama descriptor or local model store."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from safetensors.torch import load_file

from training_suite.models.phonon_hybrid import PhononHybrid, digest


def load_descriptor(descriptor):
    directory = Path(descriptor["directory"])
    weights = {}
    for blob in descriptor["tensor_blobs"]:
        path = Path(blob["path"])
        if "sha256:" + digest(path) != blob["digest"]:
            raise ValueError("Ollama tensor blob failed integrity verification")
        for name, value in load_file(str(path)).items():
            if name in weights:
                raise ValueError("duplicate tensor in Ollama hybrid model: " + name)
            weights[name] = value
    config = json.loads((directory / "config.json").read_text())
    model = PhononHybrid.from_state(directory, config, weights)
    return model, sum(t.numel() * t.element_size() for t in weights.values())


def load_ollama_model(models_directory: Path, tag: str):
    """Materialize only metadata; all parameters come from verified tensor blobs."""
    parts = tag.split(":")
    if len(parts) > 2:
        raise ValueError("use a local namespace/model:tag name")
    path, revision = parts[0], parts[1] if len(parts) == 2 else "latest"
    names = path.split("/")
    if len(names) == 1:
        names.insert(0, "library")
    if len(names) != 2 or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", x) for x in [*names, revision]):
        raise ValueError("invalid local Ollama model name")
    models_directory = models_directory.resolve()

    def read_blob(value):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
            raise ValueError("invalid Ollama blob digest")
        blob = models_directory / "blobs" / value.replace(":", "-")
        if "sha256:" + digest(blob) != value:
            raise ValueError("Ollama blob failed integrity verification")
        return blob

    manifest_file = next((p for directory in ["manifests-v2", "manifests"]
                          for registry in ["ollama.com", "registry.ollama.ai", "registry.ollama.com"]
                          if (p := models_directory / directory / registry / names[0] / names[1] / revision).is_file()), None)
    if manifest_file is None:
        raise FileNotFoundError("hybrid model is not installed in the selected Ollama model store")
    resolved = manifest_file.resolve()
    if not resolved.is_relative_to(models_directory):
        raise ValueError("Ollama manifest escapes its model store")
    if resolved.name.startswith("sha256-"):
        read_blob(resolved.name.replace("-", ":", 1))
    manifest = json.loads(manifest_file.read_text())
    if "manifests" in manifest:
        child = next((m for m in manifest["manifests"] if m.get("runner") == "phonon_hybrid"), None)
        if child is None:
            raise ValueError("model has no Phonon hybrid runner variant")
        manifest = json.loads(read_blob(child["digest"]).read_text())
    if manifest.get("runner") != "phonon_hybrid":
        raise ValueError("voice profiles require a Phonon hybrid model")
    with tempfile.TemporaryDirectory(prefix="phonon-echo-bake-") as temporary:
        directory = Path(temporary)
        tensors, seen = [], set()
        for layer in manifest["layers"]:
            blob = read_blob(layer["digest"])
            if layer["mediaType"] == "application/vnd.ollama.image.tensor" and layer["digest"] not in seen:
                tensors.append({"path": str(blob), "digest": layer["digest"]})
                seen.add(layer["digest"])
            elif layer["mediaType"] == "application/vnd.ollama.image.json":
                target = directory / layer["name"]
                if not target.resolve().is_relative_to(directory):
                    raise ValueError("invalid hybrid metadata path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(blob)
        model, _ = load_descriptor({"directory": str(directory), "tensor_blobs": tensors})
    return model
