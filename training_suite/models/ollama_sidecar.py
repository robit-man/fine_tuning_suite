from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from training_suite.models.single_gguf import (
    inspect_monolithic_gguf,
    materialize_component_view,
)

OMNI_LAYER_MEDIA_TYPE = "application/vnd.robit.ollama.omni.bundle.v1+gguf"
DEFAULT_REGISTRY = "registry.ollama.ai"
RUNTIME_VIEWS = (
    "comprehension_model",
    "comprehension_projector",
    "tts_model",
    "tts_projector",
)


class OllamaSidecarError(RuntimeError):
    """Raised when an Omni sidecar cannot be resolved or attached safely."""


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_models_dir(explicit: Path | None = None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["OLLAMA_MODELS"]) if os.environ.get("OLLAMA_MODELS") else None,
        Path("/srv/ollama/models"),
        Path.home() / ".ollama" / "models",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_dir():
            return candidate.expanduser().resolve()
    raise OllamaSidecarError("could not locate the Ollama models directory")


def manifest_path(model: str, models_dir: Path) -> Path:
    reference = model.strip()
    if not reference:
        raise OllamaSidecarError("model reference cannot be empty")
    if ":" in reference.rsplit("/", 1)[-1]:
        repository, tag = reference.rsplit(":", 1)
    else:
        repository, tag = reference, "latest"
    parts = repository.split("/")
    if "." in parts[0] or ":" in parts[0] or parts[0] == "localhost":
        registry = parts.pop(0)
    else:
        registry = DEFAULT_REGISTRY
    if len(parts) == 1:
        parts.insert(0, "library")
    if len(parts) < 2 or not all(parts) or not tag:
        raise OllamaSidecarError(f"invalid Ollama model reference: {model!r}")
    return models_dir / "manifests" / registry / Path(*parts) / tag


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise OllamaSidecarError(f"Ollama manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OllamaSidecarError(f"cannot read Ollama manifest {path}: {exc}") from exc
    if not isinstance(manifest.get("layers"), list):
        raise OllamaSidecarError(f"Ollama manifest has no layer list: {path}")
    return manifest


def _install_blob(source: Path, destination: Path) -> str:
    digest = _sha256(source)
    expected = destination / f"sha256-{digest}"
    if expected.exists():
        if expected.stat().st_size != source.stat().st_size or _sha256(expected) != digest:
            raise OllamaSidecarError(f"existing Ollama blob failed digest validation: {expected}")
        return digest
    expected.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, expected)
    except OSError:
        partial = expected.with_name(expected.name + ".partial")
        try:
            with source.open("rb") as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            if _sha256(partial) != digest:
                raise OllamaSidecarError("copied sidecar blob failed digest validation")
            os.replace(partial, expected)
        finally:
            if partial.exists():
                partial.unlink()
    return digest


def attach_ollama_sidecar(
    *,
    model: str,
    bundle_gguf: Path,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Attach a custom Omni GGUF layer to an existing runnable Ollama tag."""
    bundle = bundle_gguf.expanduser().resolve()
    inspection = inspect_monolithic_gguf(bundle)
    if not inspection["valid"]:
        raise OllamaSidecarError(f"invalid Omni bundle: {inspection['errors']}")
    root = resolve_models_dir(models_dir)
    path = manifest_path(model, root)
    manifest = _read_manifest(path)
    digest = _install_blob(bundle, root / "blobs")
    layer = {
        "mediaType": OMNI_LAYER_MEDIA_TYPE,
        "digest": f"sha256:{digest}",
        "size": bundle.stat().st_size,
        "annotations": {
            "org.opencontainers.image.title": bundle.name,
            "io.robit.omni.schema": inspection["manifest"]["schema"],
        },
    }
    previous_sidecars = sum(
        existing.get("mediaType") == OMNI_LAYER_MEDIA_TYPE for existing in manifest["layers"]
    )
    retained = [
        existing
        for existing in manifest["layers"]
        if existing.get("mediaType") != OMNI_LAYER_MEDIA_TYPE
    ]
    manifest["layers"] = [*retained, layer]
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.replace(partial, path)
    return {
        "model": model,
        "manifest": str(path),
        "models_dir": str(root),
        "layer": layer,
        "replaced_existing_sidecar": previous_sidecars > 0,
        "bundle": str(bundle),
        "bundle_inspection": {
            "valid": inspection["valid"],
            "tensor_count": inspection["tensor_count"],
            "view_tensor_counts": inspection["view_tensor_counts"],
        },
    }


def resolve_ollama_sidecar(
    *,
    model: str,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve and validate the Omni sidecar layer for a local Ollama tag."""
    root = resolve_models_dir(models_dir)
    path = manifest_path(model, root)
    manifest = _read_manifest(path)
    layers = [
        layer
        for layer in manifest["layers"]
        if layer.get("mediaType") == OMNI_LAYER_MEDIA_TYPE
    ]
    if len(layers) != 1:
        raise OllamaSidecarError(
            f"expected exactly one Omni sidecar layer for {model}, found {len(layers)}"
        )
    layer = layers[0]
    digest = str(layer.get("digest") or "")
    if not digest.startswith("sha256:"):
        raise OllamaSidecarError("Omni sidecar layer has no SHA-256 digest")
    blob = root / "blobs" / digest.replace(":", "-", 1)
    if not blob.is_file() or blob.stat().st_size != layer.get("size"):
        raise OllamaSidecarError(f"Omni sidecar blob is missing or has the wrong size: {blob}")
    inspection = inspect_monolithic_gguf(blob)
    if not inspection["valid"]:
        raise OllamaSidecarError(f"Omni sidecar failed inspection: {inspection['errors']}")
    return {
        "model": model,
        "manifest": str(path),
        "layer": layer,
        "bundle": str(blob),
        "inspection": inspection,
    }


def prepare_ollama_sidecar(
    *,
    model: str,
    output_dir: Path,
    views: tuple[str, ...] = RUNTIME_VIEWS,
    models_dir: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize disposable runtime views directly from an installed Ollama tag."""
    unknown = sorted(set(views) - set(RUNTIME_VIEWS))
    if unknown:
        raise OllamaSidecarError(f"unsupported runtime views: {', '.join(unknown)}")
    if not views:
        raise OllamaSidecarError("at least one runtime view is required")
    resolved = resolve_ollama_sidecar(model=model, models_dir=models_dir)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for view in views:
        output = destination / (view.replace("_", "-") + ".gguf")
        outputs[view] = materialize_component_view(
            bundle_gguf=Path(resolved["bundle"]),
            view=view,
            out_gguf=output,
            overwrite=overwrite,
        )
    return {
        "model": model,
        "bundle": resolved["bundle"],
        "bundle_digest": resolved["layer"]["digest"],
        "output_dir": str(destination),
        "disposable_cache": True,
        "views": outputs,
    }
