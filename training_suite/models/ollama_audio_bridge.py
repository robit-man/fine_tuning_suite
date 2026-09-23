from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from training_suite.models.ollama_sidecar import (
    OMNI_LAYER_MEDIA_TYPE,
    _install_blob,
    _read_manifest,
    manifest_path,
    resolve_models_dir,
)
from training_suite.models.single_gguf import (
    LIGHTWEIGHT_AUDIO_BRIDGE_SCHEMA,
    inspect_monolithic_gguf,
)

OLLAMA_PROJECTOR_MEDIA_TYPE = "application/vnd.ollama.image.projector"
OLLAMA_AUDIO_BRIDGE_SCHEMA = "robit.ollama-audio-bridge-tag.v1"


class OllamaAudioBridgeError(RuntimeError):
    """Raised when a trained audio-bridge tag cannot be assembled safely."""


def create_audio_bridge_tag(
    *,
    source_model: str,
    target_model: str,
    combined_projector_gguf: Path,
    tts_sidecar_gguf: Path,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Clone a stock tag, replace its projector, and attach the TTS-only sidecar."""
    if source_model == target_model:
        raise OllamaAudioBridgeError("target tag must differ from the source tag")
    root = resolve_models_dir(models_dir)
    source_manifest_path = manifest_path(source_model, root)
    target_manifest_path = manifest_path(target_model, root)
    if target_manifest_path.exists():
        raise OllamaAudioBridgeError(f"target tag already exists: {target_model}")
    projector = combined_projector_gguf.expanduser().resolve()
    sidecar = tts_sidecar_gguf.expanduser().resolve()
    if not projector.is_file():
        raise FileNotFoundError(projector)
    if not sidecar.is_file():
        raise FileNotFoundError(sidecar)
    inspection = inspect_monolithic_gguf(sidecar)
    if not inspection["valid"]:
        raise OllamaAudioBridgeError(f"invalid TTS sidecar: {inspection['errors']}")
    if inspection["manifest"].get("schema") != LIGHTWEIGHT_AUDIO_BRIDGE_SCHEMA:
        raise OllamaAudioBridgeError("sidecar is not the lightweight audio-bridge profile")

    source_manifest = _read_manifest(source_manifest_path)
    manifest = deepcopy(source_manifest)
    projector_layers = [
        layer
        for layer in manifest["layers"]
        if layer.get("mediaType") == OLLAMA_PROJECTOR_MEDIA_TYPE
    ]
    if len(projector_layers) != 1:
        raise OllamaAudioBridgeError(
            f"source tag must contain exactly one projector layer; found {len(projector_layers)}"
        )
    projector_digest = _install_blob(projector, root / "blobs")
    sidecar_digest = _install_blob(sidecar, root / "blobs")
    replacement = {
        "mediaType": OLLAMA_PROJECTOR_MEDIA_TYPE,
        "digest": f"sha256:{projector_digest}",
        "size": projector.stat().st_size,
        "annotations": {
            "org.opencontainers.image.title": projector.name,
            "io.robit.audio-bridge.schema": OLLAMA_AUDIO_BRIDGE_SCHEMA,
            "io.robit.audio-bridge.modalities": "vision,audio",
        },
    }
    sidecar_layer = {
        "mediaType": OMNI_LAYER_MEDIA_TYPE,
        "digest": f"sha256:{sidecar_digest}",
        "size": sidecar.stat().st_size,
        "annotations": {
            "org.opencontainers.image.title": sidecar.name,
            "io.robit.omni.schema": LIGHTWEIGHT_AUDIO_BRIDGE_SCHEMA,
            "io.robit.omni.profile": "trained-audio-bridge",
        },
    }
    new_layers = []
    for layer in manifest["layers"]:
        media_type = layer.get("mediaType")
        if media_type == OLLAMA_PROJECTOR_MEDIA_TYPE:
            new_layers.append(replacement)
        elif media_type != OMNI_LAYER_MEDIA_TYPE:
            new_layers.append(layer)
    new_layers.append(sidecar_layer)
    manifest["layers"] = new_layers

    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    partial = target_manifest_path.with_name(target_manifest_path.name + ".partial")
    partial.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    os.replace(partial, target_manifest_path)
    verified = _read_manifest(target_manifest_path)
    if verified != manifest:
        raise OllamaAudioBridgeError("written target manifest failed round-trip validation")

    model_layers = [
        layer
        for layer in manifest["layers"]
        if layer.get("mediaType") == "application/vnd.ollama.image.model"
    ]
    if len(model_layers) != 1:
        raise OllamaAudioBridgeError("target manifest does not contain one language model layer")
    resident_weight_bytes = (
        int(model_layers[0]["size"])
        + projector.stat().st_size
        + sidecar.stat().st_size
    )
    return {
        "schema": OLLAMA_AUDIO_BRIDGE_SCHEMA,
        "source_model": source_model,
        "target_model": target_model,
        "source_manifest": str(source_manifest_path),
        "target_manifest": str(target_manifest_path),
        "projector_layer": replacement,
        "sidecar_layer": sidecar_layer,
        "language_model_digest": model_layers[0]["digest"],
        "resident_weight_bytes": resident_weight_bytes,
        "resident_weight_gib": resident_weight_bytes / (1024**3),
        "language_trunk_copies": 1,
        "omni_thinker_included": False,
    }
