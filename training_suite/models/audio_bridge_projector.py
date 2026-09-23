from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

AUDIO_BRIDGE_PROJECTOR_SCHEMA = "robit.qwen3a-audio-bridge-projector.v1"
HOUSEKEEPING_FIELDS = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
}


class AudioBridgeProjectorError(RuntimeError):
    """Raised when a target-vision plus Omni-audio projector cannot be assembled."""


def _require_dependencies():
    try:
        import gguf  # type: ignore
        import ml_dtypes  # type: ignore
        import numpy as np  # type: ignore
        from safetensors.torch import load_file
    except Exception as exc:  # pragma: no cover - environment guard
        raise AudioBridgeProjectorError(
            "assembly requires gguf, ml_dtypes, numpy, torch, and safetensors"
        ) from exc
    return gguf, ml_dtypes, np, load_file


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _field_value(field: Any) -> Any:
    return field.contents()


def _metadata(reader: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, field in reader.fields.items():
        try:
            result[key] = _field_value(field)
        except Exception:  # noqa: BLE001,S112 - third-party decoder errors vary
            continue
    return result


def _copy_tensor(writer: Any, tensor: Any, np: Any) -> None:
    writer.add_tensor(
        tensor.name,
        np.asarray(tensor.data),
        raw_dtype=tensor.tensor_type,
    )


def _merged_metadata(base: Any, donor: Any, target_name: str) -> dict[str, tuple[Any, Any]]:
    merged: dict[str, tuple[Any, Any]] = {}
    for key, field in base.fields.items():
        if key in HOUSEKEEPING_FIELDS or key == "clip.projector_type":
            continue
        try:
            merged[key] = (_field_value(field), field.types[0])
        except Exception:  # noqa: BLE001,S112 - preserve readable metadata only
            continue

    base_values = _metadata(base)
    vision_projector_type = base_values.get("clip.vision.projector_type")
    if not vision_projector_type:
        vision_projector_type = base_values.get("clip.projector_type")
    if not vision_projector_type:
        raise AudioBridgeProjectorError("base projector has no vision projector type")

    for key, field in donor.fields.items():
        if key.startswith("clip.audio.") or key == "clip.has_audio_encoder":
            try:
                merged[key] = (_field_value(field), field.types[0])
            except Exception:  # noqa: BLE001,S112 - preserve readable metadata only
                continue

    gguf, _, _, _ = _require_dependencies()
    merged["general.name"] = (target_name, gguf.GGUFValueType.STRING)
    merged["clip.has_vision_encoder"] = (True, gguf.GGUFValueType.BOOL)
    merged["clip.has_audio_encoder"] = (True, gguf.GGUFValueType.BOOL)
    merged["clip.vision.projector_type"] = (
        vision_projector_type,
        gguf.GGUFValueType.STRING,
    )
    merged["clip.audio.projector_type"] = ("qwen3a", gguf.GGUFValueType.STRING)
    merged["robit.audio_bridge.schema"] = (
        AUDIO_BRIDGE_PROJECTOR_SCHEMA,
        gguf.GGUFValueType.STRING,
    )
    return merged


def _load_bridge(checkpoint_dir: Path, load_file: Any) -> tuple[Any, Any, dict[str, Any]]:
    config_path = checkpoint_dir / "bridge_config.json"
    weights_path = checkpoint_dir / "bridge.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise AudioBridgeProjectorError(
            "checkpoint directory must contain bridge_config.json and bridge.safetensors"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bridge = config.get("bridge") or {}
    if bridge.get("architecture") != "qwen3a-final-projector":
        raise AudioBridgeProjectorError(
            "checkpoint is not a deployable qwen3a-final-projector"
        )
    state = load_file(str(weights_path), device="cpu")
    expected = {"projection.weight", "projection.bias"}
    if set(state) != expected:
        raise AudioBridgeProjectorError(
            f"unexpected bridge tensors: expected {sorted(expected)}, got {sorted(state)}"
        )
    weight = state["projection.weight"].detach().cpu()
    bias = state["projection.bias"].detach().cpu()
    source_width = int(bridge["source_width"])
    target_width = int(bridge["target_width"])
    if tuple(weight.shape) != (target_width, source_width):
        raise AudioBridgeProjectorError(
            f"bridge weight shape {tuple(weight.shape)} does not match "
            f"({target_width}, {source_width})"
        )
    if tuple(bias.shape) != (target_width,):
        raise AudioBridgeProjectorError("bridge bias shape does not match target width")
    return weight, bias, config


def build_audio_bridge_projector(
    *,
    base_projector_gguf: Path,
    omni_projector_gguf: Path,
    bridge_checkpoint: Path,
    out_gguf: Path,
    target_name: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Combine target-native vision with Omni audio and a trained final projection."""
    gguf, ml_dtypes, np, load_file = _require_dependencies()
    base_path = base_projector_gguf.expanduser().resolve()
    donor_path = omni_projector_gguf.expanduser().resolve()
    checkpoint = bridge_checkpoint.expanduser().resolve()
    out = out_gguf.expanduser().resolve()
    for path in (base_path, donor_path):
        if not path.is_file():
            raise AudioBridgeProjectorError(f"projector does not exist: {path}")
    if out in (base_path, donor_path):
        raise AudioBridgeProjectorError("output must not overwrite an input projector")
    if out.exists() and not overwrite:
        raise AudioBridgeProjectorError(f"output already exists: {out}")

    weight, bias, checkpoint_config = _load_bridge(checkpoint, load_file)
    base = gguf.GGUFReader(str(base_path), "r")
    donor = gguf.GGUFReader(str(donor_path), "r")
    base_metadata = _metadata(base)
    donor_metadata = _metadata(donor)
    if not base_metadata.get("clip.has_vision_encoder"):
        raise AudioBridgeProjectorError("base projector has no vision encoder")
    if not donor_metadata.get("clip.has_audio_encoder"):
        raise AudioBridgeProjectorError("Omni projector has no audio encoder")

    base_names = {tensor.name for tensor in base.tensors}
    if any(name.startswith(("a.", "mm.a.")) for name in base_names):
        raise AudioBridgeProjectorError("base projector already contains audio tensors")
    donor_audio = [
        tensor
        for tensor in donor.tensors
        if tensor.name.startswith(("a.", "mm.a."))
        and tensor.name not in {"mm.a.mlp.2.weight", "mm.a.mlp.2.bias"}
    ]
    if not donor_audio or not any(tensor.name.startswith("a.") for tensor in donor_audio):
        raise AudioBridgeProjectorError("Omni projector has no reusable audio tower tensors")

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        partial.unlink()
    writer = gguf.GGUFWriter(str(partial), arch="clip")
    for key, (value, value_type) in _merged_metadata(base, donor, target_name).items():
        if isinstance(value, list) and not value:
            continue
        writer.add_key_value(key, value, value_type)

    for tensor in base.tensors:
        _copy_tensor(writer, tensor, np)
    for tensor in donor_audio:
        if tensor.name in base_names:
            raise AudioBridgeProjectorError(f"duplicate tensor name: {tensor.name}")
        _copy_tensor(writer, tensor, np)

    weight_array = weight.float().numpy().astype(ml_dtypes.bfloat16)
    bias_array = bias.float().numpy().astype(np.float32)
    writer.add_tensor(
        "mm.a.mlp.2.weight",
        weight_array,
        raw_dtype=gguf.GGMLQuantizationType.BF16,
    )
    writer.add_tensor(
        "mm.a.mlp.2.bias",
        bias_array,
        raw_dtype=gguf.GGMLQuantizationType.F32,
    )
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    partial.replace(out)

    inspection = gguf.GGUFReader(str(out), "r")
    tensors = {tensor.name: tensor for tensor in inspection.tensors}
    final_weight = tensors.get("mm.a.mlp.2.weight")
    if final_weight is None or tuple(int(v) for v in final_weight.shape) != (
        int(weight.shape[1]),
        int(weight.shape[0]),
    ):
        raise AudioBridgeProjectorError("written final projector failed shape validation")
    report = {
        "schema": AUDIO_BRIDGE_PROJECTOR_SCHEMA,
        "output": str(out),
        "size_bytes": out.stat().st_size,
        "sha256": _sha256(out),
        "base_projector": {
            "path": str(base_path),
            "size_bytes": base_path.stat().st_size,
            "sha256": _sha256(base_path),
            "tensor_count": len(base.tensors),
        },
        "omni_audio_donor": {
            "path": str(donor_path),
            "size_bytes": donor_path.stat().st_size,
            "sha256": _sha256(donor_path),
            "copied_tensor_count": len(donor_audio),
            "thinker_included": False,
            "vision_tensors_included": False,
        },
        "bridge": checkpoint_config["bridge"],
        "output_tensor_count": len(inspection.tensors),
        "modalities": ["vision", "audio"],
    }
    report_path = out.with_suffix(out.suffix + ".report.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report"] = str(report_path)
    return report
