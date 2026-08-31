from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GGUFInspection:
    path: str | None = None
    architecture: str | None = None
    context_length: int | None = None
    quantization: str | None = None
    tensor_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tensor_prefix_counts: dict[str, int] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "architecture": self.architecture,
            "context_length": self.context_length,
            "quantization": self.quantization,
            "tensor_count": self.tensor_count,
            "metadata": self.metadata,
            "tensor_prefix_counts": self.tensor_prefix_counts,
            "capabilities": self.capabilities,
            "flags": self.flags,
            "error": self.error,
        }


def _metadata_str(metadata: dict[str, Any]) -> str:
    return "\n".join(f"{k}={v}" for k, v in metadata.items()).lower()


def infer_capabilities(
    *,
    architecture: str | None,
    metadata: dict[str, Any],
    tensor_names: list[str] | None = None,
    chat_template: str | None = None,
) -> tuple[list[str], list[str]]:
    tensor_names = tensor_names or []
    meta_text = _metadata_str(metadata)
    template = (chat_template or str(metadata.get("tokenizer.chat_template", ""))).lower()
    capabilities = ["completion"] if architecture or metadata else []
    flags: list[str] = []

    has_vision_meta = ".vision." in meta_text or "vision_start_token_id" in meta_text
    has_vision_tensors = any(name.startswith(("v.", "clip.", "vision.")) for name in tensor_names)
    if has_vision_meta and has_vision_tensors:
        capabilities.append("vision")
    elif has_vision_meta or has_vision_tensors:
        flags.append("missing vision tensor/metadata counterpart")

    if "video_token_id" in meta_text and has_vision_tensors:
        capabilities.append("video-input")

    has_audio_meta = any(
        marker in meta_text
        for marker in ("audio_token_id", ".audio.", "audio_encoder", "audio.projector")
    )
    has_audio_tensors = any(
        name.startswith(("a.", "audio.", "audio_encoder.", "mm.audio."))
        for name in tensor_names
    )
    if has_audio_meta and has_audio_tensors:
        capabilities.append("audio-input")
    elif has_audio_meta or has_audio_tensors:
        flags.append("missing audio tensor/metadata counterpart")

    has_talker = "talker" in meta_text or any(name.startswith("talker.") for name in tensor_names)
    has_code2wav = "code2wav" in meta_text or any(
        name.startswith(("code2wav.", "codec.")) for name in tensor_names
    )
    if has_talker and has_code2wav:
        capabilities.append("audio-output")
    elif has_talker or has_code2wav:
        flags.append("incomplete audio-output Talker/code2wav pair")

    if "tool_call" in template or "<tools>" in template:
        capabilities.append("tools")
    else:
        flags.append("missing tool-aware chat template")

    if "<think>" in template or "reasoning" in template:
        capabilities.append("thinking")

    return sorted(set(capabilities)), flags


def inspect_metadata(
    *,
    architecture: str | None,
    metadata: dict[str, Any],
    tensor_names: list[str] | None = None,
    path: str | None = None,
    chat_template: str | None = None,
) -> GGUFInspection:
    capabilities, flags = infer_capabilities(
        architecture=architecture,
        metadata=metadata,
        tensor_names=tensor_names,
        chat_template=chat_template,
    )
    prefix_counts: dict[str, int] = {}
    for name in tensor_names or []:
        prefix = name.split(".", 1)[0] if "." in name else name
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    return GGUFInspection(
        path=path,
        architecture=architecture,
        context_length=_int_or_none(
            metadata.get(f"{architecture}.context_length") if architecture else None
        ),
        quantization=str(metadata.get("general.file_type", "")) or None,
        tensor_count=len(tensor_names or []),
        metadata=metadata,
        tensor_prefix_counts=prefix_counts,
        capabilities=capabilities,
        flags=flags,
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def inspect_gguf(path: Path) -> GGUFInspection:
    if not path.exists():
        return GGUFInspection(path=str(path), error=f"GGUF not found: {path}")
    try:
        import gguf  # type: ignore
    except Exception as exc:
        return GGUFInspection(
            path=str(path),
            error=f"gguf Python package unavailable: {type(exc).__name__}: {exc}",
        )

    try:
        reader = gguf.GGUFReader(str(path))
        metadata: dict[str, Any] = {}
        for key, field in reader.fields.items():
            try:
                metadata[key] = field.contents()
            except Exception:
                metadata[key] = str(field)
        architecture = str(metadata.get("general.architecture") or "")
        tensor_names = [tensor.name for tensor in reader.tensors]
        inspection = inspect_metadata(
            architecture=architecture,
            metadata=metadata,
            tensor_names=tensor_names,
            path=str(path),
        )
        file_type = metadata.get("general.file_type")
        inspection.quantization = str(file_type) if file_type is not None else None
        return inspection
    except Exception as exc:
        return GGUFInspection(
            path=str(path),
            error=f"{type(exc).__name__}: {exc}",
        )


def compatible_for_splice(source: dict[str, Any], target: dict[str, Any]) -> tuple[bool, list[str]]:
    """Conservative architecture compatibility gate for text/vision splicing."""
    errors: list[str] = []
    checks = [
        ("architecture", source.get("architecture"), target.get("architecture")),
        ("hidden_size", source.get("hidden_size"), target.get("hidden_size")),
        ("text_hidden_size", source.get("text_hidden_size"), target.get("text_hidden_size")),
        ("num_hidden_layers", source.get("num_hidden_layers"), target.get("num_hidden_layers")),
        ("vocab_size", source.get("vocab_size"), target.get("vocab_size")),
        ("vision_out_hidden_size", source.get("vision_out_hidden_size"), target.get("vision_out_hidden_size")),
    ]
    for label, left, right in checks:
        if left is None or right is None:
            continue
        if left != right:
            errors.append(f"{label}: {left!r} != {right!r}")
    return not errors, errors
