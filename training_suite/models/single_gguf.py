from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from training_suite.models.omni_adapter import adapter_contract


BUNDLE_SCHEMA = "robit.ollama-monolithic-audio.v1"
BUNDLE_NAMESPACE = "robit.audio_bundle"
MAX_GGML_TENSOR_NAME_BYTES = 127


@dataclass(frozen=True)
class EmbeddedComponent:
    name: str
    role: str
    tensor_prefix: str
    metadata_prefix: str
    source: str
    input_modalities: tuple[str, ...]
    output_modalities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["input_modalities"] = list(self.input_modalities)
        data["output_modalities"] = list(self.output_modalities)
        return data


COMPREHENSION_COMPONENT = EmbeddedComponent(
    name="comprehension",
    role="audio-video-understanding",
    tensor_prefix="a.c.",
    metadata_prefix=f"{BUNDLE_NAMESPACE}.component.comprehension.kv.",
    source="",
    input_modalities=("audio", "video", "text"),
    output_modalities=("text",),
)

TTS_COMPONENT = EmbeddedComponent(
    name="tts",
    role="text-to-speech",
    tensor_prefix="s.t.",
    metadata_prefix=f"{BUNDLE_NAMESPACE}.component.tts.kv.",
    source="",
    input_modalities=("text",),
    output_modalities=("audio",),
)


class SingleGGUFError(RuntimeError):
    """Raised when a monolithic Ollama GGUF cannot be packed safely."""


def audio_router_contract() -> dict[str, Any]:
    """Return the wire ABI plus its binding to the one-file GGUF layout."""
    contract = adapter_contract()
    contract["artifact"] = {
        "bundle_schema": BUNDLE_SCHEMA,
        "base_tensor_view": "unprefixed tensors",
        "comprehension_tensor_view": "a.c.* with prefix stripped",
        "tts_tensor_view": "s.t.* with prefix stripped",
    }
    return contract


def monolithic_bundle_manifest(
    *,
    base_source: str,
    comprehension_source: str,
    tts_source: str,
    base_architecture: str | None = None,
) -> dict[str, Any]:
    comprehension = EmbeddedComponent(
        **{**asdict(COMPREHENSION_COMPONENT), "source": comprehension_source}
    )
    tts = EmbeddedComponent(**{**asdict(TTS_COMPONENT), "source": tts_source})
    return {
        "schema": BUNDLE_SCHEMA,
        "physical_artifacts": 1,
        "container": "GGUF v3",
        "ollama_import": "FROM ./model.gguf",
        "base": {
            "source": base_source,
            "architecture": base_architecture,
            "tensor_namespace": "unmodified",
            "role": "reasoning-tools-thinking",
        },
        "components": [comprehension.to_dict(), tts.to_dict()],
        "contract": audio_router_contract(),
        "runtime": {
            "stock_ollama_text_import": True,
            "custom_audio_handler_required": True,
            "loader_behavior": (
                "Keep base tensors unmodified; expose filtered GGUF views by stripping "
                "a.c. and s.t. tensor prefixes plus their metadata namespaces."
            ),
            "memory_policy": "load components lazily and evict independently",
        },
        "limitations": [
            "One physical GGUF contains multiple executable graphs; it is not one fused transformer.",
            "The comprehension component must be self-contained unless a learned target-language bridge exists.",
            "TTS must be text-conditioned; an Omni Talker tied to a mismatched Thinker cannot consume target hidden states.",
            "Stock Ollama does not interpret the custom audio request or response fields.",
        ],
    }


def _require_gguf() -> tuple[Any, Any]:
    try:
        import gguf  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional runtime deps
        raise SingleGGUFError(
            "GGUF packing requires the gguf and numpy packages; run with the suite virtual environment"
        ) from exc
    return gguf, np


def _field_value(field: Any) -> Any:
    return field.contents()


def _reader_metadata(reader: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, field in reader.fields.items():
        try:
            values[key] = _field_value(field)
        except Exception:
            continue
    return values


def _architecture(reader: Any) -> str:
    return str(_reader_metadata(reader).get("general.architecture") or "")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_quantized(gguf: Any, tensor: Any) -> bool:
    plain = {"F32", "F16", "F64", "I8", "I16", "I32", "I64"}
    return tensor.tensor_type.name not in plain


def _add_reader_tensor(writer: Any, tensor: Any, out_name: str, gguf: Any, np: Any) -> None:
    if len(out_name.encode("utf-8")) > MAX_GGML_TENSOR_NAME_BYTES:
        raise SingleGGUFError(
            f"tensor name exceeds {MAX_GGML_TENSOR_NAME_BYTES} bytes after namespacing: {out_name}"
        )
    data = np.asarray(tensor.data)
    if _is_quantized(gguf, tensor):
        writer.add_tensor(out_name, data, raw_dtype=tensor.tensor_type)
    else:
        # GGUFReader exposes logical shape separately; its data array is already
        # in the storage order GGUFWriter expects.
        writer.add_tensor(out_name, data, raw_dtype=tensor.tensor_type)


def _copy_metadata(
    *,
    reader: Any,
    writer: Any,
    gguf: Any,
    prefix: str = "",
    skip_existing_bundle: bool = False,
) -> tuple[int, list[str]]:
    housekeeping = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
    }
    if not prefix:
        housekeeping.add("general.architecture")
    copied = 0
    skipped: list[str] = []
    for key, field in reader.fields.items():
        if key in housekeeping:
            skipped.append(key)
            continue
        if skip_existing_bundle and key.startswith(BUNDLE_NAMESPACE + "."):
            skipped.append(key)
            continue
        try:
            value = _field_value(field)
            if isinstance(value, list) and not value:
                skipped.append(key)
                continue
            writer.add_key_value(prefix + key, value, field.types[0])
            copied += 1
        except Exception as exc:
            skipped.append(f"{key}: {type(exc).__name__}: {exc}")
    return copied, skipped


def _component_summary(path: Path, reader: Any, prefix: str, source: str) -> dict[str, Any]:
    return {
        "source": source,
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "architecture": _architecture(reader),
        "tensor_count": len(reader.tensors),
        "tensor_prefix": prefix,
    }


def pack_monolithic_gguf(
    *,
    base_gguf: Path,
    comprehension_gguf: Path,
    tts_gguf: Path,
    out_gguf: Path,
    base_source: str | None = None,
    comprehension_source: str | None = None,
    tts_source: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Pack three GGUF graphs into one Ollama-importable monolithic GGUF.

    The base model remains in its original tensor namespace. The custom runner
    exposes filtered views of `a.c.*` and `s.t.*` to the comprehension and TTS
    loaders. No attempt is made to splice incompatible hidden states.
    """
    gguf, np = _require_gguf()
    inputs = [path.expanduser().resolve() for path in (base_gguf, comprehension_gguf, tts_gguf)]
    for path in inputs:
        if not path.is_file():
            raise SingleGGUFError(f"GGUF input does not exist: {path}")
    out = out_gguf.expanduser().resolve()
    if out in inputs:
        raise SingleGGUFError("output GGUF must not overwrite an input component")
    if out.exists() and not overwrite:
        raise SingleGGUFError(f"output already exists; pass overwrite=True to replace it: {out}")

    base_reader = gguf.GGUFReader(str(inputs[0]))
    comprehension_reader = gguf.GGUFReader(str(inputs[1]))
    tts_reader = gguf.GGUFReader(str(inputs[2]))
    base_arch = _architecture(base_reader)
    if not base_arch:
        raise SingleGGUFError("base GGUF has no general.architecture metadata")
    embedded_prefixes = (
        COMPREHENSION_COMPONENT.tensor_prefix,
        TTS_COMPONENT.tensor_prefix,
    )
    already_embedded = [
        tensor.name
        for tensor in base_reader.tensors
        if tensor.name.startswith(embedded_prefixes)
    ]
    if already_embedded:
        raise SingleGGUFError(
            "base GGUF already contains reserved monolithic component tensors; "
            f"first: {already_embedded[:3]}"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        partial.unlink()

    started = time.time()
    writer = gguf.GGUFWriter(str(partial), arch=base_arch)
    try:
        base_meta_count, base_meta_skipped = _copy_metadata(
            reader=base_reader,
            writer=writer,
            gguf=gguf,
            skip_existing_bundle=True,
        )
        comp_meta_count, comp_meta_skipped = _copy_metadata(
            reader=comprehension_reader,
            writer=writer,
            gguf=gguf,
            prefix=COMPREHENSION_COMPONENT.metadata_prefix,
        )
        tts_meta_count, tts_meta_skipped = _copy_metadata(
            reader=tts_reader,
            writer=writer,
            gguf=gguf,
            prefix=TTS_COMPONENT.metadata_prefix,
        )

        components = {
            "base": _component_summary(
                inputs[0], base_reader, "", base_source or inputs[0].name
            ),
            "comprehension": _component_summary(
                inputs[1],
                comprehension_reader,
                COMPREHENSION_COMPONENT.tensor_prefix,
                comprehension_source or inputs[1].name,
            ),
            "tts": _component_summary(
                inputs[2], tts_reader, TTS_COMPONENT.tensor_prefix, tts_source or inputs[2].name
            ),
        }
        manifest = monolithic_bundle_manifest(
            base_source=components["base"]["source"],
            comprehension_source=components["comprehension"]["source"],
            tts_source=components["tts"]["source"],
            base_architecture=base_arch,
        )
        manifest["component_files"] = components

        writer.add_key_value(
            f"{BUNDLE_NAMESPACE}.schema",
            BUNDLE_SCHEMA,
            gguf.GGUFValueType.STRING,
        )
        writer.add_key_value(
            f"{BUNDLE_NAMESPACE}.manifest",
            json.dumps(manifest, separators=(",", ":"), sort_keys=True),
            gguf.GGUFValueType.STRING,
        )
        writer.add_key_value(
            f"{BUNDLE_NAMESPACE}.tensor_prefixes",
            json.dumps(
                [COMPREHENSION_COMPONENT.tensor_prefix, TTS_COMPONENT.tensor_prefix],
                separators=(",", ":"),
            ),
            gguf.GGUFValueType.STRING,
        )

        for tensor in base_reader.tensors:
            _add_reader_tensor(writer, tensor, tensor.name, gguf, np)
        for tensor in comprehension_reader.tensors:
            _add_reader_tensor(
                writer,
                tensor,
                COMPREHENSION_COMPONENT.tensor_prefix + tensor.name,
                gguf,
                np,
            )
        for tensor in tts_reader.tensors:
            _add_reader_tensor(
                writer,
                tensor,
                TTS_COMPONENT.tensor_prefix + tensor.name,
                gguf,
                np,
            )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        os.replace(partial, out)
    except Exception:
        try:
            writer.close()
        except Exception:
            pass
        if partial.exists():
            partial.unlink()
        raise

    inspection = inspect_monolithic_gguf(out)
    if not inspection["valid"]:
        raise SingleGGUFError(f"post-write bundle inspection failed: {inspection['errors']}")
    return {
        "schema": BUNDLE_SCHEMA,
        "output": str(out),
        "output_size_bytes": out.stat().st_size,
        "elapsed_s": round(time.time() - started, 2),
        "components": components,
        "metadata": {
            "base_copied": base_meta_count,
            "comprehension_copied": comp_meta_count,
            "tts_copied": tts_meta_count,
            "skipped": {
                "base": base_meta_skipped,
                "comprehension": comp_meta_skipped,
                "tts": tts_meta_skipped,
            },
        },
        "inspection": inspection,
    }


def inspect_monolithic_gguf(path: Path) -> dict[str, Any]:
    gguf, _ = _require_gguf()
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SingleGGUFError(f"GGUF does not exist: {resolved}")
    reader = gguf.GGUFReader(str(resolved))
    metadata = _reader_metadata(reader)
    manifest_raw = metadata.get(f"{BUNDLE_NAMESPACE}.manifest")
    errors: list[str] = []
    try:
        manifest = json.loads(str(manifest_raw)) if manifest_raw else None
    except json.JSONDecodeError as exc:
        manifest = None
        errors.append(f"invalid bundle manifest JSON: {exc}")
    if metadata.get(f"{BUNDLE_NAMESPACE}.schema") != BUNDLE_SCHEMA:
        errors.append("missing or unsupported bundle schema")
    counts = {
        "base": 0,
        "comprehension": 0,
        "tts": 0,
    }
    for tensor in reader.tensors:
        if tensor.name.startswith(COMPREHENSION_COMPONENT.tensor_prefix):
            counts["comprehension"] += 1
        elif tensor.name.startswith(TTS_COMPONENT.tensor_prefix):
            counts["tts"] += 1
        else:
            counts["base"] += 1
    for component in ("base", "comprehension", "tts"):
        if counts[component] == 0:
            errors.append(f"bundle has no {component} tensors")
    return {
        "valid": not errors,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "architecture": _architecture(reader),
        "tensor_count": len(reader.tensors),
        "tensor_counts": counts,
        "manifest": manifest,
        "errors": errors,
    }
