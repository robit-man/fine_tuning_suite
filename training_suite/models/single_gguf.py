from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training_suite.models.omni_adapter import adapter_contract

BUNDLE_SCHEMA = "robit.ollama-monolithic-omni.v3"
BUNDLE_NAMESPACE = "robit.audio_bundle"
CONTAINER_FORMAT = "robit-namespaced-multigraph-gguf-v1"
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


BASE_PROJECTOR_COMPONENT = EmbeddedComponent(
    name="base_projector",
    role="base-vision-projector",
    tensor_prefix="b.p.",
    metadata_prefix=f"{BUNDLE_NAMESPACE}.component.base_projector.kv.",
    source="",
    input_modalities=("image", "video"),
    output_modalities=("embedding",),
)

COMPREHENSION_COMPONENT = EmbeddedComponent(
    name="comprehension_model",
    role="audio-video-understanding",
    tensor_prefix="a.c.m.",
    metadata_prefix=f"{BUNDLE_NAMESPACE}.component.comprehension_model.kv.",
    source="",
    input_modalities=("audio", "image", "video", "text"),
    output_modalities=("text",),
)

COMPREHENSION_PROJECTOR_COMPONENT = EmbeddedComponent(
    name="comprehension_projector",
    role="audio-vision-projector",
    tensor_prefix="a.c.p.",
    metadata_prefix=f"{BUNDLE_NAMESPACE}.component.comprehension_projector.kv.",
    source="",
    input_modalities=("audio", "image", "video"),
    output_modalities=("embedding",),
)

TTS_COMPONENT = EmbeddedComponent(
    name="tts_model",
    role="text-to-speech",
    tensor_prefix="s.t.m.",
    metadata_prefix=f"{BUNDLE_NAMESPACE}.component.tts_model.kv.",
    source="",
    input_modalities=("text",),
    output_modalities=("codec_tokens",),
)

TTS_PROJECTOR_COMPONENT = EmbeddedComponent(
    name="tts_projector",
    role="codec-and-waveform-generator",
    tensor_prefix="s.t.p.",
    metadata_prefix=f"{BUNDLE_NAMESPACE}.component.tts_projector.kv.",
    source="",
    input_modalities=("codec_tokens",),
    output_modalities=("audio",),
)

EMBEDDED_COMPONENTS = (
    BASE_PROJECTOR_COMPONENT,
    COMPREHENSION_COMPONENT,
    COMPREHENSION_PROJECTOR_COMPONENT,
    TTS_COMPONENT,
    TTS_PROJECTOR_COMPONENT,
)
COMPONENT_BY_NAME = {component.name: component for component in EMBEDDED_COMPONENTS}
RESERVED_TENSOR_PREFIXES = tuple(component.tensor_prefix for component in EMBEDDED_COMPONENTS)


class SingleGGUFError(RuntimeError):
    """Raised when a monolithic Omni sidecar GGUF cannot be processed safely."""


def audio_router_contract() -> dict[str, Any]:
    """Return the wire ABI plus its binding to the one-file sidecar layout."""
    contract = adapter_contract()
    contract["artifact"] = {
        "bundle_schema": BUNDLE_SCHEMA,
        "base_tensor_view": "unprefixed tensors",
        "base_projector_tensor_view": "b.p.* with prefix stripped",
        "comprehension_tensor_views": "a.c.m.* and a.c.p.* with prefixes stripped",
        "tts_tensor_views": "s.t.m.* and s.t.p.* with prefixes stripped",
        "ollama_delivery": "custom sidecar layer on a stock-runnable logical model manifest",
    }
    return contract


def monolithic_bundle_manifest(
    *,
    base_source: str,
    comprehension_source: str,
    tts_source: str,
    base_architecture: str | None = None,
    base_projector_source: str | None = None,
    comprehension_projector_source: str | None = None,
    tts_projector_source: str | None = None,
) -> dict[str, Any]:
    sources = {
        "base_projector": base_projector_source,
        "comprehension_model": comprehension_source,
        "comprehension_projector": comprehension_projector_source,
        "tts_model": tts_source,
        "tts_projector": tts_projector_source,
    }
    components = [
        EmbeddedComponent(**{**asdict(component), "source": str(sources[component.name])})
        for component in EMBEDDED_COMPONENTS
        if sources[component.name]
    ]
    return {
        "schema": BUNDLE_SCHEMA,
        "physical_bundle_artifacts": 1,
        "container": "GGUF v3",
        "container_format": CONTAINER_FORMAT,
        "ollama_delivery": "attach as application/vnd.robit.ollama.omni.bundle.v1+gguf layer",
        "base": {
            "source": base_source,
            "architecture": base_architecture,
            "tensor_namespace": "unmodified",
            "role": "reasoning-tools-thinking",
        },
        "components": [component.to_dict() for component in components],
        "contract": audio_router_contract(),
        "runtime": {
            "stock_ollama_direct_bundle_import": False,
            "stock_ollama_logical_tag": True,
            "custom_media_handler_required": True,
            "loader_behavior": (
                "The stock Ollama manifest uses standard base/projector layers. The adapter "
                "materializes filtered views from this namespaced GGUF sidecar."
            ),
            "memory_policy": "load components lazily and evict independently",
        },
        "limitations": [
            "Standard GGUF requires one contiguous tensor inventory for one architecture.",
            "The sidecar is a valid GGUF container but is not directly executable by stock Ollama.",
            "The logical Ollama tag therefore retains separate standard model and projector layers.",
            "The comprehension component is self-contained rather than a hidden-state graft.",
            "TTS is text-conditioned because a mismatched Talker cannot consume base-model hidden states.",
        ],
    }


def _require_gguf() -> tuple[Any, Any]:
    try:
        import gguf  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - optional runtime dependencies
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
        except Exception:  # noqa: BLE001,S112 - GGUF field decoders expose backend errors
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


def _is_quantized(tensor: Any) -> bool:
    plain = {"F32", "F16", "F64", "I8", "I16", "I32", "I64"}
    return tensor.tensor_type.name not in plain


def _add_reader_tensor(writer: Any, tensor: Any, out_name: str, np: Any) -> None:
    if len(out_name.encode("utf-8")) > MAX_GGML_TENSOR_NAME_BYTES:
        raise SingleGGUFError(
            f"tensor name exceeds {MAX_GGML_TENSOR_NAME_BYTES} bytes after namespacing: {out_name}"
        )
    data = np.asarray(tensor.data)
    writer.add_tensor(out_name, data, raw_dtype=tensor.tensor_type)


def _copy_metadata(
    *,
    reader: Any,
    writer: Any,
    prefix: str = "",
    skip_existing_bundle: bool = False,
) -> tuple[int, list[str]]:
    housekeeping = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
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
        except Exception as exc:  # noqa: BLE001 - retain per-field conversion failures
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


def _embedded_inputs(
    *,
    base_projector_gguf: Path | None,
    comprehension_gguf: Path,
    comprehension_projector_gguf: Path | None,
    tts_gguf: Path,
    tts_projector_gguf: Path | None,
    base_projector_source: str | None,
    comprehension_source: str | None,
    comprehension_projector_source: str | None,
    tts_source: str | None,
    tts_projector_source: str | None,
) -> list[tuple[EmbeddedComponent, Path, str]]:
    raw = (
        (BASE_PROJECTOR_COMPONENT, base_projector_gguf, base_projector_source),
        (COMPREHENSION_COMPONENT, comprehension_gguf, comprehension_source),
        (
            COMPREHENSION_PROJECTOR_COMPONENT,
            comprehension_projector_gguf,
            comprehension_projector_source,
        ),
        (TTS_COMPONENT, tts_gguf, tts_source),
        (TTS_PROJECTOR_COMPONENT, tts_projector_gguf, tts_projector_source),
    )
    return [
        (component, path.expanduser().resolve(), source or path.name)
        for component, path, source in raw
        if path is not None
    ]


def pack_monolithic_gguf(
    *,
    base_gguf: Path,
    base_projector_gguf: Path | None = None,
    comprehension_gguf: Path,
    comprehension_projector_gguf: Path | None = None,
    tts_gguf: Path,
    tts_projector_gguf: Path | None = None,
    out_gguf: Path,
    base_source: str | None = None,
    base_projector_source: str | None = None,
    comprehension_source: str | None = None,
    comprehension_projector_source: str | None = None,
    tts_source: str | None = None,
    tts_projector_source: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Pack six independently executable views into one namespaced GGUF sidecar."""
    gguf, np = _require_gguf()
    base_path = base_gguf.expanduser().resolve()
    embedded_inputs = _embedded_inputs(
        base_projector_gguf=base_projector_gguf,
        comprehension_gguf=comprehension_gguf,
        comprehension_projector_gguf=comprehension_projector_gguf,
        tts_gguf=tts_gguf,
        tts_projector_gguf=tts_projector_gguf,
        base_projector_source=base_projector_source,
        comprehension_source=comprehension_source,
        comprehension_projector_source=comprehension_projector_source,
        tts_source=tts_source,
        tts_projector_source=tts_projector_source,
    )
    inputs = [base_path, *(path for _, path, _ in embedded_inputs)]
    for path in inputs:
        if not path.is_file():
            raise SingleGGUFError(f"GGUF input does not exist: {path}")
    out = out_gguf.expanduser().resolve()
    if out in inputs:
        raise SingleGGUFError("output GGUF must not overwrite an input component")
    if out.exists() and not overwrite:
        raise SingleGGUFError(f"output already exists; pass overwrite=True to replace it: {out}")

    base_reader = gguf.GGUFReader(str(base_path))
    embedded_readers = [
        (component, path, source, gguf.GGUFReader(str(path)))
        for component, path, source in embedded_inputs
    ]
    base_arch = _architecture(base_reader)
    if not base_arch:
        raise SingleGGUFError("base GGUF has no general.architecture metadata")
    already_embedded = [
        tensor.name
        for tensor in base_reader.tensors
        if tensor.name.startswith(RESERVED_TENSOR_PREFIXES)
    ]
    if already_embedded:
        raise SingleGGUFError(
            "base GGUF already contains reserved component tensors; "
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
            skip_existing_bundle=True,
        )
        metadata_counts: dict[str, int] = {"base": base_meta_count}
        metadata_skipped: dict[str, list[str]] = {"base": base_meta_skipped}
        for component, _, _, reader in embedded_readers:
            copied, skipped = _copy_metadata(
                reader=reader,
                writer=writer,
                prefix=component.metadata_prefix,
            )
            metadata_counts[component.name] = copied
            metadata_skipped[component.name] = skipped

        components = {
            "base": _component_summary(base_path, base_reader, "", base_source or base_path.name)
        }
        for component, path, source, reader in embedded_readers:
            components[component.name] = _component_summary(
                path,
                reader,
                component.tensor_prefix,
                source,
            )
        manifest = monolithic_bundle_manifest(
            base_source=components["base"]["source"],
            base_projector_source=(
                components["base_projector"]["source"] if "base_projector" in components else None
            ),
            comprehension_source=components["comprehension_model"]["source"],
            comprehension_projector_source=(
                components["comprehension_projector"]["source"]
                if "comprehension_projector" in components
                else None
            ),
            tts_source=components["tts_model"]["source"],
            tts_projector_source=(
                components["tts_projector"]["source"] if "tts_projector" in components else None
            ),
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
        for tensor in base_reader.tensors:
            _add_reader_tensor(writer, tensor, tensor.name, np)
        for component, _, _, reader in embedded_readers:
            for tensor in reader.tensors:
                _add_reader_tensor(writer, tensor, component.tensor_prefix + tensor.name, np)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        os.replace(partial, out)
    except Exception:
        try:
            writer.close()
        except Exception:  # noqa: BLE001,S110 - preserve original pack failure
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
        "metadata": {"copied": metadata_counts, "skipped": metadata_skipped},
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
    view_counts = {"base": 0, **{component.name: 0 for component in EMBEDDED_COMPONENTS}}
    for tensor in reader.tensors:
        for component in EMBEDDED_COMPONENTS:
            if tensor.name.startswith(component.tensor_prefix):
                view_counts[component.name] += 1
                break
        else:
            view_counts["base"] += 1
    counts = {
        "base": view_counts["base"] + view_counts["base_projector"],
        "comprehension": (
            view_counts["comprehension_model"] + view_counts["comprehension_projector"]
        ),
        "tts": view_counts["tts_model"] + view_counts["tts_projector"],
    }
    for component in ("base", "comprehension", "tts"):
        if counts[component] == 0:
            errors.append(f"bundle has no {component} tensors")
    declared = {
        str(component.get("name"))
        for component in (manifest or {}).get("components", [])
        if isinstance(component, Mapping)
    }
    for component_name in sorted(declared):
        if component_name not in view_counts:
            errors.append(f"bundle manifest declares unknown component {component_name}")
        elif view_counts[component_name] == 0:
            errors.append(f"bundle has no {component_name} tensors")
    return {
        "valid": not errors,
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "architecture": _architecture(reader),
        "tensor_count": len(reader.tensors),
        "tensor_counts": counts,
        "view_tensor_counts": view_counts,
        "manifest": manifest,
        "errors": errors,
    }


def _copy_view_metadata(
    *,
    reader: Any,
    writer: Any,
    component: EmbeddedComponent | None,
) -> tuple[int, list[str]]:
    copied = 0
    skipped: list[str] = []
    prefix = component.metadata_prefix if component else ""
    for key, field in reader.fields.items():
        if key in {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"}:
            continue
        if component:
            if not key.startswith(prefix):
                continue
            out_key = key.removeprefix(prefix)
        else:
            if key.startswith(BUNDLE_NAMESPACE + "."):
                continue
            out_key = key
        if out_key == "general.architecture":
            continue
        try:
            value = _field_value(field)
            if isinstance(value, list) and not value:
                skipped.append(key)
                continue
            writer.add_key_value(out_key, value, field.types[0])
            copied += 1
        except Exception as exc:  # noqa: BLE001 - retain per-field conversion failures
            skipped.append(f"{key}: {type(exc).__name__}: {exc}")
    return copied, skipped


def materialize_component_view(
    *,
    bundle_gguf: Path,
    view: str,
    out_gguf: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Materialize one byte-preserving executable GGUF view from a sidecar."""
    gguf, np = _require_gguf()
    source = bundle_gguf.expanduser().resolve()
    out = out_gguf.expanduser().resolve()
    if not source.is_file():
        raise SingleGGUFError(f"bundle GGUF does not exist: {source}")
    if source == out:
        raise SingleGGUFError("materialized output must not overwrite the bundle")
    if out.exists() and not overwrite:
        raise SingleGGUFError(f"output already exists; pass overwrite=True to replace it: {out}")
    component = None if view == "base" else COMPONENT_BY_NAME.get(view)
    if view != "base" and component is None:
        allowed = ", ".join(["base", *COMPONENT_BY_NAME])
        raise SingleGGUFError(f"unknown component view {view!r}; expected one of: {allowed}")

    reader = gguf.GGUFReader(str(source))
    metadata = _reader_metadata(reader)
    if metadata.get(f"{BUNDLE_NAMESPACE}.schema") != BUNDLE_SCHEMA:
        raise SingleGGUFError(f"unsupported or missing bundle schema in {source}")
    if component:
        architecture = str(metadata.get(component.metadata_prefix + "general.architecture") or "")
        tensor_prefix = component.tensor_prefix
    else:
        architecture = _architecture(reader)
        tensor_prefix = ""
    if not architecture:
        raise SingleGGUFError(f"view {view!r} has no general.architecture metadata")

    selected: list[tuple[Any, str]] = []
    for tensor in reader.tensors:
        if component:
            if tensor.name.startswith(tensor_prefix):
                selected.append((tensor, tensor.name.removeprefix(tensor_prefix)))
        elif not tensor.name.startswith(RESERVED_TENSOR_PREFIXES):
            selected.append((tensor, tensor.name))
    if not selected:
        raise SingleGGUFError(f"view {view!r} has no tensors")

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        partial.unlink()
    writer = gguf.GGUFWriter(str(partial), arch=architecture)
    started = time.time()
    try:
        metadata_count, metadata_skipped = _copy_view_metadata(
            reader=reader,
            writer=writer,
            component=component,
        )
        for tensor, out_name in selected:
            _add_reader_tensor(writer, tensor, out_name, np)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()
        os.replace(partial, out)
    except Exception:
        try:
            writer.close()
        except Exception:  # noqa: BLE001,S110 - preserve original materialization failure
            pass
        if partial.exists():
            partial.unlink()
        raise

    output_reader = gguf.GGUFReader(str(out))
    output_names = [tensor.name for tensor in output_reader.tensors]
    expected_names = [name for _, name in selected]
    if output_names != expected_names:
        raise SingleGGUFError(f"materialized tensor inventory mismatch for view {view!r}")
    return {
        "schema": BUNDLE_SCHEMA,
        "bundle": str(source),
        "view": view,
        "output": str(out),
        "output_size_bytes": out.stat().st_size,
        "architecture": _architecture(output_reader),
        "tensor_count": len(output_reader.tensors),
        "metadata_copied": metadata_count,
        "metadata_skipped": metadata_skipped,
        "elapsed_s": round(time.time() - started, 2),
    }
