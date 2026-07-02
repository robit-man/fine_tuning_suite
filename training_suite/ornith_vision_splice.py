#!/usr/bin/env python3
"""Build Ornith vision variants by transplanting text tensors into a vision GGUF.

The pipeline is intentionally GGUF-native:

1. Read a text/tool Ornith GGUF from local Ollama storage.
2. Read a compatible multimodal Qwen-family GGUF from local Ollama storage.
3. Write a new combined GGUF using Ornith tensors where names/shapes match,
   and donor tensors everywhere else (vision tower, MTP/extras, metadata).
4. Create an Ollama model with qwen3.5 renderer/parser so Ollama advertises
   and serves vision + tools + thinking.

This avoids converting to F16 and preserves packed quantized tensor bytes for
normal replacements.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gguf  # type: ignore
except Exception as exc:  # pragma: no cover - exercised only outside suite venv
    raise SystemExit(
        "The gguf package is required. Run with training_suite/.venv/bin/python "
        "or install training_suite/requirements.txt."
    ) from exc


ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = ROOT / "training_suite"
OUT_ROOT = PACKAGE_ROOT / "outputs" / "ornith_vision"
LOG_ROOT = PACKAGE_ROOT / "logs"


@dataclass(frozen=True)
class Preset:
    size: str
    source_model: str
    donor_model: str
    local_tag: str
    remote_tag: str
    min_replaced: int
    note: str


PRESETS: dict[str, Preset] = {
    "9b": Preset(
        size="9b",
        source_model="ornith-1.0-9b-tools:q4km",
        donor_model="qwen3.5:9b",
        local_tag="ornith-vision:9b",
        remote_tag="robit/ornith-vision:9b",
        min_replaced=420,
        note="Ornith 9B text into qwen3.5:9b combined vision GGUF.",
    ),
    "35b": Preset(
        size="35b",
        source_model="ornith-1.0-35b-tools:q4km",
        donor_model="qwen3.6:35b",
        local_tag="ornith-vision:35b",
        remote_tag="robit/ornith-vision:35b",
        min_replaced=650,
        note="Ornith 35B qwen35moe text into qwen3.6:35b combined vision GGUF.",
    ),
}


HOUSEKEEPING_FIELDS = {
    "GGUF.version",
    "GGUF.tensor_count",
    "GGUF.kv_count",
    "general.architecture",
}


QUANTIZED_TYPES = {
    q
    for q in gguf.GGMLQuantizationType
    if q.name not in {"F32", "F16", "F64", "I8", "I16", "I32", "I64"}
}


def log(msg: str) -> None:
    print(f"[ornith-vision] {msg}", flush=True)


def run(cmd: list[str], *, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout[-2000:]}\n\nstderr:\n{proc.stderr[-2000:]}"
        )
    return proc


def ollama_models_root(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("OLLAMA_MODELS"):
        candidates.append(Path(os.environ["OLLAMA_MODELS"]).expanduser())
    candidates.extend([Path("/srv/ollama/models"), Path.home() / ".ollama" / "models"])
    for candidate in candidates:
        if (candidate / "manifests").exists() and (candidate / "blobs").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Ollama model storage. Pass --ollama-models-root or set OLLAMA_MODELS."
    )


def manifest_candidates(model: str, root: Path) -> list[Path]:
    name, tag = model.rsplit(":", 1) if ":" in model else (model, "latest")
    manifests = root / "manifests"
    parts = name.split("/")
    candidates: list[Path] = []
    if "/" not in name:
        candidates.append(manifests / "registry.ollama.ai" / "library" / name / tag)
    else:
        if "." in parts[0]:
            candidates.append(manifests.joinpath(*parts, tag))
        candidates.append(manifests / "registry.ollama.ai" / Path(*parts) / tag)
    return candidates


def find_manifest(model: str, root: Path) -> Path:
    for candidate in manifest_candidates(model, root):
        if candidate.exists():
            return candidate

    # Fallback for unusual local names: scan manifests and reconstruct Ollama tags.
    manifests = root / "manifests"
    for path in manifests.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(manifests).parts
        if len(rel) < 3:
            continue
        tag = rel[-1]
        prefix = rel[:-1]
        if prefix[:2] == ("registry.ollama.ai", "library"):
            found = "/".join(prefix[2:]) + f":{tag}"
        elif prefix[:1] == ("registry.ollama.ai",):
            found = "/".join(prefix[1:]) + f":{tag}"
        else:
            found = "/".join(prefix) + f":{tag}"
        if found == model or (":" not in model and found == f"{model}:latest"):
            return path
    raise FileNotFoundError(f"Ollama manifest not found for {model!r}")


def model_blob_path(model: str, root: Path) -> Path:
    manifest = find_manifest(model, root)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    for layer in data.get("layers", []):
        if layer.get("mediaType") == "application/vnd.ollama.image.model":
            digest = layer["digest"].replace(":", "-")
            path = root / "blobs" / digest
            if not path.exists():
                raise FileNotFoundError(f"blob from {manifest} is missing: {path}")
            return path
    raise RuntimeError(f"manifest has no model layer: {manifest}")


def metadata(reader: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, field in reader.fields.items():
        try:
            out[key] = field.contents()
        except Exception:
            out[key] = str(field)
    return out


def architecture(reader: Any) -> str:
    value = metadata(reader).get("general.architecture")
    return str(value or "")


def shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in tensor.shape)


def same_numel(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return prod(left) == prod(right)


def storage_shape_for_logical(logical_shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(logical_shape))


def add_aliases(tensors: list[Any]) -> dict[str, Any]:
    by_name: dict[str, Any] = {}
    for tensor in tensors:
        by_name[tensor.name] = tensor
        if tensor.name.endswith(".ssm_dt.bias"):
            by_name[tensor.name[: -len(".bias")]] = tensor
    return by_name


def is_quantized(tensor: Any) -> bool:
    return tensor.tensor_type in QUANTIZED_TYPES


def add_reader_tensor(writer: Any, out_name: str, tensor: Any, *, logical_shape: tuple[int, ...] | None = None) -> None:
    arr = np.asarray(tensor.data)
    raw_dtype = tensor.tensor_type

    if is_quantized(tensor):
        if logical_shape is not None and logical_shape != shape(tensor):
            raise ValueError(f"cannot reshape packed quantized tensor {tensor.name}")
        # For packed uint8 tensors, GGUFWriter expects the byte shape and derives
        # the logical shape from raw_dtype. Passing tensor.shape here is wrong.
        writer.add_tensor(out_name, arr, raw_dtype=raw_dtype)
        return

    if logical_shape is not None:
        storage_shape = storage_shape_for_logical(logical_shape)
        if tuple(arr.shape) != storage_shape:
            arr = arr.reshape(storage_shape)

    # GGUFReader exposes tensor.shape in logical order, while tensor.data is in
    # the storage order expected by GGUFWriter. Passing raw_shape for F16/F32
    # tensors double-reverses multi-dimensional tensors and produces models
    # that Ollama can advertise but cannot load.
    writer.add_tensor(out_name, arr, raw_dtype=raw_dtype)


def copy_metadata(donor: Any, writer: Any, *, size: str, source_model: str, donor_model: str) -> dict[str, Any]:
    copied = 0
    skipped: list[str] = []
    for field_name, field in donor.fields.items():
        if field_name in HOUSEKEEPING_FIELDS:
            skipped.append(field_name)
            continue
        try:
            values = field.contents()
        except Exception as exc:
            skipped.append(f"{field_name}: unreadable ({exc})")
            continue
        if isinstance(values, list) and not values:
            skipped.append(f"{field_name}: empty array")
            continue
        if field_name.endswith(".rope.dimension_sections") and isinstance(values, list) and len(values) == 3:
            values = list(values) + [0]
        try:
            writer.add_key_value(field_name, values, field.types[0])
            copied += 1
        except Exception as exc:
            skipped.append(f"{field_name}: {exc}")

    writer.add_key_value("general.name", f"Ornith Vision {size.upper()}", gguf.GGUFValueType.STRING)
    writer.add_key_value("general.basename", "Ornith-Vision", gguf.GGUFValueType.STRING)
    writer.add_key_value("general.finetune", f"{source_model} text + {donor_model} vision", gguf.GGUFValueType.STRING)
    writer.add_key_value("clip.has_vision_encoder", True, gguf.GGUFValueType.BOOL)
    return {"copied": copied, "skipped": skipped}


def tensor_kind(name: str) -> str:
    if name.startswith("v."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith(("blk.", "token_embd.", "output.", "output_norm.")):
        return "text"
    return "other"


def splice_gguf(
    *,
    source_path: Path,
    donor_path: Path,
    out_path: Path,
    size: str,
    source_model: str,
    donor_model: str,
    min_replaced: int,
    allow_same_numel_reshape: bool = True,
) -> dict[str, Any]:
    if out_path.exists():
        log(f"removing previous output {out_path}")
        out_path.unlink()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"reading source text GGUF: {source_path}")
    source = gguf.GGUFReader(str(source_path))
    log(f"  source tensors={len(source.tensors)} arch={architecture(source)}")

    log(f"reading donor vision GGUF: {donor_path}")
    donor = gguf.GGUFReader(str(donor_path))
    donor_arch = architecture(donor)
    log(f"  donor tensors={len(donor.tensors)} arch={donor_arch}")

    source_arch = architecture(source)
    if source_arch != donor_arch:
        raise RuntimeError(f"architecture mismatch: source={source_arch} donor={donor_arch}")

    source_by_name = add_aliases(source.tensors)
    writer = gguf.GGUFWriter(str(out_path), arch=donor_arch)
    meta_report = copy_metadata(
        donor,
        writer,
        size=size,
        source_model=source_model,
        donor_model=donor_model,
    )

    replaced = 0
    kept = 0
    reshaped: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    kept_by_kind: dict[str, int] = {}
    replaced_by_kind: dict[str, int] = {}

    log(f"writing combined GGUF: {out_path}")
    t0 = time.time()
    for i, donor_tensor in enumerate(donor.tensors, 1):
        source_tensor = source_by_name.get(donor_tensor.name)
        if source_tensor is None:
            add_reader_tensor(writer, donor_tensor.name, donor_tensor)
            kept += 1
            kept_by_kind[tensor_kind(donor_tensor.name)] = kept_by_kind.get(tensor_kind(donor_tensor.name), 0) + 1
        else:
            donor_shape = shape(donor_tensor)
            source_shape = shape(source_tensor)
            logical_shape: tuple[int, ...] | None = None
            if source_shape == donor_shape:
                pass
            elif allow_same_numel_reshape and same_numel(source_shape, donor_shape) and not is_quantized(source_tensor):
                logical_shape = donor_shape
                reshaped.append(
                    {
                        "name": donor_tensor.name,
                        "source_shape": source_shape,
                        "donor_shape": donor_shape,
                    }
                )
            else:
                mismatches.append(
                    {
                        "name": donor_tensor.name,
                        "source_shape": source_shape,
                        "donor_shape": donor_shape,
                        "source_type": source_tensor.tensor_type.name,
                        "donor_type": donor_tensor.tensor_type.name,
                    }
                )
                add_reader_tensor(writer, donor_tensor.name, donor_tensor)
                kept += 1
                kept_by_kind[tensor_kind(donor_tensor.name)] = kept_by_kind.get(tensor_kind(donor_tensor.name), 0) + 1
                continue

            add_reader_tensor(writer, donor_tensor.name, source_tensor, logical_shape=logical_shape)
            replaced += 1
            replaced_by_kind[tensor_kind(donor_tensor.name)] = replaced_by_kind.get(tensor_kind(donor_tensor.name), 0) + 1

        if i % 100 == 0:
            log(f"  ...{i}/{len(donor.tensors)} tensors")

    if mismatches:
        raise RuntimeError(f"{len(mismatches)} tensor shape mismatches; first: {mismatches[:5]}")
    if replaced < min_replaced:
        raise RuntimeError(f"only replaced {replaced} tensors, below safety threshold {min_replaced}")

    log("writing header, metadata, and tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    elapsed = round(time.time() - t0, 1)
    log(f"wrote {out_path} ({out_path.stat().st_size / 1e9:.2f} GB) in {elapsed}s")

    verify = gguf.GGUFReader(str(out_path))
    names = [tensor.name for tensor in verify.tensors]
    donor_shapes = {tensor.name: shape(tensor) for tensor in donor.tensors}
    shape_mismatches = [
        {
            "name": tensor.name,
            "output_shape": shape(tensor),
            "donor_shape": donor_shapes.get(tensor.name),
        }
        for tensor in verify.tensors
        if donor_shapes.get(tensor.name) != shape(tensor)
    ]
    if shape_mismatches:
        raise RuntimeError(
            f"post-write shape verification failed for {len(shape_mismatches)} tensors; "
            f"first: {shape_mismatches[:5]}"
        )
    report = {
        "source": str(source_path),
        "donor": str(donor_path),
        "output": str(out_path),
        "source_model": source_model,
        "donor_model": donor_model,
        "size": size,
        "architecture": donor_arch,
        "replaced": replaced,
        "kept": kept,
        "replaced_by_kind": replaced_by_kind,
        "kept_by_kind": kept_by_kind,
        "reshaped": reshaped,
        "metadata": meta_report,
        "output_size_bytes": out_path.stat().st_size,
        "elapsed_s": elapsed,
        "verify": {
            "tensor_count": len(verify.tensors),
            "text_tensors": sum(1 for name in names if tensor_kind(name) == "text"),
            "vision_tensors": sum(1 for name in names if tensor_kind(name) == "vision"),
            "mtp_tensors": sum(1 for name in names if tensor_kind(name) == "mtp"),
            "other_tensors": sum(1 for name in names if tensor_kind(name) == "other"),
            "shape_mismatches_vs_donor": 0,
        },
    }
    return report


def write_modelfile(path: Path, gguf_path: Path, *, tag: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Generated by training_suite/ornith_vision_splice.py
# {tag}: Ornith text tensors with Qwen multimodal vision tensors.

FROM ./{gguf_path.name}

TEMPLATE {{{{ .Prompt }}}}

RENDERER qwen3.5
PARSER qwen3.5
REQUIRES 0.17.1

PARAMETER num_ctx 262144
PARAMETER num_predict 16384
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER stop "<|im_end|>"
""",
        encoding="utf-8",
    )
    return path


def create_ollama_model(tag: str, modelfile: Path) -> None:
    run(["ollama", "create", tag, "-f", str(modelfile)], timeout=1800)


def copy_remote(local_tag: str, remote_tag: str) -> None:
    run(["ollama", "cp", local_tag, remote_tag], timeout=600)


def push_remote(remote_tag: str) -> None:
    run(["ollama", "push", remote_tag], timeout=10800)


def render_probe_image(path: Path, text: str = "42") -> None:
    from PIL import Image, ImageDraw, ImageFont

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (512, 512), (200, 30, 30))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 96)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text(((512 - width) / 2, (512 - height) / 2 - 16), text, font=font, fill=(255, 255, 255))
    image.save(path, "PNG")


def chat(model: str, messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    import httpx

    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
    payload.update(extra)
    response = httpx.post("http://localhost:11434/api/chat", json=payload, timeout=900)
    if response.status_code >= 400:
        raise RuntimeError(
            f"Ollama /api/chat failed for {model} with HTTP {response.status_code}:\n"
            f"{response.text[-2000:]}"
        )
    return response.json()


def test_model(tag: str, *, out: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from training_suite.evals.runner import capability_gate, tool_smoke

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    image_path = LOG_ROOT / "ornith_vision_probe_42.png"
    render_probe_image(image_path)
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    log(f"testing capabilities for {tag}")
    capability = capability_gate(tag, target=["vision", "tools", "thinking"])

    log(f"testing tool calls for {tag}")
    tools = tool_smoke(tag)

    log(f"testing thinking for {tag}")
    thinking_response = chat(
        tag,
        [{"role": "user", "content": "Think briefly, then answer: what is 19 + 23?"}],
        options={"temperature": 0, "num_predict": 256},
    )
    thinking_msg = thinking_response.get("message") or {}
    thinking_text = thinking_msg.get("thinking") or thinking_msg.get("content") or ""
    thinking_ok = bool(thinking_msg.get("thinking")) or "<think" in thinking_text.lower()

    log(f"testing vision for {tag}")
    vision_response = chat(
        tag,
        [
            {
                "role": "user",
                "content": "What number is written in this image? Reply with only the number.",
                "images": [image_b64],
            }
        ],
        # Thinking-capable models may spend the first tokens in the parsed
        # thinking field before emitting visible answer content.
        options={"temperature": 0, "num_predict": 512},
    )
    vision_msg = vision_response.get("message") or {}
    vision_content = vision_msg.get("content") or ""
    vision_ok = "42" in vision_content

    report = {
        "model": tag,
        "ok": bool(capability.get("ok")) and bool(tools.get("ok")) and thinking_ok and vision_ok,
        "capability_gate": capability,
        "tool_smoke": tools,
        "thinking": {
            "ok": thinking_ok,
            "thinking": (thinking_msg.get("thinking") or "")[:500],
            "content": (thinking_msg.get("content") or "")[:500],
        },
        "vision": {
            "ok": vision_ok,
            "image": str(image_path),
            "content": vision_content[:500],
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log(f"test report written: {out}")
    return report


def build_one(args: argparse.Namespace, preset: Preset) -> bool:
    root = ollama_models_root(args.ollama_models_root)
    source_model = args.source_model or preset.source_model
    donor_model = args.donor_model or preset.donor_model
    local_tag = args.local_tag or preset.local_tag
    remote_tag = args.remote_tag or preset.remote_tag
    out_dir = OUT_ROOT / preset.size
    out_path = Path(args.out) if args.out else out_dir / f"ornith-vision-{preset.size}.q4km.gguf"
    modelfile = out_dir / "Modelfile"
    report_path = out_dir / "splice_report.json"

    log("=" * 72)
    log(f"{preset.size}: {preset.note}")
    source_path = Path(args.source_gguf) if args.source_gguf else model_blob_path(source_model, root)
    donor_path = Path(args.donor_gguf) if args.donor_gguf else model_blob_path(donor_model, root)

    if args.reuse_existing and out_path.exists():
        log(f"reusing existing GGUF: {out_path}")
        if report_path.exists():
            log(f"existing splice report: {report_path}")
    else:
        report = splice_gguf(
            source_path=source_path,
            donor_path=donor_path,
            out_path=out_path,
            size=preset.size,
            source_model=source_model,
            donor_model=donor_model,
            min_replaced=args.min_replaced or preset.min_replaced,
            allow_same_numel_reshape=not args.no_same_numel_reshape,
        )
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log(f"splice report written: {report_path}")

    write_modelfile(modelfile, out_path, tag=local_tag)
    log(f"Modelfile written: {modelfile}")

    ok = True
    if args.create:
        create_ollama_model(local_tag, modelfile)
    if args.test:
        test_report = test_model(local_tag, out=LOG_ROOT / f"ornith_vision_test_{preset.size}.json")
        ok = bool(test_report.get("ok"))
        log(f"test ok={ok}")
    if args.copy_remote:
        copy_remote(local_tag, remote_tag)
    if args.push:
        push_remote(remote_tag)
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Splice vision tensors into Ornith 9B/35B and register Ollama variants.",
    )
    parser.add_argument("sizes", nargs="*", choices=["9b", "35b", "both"], default=["both"])
    parser.add_argument("--ollama-models-root", help="Path to Ollama models directory")
    parser.add_argument("--source-model", help="Override source text Ollama model (single-size runs only)")
    parser.add_argument("--donor-model", help="Override multimodal donor Ollama model (single-size runs only)")
    parser.add_argument("--source-gguf", help="Override source text GGUF path (single-size runs only)")
    parser.add_argument("--donor-gguf", help="Override multimodal donor GGUF path (single-size runs only)")
    parser.add_argument("--out", help="Override output GGUF path (single-size runs only)")
    parser.add_argument("--local-tag", help="Override local Ollama tag (single-size runs only)")
    parser.add_argument("--remote-tag", help="Override remote Ollama tag (single-size runs only)")
    parser.add_argument("--min-replaced", type=int, help="Override minimum replaced tensor safety threshold")
    parser.add_argument("--no-same-numel-reshape", action="store_true", help="Do not reshape same-element non-quant tensors")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse an existing output GGUF instead of splicing again")
    parser.add_argument("--create", action="store_true", help="Run ollama create after splicing")
    parser.add_argument("--test", action="store_true", help="Run capability, tool, thinking, and vision smoke tests")
    parser.add_argument("--copy-remote", action="store_true", help="Run ollama cp local tag to remote tag")
    parser.add_argument("--push", action="store_true", help="Run ollama push for the remote tag")
    args = parser.parse_args()

    sizes: list[str] = []
    for item in args.sizes or ["both"]:
        sizes.extend(["9b", "35b"] if item == "both" else [item])
    sizes = list(dict.fromkeys(sizes))

    override_fields = [
        args.source_model,
        args.donor_model,
        args.source_gguf,
        args.donor_gguf,
        args.out,
        args.local_tag,
        args.remote_tag,
    ]
    if len(sizes) > 1 and any(override_fields):
        raise SystemExit("source/donor/output/tag overrides are allowed only for a single size run")

    ok = True
    for size in sizes:
        if not build_one(args, PRESETS[size]):
            ok = False
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
