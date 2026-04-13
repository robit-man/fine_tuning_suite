#!/usr/bin/env python3
"""GGUF-level text-tensor substitution.

Why this exists
---------------
Ollama's native Qwen3.5 multimodal support expects a single combined GGUF
with architecture='qwen35' containing BOTH text (`blk.*`) and vision
(`v.*`) tensors, plus `qwen35.vision.*` metadata. llama.cpp's current
`convert_hf_to_gguf.py --mmproj` flag produces a separate `clip.*`
architecture mmproj file which is not compatible with Ollama's qwen35
loader.

So: start from base `qwen3.5:9b` (which has the correct combined layout)
and swap out only the 427 text tensors with our fine-tuned weights, keeping
all 441 vision tensors + 15 MTP tensors untouched.

Tensor name maps perfectly between our text-only GGUF and the base
combined GGUF (both were produced by llama.cpp and use identical
`blk.{idx}.{kind}.weight` naming). Only the data bytes change.

Output
------
A new combined GGUF under outputs/gguf_combined/<output_name>/ with:
  - base vision tensors (unchanged, already Q4_K_M)
  - base MTP tensors (unchanged)
  - OUR text tensors, re-quantized to Q4_K_M to match base format
  - all KV metadata from base (including qwen35.vision.*)

Then we quantize the text side via llama-quantize to match the base's
Q4_K_M format.

Practical approach: produce a combined F16 GGUF first, then use
llama-quantize to convert the whole thing to Q4_K_M. This avoids
tensor-by-tensor requantization which would require re-implementing
Q4_K_M kernels in Python.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
import os

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# Allow overriding paths via environment variables for flexible reuse
BASE_GGUF = Path(os.environ.get(
    "BASE_GGUF",
    "/srv/ollama/models/blobs/sha256-dec52a44569a2a25341c4e4d3fee25846eed4f6f0b936278e3a3c900bb99d37c",
))
OURS_TEXT_F16 = Path(os.environ.get(
    "OURS_TEXT_F16",
    str(ROOT / "outputs" / "gguf_vision" / "qwen3.5-9b-qwen3.6-reasoning-distilled" / "model.f16.gguf"),
))
OUT_DIR = Path(os.environ.get(
    "OUT_DIR",
    str(ROOT / "outputs" / "gguf_combined" / "qwen3.5-9b-qwen3.6-reasoning-distilled"),
))
OUT_F16 = OUT_DIR / "combined.f16.gguf"


def log(msg: str) -> None:
    print(f"[gguf-surgery] {msg}", flush=True)


def dequantize_preserve(tensor):
    """Return (numpy_array, target_dtype). Preserve native types (F32/F16)
    and dequantize quantized types to F32 (loader-safe for norms/embeddings).
    """
    tt = tensor.tensor_type
    if tt == gguf.GGMLQuantizationType.F32:
        arr = np.asarray(tensor.data, dtype=np.float32)
        return arr, gguf.GGMLQuantizationType.F32
    if tt == gguf.GGMLQuantizationType.F16:
        arr = np.asarray(tensor.data, dtype=np.float16)
        return arr, gguf.GGMLQuantizationType.F16
    if tt == gguf.GGMLQuantizationType.BF16:
        arr = np.frombuffer(tensor.data.tobytes(), dtype=np.uint16).astype(np.float32)
        # Reinterpret uint16 → bfloat16 via bit shift to float32
        arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr, gguf.GGMLQuantizationType.F32
    # Quantized types: dequantize to F32 for loader safety
    arr = gguf.quants.dequantize(tensor.data, tt)
    return arr.astype(np.float32), gguf.GGMLQuantizationType.F32


def main() -> None:
    if not BASE_GGUF.exists():
        sys.exit(f"base gguf not found: {BASE_GGUF}")
    if not OURS_TEXT_F16.exists():
        sys.exit(f"ours text gguf not found: {OURS_TEXT_F16}")

    if OUT_DIR.exists():
        log(f"cleaning {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    log(f"reading base combined GGUF: {BASE_GGUF}")
    base = gguf.GGUFReader(str(BASE_GGUF))
    log(f"  KV fields: {len(base.fields)},  tensors: {len(base.tensors)}")

    log(f"reading our text-only F16 GGUF: {OURS_TEXT_F16}")
    ours = gguf.GGUFReader(str(OURS_TEXT_F16))
    log(f"  KV fields: {len(ours.fields)},  tensors: {len(ours.tensors)}")

    # Index our tensors by name, but also apply the known name drift:
    # newer llama.cpp uses `blk.X.ssm_dt.bias` for what base calls `blk.X.ssm_dt`
    ours_by_name: dict[str, object] = {}
    for t in ours.tensors:
        ours_by_name[t.name] = t
        # Map to the older base name for Gated DeltaNet delta-time bias
        if t.name.endswith(".ssm_dt.bias"):
            alt = t.name.replace(".ssm_dt.bias", ".ssm_dt")
            ours_by_name[alt] = t
    log(f"  indexed {len(ours_by_name)} tensors from ours (incl. name-drift aliases)")

    # Match tensors: which base tensors have a corresponding name in ours?
    base_text_names = [t.name for t in base.tensors if t.name in ours_by_name]
    base_other_names = [t.name for t in base.tensors if t.name not in ours_by_name]
    log(f"matched {len(base_text_names)} text tensors in base that exist in ours")
    log(f"  {len(base_other_names)} other tensors stay from base (vision + mtp + extras)")
    unmatched_in_ours = [n for n in ours_by_name if n not in {t.name for t in base.tensors}]
    if unmatched_in_ours:
        log(f"  WARN: {len(unmatched_in_ours)} tensors in ours not present in base (first 5):")
        for n in unmatched_in_ours[:5]:
            log(f"    {n}")

    # Shape check on matched tensors
    mismatches = []
    for t in base.tensors:
        if t.name not in ours_by_name:
            continue
        our_t = ours_by_name[t.name]
        if tuple(t.shape) != tuple(our_t.shape):
            mismatches.append((t.name, tuple(t.shape), tuple(our_t.shape)))
    if mismatches:
        log(f"SHAPE MISMATCHES on {len(mismatches)} tensors:")
        for n, a, b in mismatches[:10]:
            log(f"  {n}: base={a} ours={b}")
        sys.exit("aborting due to shape mismatches")
    log("all shapes match")

    # Build the output writer
    log(f"writing combined F16 GGUF → {OUT_F16}")
    writer = gguf.GGUFWriter(str(OUT_F16), arch="qwen35")

    # Copy ALL KV fields from base (including qwen35.vision.*)
    skip_meta = {
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "general.architecture",
        "general.file_type",
        "general.quantization_version",
    }
    log(f"copying {len(base.fields)} KV fields from base (skipping housekeeping + empty arrays)")
    for field_name, field in base.fields.items():
        if field_name in skip_meta:
            continue
        # Extract typed value from the field
        try:
            values = field.contents()
        except Exception as e:
            log(f"  warn: could not read field {field_name}: {e}")
            continue
        # gguf writer can't handle empty arrays — skip them
        if isinstance(values, list) and len(values) == 0:
            log(f"  skip empty-array field: {field_name}")
            continue
        # llama.cpp version drift: current build expects 4-element
        # rope.dimension_sections; base was produced with 3. Pad with 0.
        if field_name == "qwen35.rope.dimension_sections" and isinstance(values, list) and len(values) == 3:
            values = list(values) + [0]
            log(f"  pad {field_name} 3→4 elements (trailing 0)")
        try:
            writer.add_key_value(field_name, values, field.types[0])
        except Exception as e:
            log(f"  warn: could not add field {field_name}: {e}")

    # Add tensors: text from ours, everything else from base
    # For each base tensor we want to preserve its NATIVE precision class:
    #   - F32 stays F32 (important for norms and embeddings)
    #   - F16 stays F16
    #   - Quantized (Q4_K/Q6_K/BF16) → dequantize to F32 (safer than F16)
    # For text tensors where ours has F16 and base expected F32, upcast ours.
    log("adding tensors...")
    t0 = time.time()
    n_ours = 0
    n_base = 0
    from collections import Counter
    out_type_counts = Counter()

    for t in base.tensors:
        if t.name in ours_by_name:
            our_t = ours_by_name[t.name]
            # Read our data as f32 (we can always downcast if needed)
            if our_t.tensor_type == gguf.GGMLQuantizationType.F16:
                our_arr = np.asarray(our_t.data, dtype=np.float16).astype(np.float32)
            elif our_t.tensor_type == gguf.GGMLQuantizationType.F32:
                our_arr = np.asarray(our_t.data, dtype=np.float32)
            else:
                our_arr = gguf.quants.dequantize(our_t.data, our_t.tensor_type).astype(np.float32)
            our_arr = our_arr.reshape(tuple(int(x) for x in our_t.shape))

            # Target dtype follows base's original tensor type class:
            #   F32 → keep F32
            #   F16 → write F16
            #   quantized → write F32 (dequantized) — llama-quantize can later
            #   pack to Q4_K_M from F32 uniformly
            if t.tensor_type == gguf.GGMLQuantizationType.F32:
                arr_out = our_arr.astype(np.float32)
                rd = gguf.GGMLQuantizationType.F32
            else:
                arr_out = our_arr.astype(np.float16)
                rd = gguf.GGMLQuantizationType.F16
            writer.add_tensor(t.name, arr_out, raw_shape=arr_out.shape, raw_dtype=rd)
            out_type_counts[rd.name] += 1
            n_ours += 1
        else:
            arr, rd = dequantize_preserve(t)
            arr = arr.reshape(tuple(int(x) for x in t.shape))
            writer.add_tensor(t.name, arr, raw_shape=arr.shape, raw_dtype=rd)
            out_type_counts[rd.name] += 1
            n_base += 1
        if (n_ours + n_base) % 100 == 0:
            log(f"  ...added {n_ours + n_base} tensors")

    log(f"total added: ours={n_ours} base={n_base}  (elapsed: {time.time() - t0:.1f}s)")
    log(f"output type counts: {dict(out_type_counts)}")

    log("writing header + KV + tensors to disk")
    t0 = time.time()
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    log(f"  wrote {OUT_F16.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.1f}s")

    log("verifying output")
    check = gguf.GGUFReader(str(OUT_F16))
    log(f"  KV: {len(check.fields)}  tensors: {len(check.tensors)}")
    names = [t.name for t in check.tensors]
    log(f"  text blk.*  : {sum(1 for n in names if n.startswith('blk.'))}")
    log(f"  vision v.*  : {sum(1 for n in names if n.startswith('v.'))}")
    log(f"  mtp mtp.*   : {sum(1 for n in names if n.startswith('mtp.'))}")
    log(f"  other       : {sum(1 for n in names if not n.startswith(('blk.','v.','mtp.')))}")


if __name__ == "__main__":
    main()
