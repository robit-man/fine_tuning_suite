#!/usr/bin/env python3
"""Patch qwen35.rope.dimension_sections to a 4-element array.

The base qwen3.5:9b GGUF stores rope.dimension_sections as a 3-element
array. The current llama.cpp main (commit ~0.9.11 at time of writing)
expects 4 elements. This patches the key to append a trailing 0 so the
file loads in newer llama.cpp builds.

Reads src, rewrites entire file to dst (cheap — only metadata changes,
tensor data is copied through).
"""
from __future__ import annotations

import sys
from pathlib import Path

import gguf


def log(msg: str) -> None:
    print(f"[patch] {msg}", flush=True)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: patch_rope_sections.py <src.gguf> <dst.gguf>", file=sys.stderr)
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    if not src.exists():
        sys.exit(f"not found: {src}")

    log(f"reading {src}")
    r = gguf.GGUFReader(str(src))
    log(f"  KV fields: {len(r.fields)}  tensors: {len(r.tensors)}")

    # Find the rope.dimension_sections field and inspect
    key = "qwen35.rope.dimension_sections"
    if key not in r.fields:
        sys.exit(f"key {key} not found in {src}")
    field = r.fields[key]
    current = field.contents()
    log(f"  {key} current = {current!r}")
    if isinstance(current, list) and len(current) == 4:
        log("already 4 elements; copying source unchanged")
        dst.write_bytes(src.read_bytes())
        return
    if not isinstance(current, list) or len(current) != 3:
        sys.exit(f"unexpected shape for {key}: {current!r}")
    new_sections = list(current) + [0]
    log(f"  patching to {new_sections!r}")

    # Write new GGUF file with everything from src except this one field
    skip_meta = {
        "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
        "general.architecture", "general.file_type", "general.quantization_version",
    }
    w = gguf.GGUFWriter(str(dst), arch="qwen35")

    for field_name, f in r.fields.items():
        if field_name in skip_meta:
            continue
        if field_name == key:
            w.add_key_value(key, new_sections, f.types[0])
            continue
        try:
            values = f.contents()
        except Exception as e:
            log(f"  warn: cannot read {field_name}: {e}")
            continue
        if isinstance(values, list) and len(values) == 0:
            log(f"  skip empty-array field: {field_name}")
            continue
        try:
            w.add_key_value(field_name, values, f.types[0])
        except Exception as e:
            log(f"  warn: cannot add {field_name}: {e}")

    # Copy tensors through, preserving types
    log("copying tensors (preserving types)...")
    for i, t in enumerate(r.tensors):
        # Keep original dtype + raw bytes; use raw_shape + tobytes via add_tensor
        import numpy as np
        tt = t.tensor_type
        arr = np.asarray(t.data)
        w.add_tensor(t.name, arr, raw_shape=tuple(int(x) for x in t.shape), raw_dtype=tt)
        if (i + 1) % 100 == 0:
            log(f"  ...added {i+1} tensors")
    log(f"  total tensors added: {len(r.tensors)}")

    log("writing header + kv + tensors")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    log(f"done: {dst.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
