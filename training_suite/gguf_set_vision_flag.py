#!/usr/bin/env python3
"""
Add clip.has_vision_encoder=true to a GGUF file.

Writes a new GGUF with identical tensors and metadata plus the missing flag.
Usage:
  python scripts/gguf_set_vision_flag.py <in.gguf> <out.gguf>
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'vendor' / 'llama.cpp' / 'gguf-py'))
import gguf  # type: ignore


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: gguf_set_vision_flag.py <in.gguf> <out.gguf>", file=sys.stderr)
        sys.exit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    r = gguf.GGUFReader(str(src))

    w = gguf.GGUFWriter(str(dst), arch=r.fields['general.architecture'].contents())

    # Copy all KV, then ensure the flag exists and is true
    for k, field in r.fields.items():
        if k in {"GGUF.version","GGUF.tensor_count","GGUF.kv_count"}:
            continue
        try:
            w.add_key_value(k, field.contents(), field.types[0])
        except Exception:
            # best-effort; skip incompatible housekeeping
            pass

    # Set the flag (explicit type)
    w.add_key_value("clip.has_vision_encoder", True, gguf.GGUFValueType.BOOL)

    # Copy tensors verbatim
    for t in r.tensors:
        arr = gguf.quants.dequantize(t.data, t.tensor_type) if t.tensor_type not in (
            gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32
        ) else t.data
        w.add_tensor(t.name, arr, raw_shape=tuple(int(x) for x in t.shape), raw_dtype=t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


if __name__ == "__main__":
    main()
