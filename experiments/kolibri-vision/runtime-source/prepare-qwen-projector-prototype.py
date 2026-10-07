"""Reuse the local pretrained Qwen vision tower with a new output projection.

The prototype projection is random and explicitly untrained. It qualifies the
real dynamic-resolution encoder in stock Ollama before any training-data pull.
The same file serves as a frozen feature tap for a differentiable training
bridge; the final trained mm.2 projection replaces this prototype at export.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / "training_suite/vendor/kolibri-port/llama.cpp/gguf-py"))
import gguf

ROOT = Path(__file__).resolve().parents[1]
DONOR = Path("/srv/ollama/models/blobs/sha256-2635637dd72b4d39a97331373a73366b69a0a6a14151b2a1da0a97b311e2461f")


def main():
    output = ROOT / "compatibility/qwen-vision-prototype32.gguf"
    if output.exists():
        raise FileExistsError(output)
    if hashlib.file_digest(DONOR.open("rb"), "sha256").hexdigest() != DONOR.name.removeprefix("sha256-"):
        raise RuntimeError("Existing donor bytes do not match their immutable blob digest")
    reader = gguf.GGUFReader(DONOR)
    if reader.fields["clip.projector_type"].contents() != "qwen3vl_merger":
        raise RuntimeError("Wrong donor projector type")
    if any(reader.fields["clip.vision.is_deepstack_layers"].contents()):
        raise RuntimeError("The donor requires decoder deepstack injection")
    random = np.random.default_rng(6110626)
    replacements = {
        "mm.2.weight": random.normal(0, .0002, (32, 4608)).astype(np.float32),
        "mm.2.bias": np.zeros(32, np.float32),
    }
    size = sum(replacements[t.name].nbytes if t.name in replacements else t.n_bytes for t in reader.tensors)
    if size > 2 * 1024**3 or shutil.disk_usage(output.parent).free < size + 2 * 1024**3:
        raise RuntimeError("Prototype exceeds its bounded storage allocation")
    partial = output.with_suffix(".gguf.partial")
    writer = gguf.GGUFWriter(partial, "clip")
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value = field.contents()
        if key == "clip.vision.projection_dim": value = 32
        if key == "general.name": value = "Kolibri vision compatibility prototype; UNTRAINED"
        writer.add_key_value(key, value, field.types[0], field.types[-1] if len(field.types) > 1 else None)
    writer.add_string("kolibri.vision.prototype_status", "untrained interface fixture; must not publish")
    writer.add_string("kolibri.vision.donor_digest", DONOR.name.removeprefix("sha256-"))
    for tensor in reader.tensors:
        if tensor.name in replacements:
            value = replacements[tensor.name]
            writer.add_tensor_info(tensor.name, value.shape, value.dtype, value.nbytes,
                                   raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor_info(tensor.name, tuple(reversed(tensor.shape.tolist())), np.dtype("float32"),
                                   tensor.n_bytes, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    fp = writer.fout[0]
    for tensor in reader.tensors:
        value = replacements.get(tensor.name, tensor.data)
        raw = memoryview(value.reshape(-1)).cast("B")
        writer.write_padding(fp, fp.tell())
        fp.write(raw)
        writer.write_padding(fp, len(raw))
    writer.close()
    partial.rename(output)
    digest = hashlib.file_digest(output.open("rb"), "sha256").hexdigest()
    report = {"donor_blob": str(DONOR), "donor_sha256": DONOR.name.removeprefix("sha256-"),
              "donor_tag": "robit/ornith-1.5:9b", "architecture": "clip", "projector_type": "qwen3vl_merger",
              "vision_encoder_unchanged": True, "deepstack_required": False,
              "output_projection": "random F32 4608 -> 32; synthetic interface fixture only",
              "output": str(output), "sha256": digest, "bytes": output.stat().st_size,
              "trained_vision_capability_claim": False}
    (ROOT / "compatibility/qwen-prototype-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
