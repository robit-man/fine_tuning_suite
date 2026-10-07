"""Bake the trained bridge into a normal stock-compatible vision GGUF.

This exports a run-local pilot candidate only. Publication remains separately
gated on learned vision quality and stock multimodal/tool/thinking evaluations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / "training_suite/vendor/kolibri-port/llama.cpp/gguf-py"))
import gguf
DONOR = Path("/srv/ollama/models/blobs/sha256-2635637dd72b4d39a97331373a73366b69a0a6a14151b2a1da0a97b311e2461f")


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def export(checkpoint, output):
    if not output.resolve().is_relative_to(ROOT) or output.exists():
        raise ValueError("Choose a fresh run-local candidate path")
    if sha(DONOR) != DONOR.name.removeprefix("sha256-"):
        raise RuntimeError("Pretrained vision donor digest changed")
    trained = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if trained.get("optimizer_updates", 0) <= 0 and not trained.get("pretrained_transfer"):
        raise ValueError("No trained vision bridge is present")
    weights = trained["state_dict"]
    replacement = {"mm.2.weight": weights["weight"].float().contiguous().numpy(),
                   "mm.2.bias": weights["bias"].float().contiguous().numpy()}
    if replacement["mm.2.weight"].shape != (2560, 4608) or replacement["mm.2.bias"].shape != (2560,):
        raise ValueError("Vision projection does not match the released Kolibri embedding width")
    if any(not np.isfinite(v).all() for v in replacement.values()):
        raise ValueError("Trained projection is not finite")
    reader = gguf.GGUFReader(DONOR)
    if reader.fields["clip.projector_type"].contents() != "qwen3vl_merger" or any(reader.fields["clip.vision.is_deepstack_layers"].contents()):
        raise ValueError("Unsupported stock tower/deepstack interface")
    size = sum(replacement[t.name].nbytes if t.name in replacement else t.n_bytes for t in reader.tensors)
    if size > 2 * 2**30 or shutil.disk_usage(ROOT).free - size < 64 * 2**30:
        raise RuntimeError("Candidate exceeds its bounded storage allocation")
    partial = output.with_suffix(".gguf.partial")
    writer = gguf.GGUFWriter(partial, "clip")
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        value = field.contents()
        if key == "clip.vision.projection_dim": value = 2560
        if key == "general.name": value = "Kolibri-1 vision training pilot; capability NOT QUALIFIED"
        writer.add_key_value(key, value, field.types[0], field.types[-1] if len(field.types)>1 else None)
    writer.add_string("kolibri.vision.training_status", "engineering pilot; do not publish before independent evaluation")
    writer.add_string("kolibri.vision.donor_sha256", sha(DONOR))
    writer.add_string("kolibri.vision.checkpoint_sha256", sha(checkpoint))
    writer.add_uint32("kolibri.vision.optimizer_updates", trained["optimizer_updates"])
    if trained.get("pretrained_transfer"):
        writer.add_string("kolibri.vision.pretrained_transfer", json.dumps(trained["pretrained_transfer"], sort_keys=True))
    if trained.get('projection_calibration'):
        writer.add_string('kolibri.vision.projection_calibration',json.dumps(trained['projection_calibration'],sort_keys=True))
    for tensor in reader.tensors:
        if tensor.name in replacement:
            value = replacement[tensor.name]
            writer.add_tensor_info(tensor.name, value.shape, value.dtype, value.nbytes, raw_dtype=gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor_info(tensor.name, tuple(reversed(tensor.shape.tolist())), np.dtype("float32"), tensor.n_bytes, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_ti_data_to_file()
    fp = writer.fout[0]
    for tensor in reader.tensors:
        raw = memoryview(replacement.get(tensor.name, tensor.data).reshape(-1)).cast("B")
        writer.write_padding(fp, fp.tell()); fp.write(raw); writer.write_padding(fp, len(raw))
    writer.close(); partial.rename(output)
    candidate = gguf.GGUFReader(output)
    target = {t.name:t for t in candidate.tensors}
    unchanged = changed = 0
    for tensor in reader.tensors:
        if tensor.name in replacement:
            if not np.array_equal(target[tensor.name].data.reshape(replacement[tensor.name].shape), replacement[tensor.name]):
                raise RuntimeError("Projection export differs from the checkpoint")
            changed += 1
        else:
            if target[tensor.name].tensor_type != tensor.tensor_type or not np.array_equal(target[tensor.name].data, tensor.data):
                raise RuntimeError("Frozen pretrained tower bytes changed")
            unchanged += 1
    report = {"passed": True, "scope": "checkpoint-to-stock-GGUF tensor export; learned vision quality not qualified",
        "output": str(output), "sha256": sha(output), "bytes": output.stat().st_size,
        "donor_sha256": sha(DONOR), "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "optimizer_updates": trained["optimizer_updates"], "unchanged_tensors": unchanged, "changed_tensors": changed,
        "language_weights_changed": False, "runtime_adapter_required": False, "published": False}
    output.with_suffix(".export.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pilot-projector.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "Kolibri-1-vision-pilot-mmproj.gguf")
    args = parser.parse_args()
    export(args.checkpoint, args.output)
