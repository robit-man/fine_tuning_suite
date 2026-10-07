"""Check learned GGUF projection against the actual training checkpoint."""
import importlib.util
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint',type=Path,default=ROOT/'pilot-projector.pt')
parser.add_argument('--projector',type=Path,default=ROOT/'Kolibri-1-vision-pilot-mmproj.gguf')
parser.add_argument('--report',type=Path,default=ROOT/'compatibility/pilot-native-projection-proof.json')
args=parser.parse_args()
torch.set_num_threads(8)
spec = importlib.util.spec_from_file_location("features", ROOT / "runtime-source/vision-features.py")
features_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(features_module)
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
extractor = features_module.VisionFeatures(ROOT, gpu=False, min_tokens=4, max_tokens=128,
    text_model=ROOT.parent / "kolibri-1-20261004/Kolibri-1-stock-BF16-release.gguf",
    projector=args.projector)
examples = [json.loads(line) for line in (ROOT / "data-pilot-counted/examples.jsonl").read_text().splitlines()]
example = next(e for e in examples if e["pilot_split"] == "development_holdout" and e["config"] == "docvqa")
with Image.open(ROOT / "data-pilot-counted" / example["image"]) as image:
    frozen = torch.from_numpy(extractor.encode(image))
actual = extractor.projected()
with torch.no_grad():
    expected = torch.nn.functional.linear(frozen, checkpoint["state_dict"]["weight"], checkpoint["state_dict"]["bias"]).numpy()
delta = actual - expected
rmse, largest = float(np.sqrt(np.mean(delta**2))), float(np.max(np.abs(delta)))
report = {"passed": rmse < 1e-5 and largest < 1e-4, "scope": "held-out image, exported learned projection matches checkpoint using the installed official stock vision library; image understanding quality unqualified",
    "checkpoint":str(args.checkpoint.resolve()),"projector":str(args.projector.resolve()),
    "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    "optimizer_updates": checkpoint["optimizer_updates"], "image": example["id"], "image_tokens": len(actual),
    "projection_width": actual.shape[1], "projection_rmse": rmse, "projection_max_error": largest,
    "text_model": "released full Kolibri BF16", "runtime_adapter_required": False, "published": False}
with args.projector.open('rb') as file:report['projector_sha256']=hashlib.file_digest(file,'sha256').hexdigest()
args.report.write_text(json.dumps(report,indent=2)+"\n")
extractor.close()
print(json.dumps(report))
raise SystemExit(0 if report["passed"] else 1)
