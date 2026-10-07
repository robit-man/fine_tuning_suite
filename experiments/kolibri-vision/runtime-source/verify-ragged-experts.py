"""Validate ragged BF16 expert outputs and the declared gradient surrogate."""
import importlib.util
import json
import os
from pathlib import Path
import time
import argparse

import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--ready-file",type=Path,default=ROOT / "compatibility/ragged-kernel.ready")
parser.add_argument("--actual-shapes",action="store_true")
args = parser.parse_args()
if not os.environ.get("OLLAMA_UNIFY_GPU_LEASE"):
    raise RuntimeError("A scoped CUDA lease is required")
spec = importlib.util.spec_from_file_location("ragged", ROOT / "runtime-source/ragged-experts.py")
ragged = importlib.util.module_from_spec(spec); spec.loader.exec_module(ragged)
control_spec = importlib.util.spec_from_file_location("qualification", ROOT / "runtime-source/qualify-full-training.py")
qualification = importlib.util.module_from_spec(control_spec); control_spec.loader.exec_module(qualification)
torch.manual_seed(6100626)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
expert_count,output_width,input_width = (384,512,2560) if args.actual_shapes else (8,64,128)
devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
resident = [(device, torch.randn((expert_count, output_width, input_width), device=device, dtype=torch.bfloat16) * .1,
    torch.randn((64, input_width), device=device, dtype=torch.float32, requires_grad=True)) for device in devices]
for device in devices:torch.cuda.synchronize(device)
args.ready_file.write_text("test CUDA tensors resident on reserved devices\n")
for _ in range(30):
    if qualification.control("heartbeat")["lease"]["state"] == "active":break
    time.sleep(.25)
else: raise RuntimeError("CUDA test lease did not activate")
qualification.control("prepare")
cases = []
for device,weights,value in resident:
 for ids in [[0]*40 + [2]*10 + [expert_count-1]*14, list(range(8))*8, [expert_count//2]*64]:
    selected = torch.tensor(ids, device=device, dtype=torch.long)[:,None]
    routing = ragged.routing_tiles(selected, expert_count)
    order = routing[0]
    sorted_value = value[order]
    sorted_ids = selected.reshape(-1)[order]
    output = ragged.linear(sorted_value, weights, routing)
    reference = torch.empty_like(output)
    upstream = torch.randn_like(output)
    expected_gradient = torch.empty_like(sorted_value)
    for expert in sorted(set(ids)):
        mask = sorted_ids == expert
        if not mask.any():continue
        # Independent FP64 accumulation distinguishes summation-order noise
        # from kernel mistakes at the actual 2560-wide expert dimensions.
        reference[mask] = (sorted_value[mask].detach().to(torch.bfloat16).double() @ weights[expert].double().T).float()
        expected_gradient[mask] = (upstream[mask].to(torch.bfloat16).double() @ weights[expert].double()).float()
    observed_gradient = torch.autograd.grad(output, sorted_value, grad_outputs=upstream)[0]
    forward_error = float((output-reference).detach().abs().max())
    gradient_error = float((observed_gradient-expected_gradient).abs().max())
    forward_relative = float((output-reference).detach().norm() / reference.norm())
    gradient_relative = float((observed_gradient-expected_gradient).norm() / expected_gradient.norm())
    passed = forward_error < 3e-5 and gradient_error < 1e-5 and forward_relative < 1e-5 and gradient_relative < 1e-5 and bool(torch.isfinite(observed_gradient).all())
    cases.append({"device": device, "nonzero_expert_counts": {str(i):ids.count(i) for i in set(ids)},
        "forward_max_error": forward_error, "gradient_max_error": gradient_error,
        "forward_relative_rmse": forward_relative, "gradient_relative_rmse": gradient_relative, "passed": passed})
qualification.control("ready")
report = {"passed": all(c["passed"] for c in cases), "scope": "ragged CUDA BF16 frozen expert GEMM and explicit F32 straight-through input gradient surrogate",
    "cases": cases, "weight_shape": [expert_count,output_width,input_width],
    "peak_allocated_bytes": [torch.cuda.max_memory_allocated(d) for d in devices], "adapter_required_at_inference": False}
name = "ragged-experts-dual-actual-proof.json" if args.actual_shapes else "ragged-experts-proof.json"
(ROOT / "compatibility" / name).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)
raise SystemExit(0 if report["passed"] else 1)
