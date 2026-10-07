"""Scoped full-weight decoder qualification and a gated projector-gradient pilot.

This is a bounded engineering check, not the large curriculum. A failed full
numerical gate prevents optimizer updates. No publication action exists here.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT.parent / "kolibri-1-20261004"
GPUS = ["GPU-3538758b-6ba7-fa1b-a80c-e315a606c3f0", "GPU-fc9d264d-8a20-bf8c-eb72-72a6dcbdaf2a"]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def control(action):
    # Lease credentials stay inside this process and are never logged.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(20)
        sock.connect("/run/ollama-unify/gpu-negotiator.sock")
        sock.sendall((json.dumps({"action": action, "token": os.environ["OLLAMA_UNIFY_GPU_LEASE"]}) + "\n").encode())
        data = bytearray()
        while b"\n" not in data:
            chunk = sock.recv(65536)
            if not chunk:
                break
            data.extend(chunk)
        result = json.loads(data)
        if not result.get("ok"):
            reason=str(result.get('error','unspecified')).replace(os.environ['OLLAMA_UNIFY_GPU_LEASE'],'[redacted]')[:300]
            raise RuntimeError(f"Broker rejected {action}: {reason}")
        return result.get("result", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--run-id", default="full-bf16-initial")
    parser.add_argument("--pilot-directory", type=Path, default=ROOT / "data-pilot-counted")
    parser.add_argument("--cpu-layers", type=int, default=0)
    parser.add_argument("--capture-layers", type=Path)
    parser.add_argument("--golden-logits", type=Path, default=ROOT / "compatibility/stock-gpu-layer-capture/logits.f32")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") != GPUS or not os.environ.get("OLLAMA_UNIFY_GPU_LEASE"):
        raise RuntimeError("The exact scoped broker reservation is required")
    if not args.run_id.replace("-", "").isalnum() or not args.ready_file.resolve().is_relative_to(ROOT):
        raise ValueError("Invalid run-local output")
    torch.set_num_threads(16)
    torch.manual_seed(6100626)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    decoder_module = module("frozen_afmoe", ROOT / "runtime-source/frozen-decoder.py")
    start = time.monotonic()
    def progress(layer, device):
        if layer % 8 == 0 or layer == 79:
            print(json.dumps({"loaded_layer": layer, "device": device, "seconds": time.monotonic() - start}), flush=True)
    decoder = decoder_module.FrozenAFMoE(str(RELEASE / "Kolibri-1-stock-BF16-release.gguf"),
        devices=("cuda:0", "cuda:1"), checkpoint_layers=True, grouped_experts=True,
        cpu_embedding=True, cpu_head=True, cpu_layers=args.cpu_layers, load_progress=progress)
    bridge = torch.nn.Linear(4608, decoder.hidden, device="cuda:0", dtype=torch.float32)
    # Explicit Adam buffers make residency/readiness cover optimizer storage.
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-4, weight_decay=0)
    for param in bridge.parameters():
        optimizer.state[param] = {"step": torch.tensor(0.), "exp_avg": torch.zeros_like(param),
            "exp_avg_sq": torch.zeros_like(param)}
    for device in decoder.devices:
        torch.cuda.synchronize(device)
    args.ready_file.write_text("models and optimizer resident\n")
    for _ in range(30):
        result = control("heartbeat")
        lease = result.get("lease", result)
        if lease.get("state") == "active":
            break
        time.sleep(.25)
    else:
        raise RuntimeError("Supervisor did not activate the resident lease")
    print(json.dumps({"resident": True, "load_seconds": time.monotonic() - start,
        "allocated_GiB": [torch.cuda.memory_allocated(d) / 2**30 for d in decoder.devices]}), flush=True)
    control("prepare")
    data = json.loads((ROOT / "compatibility/full-bf16-stock-input.json").read_text())
    reference_paths = list(dict.fromkeys([args.golden_logits,
        ROOT / "compatibility/full-bf16-stock.logits.bin",
        ROOT / "compatibility/stock-layer-capture/logits.f32"]))
    references = {str(p): torch.from_numpy(np.fromfile(p, dtype=np.float32)) for p in reference_paths}
    parity_start = time.monotonic()
    layer_report = []
    if args.capture_layers:
        def captured(name, layer, value):
            path = args.capture_layers / f"{name}-{layer}.f32"
            if not path.exists():
                return
            expected = torch.from_numpy(np.fromfile(path, dtype=np.float32)).reshape(value.shape)
            actual = value.detach().float().cpu()
            error = actual - expected
            layer_report.append({"name": name, "layer": layer, "rmse": float(error.square().mean().sqrt()),
                "relative_rmse": float(error.norm() / expected.norm()), "max_error": float(error.abs().max())})
        decoder.capture = captured
    reference_comparisons = []
    with torch.no_grad():
        embeddings = decoder.embeddings(torch.tensor(data["tokens"], dtype=torch.long))
        repeated_logits = [decoder(embeddings, output_positions=[len(data["tokens"]) - 1])[0].cpu().float() for _ in range(3)]
    for repetition, current in enumerate(repeated_logits):
        for reference_path, expected in references.items():
            delta = current - expected
            logprob = expected.log_softmax(-1)
            current_kl = float((logprob.exp() * (logprob - current.log_softmax(-1))).sum())
            current_rmse = float(delta.square().mean().sqrt())
            current_passed = current_rmse <= .05 and current_kl <= .0005 and int(expected.argmax()) == int(current.argmax())
            reference_comparisons.append({"repetition": repetition, "reference": reference_path,
                "reference_sha256": hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
                "logit_rmse": current_rmse, "kl": current_kl, "same_top1": int(expected.argmax()) == int(current.argmax()),
                "passed": current_passed})
    passed = all(any(c["passed"] and c["repetition"] == i for c in reference_comparisons) for i in range(3))
    closest = min((c for c in reference_comparisons if c["repetition"] == 0), key=lambda c:c["logit_rmse"])
    logits = repeated_logits[0]
    golden = references[closest["reference"]]
    difference = logits - golden
    teacher_logprob = golden.log_softmax(-1)
    kl = float((teacher_logprob.exp() * (teacher_logprob - logits.log_softmax(-1))).sum())
    rmse = float(difference.square().mean().sqrt())
    report = {"utc": datetime.now(timezone.utc).isoformat(), "passed": passed,
        "scope": "full BF16 weights, one 20-token prompt repeated three times, matching the legitimate installed stock CPU/GPU F32-KV reference envelope; broader training parity still required",
        "closest_native_reference": closest["reference"], "reference_comparisons": reference_comparisons,
        "logit_rmse": rmse, "logit_max_error": float(difference.abs().max()), "kl": kl,
        "stock_top1": int(golden.argmax()), "training_top1": int(logits.argmax()),
        "thresholds": {"rmse": .05, "kl": .0005, "same_top1": True},
        "parity_seconds": time.monotonic() - parity_start, "load_seconds": parity_start - start,
        "peak_allocated_GiB": [torch.cuda.max_memory_allocated(d) / 2**30 for d in decoder.devices],
        "cpu_layers": args.cpu_layers, "cpu_head": True, "optimizer_updates": 0}
    if layer_report:
        report["layer_diagnostics"] = layer_report
    path = ROOT / "compatibility" / (args.run_id + "-training-qualification.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ["layer_diagnostics", "reference_comparisons"]}), flush=True)
    np.save(ROOT / "compatibility" / (args.run_id + "-training-logits.npy"), logits.numpy())
    control("ready")
    if not passed:
        raise RuntimeError("Full numerical qualification failed; no optimizer update was performed")
    decoder.capture = None
    # The sampled development holdout is never used for gradient updates.
    examples = [json.loads(line) for line in (args.pilot_directory / "examples.jsonl").read_text().splitlines()]
    selected = [next(e for e in examples if e["config"] == name and e["pilot_split"] == "training")
                for name in ["docvqa", "sharegpt4v(coco)"]]
    tokenizer = Tokenizer.from_file(str(RELEASE / "source/tokenizer.json"))
    feature_module = module("vision_features", ROOT / "runtime-source/vision-features.py")
    extractor = feature_module.VisionFeatures(ROOT, gpu=False, min_tokens=1024, max_tokens=1024)
    steps = []
    for example in selected:
        with Image.open(args.pilot_directory / example["image"]) as image:
            features = torch.from_numpy(extractor.encode(image)).to("cuda:0")
        if not steps:
            # Keep initial image embedding magnitude comparable to the frozen
            # text embeddings after stock MuP scaling, using training data only.
            tokens_for_scale = tokenizer.encode(example["user"], add_special_tokens=False).ids
            target_rms = float(decoder.embeddings(torch.tensor(tokens_for_scale)).float().square().mean().sqrt())
            feature_rms = float(features.float().square().mean().sqrt())
            with torch.no_grad():
                bridge.weight.normal_(std=target_rms / max(feature_rms * math.sqrt(4608), 1e-8))
                bridge.bias.zero_()
            print(json.dumps({"initialized_bridge_embedding_rms": target_rms, "frozen_feature_rms": feature_rms}), flush=True)
        # Match the same literal vision wrappers used by stock Qwen mtmd.
        prefix = "<|im_start|>system\n# Reasoning effort\n\nReasoning is disabled. Proceed straight to answering according to the user's instructions.<|im_end|>\n<|im_start|>user\n<|vision_start|>"
        suffix = "<|vision_end|>" + example["user"].replace("<image>", "") + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ids0 = tokenizer.encode(prefix, add_special_tokens=False).ids
        ids1 = tokenizer.encode(suffix, add_special_tokens=False).ids
        answer = tokenizer.encode(example["assistant"] + "<|im_end|>", add_special_tokens=False).ids[:256]
        control("prepare")
        step_start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        before = decoder.embeddings(torch.tensor(ids0))
        after = decoder.embeddings(torch.tensor(ids1 + answer[:-1]))
        projected = bridge(features)
        inputs = torch.cat([before, projected, after], 0)
        answer_start = len(ids0) + features.shape[0] + len(ids1) - 1
        outputs = decoder(inputs, output_positions=list(range(answer_start, answer_start + len(answer))))
        loss = torch.nn.functional.cross_entropy(outputs.float(), torch.tensor(answer, device=outputs.device))
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0))
        if not torch.isfinite(loss) or not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError("Vision bridge has invalid loss or gradients")
        optimizer.step()
        for device in decoder.devices:
            torch.cuda.synchronize(device)
        step = {"id": example["id"], "config": example["config"], "loss": float(loss.detach()),
            "gradient_norm": gradient_norm, "image_tokens": int(features.shape[0]), "answer_tokens": len(answer),
            "sequence_tokens": len(inputs), "seconds": time.monotonic() - step_start,
            "peak_allocated_GiB": [torch.cuda.max_memory_allocated(d) / 2**30 for d in decoder.devices]}
        steps.append(step)
        print(json.dumps(step), flush=True)
        checkpoint = ROOT / "pilot-projector.pt"
        partial = checkpoint.with_suffix(".pt.partial")
        torch.save({"state_dict": {k:v.detach().cpu() for k,v in bridge.state_dict().items()}, "optimizer_updates": len(steps),
            "scope": "engineering pilot only; unqualified image capability, do not publish", "steps": steps}, partial)
        partial.replace(checkpoint)
        report["pilot_steps"] = steps
        report["optimizer_updates"] = len(steps)
        report["pilot_checkpoint"] = str(checkpoint)
        path.write_text(json.dumps(report, indent=2) + "\n")
        control("ready")
    extractor.close()
    checkpoint = ROOT / "pilot-projector.pt"
    torch.save({"state_dict": {k:v.detach().cpu() for k,v in bridge.state_dict().items()}, "optimizer_updates": len(steps),
        "scope": "engineering pilot only; unqualified image capability, do not publish", "steps": steps}, checkpoint)
    report["pilot_steps"] = steps
    report["optimizer_updates"] = len(steps)
    report["pilot_checkpoint"] = str(checkpoint)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
