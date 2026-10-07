"""Register a synthetic model/projector with native Ollama and test image input.

This only qualifies the stock packaging and image-injection interface. Random
fixture outputs are deliberately not scored as learned vision capability.
"""
from datetime import datetime, timezone
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TEXT_RELEASE = ROOT.parent / "kolibri-1-20261004"
TAG = "kolibri-vision-interface-fixture:head128-20261006"
GPU = "GPU-3538758b-6ba7-fa1b-a80c-e315a606c3f0"


def api(port, path, value=None, headers=None, timeout=180):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(value).encode() if value is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projector", type=Path, default=ROOT / "compatibility/head128/clip-interface-fixture.gguf")
    parser.add_argument("--tag", default=TAG)
    parser.add_argument("--proof-prefix", default="stock")
    args = parser.parse_args()
    if not args.projector.resolve().is_relative_to(ROOT):
        raise ValueError("Only this run's experimental projector may be registered")
    if not args.tag.startswith("kolibri-vision-interface-fixture:"):
        raise ValueError("Synthetic experiments must never use a publication name")
    if not args.proof_prefix.replace("-", "").isalnum():
        raise ValueError("Invalid proof prefix")
    proof = json.loads((ROOT / "compatibility/head128/stock-cpu-interface-proof.json").read_text())
    if not proof["passed"]:
        raise RuntimeError("CPU interface qualification has not passed")
    discovered = subprocess.run(["docker", "gpu", "discover"], check=True,
                                capture_output=True, text=True)
    discovery = json.loads(discovered.stdout)
    (ROOT / f"compatibility/gpu-discovery-before-{args.proof_prefix}-api-fixture.json").write_text(discovered.stdout)
    if not any(g["uuid"] == GPU and g["selected_for_ollama"] for g in discovery["gpus"]):
        raise RuntimeError("Expected GPU is not selected by the broker")
    print(json.dumps({"warnings": discovery.get("warnings"),
                      "active_lease_summaries": discovery.get("active_leases")}), flush=True)
    files = {}
    for path in [ROOT / "compatibility/head128/afmoe-interface-fixture.gguf", args.projector]:
        name = path.name
        if path.stat().st_size > 2 * 2**30:
            raise ValueError("Fixture blob exceeds the bounded import size")
        with path.open("rb") as source:
            digest = "sha256:" + hashlib.file_digest(source, "sha256").hexdigest()
        blob_url = f"http://127.0.0.1:11436/api/blobs/{digest}"
        try:
            with urllib.request.urlopen(urllib.request.Request(blob_url, method="HEAD"), timeout=20):
                exists = True
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            exists = False
        if not exists:
            with path.open("rb") as source:
                request = urllib.request.Request(blob_url, data=source, method="POST",
                    headers={"Content-Length": str(path.stat().st_size)})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 201:
                        raise RuntimeError("Native fixture blob import did not succeed")
        files[name] = digest
    create_request = {
        "model": args.tag, "files": files, "stream": False, "requires": "0.35.0",
        "template": (TEXT_RELEASE / "kolibri-ollama-tools.gotmpl").read_text(),
        "info": {"capabilities": ["completion", "vision", "tools", "thinking"]},
        "parameters": {"num_ctx": 2048, "temperature": 0, "stop": ["<|im_end|>"]},
    }
    (ROOT / f"compatibility/{args.proof_prefix}-fixture-create-request.json").write_text(
        json.dumps(create_request, indent=2) + "\n")
    created = api(11436, "/api/create", create_request)
    if created.get("status") != "success":
        raise RuntimeError("Native model/projector create did not succeed")
    shown = api(11434, "/api/show", {"model": args.tag})
    if not {"completion", "vision", "tools", "thinking"}.issubset(shown["capabilities"]):
        raise RuntimeError("Packed capabilities are incomplete")
    responses = []
    for color in ["red", "blue", "red"]:
        path = ROOT / "compatibility" / f"{color}.png"
        Image.new("RGB", (28, 28), color).save(path)
        response = api(11434, "/api/chat", {
            "model": args.tag, "messages": [{"role": "user", "content": "Describe this image.",
                "images": [base64.b64encode(path.read_bytes()).decode()]}],
            "think": False, "stream": False, "keep_alive": "5m",
            "options": {"num_gpu": -1, "num_ctx": 2048, "num_predict": 8,
                        "temperature": 0, "seed": 6100626},
        }, headers={"X-Ollama-Unify-GPU-UUIDs": GPU,
                    "X-Ollama-Unify-Workload-Class": "background"})
        if not response.get("done") or response.get("prompt_eval_count", 0) < 4:
            raise RuntimeError("Stock image request did not complete")
        responses.append({"color": color, "response": response})
    report = {
        "utc": datetime.now(timezone.utc).isoformat(), "passed": True,
        "scope": "stock API packaging and image-input interface; synthetic random weights",
        "trained_vision_capability_claim": False, "model": args.tag,
        "capabilities": shown["capabilities"], "files": files,
        "create_response": created, "responses": responses,
        "public_api": "http://127.0.0.1:11434", "gpu_allowlist": [GPU],
        "stock_ollama_version": api(11436, "/api/version")["version"],
    }
    (ROOT / f"compatibility/{args.proof_prefix}-api-interface-proof.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "model": args.tag, "capabilities": shown["capabilities"],
                      "image_requests_completed": len(responses)}))


if __name__ == "__main__":
    main()
