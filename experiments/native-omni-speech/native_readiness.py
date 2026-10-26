"""Broker readiness for an official Ollama fixture after its CUDA model loads."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import httpx


def descendants(pid):
    result = {pid}
    # Go can spawn its runner from any OS thread, not just the main thread.
    for children in Path(f"/proc/{pid}/task").glob("*/children"):
        try:
            child_pids = children.read_text().split()
        except FileNotFoundError:
            continue
        for child in child_pids:
            result.update(descendants(int(child)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pid-file", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        pid = int(args.pid_file.read_text())
        scope = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        with httpx.Client(timeout=2) as client:
            result = client.get(args.url + "/api/ps")
            result.raise_for_status()
            models = result.json().get("models", [])
        if not any(m.get("name") == args.model for m in models):
            return 1
        pids = descendants(pid)
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        resident = []
        for row in result.stdout.splitlines():
            process, gpu, memory = (part.strip() for part in row.split(","))
            if int(process) in pids:
                resident.append({"pid": int(process), "gpu_uuid": gpu, "vram_mib": int(memory)})
        if not resident or {item["gpu_uuid"] for item in resident} != set(scope):
            return 1
        proof = {"server_pid": pid, "gpu_uuids": scope, "resident": resident, "models": models}
        args.out.write_text(json.dumps(proof, indent=2) + "\n")
        return 0
    except (OSError, ValueError, KeyError, httpx.HTTPError, subprocess.SubprocessError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
