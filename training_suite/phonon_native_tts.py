"""Prepare and check Phonon TTS for the unmodified official Ollama binary."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import wave
from pathlib import Path

import httpx

REPOSITORY = "gradium/phonon-31m-en"
REVISION = "92741c5b07d085ff840e26fea8ce43110b87852f"
FILES = {
    "config.json": "b5d0f88353638370d94035b7d73d7d6577c0554d78f8d7ecfd40490aae39ae85",
    "model.safetensors": "8a04cf833cfe69b67c2ec968b2d31244655b9accc84ade9c0ebcc176f2074607",
    "default-voice.safetensors": "dc537478eaadc1a5257fcda2e3501ab81e14c111a5d95a5838f350db2afd0f1a",
    "tokenizer.json": "691a00068fa7b989f2f3e5ae5daa2d1a5bb319f2d5360deafcfb19a8bb6e7936",
    "tokenizer_config.json": "17d12cf8dee2891506648bb1d5657e26fb165263686d0e2ef311513488d6446f",
}


def prepare(directory: Path):
    source = directory / "source"
    source.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for name, expected in FILES.items():
            target = source / name
            if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
                continue
            partial = target.with_suffix(target.suffix + ".partial")
            hasher = hashlib.sha256()
            try:
                with client.stream("GET", f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}") as response:
                    response.raise_for_status()
                    with partial.open("wb") as output:
                        for chunk in response.iter_bytes(1024 * 1024):
                            output.write(chunk)
                            hasher.update(chunk)
                if hasher.hexdigest() != expected:
                    raise ValueError(f"source SHA-256 mismatch: {name}")
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
    (source / "generation_config.json").write_text(json.dumps({"temperature": 0.0}, indent=2) + "\n")
    license_text = (Path(__file__).parent / "licenses/Apache-2.0.txt").read_text()
    (directory / "Modelfile").write_text(f'FROM {source.resolve()}\nLICENSE """{license_text}\n"""\n')
    receipt = {"repository": REPOSITORY, "revision": REVISION, "sha256": FILES,
               "preparation": {"temperature": 0.0, "neural_weights_changed": False},
               "architecture": "GradiumPhononTTSForConditionalGeneration",
               "runtime": "official Ollama 0.40.11", "capabilities": ["speech"]}
    (directory / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def ready(url: str, model: str):
    with httpx.Client(base_url=url, timeout=240) as client:
        for attempt in range(80):
            try:
                version = client.get("/api/version").raise_for_status().json()
                break
            except httpx.ConnectError:
                if attempt == 79:
                    raise
                time.sleep(0.25)
        if version != {"version": "0.40.11"}:
            raise ValueError(f"expected official Ollama 0.40.11, got {version}")
        response = client.post("/v1/audio/speech", json={
            "model": model, "input": "The small robot speaks clearly.", "voice": "default",
        }).raise_for_status()
        with wave.open(io.BytesIO(response.content), "rb") as audio:
            if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) != (1, 24000, 2):
                raise ValueError("expected 24 kHz mono PCM16 WAV")
            samples = audio.readframes(audio.getnframes())
        if len(samples) < 24000 or len(samples) % 2 or not any(samples):
            raise ValueError("official speech endpoint returned incomplete or silent PCM")
        return {"version": version, "model": model, "pcm_bytes": len(samples)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("ready")
    check.add_argument("--url", default="http://127.0.0.1:11470")
    check.add_argument("--model", default="robit/phonon-echo:hybrid")
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.out) if args.command == "prepare" else ready(args.url, args.model), indent=2))


if __name__ == "__main__":
    main()
