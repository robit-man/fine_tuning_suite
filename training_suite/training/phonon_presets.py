"""Bake named reference speaker profiles into a self-contained hybrid checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def voice_spec(value: str) -> tuple[str, Path]:
    name, separator, recording = value.partition("=")
    if (not separator or not recording or name == "default"
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name)):
        raise argparse.ArgumentTypeError("--voice requires NAME=RECORDING; default is reserved")
    return name, Path(recording)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models-directory", type=Path,
                        default=Path(os.environ.get("OLLAMA_MODELS", Path.home() / ".ollama/models")))
    parser.add_argument("--source-model", default="robit/phonon-echo:clone-asr-v1")
    parser.add_argument("--voice", type=voice_spec, action="append", required=True, metavar="NAME=RECORDING")
    parser.add_argument("--out", type=Path, required=True)


def bake(*, models_directory: Path, source_model: str, voices: dict[str, Path], output: Path) -> dict:
    # Keep CLI help and the control plane usable without the optional audio stack.
    import numpy as np
    import torch

    from training_suite.models.phonon_hybrid import digest, read_audio
    from training_suite.models.phonon_hybrid_storage import load_ollama_model

    if not voices:
        raise ValueError("at least one named voice is required")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty directory")
    torch.set_num_threads(4)
    model = load_ollama_model(models_directory, source_model)
    receipts = {}
    for name, recording in voices.items():
        audio, rate = read_audio(recording.read_bytes(), max_seconds=30)
        # Trim outer silence and normalize amplitude before the reference encoder.
        active = np.flatnonzero(np.abs(audio) > max(float(np.max(np.abs(audio))) * 0.03, 0.0005))
        if not len(active):
            raise ValueError("reference has no active speech")
        padding = int(rate * 0.05)
        audio = audio[max(0, active[0] - padding):min(len(audio), active[-1] + padding + 1)]
        if len(audio) / rate < 0.25:
            raise ValueError("reference active speech is too short")
        audio = audio * (0.9 / float(np.max(np.abs(audio))))
        embedding = model.speaker_embedding(audio, rate)
        model.add_voice_profile(name, embedding)
        receipts[name] = {
            "reference_sha256": digest(recording), "active_seconds": len(audio) / rate,
            "embedding_shape": list(embedding.shape), "embedding_norm": float(embedding.norm()),
        }
        print(json.dumps({"baked_voice": name, "active_seconds": round(len(audio) / rate, 3)}), flush=True)
    manifest = model.save_pretrained(output)
    card = {
        "name": "Phonon Echo", "source_model": source_model, "available_voices": model.voice_names(),
        "source_recordings_in_checkpoint": False, "requires_extension": "phonon-echo.2",
        "official_binary_supported": False,
        "setup": "https://github.com/robit-man/fine_tuning_suite/blob/main/docs/phonon-hybrid.md",
        "composition": ["Phonon", "Mimi", "OpenVoice V2", "Whisper tiny.en"], "jointly_trained": False,
    }
    (output / "model-card.json").write_text(json.dumps(card, indent=2) + "\n")
    manifest["files"]["model-card.json"] = digest(output / "model-card.json")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = {
        "schema": "phonon-echo.baked-voices.v1", "source_model": source_model,
        "profiles": receipts, "available_voices": model.voice_names(),
        "source_recordings_in_checkpoint": False, "cuda_initialized": torch.cuda.is_initialized(),
        "checkpoint_sha256": manifest["files"]["model.safetensors"],
    }
    (output.parent / "bake-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def run(args: argparse.Namespace) -> None:
    voices = dict(args.voice)
    if len(voices) != len(args.voice):
        raise SystemExit("duplicate voice name")
    print(json.dumps(bake(models_directory=args.models_directory, source_model=args.source_model,
                          voices=voices, output=args.out), indent=2))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
