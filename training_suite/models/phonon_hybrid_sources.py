"""Pinned preparation inputs for the self-contained Phonon hybrid export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

SOURCES = {
    "phonon": {"repo": "gradium/phonon-31m-en", "revision": "92741c5b07d085ff840e26fea8ce43110b87852f",
               "files": ["config.json", "generation_config.json", "model.safetensors", "default-voice.safetensors",
                         "tokenizer.json", "tokenizer.5.0.0.json", "tokenizer_config.json", "chat_template.jinja"]},
    "codec": {"repo": "gradium/phonon-mimi", "revision": "33519be1a8e61c6fa00c6e154e2d38e76a367919",
              "files": ["config.json", "model.safetensors"]},
    "converter": {"repo": "myshell-ai/OpenVoiceV2", "revision": "f36e7edfe1684461a8343844af60babc2efbb727",
                  "prefix": "converter/", "files": ["config.json", "checkpoint.pth"]},
    "asr": {"repo": "openai/whisper-tiny.en", "revision": "87c7102498dcde7456f24cfd30239ca606ed9063",
            "files": ["config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json",
                      "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "normalizer.json"]},
}


def download_components(directory: Path):
    receipts = {}
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for component, source in SOURCES.items():
            target = directory / component
            target.mkdir(parents=True, exist_ok=True)
            hashes = {}
            for name in source["files"]:
                url = f"https://huggingface.co/{source['repo']}/resolve/{source['revision']}/{source.get('prefix', '')}{name}"
                path = target / name
                partial = path.with_suffix(path.suffix + ".partial")
                result = hashlib.sha256()
                # Always verify a complete download; never mistake a partial or
                # an arbitrary pre-existing local file for the pinned source.
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with partial.open("wb") as output:
                        for chunk in response.iter_bytes(1024 * 1024):
                            output.write(chunk)
                            result.update(chunk)
                partial.replace(path)
                hashes[name] = result.hexdigest()
                print(f"{component}/{name}: {path.stat().st_size} bytes", flush=True)
            receipts[component] = {**source, "sha256": hashes}
    (directory / "sources.json").write_text(json.dumps(receipts, indent=2) + "\n")
    return receipts
