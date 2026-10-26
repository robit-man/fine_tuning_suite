"""Ollama-managed PhononHybrid runner: load the selected tag's actual blobs."""

import argparse
import base64
import json
import os
from pathlib import Path

import torch
from flask import Response, jsonify, request
from safetensors.torch import load_file
from waitress import serve

from training_suite.models.phonon_hybrid import PhononHybrid, digest, read_audio
from training_suite.phonon_hybrid import create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    descriptor = json.loads(args.descriptor.read_text())
    directory = Path(descriptor["directory"])
    weights = {}
    for blob in descriptor["tensor_blobs"]:
        path = Path(blob["path"])
        if "sha256:" + digest(path) != blob["digest"]:
            raise ValueError("Ollama tensor blob failed integrity verification")
        for name, value in load_file(str(path)).items():
            if name in weights:
                raise ValueError("duplicate tensor in Ollama hybrid model: " + name)
            weights[name] = value
    config = json.loads((directory / "config.json").read_text())
    model = PhononHybrid.from_state(directory, config, weights)
    memory = sum(t.numel() * t.element_size() for t in weights.values())
    del weights
    app = create_app(model, voices_dir=directory / "voices", managed_by_ollama=True)

    @app.get("/v1/status")
    def status():
        return jsonify({"Status": 0, "Progress": 100, "ContextLength": 8192,
                        "Memory": memory, "runtime": "phonon-hybrid", "model": descriptor["model"],
                        "pid": os.getpid(), "cuda_initialized": torch.cuda.is_initialized()})

    @app.post("/v1/tokenize")
    def tokenize():
        return jsonify(model.tokenizer.encode(request.get_data(as_text=True), add_special_tokens=False))

    @app.post("/v1/detokenize")
    def detokenize():
        body = request.get_json()
        ids = body if isinstance(body, list) else body.get("tokens", body.get("Tokens", []))
        return jsonify(model.tokenizer.decode(ids))

    @app.post("/completion")
    def completion():
        body = request.get_json()
        media = body.get("Media", body.get("media", []))
        if not media:
            raise ValueError("PhononHybrid text output requires input audio for transcription")
        texts = []
        for item in media:
            data = item.get("data", item.get("Data"))
            if not isinstance(data, str):
                raise ValueError("transcription media requires base64 audio")
            audio, rate = read_audio(base64.b64decode(data, validate=True), max_seconds=30, allow_silence=True)
            texts.append(model.transcribe(audio, rate) if audio.any() else "")
        result = {"Content": " ".join(texts), "Done": True, "DoneReason": 0,
                  "PromptEvalCount": 1, "EvalCount": len(" ".join(texts).split())}
        return Response(json.dumps(result) + "\n", content_type="application/x-ndjson")

    print(json.dumps({"runtime": "Ollama-managed PhononHybrid", "model": descriptor["model"],
                      "pid": os.getpid(), "tensor_source": "Ollama manifest blobs", "memory": memory}), flush=True)
    serve(app, host="127.0.0.1", port=args.port, threads=4)


if __name__ == "__main__":
    main()
