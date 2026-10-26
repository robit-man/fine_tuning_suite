"""Build and serve the local Phonon voice-cloning and ASR hybrid."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import re
import struct
import threading
import time
import wave
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, stream_with_context
from werkzeug.exceptions import HTTPException

from training_suite.models.phonon_hybrid import MODEL_ID, SAMPLE_RATE, SCHEMA, read_audio


class VoiceStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    @staticmethod
    def validate_id(name):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or name == "default":
            raise ValueError("voice name must be 1–64 letters, digits, hyphens or underscores, excluding default")
        return name

    def put(self, name, embedding):
        from safetensors.numpy import save_file

        name = self.validate_id(name)
        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.shape != (1, 256, 1) or not np.isfinite(embedding).all():
            raise ValueError("invalid reference speaker embedding")
        with self.lock:
            path = self.directory / f"{name}.safetensors"
            if path.exists():
                raise FileExistsError("voice name already exists; choose a new name")
            partial = path.with_suffix(".partial")
            save_file({"speaker_embedding": embedding}, str(partial), metadata={"schema": SCHEMA})
            partial.replace(path)
        return name

    def get(self, name):
        from safetensors.numpy import load_file

        name = self.validate_id(name)
        with self.lock:
            path = self.directory / f"{name}.safetensors"
            if not path.is_file():
                raise ValueError(f"unknown voice: {name}")
            return load_file(str(path))["speaker_embedding"].copy()

    def names(self):
        return ["default", *sorted(path.stem for path in self.directory.glob("*.safetensors"))]


def pcm_bytes(audio, *, allow_silence=False):
    audio = np.asarray(audio, dtype=np.float32)
    if (audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or
            (not allow_silence and np.max(np.abs(audio)) < 1e-5)):
        raise ValueError("speech model returned invalid or silent waveform")
    return np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def wav_bytes(pcm):
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    return output.getvalue()


def streaming_wav_header():
    return (b"RIFF" + struct.pack("<I", 0xffffffff) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16) +
            b"data" + struct.pack("<I", 0xffffffff))


def _reference_bytes(value):
    if not isinstance(value, dict) or set(value) - {"data", "format"}:
        raise ValueError("reference_audio must contain base64 data and optional format=wav")
    if value.get("format", "wav") != "wav" or not isinstance(value.get("data"), str):
        raise ValueError("reference_audio requires WAV data encoded as base64")
    try:
        return base64.b64decode(value["data"], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("reference_audio contains invalid base64") from exc


def create_app(engine, *, voices_dir: Path, model_id=MODEL_ID, managed_by_ollama=False):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    store = VoiceStore(voices_dir)

    def model_requested(value):
        if value != model_id:
            raise ValueError(f"model must be {model_id}")

    @app.errorhandler(Exception)
    def error(exc):
        status = (409 if isinstance(exc, FileExistsError) else
                  400 if isinstance(exc, ValueError) else
                  exc.code if isinstance(exc, HTTPException) else 500)
        if status == 500:
            app.logger.exception("hybrid request failed")
        return jsonify({"error": {"message": str(exc) if status != 500 else "hybrid inference failed",
                                  "type": "invalid_request_error" if status < 500 else "server_error"}}), status

    @app.get("/health")
    def health():
        return jsonify({"status": "ready", "model": model_id, "runtime": "phonon-hybrid",
                        "official_ollama": False, "device": str(engine.device),
                        "capabilities": ["speech", "voice-cloning", "transcription"], "voices": store.names()})

    @app.get("/v1/models")
    def models():
        return jsonify({"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]})

    @app.route("/v1/audio/voices", methods=["GET", "POST"])
    def voices():
        if request.method == "GET":
            return jsonify({"voices": store.names()})
        name = VoiceStore.validate_id(request.form.get("name", ""))
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("file is required")
        audio, rate = read_audio(upload.read(), max_seconds=30)
        embedding = engine.speaker_embedding(audio, rate)
        store.put(name, embedding)
        return jsonify({"id": name, "object": "voice", "reference_seconds": len(audio) / rate}), 201

    @app.post("/v1/audio/transcriptions")
    def transcriptions():
        model_requested(request.form.get("model"))
        upload = request.files.get("file")
        if upload is None:
            raise ValueError("file is required")
        fmt = request.form.get("response_format", "json")
        if fmt not in {"json", "text", "verbose_json"}:
            raise ValueError("response_format must be json, text or verbose_json")
        language = request.form.get("language", "en")
        if language not in {"en", "english", ""}:
            raise ValueError("this checkpoint supports English ASR")
        audio, rate = read_audio(upload.read(), max_seconds=30, allow_silence=True)
        text = "" if np.max(np.abs(audio)) < 1e-5 else engine.transcribe(audio, rate)
        if fmt == "text":
            return Response(text, content_type="text/plain; charset=utf-8")
        result = {"text": text}
        if fmt == "verbose_json":
            result.update({"language": "english", "duration": len(audio) / rate})
        return jsonify(result)

    @app.post("/v1/audio/speech")
    def speech():
        body = request.get_json()
        if not isinstance(body, dict):
            raise ValueError("request must be a JSON object")
        if managed_by_ollama:
            body = dict(body, model=model_id, response_format="pcm")
        supported = {"model", "input", "voice", "response_format", "speed", "instructions", "stream_format", "reference_audio"}
        if set(body) - supported:
            raise ValueError("unsupported speech fields: " + ", ".join(sorted(set(body) - supported)))
        model_requested(body.get("model"))
        text = body.get("input")
        if not isinstance(text, str) or not text.strip() or len(text) > 4096:
            raise ValueError("input must contain 1–4096 characters")
        fmt = body.get("response_format", "wav")
        if fmt not in {"wav", "pcm"}:
            raise ValueError("response_format must be wav or pcm")
        if body.get("speed", 1) != 1 or body.get("instructions", "") or body.get("stream_format", "audio") != "audio":
            raise ValueError("only speed=1, empty instructions and stream_format=audio are supported")
        voice = body.get("voice", "default")
        if not isinstance(voice, str):
            raise ValueError("voice must be a name")
        speaker = None
        if "reference_audio" in body:
            if voice != "default":
                raise ValueError("choose an enrolled voice or reference_audio, not both")
            audio, rate = read_audio(_reference_bytes(body["reference_audio"]), max_seconds=30)
            speaker = engine.speaker_embedding(audio, rate)
        elif voice != "default":
            speaker = store.get(voice)
        started = time.monotonic()
        chunks = iter(engine.iter_speech(text, speaker=speaker)) if hasattr(engine, "iter_speech") else iter([engine.synthesize(text, speaker=speaker)])
        # Generate the first chunk before committing HTTP 200, so inference
        # failures still return a proper JSON error instead of an empty WAV.
        first = pcm_bytes(next(chunks))

        def stream():
            try:
                if fmt == "wav":
                    yield streaming_wav_header()
                yield first
                for chunk in chunks:
                    yield pcm_bytes(chunk, allow_silence=True)
            finally:
                close = getattr(chunks, "close", None)
                if close:
                    close()

        return Response(stream_with_context(stream()), content_type="audio/" + fmt,
                        headers={"X-Speech-Runtime": "phonon-hybrid",
                                 "X-Audio-Sample-Rate": str(SAMPLE_RATE),
                                 "X-First-Chunk-Seconds": f"{time.monotonic() - started:.3f}"})

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download the pinned model preparation inputs")
    download.add_argument("--out", required=True, type=Path)
    build = sub.add_parser("build", help="Export all trained components into one hybrid checkpoint")
    for name in ["phonon", "codec", "converter", "asr"]:
        build.add_argument("--" + name, required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    serve = sub.add_parser("serve")
    serve.add_argument("--model", required=True, type=Path)
    serve.add_argument("--voices", required=True, type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=11471)
    serve.add_argument("--device", default="cpu")
    serve.add_argument("--threads", type=int, default=4)
    serve.add_argument("--ready-file", type=Path)
    args = parser.parse_args(argv)
    if args.command == "download":
        from training_suite.models.phonon_hybrid_sources import download_components

        download_components(args.out)
        return
    import torch
    from training_suite.models.phonon_hybrid import PhononHybrid

    torch.set_num_threads(4 if args.command == "build" else args.threads)
    if args.command == "build":
        model = PhononHybrid.from_components(phonon_dir=args.phonon, codec_dir=args.codec,
                                             converter_dir=args.converter, asr_dir=args.asr)
        print(json.dumps(model.save_pretrained(args.out), indent=2))
        return
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not os.environ.get("OLLAMA_UNIFY_GPU_LEASE"):
        raise ValueError("CUDA serving requires a scoped docker gpu lease")
    model = PhononHybrid.from_pretrained(args.model, device=args.device)
    if args.ready_file:
        # Materialize real speech and ASR CUDA allocations before readiness.
        probe = model.synthesize("The hybrid model is ready.")
        model.transcribe(probe, SAMPLE_RATE)
        speaker = model.speaker_embedding(probe, SAMPLE_RATE)
        model.synthesize("Reference voice conversion is ready.", speaker=speaker)
        record = {"pid": os.getpid(), "device": args.device, "model": str(args.model.resolve()),
                  "gpu_uuids": os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if args.device.startswith("cuda") else []}
        args.ready_file.parent.mkdir(parents=True, exist_ok=True)
        partial = args.ready_file.with_suffix(".partial")
        partial.write_text(json.dumps(record) + "\n")
        partial.replace(args.ready_file)
    from waitress import serve as run_server
    run_server(create_app(model, voices_dir=args.voices), host=args.host, port=args.port, threads=4)


if __name__ == "__main__":
    main()
