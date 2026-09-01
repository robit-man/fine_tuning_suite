"""Small HTTP wrapper around llama.cpp's Qwen3-TTS reference binary.

The server is intentionally serial: upstream ``llama-tts`` is currently a
single-shot validation tool. Production deployments should replace this
process wrapper with a persistent libmtmd worker while preserving this HTTP
contract.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training_suite.models.audio import DEFAULT_AUDIO_CONTRACT, decode_wav_payload
from training_suite.models.ollama_sidecar import resolve_ollama_sidecar
from training_suite.models.single_gguf import materialize_component_view


@dataclass(frozen=True)
class Config:
    binary: Path
    model: Path
    projector: Path
    timeout_s: float = 900
    max_text_chars: int = 4096
    max_frames: int = 512
    gpu_layers: int = -1

    @classmethod
    def from_environment(cls) -> Config:
        binary = Path(
            os.environ.get(
                "LLAMA_TTS_BIN",
                "training_suite/vendor/llama.cpp/build/bin/llama-tts",
            )
        ).expanduser()
        model_value = os.environ.get("OMNI_TTS_MODEL_GGUF")
        projector_value = os.environ.get("OMNI_TTS_PROJECTOR_GGUF")
        bundle_value = os.environ.get("OMNI_BUNDLE_GGUF")
        ollama_model = os.environ.get("OMNI_OLLAMA_MODEL")
        cache_dir = Path(
            os.environ.get("OMNI_COMPONENT_CACHE", "training_suite/outputs/omni-cache")
        ).expanduser()
        if ollama_model and not bundle_value and not (model_value or projector_value):
            bundle_value = resolve_ollama_sidecar(model=ollama_model)["bundle"]
        if bundle_value and not (model_value and projector_value):
            bundle = Path(bundle_value).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            model = cache_dir / "tts-model.gguf"
            projector = cache_dir / "tts-projector.gguf"
            if not model.exists():
                materialize_component_view(
                    bundle_gguf=bundle,
                    view="tts_model",
                    out_gguf=model,
                )
            if not projector.exists():
                materialize_component_view(
                    bundle_gguf=bundle,
                    view="tts_projector",
                    out_gguf=projector,
                )
        elif model_value and projector_value:
            model = Path(model_value).expanduser()
            projector = Path(projector_value).expanduser()
        else:
            raise RuntimeError(
                "set OMNI_OLLAMA_MODEL, OMNI_BUNDLE_GGUF, or both "
                "OMNI_TTS_MODEL_GGUF and OMNI_TTS_PROJECTOR_GGUF"
            )
        return cls(
            binary=binary,
            model=model,
            projector=projector,
            timeout_s=float(os.environ.get("OMNI_TTS_TIMEOUT_S", "900")),
            max_text_chars=int(os.environ.get("OMNI_TTS_MAX_TEXT_CHARS", "4096")),
            max_frames=int(os.environ.get("OMNI_TTS_MAX_FRAMES", "512")),
            gpu_layers=int(os.environ.get("OMNI_TTS_GPU_LAYERS", "-1")),
        )


class TTSError(RuntimeError):
    pass


def synthesize(config: Config, body: dict[str, Any]) -> bytes:
    text = str(body.get("text") or "").strip()
    if not text:
        raise TTSError("text is required")
    if len(text) > config.max_text_chars:
        raise TTSError(f"text exceeds {config.max_text_chars} characters")
    language = str(body.get("language") or body.get("lang") or "en").strip()
    speaker = str(body.get("speaker_file") or "").strip()
    frames = min(int(body.get("max_frames") or config.max_frames), config.max_frames)

    with tempfile.TemporaryDirectory(prefix="robit-omni-tts-") as temp_dir:
        output = Path(temp_dir) / "speech.wav"
        command = [
            str(config.binary),
            "-m",
            str(config.model),
            "--mmproj",
            str(config.projector),
            "--prompt",
            text,
            "--tts-lang",
            language,
            "--output",
            str(output),
            "--n-predict",
            str(frames),
            "--gpu-layers",
            str(config.gpu_layers),
        ]
        if speaker:
            command.extend(["--tts-speaker-file", speaker])
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=config.timeout_s,
        )
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout)[-2000:]
            raise TTSError(f"llama-tts exited {completed.returncode}: {diagnostic}")
        if not output.is_file():
            raise TTSError("llama-tts returned success without a WAV file")
        wav = output.read_bytes()

    envelope = {
        "mime_type": "audio/wav",
        "encoding": "base64",
        "data": base64.b64encode(wav).decode("ascii"),
    }
    decoded = decode_wav_payload(envelope, max_bytes=max(len(wav), 32 * 1024 * 1024))
    expected = DEFAULT_AUDIO_CONTRACT.output
    if (
        decoded.sample_rate_hz != expected.sample_rate_hz
        or decoded.channels != expected.channels
        or decoded.sample_width_bits != expected.sample_width_bits
    ):
        raise TTSError(
            "llama-tts output does not satisfy the 24 kHz mono PCM16 adapter contract"
        )
    return wav


def create_app(config: Config | None = None) -> Flask:
    app = Flask(__name__)
    runtime = config or Config.from_environment()
    lock = threading.Lock()

    @app.get("/healthz")
    def healthz():
        missing = [
            str(path)
            for path in (runtime.binary, runtime.model, runtime.projector)
            if not path.is_file()
        ]
        return jsonify({"ok": not missing, "missing": missing}), 200 if not missing else 503

    @app.post("/synthesize")
    def synthesize_route():
        try:
            body = request.get_json(force=True)
            if not isinstance(body, dict):
                raise TTSError("request body must be a JSON object")
            with lock:
                wav = synthesize(runtime, body)
            return Response(wav, content_type="audio/wav")
        except (TTSError, ValueError, subprocess.TimeoutExpired) as exc:
            return jsonify({"error": str(exc)}), 422

    return app


if __name__ == "__main__":
    create_app().run(
        host=os.environ.get("OMNI_TTS_HOST", "127.0.0.1"),
        port=int(os.environ.get("OMNI_TTS_PORT", "8091")),
        debug=False,
    )
