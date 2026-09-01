"""Small HTTP wrapper around llama.cpp's Qwen3-TTS reference binary.

The server is intentionally serial: upstream ``llama-tts`` is currently a
single-shot validation tool. Production deployments should replace this
process wrapper with a persistent libmtmd worker while preserving this HTTP
contract.
"""

from __future__ import annotations

import base64
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
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
    lease_token: str = ""
    gpu_uuid: str = ""
    active_pid_file: Path | None = None
    residency_timeout_s: float = 120

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
            lease_token=os.environ.get("OLLAMA_UNIFY_GPU_LEASE", "").strip(),
            gpu_uuid=os.environ.get("OMNI_TTS_GPU_UUID", "").strip(),
            active_pid_file=(
                Path(os.environ["OMNI_TTS_ACTIVE_PID_FILE"]).expanduser()
                if os.environ.get("OMNI_TTS_ACTIVE_PID_FILE")
                else None
            ),
            residency_timeout_s=float(
                os.environ.get("OMNI_TTS_RESIDENCY_TIMEOUT_S", "120")
            ),
        )


class TTSError(RuntimeError):
    pass


def _broker_transition(action: str, token: str) -> None:
    completed = subprocess.run(
        ["docker", "gpu", action, token],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()[-1000:]
        raise TTSError(f"GPU broker {action} failed: {diagnostic}")


def _cuda_process_is_resident(pid: int, gpu_uuid: str) -> bool:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3:
            continue
        try:
            process_id = int(fields[0])
            used_mib = int(fields[2])
        except ValueError:
            continue
        if process_id == pid and fields[1] == gpu_uuid and used_mib > 0:
            return True
    return False


def _wait_for_cuda_residency(
    process: subprocess.Popen[str], gpu_uuid: str, timeout_s: float
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _cuda_process_is_resident(process.pid, gpu_uuid):
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            diagnostic = (stderr or stdout)[-2000:]
            raise TTSError(
                "llama-tts exited before CUDA residency was verified: " + diagnostic
            )
        time.sleep(0.25)
    raise TTSError(
        f"llama-tts did not become resident on reserved GPU {gpu_uuid} "
        f"within {timeout_s:g}s"
    )


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


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
        if config.lease_token and not config.gpu_uuid:
            raise TTSError("OMNI_TTS_GPU_UUID is required with a scoped GPU lease")
        if config.lease_token:
            _broker_transition("prepare", config.lease_token)

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        if config.active_pid_file:
            config.active_pid_file.parent.mkdir(parents=True, exist_ok=True)
            config.active_pid_file.write_text(f"{process.pid}\n")
        broker_ready = False
        try:
            if config.lease_token:
                _wait_for_cuda_residency(
                    process, config.gpu_uuid, config.residency_timeout_s
                )
                _broker_transition("ready", config.lease_token)
                broker_ready = True
            try:
                stdout, stderr = process.communicate(timeout=config.timeout_s)
            except subprocess.TimeoutExpired:
                _stop_process_group(process)
                raise
        finally:
            if config.active_pid_file:
                config.active_pid_file.unlink(missing_ok=True)
            if config.lease_token and not broker_ready:
                try:
                    # Comprehension remains resident, so restore the scoped lease
                    # if TTS failed between prepare and its own residency signal.
                    _broker_transition("ready", config.lease_token)
                except TTSError as exc:
                    print(f"warning: {exc}", file=sys.stderr)
        if process.returncode != 0:
            diagnostic = (stderr or stdout)[-2000:]
            raise TTSError(f"llama-tts exited {process.returncode}: {diagnostic}")
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
