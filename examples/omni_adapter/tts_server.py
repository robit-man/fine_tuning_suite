"""Small HTTP wrapper around llama.cpp's Qwen3-TTS reference binary.

The server is intentionally serial: upstream ``llama-tts`` is currently a
per-request validation tool. The private-fork streaming mode exposes completed
libmtmd decoder windows as raw PCM; it does not slice a completed WAV.
Production deployments should replace this process wrapper with a persistent
libmtmd worker while preserving these HTTP contracts.
"""

from __future__ import annotations

import atexit
import base64
import io
import os
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, stream_with_context

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training_suite.models.audio import (
    DEFAULT_AUDIO_CONTRACT,
    AudioContractError,
    decode_wav_payload,
)
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
    stream_frames: int = 2
    gpu_layers: int = -1
    lease_token: str = ""
    gpu_uuid: str = ""
    active_pid_file: Path | None = None
    residency_timeout_s: float = 120
    broker_transition_timeout_s: float = 330
    persistent: bool = True
    warm_speaker_file: str = ""

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
            stream_frames=int(os.environ.get("OMNI_TTS_STREAM_FRAMES", "2")),
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
            broker_transition_timeout_s=float(
                os.environ.get("OMNI_TTS_BROKER_TRANSITION_TIMEOUT_S", "330")
            ),
            persistent=os.environ.get("OMNI_TTS_PERSISTENT", "1").strip().lower()
            not in {"0", "false", "no"},
            warm_speaker_file=os.environ.get(
                "OMNI_TTS_WARM_SPEAKER_FILE", ""
            ).strip(),
        )


class TTSError(RuntimeError):
    pass


@dataclass(frozen=True)
class SynthesisSpec:
    text: str
    language: str
    speaker: str
    speaker_audio: bytes | None
    frames: int
    temperature: float
    top_k: int
    top_p: float
    seed: int
    stream_frames: int


def _broker_transition(action: str, token: str, timeout_s: float = 330) -> None:
    completed = subprocess.run(
        ["docker", "gpu", action, token],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
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
            diagnostic_value = stderr or stdout or ""
            if isinstance(diagnostic_value, bytes):
                diagnostic_value = diagnostic_value.decode("utf-8", errors="replace")
            diagnostic = diagnostic_value[-2000:]
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


def _synthesis_spec(config: Config, body: dict[str, Any]) -> SynthesisSpec:
    text = str(body.get("text") or "").strip()
    if not text:
        raise TTSError("text is required")
    if len(text) > config.max_text_chars:
        raise TTSError(f"text exceeds {config.max_text_chars} characters")
    language = str(body.get("language") or body.get("lang") or "en").strip()
    speaker = str(body.get("speaker_file") or "").strip()
    speaker_audio: bytes | None = None
    if body.get("speaker_audio") is not None:
        if speaker:
            raise TTSError("speaker_file and speaker_audio are mutually exclusive")
        try:
            decoded_speaker = decode_wav_payload(
                body["speaker_audio"], max_bytes=10 * 1024 * 1024
            )
        except AudioContractError as exc:
            raise TTSError(f"invalid speaker_audio: {exc}") from exc
        if decoded_speaker.duration_ms < 500:
            raise TTSError("speaker_audio must be at least 0.5 seconds")
        if decoded_speaker.duration_ms > 30_000:
            raise TTSError("speaker_audio must be no longer than 30 seconds")
        speaker_audio = decoded_speaker.data
    frames = min(int(body.get("max_frames") or config.max_frames), config.max_frames)
    temperature = float(body.get("temperature", 0.7))
    top_k = int(body.get("top_k", 40))
    top_p = float(body.get("top_p", 0.9))
    seed = int(body.get("seed", 42))
    stream_frames_value = body.get("stream_frames")
    stream_frames = int(
        config.stream_frames if stream_frames_value is None else stream_frames_value
    )
    if not 0 <= temperature <= 2:
        raise TTSError("temperature must be between 0 and 2")
    if not 0 <= top_k <= 1000:
        raise TTSError("top_k must be between 0 and 1000")
    if not 0 <= top_p <= 1:
        raise TTSError("top_p must be between 0 and 1")
    if not -1 <= seed <= 2_147_483_647:
        raise TTSError("seed must be -1 or a 32-bit non-negative integer")
    if not 1 <= stream_frames <= 72:
        raise TTSError("stream_frames must be between 1 and 72")
    return SynthesisSpec(
        text=text,
        language=language,
        speaker=speaker,
        speaker_audio=speaker_audio,
        frames=frames,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        stream_frames=stream_frames,
    )


def _command(
    config: Config,
    spec: SynthesisSpec,
    output: Path,
    *,
    stream: bool = False,
    persistent: bool = False,
    speaker_file: str | None = None,
) -> list[str]:
    command = [
        str(config.binary),
        "-m",
        str(config.model),
        "--mmproj",
        str(config.projector),
        "--prompt",
        spec.text,
        "--tts-lang",
        spec.language,
        "--output",
        str(output),
        "--n-predict",
        str(spec.frames),
        "--gpu-layers",
        str(config.gpu_layers),
        "--temp",
        str(spec.temperature),
        "--top-k",
        str(spec.top_k),
        "--top-p",
        str(spec.top_p),
        "--seed",
        str(spec.seed),
    ]
    selected_speaker = speaker_file if speaker_file is not None else spec.speaker
    if selected_speaker:
        command.extend(["--tts-speaker-file", selected_speaker])
    if stream:
        command.extend(["--tts-stream", "--tts-stream-frames", str(spec.stream_frames)])
    if persistent:
        command.append("--tts-persistent")
    return command


def _speaker_path(spec: SynthesisSpec, temp_dir: str) -> str:
    if spec.speaker_audio is None:
        return spec.speaker
    path = Path(temp_dir) / "speaker-reference.wav"
    path.write_bytes(spec.speaker_audio)
    return str(path)


def _validate_wav(wav: bytes) -> None:
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


def _pcm_wav(pcm: bytes) -> bytes:
    if not pcm or len(pcm) % 2:
        raise TTSError("TTS PCM must contain complete 16-bit samples")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(pcm)
    return output.getvalue()


def synthesize(config: Config, body: dict[str, Any]) -> bytes:
    spec = _synthesis_spec(config, body)

    with tempfile.TemporaryDirectory(prefix="robit-omni-tts-") as temp_dir:
        output = Path(temp_dir) / "speech.wav"
        command = _command(
            config, spec, output, speaker_file=_speaker_path(spec, temp_dir)
        )
        if config.lease_token and not config.gpu_uuid:
            raise TTSError("OMNI_TTS_GPU_UUID is required with a scoped GPU lease")
        if config.lease_token:
            try:
                _broker_transition(
                    "prepare",
                    config.lease_token,
                    config.broker_transition_timeout_s,
                )
            except (TTSError, subprocess.TimeoutExpired):
                # The client can time out while the broker is still draining.
                # Restore the stable comprehension reservation before failing.
                try:
                    _broker_transition("ready", config.lease_token, 60)
                except (TTSError, subprocess.TimeoutExpired) as exc:
                    print(f"warning: GPU broker ready rollback failed: {exc}", file=sys.stderr)
                raise

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
                _broker_transition(
                    "ready",
                    config.lease_token,
                    config.broker_transition_timeout_s,
                )
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
                    _broker_transition("ready", config.lease_token, 60)
                except (TTSError, subprocess.TimeoutExpired) as exc:
                    print(f"warning: {exc}", file=sys.stderr)
        if process.returncode != 0:
            diagnostic = (stderr or stdout)[-2000:]
            raise TTSError(f"llama-tts exited {process.returncode}: {diagnostic}")
        if not output.is_file():
            raise TTSError("llama-tts returned success without a WAV file")
        wav = output.read_bytes()

    _validate_wav(wav)
    return wav


def _stream_synthesize(config: Config, spec: SynthesisSpec) -> Iterator[bytes]:
    with tempfile.TemporaryDirectory(prefix="robit-omni-tts-stream-") as temp_dir:
        output = Path(temp_dir) / "speech.wav"
        command = _command(
            config,
            spec,
            output,
            stream=True,
            speaker_file=_speaker_path(spec, temp_dir),
        )
        if config.lease_token and not config.gpu_uuid:
            raise TTSError("OMNI_TTS_GPU_UUID is required with a scoped GPU lease")
        if config.lease_token:
            try:
                _broker_transition(
                    "prepare",
                    config.lease_token,
                    config.broker_transition_timeout_s,
                )
            except (TTSError, subprocess.TimeoutExpired):
                try:
                    _broker_transition("ready", config.lease_token, 60)
                except (TTSError, subprocess.TimeoutExpired) as exc:
                    print(f"warning: GPU broker ready rollback failed: {exc}", file=sys.stderr)
                raise

        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=stderr_log,
                start_new_session=True,
            )
            if process.stdout is None:
                _stop_process_group(process)
                raise TTSError("failed to open llama-tts PCM stream")
            if config.active_pid_file:
                config.active_pid_file.parent.mkdir(parents=True, exist_ok=True)
                config.active_pid_file.write_text(f"{process.pid}\n")
            broker_ready = False
            try:
                if config.lease_token:
                    _wait_for_cuda_residency(
                        process, config.gpu_uuid, config.residency_timeout_s
                    )
                    _broker_transition(
                        "ready",
                        config.lease_token,
                        config.broker_transition_timeout_s,
                    )
                    broker_ready = True

                deadline = time.monotonic() + config.timeout_s
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        _stop_process_group(process)
                        raise subprocess.TimeoutExpired(command, config.timeout_s)
                    readable, _, _ = select.select(
                        [process.stdout], [], [], min(1.0, remaining)
                    )
                    if readable:
                        chunk = os.read(process.stdout.fileno(), 64 * 1024)
                        if chunk:
                            yield chunk
                            continue
                        break
                    if process.poll() is not None:
                        continue

                process.wait(timeout=max(0.1, deadline - time.monotonic()))
                stderr_log.seek(0)
                diagnostic = stderr_log.read()[-2000:]
                if process.returncode != 0:
                    raise TTSError(
                        f"llama-tts exited {process.returncode}: {diagnostic}"
                    )
                if not output.is_file():
                    raise TTSError("llama-tts returned success without a WAV file")
                _validate_wav(output.read_bytes())
            finally:
                if process.poll() is None:
                    _stop_process_group(process)
                if config.active_pid_file:
                    config.active_pid_file.unlink(missing_ok=True)
                if config.lease_token and not broker_ready:
                    try:
                        _broker_transition("ready", config.lease_token, 60)
                    except (TTSError, subprocess.TimeoutExpired) as exc:
                        print(f"warning: {exc}", file=sys.stderr)


def stream_synthesize(
    config: Config, body: dict[str, Any]
) -> tuple[SynthesisSpec, Iterator[bytes]]:
    spec = _synthesis_spec(config, body)
    return spec, _stream_synthesize(config, spec)


class PersistentTTSWorker:
    """Keep one llama.cpp TTS graph resident and exchange framed PCM over pipes."""

    def __init__(self, config: Config):
        self.config = config
        self.process: subprocess.Popen[bytes] | None = None
        self.profile: tuple[Any, ...] | None = None
        self.stderr_log: Any = None

    @staticmethod
    def _profile(spec: SynthesisSpec) -> tuple[Any, ...]:
        return (
            spec.language,
            spec.speaker,
            spec.frames,
            spec.temperature,
            spec.top_k,
            spec.top_p,
            spec.seed,
            spec.stream_frames,
        )

    @property
    def ready(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _diagnostic(self) -> str:
        if self.stderr_log is None:
            return ""
        self.stderr_log.flush()
        self.stderr_log.seek(0)
        value = self.stderr_log.read()[-2000:]
        self.stderr_log.seek(0, os.SEEK_END)
        return value

    @staticmethod
    def _read_exact(
        process: subprocess.Popen[bytes], size: int, deadline: float
    ) -> bytes:
        if process.stdout is None:
            raise TTSError("persistent TTS stdout is unavailable")
        result = bytearray()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired("persistent llama-tts", 0)
            readable, _, _ = select.select(
                [process.stdout], [], [], min(1.0, remaining)
            )
            if not readable:
                if process.poll() is not None:
                    raise TTSError("persistent llama-tts exited unexpectedly")
                continue
            chunk = os.read(process.stdout.fileno(), size - len(result))
            if not chunk:
                raise TTSError("persistent llama-tts closed its output")
            result.extend(chunk)
        return bytes(result)

    @classmethod
    def _read_frame(
        cls, process: subprocess.Popen[bytes], deadline: float
    ) -> tuple[str, bytes]:
        header = cls._read_exact(process, 9, deadline)
        size = int.from_bytes(header[1:], "little")
        if size > 128 * 1024 * 1024:
            raise TTSError("persistent TTS emitted an oversized frame")
        payload = cls._read_exact(process, size, deadline) if size else b""
        return chr(header[0]), payload

    def close(self) -> None:
        process = self.process
        self.process = None
        self.profile = None
        if process is not None:
            _stop_process_group(process)
        if self.stderr_log is not None:
            self.stderr_log.close()
            self.stderr_log = None
        if self.config.active_pid_file:
            self.config.active_pid_file.unlink(missing_ok=True)

    def ensure(self, spec: SynthesisSpec) -> None:
        wanted = self._profile(spec)
        if self.ready and self.profile == wanted:
            return
        self.close()
        if spec.speaker_audio is not None:
            raise TTSError("inline speaker audio requires the single-shot TTS path")
        if self.config.lease_token and not self.config.gpu_uuid:
            raise TTSError("OMNI_TTS_GPU_UUID is required with a scoped GPU lease")
        if self.config.lease_token:
            _broker_transition(
                "prepare",
                self.config.lease_token,
                self.config.broker_transition_timeout_s,
            )
        command = _command(
            self.config,
            spec,
            Path(os.devnull),
            stream=True,
            persistent=True,
        )
        self.stderr_log = tempfile.TemporaryFile(  # noqa: SIM115 - worker lifetime owns it
            mode="w+t", encoding="utf-8"
        )
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_log,
            start_new_session=True,
        )
        self.process = process
        if self.config.active_pid_file:
            self.config.active_pid_file.parent.mkdir(parents=True, exist_ok=True)
            self.config.active_pid_file.write_text(f"{process.pid}\n")
        broker_ready = False
        try:
            if self.config.lease_token:
                _wait_for_cuda_residency(
                    process, self.config.gpu_uuid, self.config.residency_timeout_s
                )
            frame_type, payload = self._read_frame(
                process, time.monotonic() + self.config.timeout_s
            )
            if frame_type != "R" or payload:
                raise TTSError("persistent llama-tts returned an invalid ready frame")
            if self.config.lease_token:
                _broker_transition(
                    "ready",
                    self.config.lease_token,
                    self.config.broker_transition_timeout_s,
                )
                broker_ready = True
            self.profile = wanted
        except Exception:
            diagnostic = self._diagnostic()
            self.close()
            if self.config.lease_token and not broker_ready:
                try:
                    _broker_transition("ready", self.config.lease_token, 60)
                except (TTSError, subprocess.TimeoutExpired) as exc:
                    print(f"warning: GPU broker ready rollback failed: {exc}", file=sys.stderr)
            if diagnostic:
                print(diagnostic, file=sys.stderr)
            raise

    def stream(self, spec: SynthesisSpec) -> Iterator[bytes]:
        self.ensure(spec)
        process = self.process
        if process is None or process.stdin is None:
            raise TTSError("persistent llama-tts is unavailable")
        encoded = base64.b64encode(spec.text.encode("utf-8")) + b"\n"
        try:
            process.stdin.write(encoded)
            process.stdin.flush()
            deadline = time.monotonic() + self.config.timeout_s
            while True:
                frame_type, payload = self._read_frame(process, deadline)
                if frame_type == "A":
                    if payload:
                        yield payload
                elif frame_type == "D":
                    if payload:
                        raise TTSError("persistent TTS done frame contained data")
                    return
                elif frame_type == "E":
                    raise TTSError(
                        payload.decode("utf-8", errors="replace")
                        or "persistent TTS generation failed"
                    )
                else:
                    raise TTSError(
                        f"persistent TTS emitted unknown frame {frame_type!r}"
                    )
        except Exception:
            self.close()
            raise


def _warm_spec(config: Config) -> SynthesisSpec | None:
    if not config.warm_speaker_file:
        return None
    speaker = str(Path(config.warm_speaker_file).expanduser().resolve())
    if not Path(speaker).is_file():
        return None
    return SynthesisSpec(
        text="warmup",
        language="en",
        speaker=speaker,
        speaker_audio=None,
        frames=config.max_frames,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        seed=42,
        stream_frames=config.stream_frames,
    )


def create_app(config: Config | None = None) -> Flask:
    app = Flask(__name__)
    runtime = config or Config.from_environment()
    lock = threading.Lock()
    persistent = PersistentTTSWorker(runtime) if runtime.persistent else None
    warm_spec = _warm_spec(runtime)

    def warm_persistent_worker() -> None:
        if persistent is None or warm_spec is None:
            return
        try:
            with lock:
                persistent.ensure(warm_spec)
        except (TTSError, subprocess.TimeoutExpired, OSError) as exc:
            print(f"warning: persistent TTS warmup failed: {exc}", file=sys.stderr)

    if persistent is not None:
        atexit.register(persistent.close)
    if persistent is not None and warm_spec is not None:
        threading.Thread(
            target=warm_persistent_worker,
            name="qwen3-tts-warmup",
            daemon=True,
        ).start()

    @app.get("/healthz")
    def healthz():
        missing = [
            str(path)
            for path in (runtime.binary, runtime.model, runtime.projector)
            if not path.is_file()
        ]
        return jsonify(
            {
                "ok": not missing,
                "missing": missing,
                "persistent": bool(persistent),
                "persistent_ready": bool(persistent and persistent.ready),
            }
        ), 200 if not missing else 503

    @app.post("/synthesize")
    def synthesize_route():
        try:
            body = request.get_json(force=True)
            if not isinstance(body, dict):
                raise TTSError("request body must be a JSON object")
            with lock:
                spec = _synthesis_spec(runtime, body)
                if persistent is not None and spec.speaker_audio is None:
                    wav = _pcm_wav(b"".join(persistent.stream(spec)))
                    _validate_wav(wav)
                else:
                    if persistent is not None:
                        persistent.close()
                    wav = synthesize(runtime, body)
                    if persistent is not None and warm_spec is not None:
                        threading.Thread(
                            target=warm_persistent_worker,
                            name="qwen3-tts-rewarm",
                            daemon=True,
                        ).start()
            return Response(wav, content_type="audio/wav")
        except (TTSError, ValueError, subprocess.TimeoutExpired) as exc:
            return jsonify({"error": str(exc)}), 422

    @app.post("/synthesize/stream")
    def synthesize_stream_route():
        try:
            body = request.get_json(force=True)
            if not isinstance(body, dict):
                raise TTSError("request body must be a JSON object")
            spec = _synthesis_spec(runtime, body)

            def generate() -> Iterator[bytes]:
                with lock:
                    if persistent is not None and spec.speaker_audio is None:
                        yield from persistent.stream(spec)
                    else:
                        if persistent is not None:
                            persistent.close()
                        yield from _stream_synthesize(runtime, spec)
                        if persistent is not None and warm_spec is not None:
                            threading.Thread(
                                target=warm_persistent_worker,
                                name="qwen3-tts-rewarm",
                                daemon=True,
                            ).start()

            response = Response(
                stream_with_context(generate()),
                content_type="audio/pcm;rate=24000;channels=1;format=s16le",
            )
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Audio-Codec"] = "pcm_s16le"
            response.headers["X-Audio-Sample-Rate"] = "24000"
            response.headers["X-Audio-Channels"] = "1"
            response.headers["X-Audio-Stream-Version"] = "1"
            response.headers["X-Audio-Stream-Frames"] = str(spec.stream_frames)
            response.headers["X-Accel-Buffering"] = "no"
            return response
        except (TTSError, ValueError, subprocess.TimeoutExpired) as exc:
            return jsonify({"error": str(exc)}), 422

    return app


if __name__ == "__main__":
    create_app().run(
        host=os.environ.get("OMNI_TTS_HOST", "127.0.0.1"),
        port=int(os.environ.get("OMNI_TTS_PORT", "8091")),
        debug=False,
    )
