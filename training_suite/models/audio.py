from __future__ import annotations

import base64
import binascii
import io
import wave
from dataclasses import asdict, dataclass
from typing import Any, Mapping


DEFAULT_MAX_AUDIO_BYTES = 32 * 1024 * 1024
WAV_MIME_TYPES = {"audio/wav", "audio/wave", "audio/x-wav"}


class AudioContractError(ValueError):
    """Raised when an audio request does not satisfy the suite wire contract."""


@dataclass(frozen=True)
class AudioFormat:
    mime_type: str
    container: str
    codec: str
    sample_rate_hz: int
    channels: int
    sample_width_bits: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioContract:
    input: AudioFormat = AudioFormat(
        mime_type="audio/wav",
        container="wav",
        codec="pcm_s16le",
        sample_rate_hz=16000,
        channels=1,
        sample_width_bits=16,
    )
    output: AudioFormat = AudioFormat(
        mime_type="audio/wav",
        container="wav",
        codec="pcm_s16le",
        sample_rate_hz=24000,
        channels=1,
        sample_width_bits=16,
    )
    transport_encoding: str = "base64"
    max_input_bytes: int = DEFAULT_MAX_AUDIO_BYTES

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input.to_dict(),
            "output": self.output.to_dict(),
            "transport_encoding": self.transport_encoding,
            "max_input_bytes": self.max_input_bytes,
        }


DEFAULT_AUDIO_CONTRACT = AudioContract()


@dataclass(frozen=True)
class DecodedAudio:
    data: bytes
    mime_type: str
    sample_rate_hz: int
    channels: int
    sample_width_bits: int
    frames: int
    duration_ms: int

    def metadata(self) -> dict[str, Any]:
        return {
            "mime_type": self.mime_type,
            "encoding": "base64",
            "container": "wav",
            "codec": f"pcm_s{self.sample_width_bits}le",
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "sample_width_bits": self.sample_width_bits,
            "frames": self.frames,
            "duration_ms": self.duration_ms,
            "decoded_bytes": len(self.data),
        }


def _payload_parts(payload: str | Mapping[str, Any]) -> tuple[str, str]:
    if isinstance(payload, str):
        value = payload.strip()
        mime_type = "audio/wav"
    elif isinstance(payload, Mapping):
        encoding = str(payload.get("encoding") or "base64").lower()
        if encoding != "base64":
            raise AudioContractError("audio encoding must be 'base64'")
        value = str(payload.get("data") or "").strip()
        mime_type = str(payload.get("mime_type") or "audio/wav").lower()
    else:
        raise AudioContractError("audio must be a base64 string or an audio envelope object")

    if value.startswith("data:"):
        header, separator, value = value.partition(",")
        if not separator or ";base64" not in header.lower():
            raise AudioContractError("audio data URI must use base64 encoding")
        mime_type = header[5:].split(";", 1)[0].lower()
    if not value:
        raise AudioContractError("audio data is empty")
    if mime_type not in WAV_MIME_TYPES:
        raise AudioContractError(
            f"unsupported audio MIME type {mime_type!r}; phase 1 accepts audio/wav"
        )
    return value, mime_type


def decode_wav_payload(
    payload: str | Mapping[str, Any],
    *,
    max_bytes: int = DEFAULT_MAX_AUDIO_BYTES,
) -> DecodedAudio:
    """Decode and inspect a base64 PCM WAV payload without writing it to disk."""
    encoded, mime_type = _payload_parts(payload)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AudioContractError(f"audio data is not valid base64: {exc}") from exc
    if len(raw) > max_bytes:
        raise AudioContractError(
            f"decoded audio is {len(raw)} bytes; limit is {max_bytes} bytes"
        )
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AudioContractError("audio data must contain a complete RIFF/WAVE header")

    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            if wav.getcomptype() != "NONE":
                raise AudioContractError("WAV input must contain uncompressed PCM samples")
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            width_bits = wav.getsampwidth() * 8
            frames = wav.getnframes()
    except (wave.Error, EOFError) as exc:
        raise AudioContractError(f"invalid WAV container: {exc}") from exc

    duration_ms = round(frames * 1000 / sample_rate) if sample_rate else 0
    return DecodedAudio(
        data=raw,
        mime_type="audio/wav",
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width_bits=width_bits,
        frames=frames,
        duration_ms=duration_ms,
    )


def validate_audio_input(
    payload: str | Mapping[str, Any],
    contract: AudioContract = DEFAULT_AUDIO_CONTRACT,
) -> DecodedAudio:
    audio = decode_wav_payload(payload, max_bytes=contract.max_input_bytes)
    expected = contract.input
    mismatches: list[str] = []
    if audio.sample_rate_hz != expected.sample_rate_hz:
        mismatches.append(
            f"sample_rate_hz={audio.sample_rate_hz}, expected {expected.sample_rate_hz}"
        )
    if audio.channels != expected.channels:
        mismatches.append(f"channels={audio.channels}, expected {expected.channels}")
    if audio.sample_width_bits != expected.sample_width_bits:
        mismatches.append(
            f"sample_width_bits={audio.sample_width_bits}, expected {expected.sample_width_bits}"
        )
    if mismatches:
        raise AudioContractError("audio format mismatch: " + "; ".join(mismatches))
    return audio


def encode_audio_response(
    wav_data: bytes,
    *,
    contract: AudioContract = DEFAULT_AUDIO_CONTRACT,
    transcript: str | None = None,
) -> dict[str, Any]:
    """Wrap a PCM WAV response in the suite's JSON-safe base64 envelope."""
    audio = decode_wav_payload(
        {
            "mime_type": "audio/wav",
            "encoding": "base64",
            "data": base64.b64encode(wav_data).decode("ascii"),
        },
        max_bytes=max(DEFAULT_MAX_AUDIO_BYTES, len(wav_data)),
    )
    expected = contract.output
    if (
        audio.sample_rate_hz,
        audio.channels,
        audio.sample_width_bits,
    ) != (
        expected.sample_rate_hz,
        expected.channels,
        expected.sample_width_bits,
    ):
        raise AudioContractError(
            "audio response must be 24 kHz mono PCM16 WAV under the default contract"
        )
    envelope: dict[str, Any] = {
        "type": "audio",
        **audio.metadata(),
        "data": base64.b64encode(wav_data).decode("ascii"),
    }
    if transcript is not None:
        envelope["transcript"] = transcript
    return envelope
