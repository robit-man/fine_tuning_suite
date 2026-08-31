from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from training_suite.core.config import DEFAULT_OLLAMA_URL
from training_suite.models.audio import (
    DEFAULT_AUDIO_CONTRACT,
    AudioContractError,
    decode_wav_payload,
    encode_audio_response,
    validate_audio_input,
)


class OmniRuntimeError(RuntimeError):
    """Raised when an Omni cascade stage fails or violates its contract."""


@dataclass(frozen=True)
class OmniRuntimeConfig:
    asr_url: str
    asr_model: str
    language_model: str
    tts_url: str
    ollama_url: str = DEFAULT_OLLAMA_URL
    timeout_s: float = 900

    @classmethod
    def from_environment(cls, *, require_tts: bool = True) -> "OmniRuntimeConfig":
        values = {
            "asr_url": os.environ.get("TRAINING_SUITE_OMNI_ASR_URL", "").strip(),
            "asr_model": os.environ.get("TRAINING_SUITE_OMNI_ASR_MODEL", "qwen3-omni").strip(),
            "language_model": os.environ.get("TRAINING_SUITE_OMNI_LANGUAGE_MODEL", "").strip(),
            "tts_url": os.environ.get("TRAINING_SUITE_OMNI_TTS_URL", "").strip(),
            "ollama_url": os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).strip(),
        }
        required = ["asr_url", "language_model"]
        if require_tts:
            required.append("tts_url")
        missing = [key for key in required if not values[key]]
        if missing:
            names = ", ".join(f"TRAINING_SUITE_OMNI_{key.upper()}" for key in missing)
            raise OmniRuntimeError(f"Omni cascade is not configured; missing {names}")
        return cls(**values)


def _response_json(response: httpx.Response, stage: str) -> dict[str, Any]:
    if response.status_code >= 400:
        raise OmniRuntimeError(
            f"{stage} failed with HTTP {response.status_code}: {response.text[:500]}"
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise OmniRuntimeError(f"{stage} did not return JSON") from exc
    if not isinstance(data, dict):
        raise OmniRuntimeError(f"{stage} returned a non-object JSON response")
    return data


def _asr_text(data: Mapping[str, Any]) -> str:
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
        if isinstance(message, Mapping) and message.get("content"):
            return str(message["content"]).strip()
    for key in ("text", "transcript"):
        if data.get(key):
            return str(data[key]).strip()
    raise OmniRuntimeError("audio-understanding stage returned no transcript/content")


def _tts_audio(response: httpx.Response) -> bytes:
    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
    if response.status_code >= 400:
        raise OmniRuntimeError(
            f"speech-synthesis stage failed with HTTP {response.status_code}: {response.text[:500]}"
        )
    if content_type in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return response.content
    try:
        data = response.json()
    except ValueError as exc:
        raise OmniRuntimeError(
            "speech-synthesis stage must return audio/wav bytes or a JSON audio envelope"
        ) from exc
    payload = data.get("audio", data) if isinstance(data, dict) else data
    try:
        return decode_wav_payload(payload).data
    except AudioContractError as exc:
        raise OmniRuntimeError(f"invalid speech-synthesis audio: {exc}") from exc


def run_http_cascade(
    *,
    audio_payload: str | Mapping[str, Any],
    prompt: str,
    config: OmniRuntimeConfig,
    client: httpx.Client | None = None,
    synthesize: bool = True,
) -> dict[str, Any]:
    """Run audio-understanding → Ollama language → TTS over explicit HTTP stages."""
    audio = validate_audio_input(audio_payload)
    encoded = base64.b64encode(audio.data).decode("ascii")
    owns_client = client is None
    session = client or httpx.Client(timeout=config.timeout_s)
    try:
        asr_response = session.post(
            config.asr_url,
            json={
                "model": config.asr_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"format": "wav", "data": encoded},
                            },
                            {"type": "text", "text": prompt or "Transcribe and understand this audio."},
                        ],
                    }
                ],
                "stream": False,
            },
        )
        transcript = _asr_text(_response_json(asr_response, "audio-understanding stage"))

        language_response = session.post(
            config.ollama_url.rstrip("/") + "/api/chat",
            json={
                "model": config.language_model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Audio transcript or semantic description:\n{transcript}\n\n"
                            f"User instruction:\n{prompt or 'Respond to the audio.'}"
                        ),
                    }
                ],
                "think": True,
                "stream": False,
            },
        )
        language_data = _response_json(language_response, "Ollama language stage")
        message = language_data.get("message")
        if not isinstance(message, Mapping) or not message.get("content"):
            raise OmniRuntimeError("Ollama language stage returned no message content")
        answer = str(message["content"]).strip()

        output_message = dict(message)
        result = {
            "mode": "cascade",
            "input_audio": audio.metadata(),
            "audio_understanding": {"content": transcript},
            "message": output_message,
            "routing": {"speech_synthesized": synthesize},
        }
        if synthesize:
            if not config.tts_url:
                raise OmniRuntimeError("speech synthesis was requested but no TTS URL is configured")
            tts_response = session.post(
                config.tts_url,
                json={
                    "text": answer,
                    "output": DEFAULT_AUDIO_CONTRACT.output.to_dict(),
                },
            )
            wav_data = _tts_audio(tts_response)
            audio_response = encode_audio_response(wav_data, transcript=answer)
            output_message["audio"] = audio_response
            result["audio"] = audio_response
        return result
    finally:
        if owns_client:
            session.close()
