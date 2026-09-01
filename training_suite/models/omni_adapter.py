from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from training_suite.models.audio import DEFAULT_AUDIO_CONTRACT, validate_audio_input

ADAPTER_SCHEMA = "robit.ollama.omni-adapter.v1"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_VIDEO_BYTES = 256 * 1024 * 1024
MAX_MEDIA_PER_REQUEST = {"audio": 8, "image": 16, "video": 4}
TASKS = {"chat", "transcribe", "describe", "synthesize"}
SPEECH_MODES = {"auto", "always", "never"}
RESPONSE_MODALITIES = {"text", "audio"}


class OmniAdapterError(ValueError):
    """Raised when a request does not satisfy the Omni adapter v1 ABI."""


@dataclass(frozen=True)
class MediaItem:
    kind: str
    mime_type: str
    data: bytes
    message_index: int
    media_index: int
    options: Mapping[str, Any] = field(default_factory=dict)

    def data_uri(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"

    def summary(self) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "mime_type": self.mime_type,
            "decoded_bytes": len(self.data),
            "message_index": self.message_index,
            "media_index": self.media_index,
        }
        if self.options:
            result["options"] = dict(self.options)
        return result


@dataclass(frozen=True)
class AdapterMessage:
    role: str
    content: str
    audios: tuple[MediaItem, ...] = ()
    images: tuple[MediaItem, ...] = ()
    videos: tuple[MediaItem, ...] = ()
    passthrough: Mapping[str, Any] = field(default_factory=dict)

    @property
    def media(self) -> tuple[MediaItem, ...]:
        return self.audios + self.images + self.videos


@dataclass(frozen=True)
class ParsedAdapterRequest:
    model: str
    messages: tuple[AdapterMessage, ...]
    task: str
    response_modalities: tuple[str, ...]
    speech_mode: str
    synthesize: bool
    include_audio_from_video: bool
    passthrough: Mapping[str, Any]
    speech: Mapping[str, Any]

    @property
    def media(self) -> tuple[MediaItem, ...]:
        return tuple(item for message in self.messages for item in message.media)

    @property
    def input_modalities(self) -> tuple[str, ...]:
        kinds = {item.kind for item in self.media}
        if any(message.content for message in self.messages):
            kinds.add("text")
        return tuple(
            kind for kind in ("text", "audio", "image", "video") if kind in kinds
        )

    @property
    def route(self) -> tuple[str, ...]:
        if self.task in {"transcribe", "describe"}:
            return ("comprehension", "tts") if self.synthesize else ("comprehension",)
        if self.task == "synthesize":
            return ("tts",)
        stages: list[str] = []
        if self.media:
            stages.append("comprehension")
        stages.append("language")
        if self.synthesize:
            stages.append("tts")
        return tuple(stages)

    def summary(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "model": self.model,
            "task": self.task,
            "input_modalities": list(self.input_modalities),
            "response_modalities": list(self.response_modalities),
            "speech_mode": self.speech_mode,
            "synthesize": self.synthesize,
            "include_audio_from_video": self.include_audio_from_video,
            "route": list(self.route),
            "media": [item.summary() for item in self.media],
        }


def adapter_contract() -> dict[str, Any]:
    """Return the versioned wire contract consumed by a patched Ollama handler."""
    return {
        "schema": ADAPTER_SCHEMA,
        "transport": {
            "endpoint": "/api/chat",
            "content_type": "application/json",
            "streaming_v1": False,
            "binary_encoding": "base64",
        },
        "compatibility": {
            "preserved_ollama_fields": [
                "model",
                "messages",
                "tools",
                "think",
                "format",
                "options",
                "keep_alive",
                "logprobs",
                "top_logprobs",
            ],
            "message_extensions": ["audios", "videos"],
            "request_extensions": [
                "omni",
                "response_modalities",
                "speech_mode",
                "speech",
            ],
            "response_extensions": ["message.audio", "adapter"],
        },
        "request": {
            "required": ["model", "messages"],
            "stream": False,
            "omni": {
                "schema": ADAPTER_SCHEMA,
                "task": "chat | transcribe | describe | synthesize",
                "include_audio_from_video": True,
            },
            "response_modalities": ["text", "audio"],
            "speech_mode": "auto | always | never",
            "speech": {
                "voice": "optional backend voice identifier",
                "language": "optional BCP-47 language hint",
                "style": "optional synthesis instruction",
                "speaker_audio": (
                    "optional base64 WAV reference for backends that support cloning"
                ),
            },
            "messages": {
                "content": "normal Ollama string content",
                "images": "normal Ollama base64 images or structured image envelopes",
                "audios": "structured audio envelopes",
                "videos": "structured video envelopes",
            },
        },
        "media": {
            "audio": {
                **DEFAULT_AUDIO_CONTRACT.input.to_dict(),
                "encoding": "base64",
                "max_decoded_bytes": DEFAULT_AUDIO_CONTRACT.max_input_bytes,
                "max_items": MAX_MEDIA_PER_REQUEST["audio"],
            },
            "image": {
                "mime_types": ["image/jpeg", "image/png", "image/webp"],
                "encoding": "base64",
                "max_decoded_bytes": MAX_IMAGE_BYTES,
                "max_items": MAX_MEDIA_PER_REQUEST["image"],
            },
            "video": {
                "mime_types": ["video/mp4", "video/webm", "image/gif"],
                "encoding": "base64",
                "max_decoded_bytes": MAX_VIDEO_BYTES,
                "max_items": MAX_MEDIA_PER_REQUEST["video"],
                "sampling": {
                    "fps": "number in (0, 30]",
                    "max_frames": "integer in [1, 1024]",
                    "include_audio": "boolean",
                },
            },
        },
        "routing": {
            "chat": "media -> comprehension -> semantic text -> language -> optional tts",
            "transcribe": "audio -> comprehension -> text -> optional tts",
            "describe": "audio/image/video -> comprehension -> text -> optional tts",
            "synthesize": "text -> tts -> audio",
            "speech_precedence": [
                "speech_mode=always",
                "speech_mode=never",
                "response_modalities contains audio",
            ],
            "tool_rule": "do not synthesize speech while unresolved tool_calls are present",
        },
        "response": {
            "ollama_fields_preserved": True,
            "message": {
                "role": "assistant",
                "content": "text response or direct transcript/description",
                "thinking": "unchanged Ollama thinking field when language stage runs",
                "tool_calls": "unchanged Ollama tool calls when language stage runs",
                "audio": {
                    "type": "audio",
                    **DEFAULT_AUDIO_CONTRACT.output.to_dict(),
                    "encoding": "base64",
                    "data": "<base64 RIFF/WAVE bytes>",
                },
            },
            "adapter": {
                "schema": ADAPTER_SCHEMA,
                "task": "selected task",
                "route": ["executed stage names"],
                "input_transcript": "tagged verbatim speech when available",
                "audio_observation": (
                    "tagged environmental and non-speech acoustic evidence when available"
                ),
            },
        },
    }


def _decode_base64_envelope(
    payload: str | Mapping[str, Any],
    *,
    default_mime: str,
    max_bytes: int,
) -> tuple[bytes, str, Mapping[str, Any]]:
    if isinstance(payload, str):
        encoded = payload.strip()
        mime_type = default_mime
        options: Mapping[str, Any] = {}
    elif isinstance(payload, Mapping):
        encoding = str(payload.get("encoding") or "base64").lower()
        if encoding != "base64":
            raise OmniAdapterError("media encoding must be 'base64'")
        encoded = str(payload.get("data") or "").strip()
        mime_type = str(payload.get("mime_type") or default_mime).lower()
        options = payload
    else:
        raise OmniAdapterError("media must be a base64 string or an envelope object")

    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header.lower():
            raise OmniAdapterError("media data URI must use base64 encoding")
        mime_type = header[5:].split(";", 1)[0].lower()
    if not encoded:
        raise OmniAdapterError("media data is empty")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise OmniAdapterError(f"media data is not valid base64: {exc}") from exc
    if len(raw) > max_bytes:
        raise OmniAdapterError(
            f"decoded media is {len(raw)} bytes; limit is {max_bytes} bytes"
        )
    return raw, mime_type, options


def _sniff_image(raw: bytes) -> str | None:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _sniff_video(raw: bytes) -> str | None:
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        return "video/mp4"
    if raw.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    return None


def _sampling_options(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("sampling") or {}
    if not isinstance(raw, Mapping):
        raise OmniAdapterError("video sampling must be an object")
    result: dict[str, Any] = {}
    if "fps" in raw:
        fps = raw["fps"]
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not 0 < fps <= 30
        ):
            raise OmniAdapterError("video sampling.fps must be a number in (0, 30]")
        result["fps"] = float(fps)
    if "max_frames" in raw:
        max_frames = raw["max_frames"]
        if (
            isinstance(max_frames, bool)
            or not isinstance(max_frames, int)
            or not 1 <= max_frames <= 1024
        ):
            raise OmniAdapterError(
                "video sampling.max_frames must be an integer in [1, 1024]"
            )
        result["max_frames"] = max_frames
    if "include_audio" in raw:
        if not isinstance(raw["include_audio"], bool):
            raise OmniAdapterError("video sampling.include_audio must be a boolean")
        result["include_audio"] = raw["include_audio"]
    return result


def _parse_media_item(
    kind: str,
    payload: str | Mapping[str, Any],
    *,
    message_index: int,
    media_index: int,
) -> MediaItem:
    if kind == "audio":
        try:
            audio = validate_audio_input(payload)
        except ValueError as exc:
            raise OmniAdapterError(str(exc)) from exc
        return MediaItem(
            kind=kind,
            mime_type="audio/wav",
            data=audio.data,
            message_index=message_index,
            media_index=media_index,
        )

    max_bytes = MAX_IMAGE_BYTES if kind == "image" else MAX_VIDEO_BYTES
    default_mime = "image/jpeg" if kind == "image" else "video/mp4"
    raw, declared_mime, envelope = _decode_base64_envelope(
        payload,
        default_mime=default_mime,
        max_bytes=max_bytes,
    )
    detected_mime = _sniff_image(raw) if kind == "image" else _sniff_video(raw)
    if (
        isinstance(payload, str)
        and not payload.strip().startswith("data:")
        and detected_mime
    ):
        # Stock Ollama image fields are bare base64 strings and carry no MIME
        # declaration, so the container signature is authoritative in that form.
        declared_mime = detected_mime
    allowed = (
        {"image/jpeg", "image/png", "image/webp"}
        if kind == "image"
        else {"video/mp4", "video/webm", "image/gif"}
    )
    if declared_mime not in allowed:
        raise OmniAdapterError(f"unsupported {kind} MIME type {declared_mime!r}")
    if detected_mime is None:
        raise OmniAdapterError(f"{kind} payload has no supported container signature")
    if detected_mime != declared_mime:
        raise OmniAdapterError(
            f"{kind} MIME type {declared_mime!r} does not match detected {detected_mime!r}"
        )
    options = _sampling_options(envelope) if kind == "video" else {}
    return MediaItem(
        kind=kind,
        mime_type=detected_mime,
        data=raw,
        message_index=message_index,
        media_index=media_index,
        options=options,
    )


def _parse_messages(messages: Any) -> tuple[AdapterMessage, ...]:
    if (
        not isinstance(messages, Sequence)
        or isinstance(messages, (str, bytes))
        or not messages
    ):
        raise OmniAdapterError("messages must be a non-empty array")
    parsed: list[AdapterMessage] = []
    counts = {kind: 0 for kind in MAX_MEDIA_PER_REQUEST}
    for message_index, raw_message in enumerate(messages):
        if not isinstance(raw_message, Mapping):
            raise OmniAdapterError(f"messages[{message_index}] must be an object")
        role = str(raw_message.get("role") or "").lower()
        if role not in {"system", "user", "assistant", "tool"}:
            raise OmniAdapterError(f"messages[{message_index}].role is unsupported")
        content = raw_message.get("content", "")
        if not isinstance(content, str):
            raise OmniAdapterError(
                f"messages[{message_index}].content must be a string for Ollama compatibility"
            )
        media_by_kind: dict[str, tuple[MediaItem, ...]] = {}
        for kind, field_name in (
            ("audio", "audios"),
            ("image", "images"),
            ("video", "videos"),
        ):
            raw_items = raw_message.get(field_name) or []
            if not isinstance(raw_items, list):
                raise OmniAdapterError(
                    f"messages[{message_index}].{field_name} must be an array"
                )
            items = tuple(
                _parse_media_item(
                    kind,
                    item,
                    message_index=message_index,
                    media_index=item_index,
                )
                for item_index, item in enumerate(raw_items)
            )
            counts[kind] += len(items)
            if counts[kind] > MAX_MEDIA_PER_REQUEST[kind]:
                raise OmniAdapterError(
                    f"request contains {counts[kind]} {kind} items; limit is "
                    f"{MAX_MEDIA_PER_REQUEST[kind]}"
                )
            media_by_kind[kind] = items
        passthrough = {
            key: value
            for key, value in raw_message.items()
            if key not in {"role", "content", "audios", "images", "videos"}
        }
        parsed.append(
            AdapterMessage(
                role=role,
                content=content,
                audios=media_by_kind["audio"],
                images=media_by_kind["image"],
                videos=media_by_kind["video"],
                passthrough=passthrough,
            )
        )
    return tuple(parsed)


def _last_user(messages: Sequence[AdapterMessage]) -> AdapterMessage:
    for message in reversed(messages):
        if message.role == "user":
            return message
    raise OmniAdapterError("messages must contain at least one user message")


def parse_adapter_request(payload: Mapping[str, Any]) -> ParsedAdapterRequest:
    """Validate an Ollama-compatible Omni request and calculate its execution route."""
    if not isinstance(payload, Mapping):
        raise OmniAdapterError("request body must be a JSON object")
    model = str(payload.get("model") or "").strip()
    if not model:
        raise OmniAdapterError("model is required")
    if payload.get("stream") is not False:
        raise OmniAdapterError("Omni adapter v1 requires stream=false")

    messages = _parse_messages(payload.get("messages"))
    last_user = _last_user(messages)
    omni = payload.get("omni") or {}
    if not isinstance(omni, Mapping):
        raise OmniAdapterError("omni must be an object")
    schema = omni.get("schema")
    if schema not in {None, "", ADAPTER_SCHEMA}:
        raise OmniAdapterError(f"unsupported omni schema {schema!r}")
    task = str(omni.get("task") or "chat").lower()
    if task not in TASKS:
        raise OmniAdapterError(f"omni.task must be one of {sorted(TASKS)}")
    include_audio = omni.get("include_audio_from_video", True)
    if not isinstance(include_audio, bool):
        raise OmniAdapterError("omni.include_audio_from_video must be a boolean")

    raw_modalities = payload.get("response_modalities") or ["text"]
    if not isinstance(raw_modalities, list) or not raw_modalities:
        raise OmniAdapterError("response_modalities must be a non-empty array")
    modalities = tuple(str(item).lower() for item in raw_modalities)
    unsupported = sorted(set(modalities) - RESPONSE_MODALITIES)
    if unsupported:
        raise OmniAdapterError(f"unsupported response modalities: {unsupported}")
    if len(set(modalities)) != len(modalities):
        raise OmniAdapterError("response_modalities must not contain duplicates")

    speech_mode = str(payload.get("speech_mode") or "auto").lower()
    if speech_mode not in SPEECH_MODES:
        raise OmniAdapterError(f"speech_mode must be one of {sorted(SPEECH_MODES)}")
    speech = payload.get("speech") or {}
    if not isinstance(speech, Mapping):
        raise OmniAdapterError("speech must be an object")
    synthesize = speech_mode == "always" or (
        speech_mode == "auto" and "audio" in modalities
    )
    if task == "synthesize":
        if not last_user.content.strip():
            raise OmniAdapterError(
                "synthesize task requires text in the last user message"
            )
        if last_user.media:
            raise OmniAdapterError("synthesize task does not accept input media")
        synthesize = True
    if task == "transcribe" and not any(
        item.kind == "audio" for item in last_user.media
    ):
        raise OmniAdapterError(
            "transcribe task requires audio in the last user message"
        )
    if task == "describe" and not last_user.media:
        raise OmniAdapterError("describe task requires media in the last user message")

    passthrough = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "model",
            "messages",
            "omni",
            "response_modalities",
            "speech_mode",
            "speech",
            "stream",
        }
    }
    return ParsedAdapterRequest(
        model=model,
        messages=messages,
        task=task,
        response_modalities=modalities,
        speech_mode=speech_mode,
        synthesize=synthesize,
        include_audio_from_video=include_audio,
        passthrough=passthrough,
        speech=speech,
    )
