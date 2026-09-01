"""Small command-line client for the Omni adapter v1 examples."""

# ruff: noqa: I001 -- direct script execution bootstraps the repository path.

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

import httpx

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training_suite.models.omni_adapter import ADAPTER_SCHEMA


IMAGE_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
VIDEO_MIME = {".mp4": "video/mp4", ".webm": "video/webm"}


def _envelope(path: Path, mime_type: str, **extra: Any) -> dict[str, Any]:
    return {
        "mime_type": mime_type,
        "encoding": "base64",
        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        **extra,
    }


def build_request(args: argparse.Namespace) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "user", "content": args.prompt}
    task = args.command
    response_modalities = ["text"]
    speech_mode = "never"

    if task in {"asr", "chat"} and getattr(args, "audio", None):
        message["audios"] = [_envelope(Path(args.audio), "audio/wav")]
    if task in {"video", "chat"} and getattr(args, "video", None):
        path = Path(args.video)
        mime = VIDEO_MIME.get(path.suffix.lower())
        if not mime:
            raise SystemExit("video input must end in .mp4 or .webm")
        message["videos"] = [
            _envelope(
                path,
                mime,
                sampling={
                    "fps": args.fps,
                    "max_frames": args.max_frames,
                    "include_audio": args.include_audio,
                },
            )
        ]
    if task == "chat" and getattr(args, "image", None):
        path = Path(args.image)
        mime = IMAGE_MIME.get(path.suffix.lower())
        if not mime:
            raise SystemExit("image input must end in .jpg, .jpeg, .png, or .webp")
        message["images"] = [_envelope(path, mime)]
    if task in {"tts", "chat"} and args.speak:
        response_modalities.append("audio")
        speech_mode = "always"

    adapter_task = {
        "asr": "transcribe",
        "video": "describe",
        "tts": "synthesize",
        "chat": "chat",
    }[task]
    return {
        "model": args.model,
        "messages": [message],
        "omni": {
            "schema": ADAPTER_SCHEMA,
            "task": adapter_task,
            "include_audio_from_video": getattr(args, "include_audio", True),
        },
        "response_modalities": response_modalities,
        "speech_mode": speech_mode,
        "speech": {"voice": args.voice} if args.voice else {},
        "think": args.think,
        "stream": False,
    }


def printable_response(data: dict[str, Any], *, include_audio_base64: bool) -> dict[str, Any]:
    """Keep normal CLI output readable while retaining an opt-in raw payload view."""
    if include_audio_base64:
        return data
    printable = json.loads(json.dumps(data))
    message = printable.get("message")
    audio = message.get("audio") if isinstance(message, dict) else None
    if isinstance(audio, dict) and isinstance(audio.get("data"), str):
        encoded_chars = len(audio["data"])
        audio["data"] = f"<base64 omitted; {encoded_chars} characters>"
    return printable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:11435/api/chat")
    parser.add_argument("--model", default="robit/qwen3.8-omni:latest")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--voice")
    parser.add_argument("--think", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-audio", default="response.wav")
    parser.add_argument(
        "--print-audio-base64",
        action="store_true",
        help="include the potentially large message.audio.data value in stdout",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    asr = sub.add_parser("asr", help="transcribe a 16 kHz mono PCM16 WAV")
    asr.add_argument("audio")
    asr.add_argument("--prompt", default="Transcribe this recording verbatim.")
    asr.set_defaults(speak=False)

    video = sub.add_parser("video", help="describe an MP4 or WebM")
    video.add_argument("video")
    video.add_argument(
        "--prompt", default="Describe the important events in temporal order."
    )
    video.add_argument("--fps", type=float, default=1.0)
    video.add_argument("--max-frames", type=int, default=32)
    video.add_argument(
        "--include-audio", action=argparse.BooleanOptionalAction, default=True
    )
    video.set_defaults(speak=False)

    tts = sub.add_parser("tts", help="synthesize the supplied text directly")
    tts.add_argument("prompt")
    tts.set_defaults(speak=True)

    chat = sub.add_parser(
        "chat", help="run media comprehension, language, and optional TTS"
    )
    chat.add_argument("--prompt", required=True)
    chat.add_argument("--audio")
    chat.add_argument("--image")
    chat.add_argument("--video")
    chat.add_argument("--fps", type=float, default=1.0)
    chat.add_argument("--max-frames", type=int, default=32)
    chat.add_argument(
        "--include-audio", action=argparse.BooleanOptionalAction, default=True
    )
    chat.add_argument(
        "--speak", action="store_true", help="also return synthesized speech"
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = build_request(args)
    response = httpx.post(args.endpoint, json=payload, timeout=args.timeout)
    response.raise_for_status()
    data = response.json()
    print(
        json.dumps(
            printable_response(data, include_audio_base64=args.print_audio_base64),
            indent=2,
            sort_keys=True,
        )
    )

    message = data.get("message") or {}
    audio = message.get("audio")
    if isinstance(audio, dict) and audio.get("data"):
        path = Path(args.output_audio).expanduser().resolve()
        path.write_bytes(base64.b64decode(audio["data"], validate=True))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
