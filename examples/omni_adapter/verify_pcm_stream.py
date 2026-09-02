#!/usr/bin/env python3
"""Probe Qwen3-TTS incremental PCM for repeated decoder-boundary artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_TEXT = (
    "The streaming decoder should preserve one continuous voice across every "
    "boundary without a repeated phantom syllable or artificial restart."
)


def _rms(values: list[float]) -> float:
    return (
        math.sqrt(sum(value * value for value in values) / len(values))
        if values
        else 0.0
    )


def analyze_pcm(
    pcm: bytes,
    *,
    sample_rate: int = 24_000,
    stream_frames: int = 2,
    samples_per_codec_frame: int = 1_920,
) -> dict[str, Any]:
    if len(pcm) % 2:
        raise ValueError("PCM16 payload has an incomplete sample")
    window_samples = stream_frames * samples_per_codec_frame
    samples = [value / 32768.0 for (value,) in struct.iter_unpack("<h", pcm)]
    windows = [
        samples[offset : offset + window_samples]
        for offset in range(0, len(samples), window_samples)
    ]
    full_windows = [window for window in windows if len(window) == window_samples]
    if len(full_windows) < 4:
        raise ValueError("at least four complete decoder windows are required")

    measured_windows = full_windows[1:]
    starts = [window[0] for window in measured_windows]
    shared_prefix: dict[str, float] = {}
    for milliseconds in (1, 10, 20):
        count = sample_rate * milliseconds // 1000
        ensemble = [
            statistics.fmean(window[index] for window in measured_windows)
            for index in range(count)
        ]
        median_window_rms = statistics.median(
            _rms(window[:count]) for window in measured_windows
        )
        shared_prefix[str(milliseconds)] = _rms(ensemble) / max(
            median_window_rms, 1e-12
        )

    local_samples = sample_rate * 20 // 1000
    boundary_ratios: list[float] = []
    for offset in range(
        window_samples, len(full_windows) * window_samples, window_samples
    ):
        local_rms = _rms(samples[offset - local_samples : offset + local_samples])
        boundary_ratios.append(
            abs(samples[offset] - samples[offset - 1]) / max(local_rms, 1e-12)
        )

    median_boundary_ratio = statistics.median(boundary_ratios)
    prefix_ratio_1ms = shared_prefix["1"]
    passed = median_boundary_ratio <= 0.35 and prefix_ratio_1ms <= 0.85
    return {
        "ok": passed,
        "sample_rate": sample_rate,
        "samples": len(samples),
        "duration_seconds": round(len(samples) / sample_rate, 3),
        "stream_frames": stream_frames,
        "window_samples": window_samples,
        "complete_windows": len(full_windows),
        "tail_samples": len(windows[-1]) if len(windows[-1]) != window_samples else 0,
        "start_sample": {
            "median": statistics.median(starts),
            "minimum": min(starts),
            "maximum": max(starts),
            "standard_deviation": statistics.pstdev(starts),
        },
        "shared_prefix_rms_ratio": shared_prefix,
        "boundary_jump_over_local_rms": {
            "median": median_boundary_ratio,
            "maximum": max(boundary_ratios),
        },
        "limits": {
            "median_boundary_jump_over_local_rms": 0.35,
            "shared_prefix_1ms_rms_ratio": 0.85,
        },
    }


def _request_pcm(args: argparse.Namespace) -> bytes:
    body: dict[str, Any] = {
        "text": args.text,
        "language": args.language,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "stream_frames": args.stream_frames,
    }
    if args.speaker_file:
        body["speaker_file"] = str(Path(args.speaker_file).expanduser().resolve())
    request = urllib.request.Request(
        args.url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        content_type = response.headers.get_content_type()
        if content_type != "audio/pcm":
            raise ValueError(f"expected audio/pcm, received {content_type}")
        if response.headers.get("X-Audio-Codec") != "pcm_s16le":
            raise ValueError("response is missing X-Audio-Codec: pcm_s16le")
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8892/synthesize/stream")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="en")
    parser.add_argument("--speaker-file")
    parser.add_argument("--stream-frames", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    try:
        report = analyze_pcm(_request_pcm(args), stream_frames=args.stream_frames)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
