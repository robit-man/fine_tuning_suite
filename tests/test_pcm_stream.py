from __future__ import annotations

import math
import struct
from pathlib import Path

from examples.omni_adapter.verify_pcm_stream import analyze_pcm


def _pcm(samples: list[float]) -> bytes:
    return b"".join(
        struct.pack("<h", max(-32768, min(32767, round(sample * 32767))))
        for sample in samples
    )


def test_pcm_continuity_probe_rejects_a_repeated_window_prefix() -> None:
    sample_rate = 24_000
    window_samples = 3_840
    samples = [
        0.15 * math.sin(2 * math.pi * 431 * index / sample_rate)
        for index in range(20 * window_samples)
    ]
    continuous = analyze_pcm(_pcm(samples))
    assert continuous["ok"] is True

    repeated_prefix = [
        -0.04 + 0.01 * math.sin(2 * math.pi * 700 * index / sample_rate)
        for index in range(240)
    ]
    corrupted = samples.copy()
    for window in range(1, 20):
        offset = window * window_samples
        corrupted[offset : offset + len(repeated_prefix)] = repeated_prefix

    report = analyze_pcm(_pcm(corrupted))
    assert report["ok"] is False
    assert report["shared_prefix_rms_ratio"]["1"] > 0.85
    assert report["boundary_jump_over_local_rms"]["median"] > 0.35


def test_stream_state_patch_uses_only_real_codec_frames() -> None:
    patch = Path(
        "training_suite/patches/llama.cpp-qwen3tts-stream-state.patch",
    ).read_text(encoding="utf-8")

    assert (
        "+                    std::vector<int32_t> codes(n_frames * n_codes);" in patch
    )
    assert "+                            codes[g * n_frames + f]" in patch
    assert (
        "-                    std::vector<int32_t> codes(n_frames_w * n_codes, 0);"
        in patch
    )
    assert "-                            codes[g * n_frames_w + f]" in patch
