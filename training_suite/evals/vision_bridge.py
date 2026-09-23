from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import httpx

VISION_EVALUATION_SCHEMA = "robit.qwen3a-vision-bridge-evaluation.v1"
VISUAL_OBSERVATION = re.compile(
    r"<visual_observation>(.*?)</visual_observation>",
    re.DOTALL,
)
VISION_SYSTEM_PROMPT = (
    "You are a visual perception encoder, not a conversational assistant. "
    "Analyze only the currently supplied image. Return exactly one "
    "visual_observation XML element and nothing else. Write plain text inside "
    "the element; never use Markdown, code fences, or nested XML. State its "
    "dominant color using exactly red or blue."
)
VISION_USER_PROMPT = "What is the dominant color in the current image?"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", binascii.crc32(body) & 0xFFFFFFFF)
    )


def solid_color_png(color: str, *, width: int = 224, height: int = 224) -> bytes:
    """Build a dependency-free RGB fixture for the temporal vision gate."""
    colors = {"red": (255, 0, 0), "blue": (0, 0, 255)}
    if color not in colors:
        raise ValueError(f"unsupported vision fixture color: {color}")
    if width <= 0 or height <= 0:
        raise ValueError("vision fixture dimensions must be positive")
    pixel = bytes(colors[color])
    scanline = b"\x00" + pixel * width
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanline * height))
        + _png_chunk(b"IEND", b"")
    )


def parse_visual_observation(content: str) -> dict[str, Any]:
    match = VISUAL_OBSERVATION.fullmatch(content.strip())
    if not match:
        return {"valid": False, "observation": "", "predicted_color": ""}
    observation = match.group(1).strip()
    if "<" in observation or ">" in observation:
        return {"valid": False, "observation": observation, "predicted_color": ""}
    colors = re.findall(r"\b(red|blue)\b", observation.casefold())
    return {
        "valid": True,
        "observation": observation,
        "predicted_color": colors[-1] if len(set(colors)) == 1 else "",
    }


def _request_payload(*, model: str, color: str, max_tokens: int) -> dict[str, Any]:
    encoded = base64.b64encode(solid_color_png(color)).decode("ascii")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": max_tokens,
        # A fresh image must always produce fresh embeddings. This is the
        # actual stale-media regression switch, not merely evaluator metadata.
        "cache_prompt": False,
    }


def evaluate_vision_bridge(
    *,
    endpoint: str,
    model: str,
    output_path: Path,
    timeout_seconds: float = 180.0,
    max_tokens: int = 128,
) -> dict[str, Any]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    expected = ("red", "blue", "red")
    results = []
    started = time.time()
    with httpx.Client(timeout=timeout_seconds) as client:
        for turn, color in enumerate(expected, 1):
            response = client.post(
                endpoint,
                json=_request_payload(model=model, color=color, max_tokens=max_tokens),
            )
            response.raise_for_status()
            payload = response.json()
            content = str(payload["choices"][0]["message"].get("content") or "")
            parsed = parse_visual_observation(content)
            results.append(
                {
                    "turn": turn,
                    "expected_color": color,
                    "predicted_color": parsed["predicted_color"],
                    "observation": parsed["observation"],
                    "tagged_output_valid": parsed["valid"],
                    "correct": parsed["predicted_color"] == color,
                    "raw_content": content,
                }
            )

    correct_sequence = all(row["correct"] for row in results)
    exact_tagged_output = all(row["tagged_output_valid"] for row in results)
    stale_media_failures = sum(
        row["predicted_color"] == results[index - 1]["expected_color"]
        and row["expected_color"] != results[index - 1]["expected_color"]
        for index, row in enumerate(results)
        if index > 0
    )
    gates = {
        "red_blue_red_sequence": correct_sequence,
        "exact_tagged_output": exact_tagged_output,
        "stale_media_failures": stale_media_failures == 0,
        "cache_prompt_disabled": True,
    }
    report = {
        "schema": VISION_EVALUATION_SCHEMA,
        "endpoint": endpoint,
        "model": model,
        "max_tokens": max_tokens,
        "sequence": list(expected),
        "metrics": {
            "samples": len(results),
            "correct": sum(row["correct"] for row in results),
            "tagged_output_rate": (
                sum(row["tagged_output_valid"] for row in results) / len(results)
            ),
            "stale_media_failures": stale_media_failures,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the native-vision red-blue-red audio-bridge release gate"
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    report = evaluate_vision_bridge(
        endpoint=args.endpoint,
        model=args.model,
        output_path=Path(args.out),
        timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
