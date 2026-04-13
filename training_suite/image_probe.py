#!/usr/bin/env python3
"""Verify image-handling capability on base vs distilled model via Ollama.

Our distilled model was converted from `Qwen3_5ForCausalLM` — the TEXT subnet
of Qwen3.5-9B. The vision tower was never loaded during training and is not
in the merged safetensors, so the GGUF contains no vision weights. The
expected outcome of this probe is:

  base  (qwen3.5:9b)                         : describes the image correctly
  tuned (qwen3.5-9b-qwen3.6-distilled:q4km)  : fails / ignores / refuses

We create three deterministic test images with known content and send each
to both models via /api/chat with the `images` field (base64-encoded PNG).
Results are scored against the expected subject string.

Usage:
  python scripts/image_probe.py
"""
from __future__ import annotations

import base64
import io
import json
import time
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
IMG_DIR = ROOT / "logs" / "image_probe_images"
OUT_JSON = ROOT / "logs" / "image_probe.json"
OLLAMA = "http://localhost:11434/api/chat"

MODELS = [
    "qwen3.5:9b",
    "qwen3.5-9b-qwen3.6-distilled:q4km",
]


def _make_image(text: str, bg: tuple, fg: tuple, path: Path) -> None:
    """Render a clear 512x512 image with a single word/phrase in large text."""
    im = Image.new("RGB", (512, 512), bg)
    d = ImageDraw.Draw(im)
    # Pick the first DejaVuSans-Bold we can find; fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
    except Exception:
        font = ImageFont.load_default()
    # Centered
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    d.text(((512 - w) / 2, (512 - h) / 2 - 20), text, font=font, fill=fg)
    im.save(path, "PNG")


def _b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def probe(model: str, image_b64: str, prompt: str, timeout: float = 600) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
    }
    t0 = time.time()
    try:
        r = httpx.post(OLLAMA, json=payload, timeout=timeout)
    except Exception as e:
        return {
            "status_code": None,
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "elapsed_s": round(time.time() - t0, 1),
        }
    out: dict = {"status_code": r.status_code, "elapsed_s": round(time.time() - t0, 1)}
    if r.status_code != 200:
        out["error"] = r.text[:400]
        return out
    d = r.json()
    msg = d.get("message", {})
    out["content"] = msg.get("content", "")
    out["thinking"] = msg.get("thinking", "")
    return out


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # Three images with known ground truth
    cases = [
        {
            "id": "C1",
            "text": "HELLO",
            "bg": (0, 0, 0),       # black background
            "fg": (255, 255, 0),   # yellow text
            "expect_any": ["hello"],
            "prompt": "What single word is written in large letters in this image? Answer with just the word.",
        },
        {
            "id": "C2",
            "text": "42",
            "bg": (200, 30, 30),   # red background
            "fg": (255, 255, 255), # white text
            "expect_any": ["42", "forty-two", "forty two"],
            "prompt": "What number is written on this red image? Just answer with the number.",
        },
        {
            "id": "C3",
            "text": "BANANA",
            "bg": (255, 240, 0),   # yellow bg
            "fg": (20, 20, 20),    # near black
            "expect_any": ["banana"],
            "prompt": "What single word is written on this yellow image? Reply with just the word, lowercased.",
        },
    ]

    # Render images
    for c in cases:
        path = IMG_DIR / f"{c['id']}.png"
        _make_image(c["text"], c["bg"], c["fg"], path)
        c["path"] = path
        c["b64"] = _b64_png(path)
        print(f"[render] {c['id']} → {path.name}  ({path.stat().st_size} bytes)", flush=True)

    results: dict = {}
    for model in MODELS:
        print(f"\n=== {model} ===", flush=True)
        rows = []
        for c in cases:
            r = probe(model, c["b64"], c["prompt"])
            content = (r.get("content") or "").lower()
            needles = [n.lower() for n in c["expect_any"]]
            correct = any(n in content for n in needles) if r.get("status_code") == 200 else False
            r["correct"] = correct
            r["expected_any"] = c["expect_any"]
            rows.append({"id": c["id"], "prompt_text": c["text"], **r})
            status = "PASS" if correct else (
                "HTTP-" + str(r.get("status_code")) if r.get("status_code") != 200
                else "WRONG"
            )
            print(
                f"  [{status:<8}] {c['id']} ({c['text']:<7}) "
                f"elapsed={r.get('elapsed_s')}s  content={r.get('content','')[:120]!r}",
                flush=True,
            )
        results[model] = rows
        OUT_JSON.write_text(json.dumps(results, indent=2, default=str))

    # Summary
    print("\n================ IMAGE HANDLING MATRIX ================\n")
    print(f"{'MODEL':<48}  {'PASS':>5}  {'FAIL':>5}  {'ERR':>5}")
    print("-" * 72)
    for m, rows in results.items():
        passed = sum(1 for r in rows if r.get("correct"))
        failed = sum(1 for r in rows if r.get("status_code") == 200 and not r.get("correct"))
        errored = sum(1 for r in rows if r.get("status_code") not in (200, None) or r.get("error"))
        print(f"{m:<48}  {passed:>5}  {failed:>5}  {errored:>5}")
    print()
    # Raw content dumps for documentation
    print("=== raw content dumps ===")
    for m, rows in results.items():
        print(f"\n[{m}]")
        for r in rows:
            print(f"  {r['id']} ({r['prompt_text']}): {r.get('content','')[:300]!r}")


if __name__ == "__main__":
    main()
