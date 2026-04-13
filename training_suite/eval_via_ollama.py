#!/usr/bin/env python3
"""Run the full eval battery on an Ollama model.

Runs against the Ollama HTTP API (native /api/chat), deterministic decoding,
fixed seeds. Produces per-row JSONL + aggregate summary.

Evals
-----
  gsm8k   — 100 locked samples from openai/gsm8k test split (seed 1337)
  mmlu    — 100 random samples from cais/mmlu all-subjects test (seed 42)
  vision  — 3 rendered text probes (HELLO / 42 / BANANA)

Usage
-----
  python scripts/eval_via_ollama.py <model_name> [--skip gsm8k,mmlu,vision]
                                                 [--out logs/eval_<name>.json]
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
GSM8K_FILE = ROOT / "data" / "gsm8k_sample.jsonl"
MMLU_FILE = ROOT / "data" / "mmlu_sample.jsonl"
IMAGE_DIR = ROOT / "logs" / "image_probe_images"
OLLAMA = "http://localhost:11434/api/chat"

# Deterministic sampling options. Note: `temperature=0` alone gives greedy
# decoding on Ollama. Adding top_k=1 is redundant and sometimes interacts
# weirdly, so we keep just temperature + seed.
DET = {"temperature": 0, "seed": 42}


# ---------------------------------------------------------------------------
# Dataset preparation (run once, then reused)
# ---------------------------------------------------------------------------

def ensure_mmlu_sample(n: int = 100, seed: int = 42) -> None:
    """Sample N MMLU questions across all subjects and lock to disk."""
    if MMLU_FILE.exists():
        return
    print(f"[prep] building MMLU sample (n={n}, seed={seed}) ...", flush=True)
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), n)
    MMLU_FILE.parent.mkdir(parents=True, exist_ok=True)
    with MMLU_FILE.open("w") as f:
        for i in sorted(idxs):
            row = ds[int(i)]
            f.write(json.dumps({
                "idx": int(i),
                "question": row["question"],
                "choices": row["choices"],   # list of 4 strings
                "answer": int(row["answer"]),  # 0-3
                "subject": row["subject"],
            }) + "\n")
    print(f"[prep] wrote {MMLU_FILE} ({n} rows)", flush=True)


# ---------------------------------------------------------------------------
# Ollama wrapper
# ---------------------------------------------------------------------------

def ollama_chat(model: str, messages: list, images: list | None = None,
                max_tokens: int = 512, timeout: float = 600) -> dict:
    msgs = [dict(m) for m in messages]
    if images:
        msgs[-1]["images"] = images
    payload = {
        "model": model,
        "messages": msgs,
        "stream": False,
        "options": {**DET, "num_predict": max_tokens},
    }
    r = httpx.post(OLLAMA, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"[Aa]nswer\s*:?\s*(-?\$?\d[\d,]*\.?\d*)")
_NUM_RE = re.compile(r"-?\$?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_number(text: str) -> float | None:
    m = _ANSWER_RE.findall(text)
    raw = m[-1] if m else None
    if raw is None:
        nums = _NUM_RE.findall(text)
        if not nums:
            return None
        raw = nums[-1]
    raw = raw.replace("$", "").replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def eval_gsm8k(model: str) -> dict:
    rows = [json.loads(l) for l in GSM8K_FILE.read_text().splitlines() if l.strip()]
    print(f"[gsm8k] {len(rows)} samples on {model}", flush=True)
    results = []
    correct = 0
    t_start = time.time()
    for i, row in enumerate(rows, 1):
        q = row["question"]
        gold_raw = row["answer"].split("####")[-1].strip().replace(",", "")
        try:
            gold = float(gold_raw)
        except ValueError:
            continue
        messages = [{
            "role": "user",
            "content": (
                f"Solve this math problem. Show your reasoning, then give the "
                f"final answer on its own line as: Answer: <number>\n\n{q}"
            ),
        }]
        t0 = time.time()
        try:
            # 4096 is enough for base qwen3.5:9b's ~2500-char thinking + ~500-char answer
            d = ollama_chat(model, messages, max_tokens=4096, timeout=900)
        except Exception as e:
            results.append({"idx": row["idx"], "error": f"{type(e).__name__}: {e}"})
            continue
        content = d["message"].get("content") or ""
        # Fallback: if the model output thinking into content (happens when
        # thinking capability not auto-extracted), strip everything up to the
        # closing </think> tag.
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>", 1)[-1]
        pred = _extract_number(content)
        is_correct = pred is not None and abs(pred - gold) < 1e-4
        if is_correct:
            correct += 1
        elapsed = round(time.time() - t0, 1)
        if i % 10 == 0 or i == len(rows):
            acc = correct / i
            print(f"  [{i}/{len(rows)}] {correct}/{i} = {acc:.3f}  last_latency={elapsed}s", flush=True)
        results.append({
            "idx": row["idx"], "gold": gold, "pred": pred, "correct": is_correct,
            "elapsed_s": elapsed, "answer_snippet": content[:300],
        })
    total_elapsed = round(time.time() - t_start, 1)
    return {
        "eval": "gsm8k",
        "model": model,
        "n": len(results),
        "correct": correct,
        "accuracy": correct / max(len(results), 1),
        "total_elapsed_s": total_elapsed,
        "rows": results,
    }


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

def eval_mmlu(model: str) -> dict:
    ensure_mmlu_sample()
    rows = [json.loads(l) for l in MMLU_FILE.read_text().splitlines() if l.strip()]
    print(f"[mmlu] {len(rows)} samples on {model}", flush=True)
    results = []
    correct = 0
    t_start = time.time()
    letters = ["A", "B", "C", "D"]
    for i, row in enumerate(rows, 1):
        choices_text = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(row["choices"]))
        prompt = (
            f"The following is a multiple choice question about "
            f"{row['subject'].replace('_', ' ')}. "
            f"Answer with only the letter of the correct choice.\n\n"
            f"Question: {row['question']}\n\n{choices_text}\n\n"
            f"Answer with ONLY one letter (A, B, C, or D):"
        )
        messages = [{"role": "user", "content": prompt}]
        t0 = time.time()
        try:
            d = ollama_chat(model, messages, max_tokens=3072, timeout=900)
        except Exception as e:
            results.append({"idx": row["idx"], "error": f"{type(e).__name__}: {e}"})
            continue
        content = (d["message"].get("content") or "")
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>", 1)[-1]
        # Find first standalone letter A/B/C/D
        m = re.search(r"\b([ABCD])\b", content.upper())
        pred_letter = m.group(1) if m else None
        pred_idx = letters.index(pred_letter) if pred_letter else -1
        is_correct = pred_idx == row["answer"]
        if is_correct:
            correct += 1
        elapsed = round(time.time() - t0, 1)
        if i % 10 == 0 or i == len(rows):
            acc = correct / i
            print(f"  [{i}/{len(rows)}] {correct}/{i} = {acc:.3f}  subject={row['subject']}  last_latency={elapsed}s", flush=True)
        results.append({
            "idx": row["idx"], "subject": row["subject"],
            "gold_idx": row["answer"], "gold_letter": letters[row["answer"]],
            "pred_letter": pred_letter, "correct": is_correct,
            "elapsed_s": elapsed, "answer_snippet": content[:200],
        })
    # Per-subject breakdown
    from collections import defaultdict
    by_subj = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if "error" in r:
            continue
        by_subj[r["subject"]]["total"] += 1
        if r["correct"]:
            by_subj[r["subject"]]["correct"] += 1
    return {
        "eval": "mmlu",
        "model": model,
        "n": len(results),
        "correct": correct,
        "accuracy": correct / max(len(results), 1),
        "total_elapsed_s": round(time.time() - t_start, 1),
        "by_subject": {s: v for s, v in by_subj.items()},
        "rows": results,
    }


# ---------------------------------------------------------------------------
# Vision probe
# ---------------------------------------------------------------------------

def eval_vision(model: str) -> dict:
    cases = [
        ("HELLO", IMAGE_DIR / "C1.png"),
        ("42", IMAGE_DIR / "C2.png"),
        ("BANANA", IMAGE_DIR / "C3.png"),
    ]
    print(f"[vision] {len(cases)} samples on {model}", flush=True)
    results = []
    correct = 0
    for label, path in cases:
        if not path.exists():
            results.append({"label": label, "error": f"image not found: {path}"})
            continue
        b64 = base64.b64encode(path.read_bytes()).decode()
        messages = [{
            "role": "user",
            "content": "What single word or number is written in this image? Reply with only that word.",
        }]
        t0 = time.time()
        try:
            d = ollama_chat(model, messages, images=[b64], max_tokens=2048, timeout=900)
        except Exception as e:
            results.append({"label": label, "error": f"{type(e).__name__}: {e}"})
            continue
        content = (d["message"].get("content") or "")
        raw = content
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>", 1)[-1].strip()
        is_correct = label.lower() in content.lower()
        if is_correct:
            correct += 1
        print(f"  [{'PASS' if is_correct else 'WRONG'}] {label:>7} → {content[:100]!r}  ({round(time.time()-t0,1)}s)", flush=True)
        results.append({
            "label": label, "correct": is_correct,
            "content_post_think": content[:200],
            "raw_content": raw[:400],
            "elapsed_s": round(time.time() - t0, 1),
        })
    return {
        "eval": "vision",
        "model": model,
        "n": len(results),
        "correct": correct,
        "accuracy": correct / max(len(results), 1),
        "rows": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--skip", default="", help="comma-separated list of evals to skip")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    results: dict = {"model": args.model, "skipped": sorted(skip), "evals": {}}
    t_start = time.time()

    # Warmup
    print(f"[warmup] {args.model}", flush=True)
    t0 = time.time()
    try:
        ollama_chat(args.model, [{"role": "user", "content": "hi"}], max_tokens=2, timeout=900)
        print(f"  warmup ok in {round(time.time()-t0,1)}s", flush=True)
    except Exception as e:
        print(f"  warmup failed: {e}", flush=True)
        sys.exit(1)

    if "gsm8k" not in skip:
        results["evals"]["gsm8k"] = eval_gsm8k(args.model)
    if "mmlu" not in skip:
        results["evals"]["mmlu"] = eval_mmlu(args.model)
    if "vision" not in skip:
        results["evals"]["vision"] = eval_vision(args.model)

    results["total_elapsed_s"] = round(time.time() - t_start, 1)

    out_path = args.out or f"logs/eval_{args.model.replace('/', '_').replace(':', '_')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[done] wrote {out_path} in {results['total_elapsed_s']}s", flush=True)
    print("\n== summary ==")
    for name, r in results["evals"].items():
        if "accuracy" in r:
            print(f"  {name}: {r['correct']}/{r['n']} = {r['accuracy']:.3f}")
        else:
            print(f"  {name}: {r}")


if __name__ == "__main__":
    main()
