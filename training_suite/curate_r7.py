#!/usr/bin/env python3
"""Curate R7 training data — ADDITIVE approach, zero regressions from R5.

CRITICAL PRINCIPLE: R5 data IS the base. We only ADD on top. Never replace.

R5 scored 84.2% on diverse eval. R6 regressed to 81.6% because it replaced
the R5 backbone. R7 takes the exact R5 training data and layers:
  - PrimeIntellect SYNTHETIC-1-SFT-Data (repetition-filtered reasoning)
  - Anti-loop corrected pairs from model's own failures

Usage:
  python scripts/curate_r7.py
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
SEED = 42


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def has_repetition(text: str, threshold: float = 0.08) -> bool:
    """Quick repetition check."""
    if not text or len(text) < 100:
        return False
    sentences = re.split(r'[.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if len(sentences) < 3:
        return False
    counts = Counter(sentences)
    repeat_ratio = sum(c - 1 for c in counts.values() if c >= 2) / len(sentences)
    return repeat_ratio > threshold


# ---------------------------------------------------------------------------
# Layer 0: Full R5 training data (UNTOUCHED)
# ---------------------------------------------------------------------------

def load_r5_full() -> list[dict]:
    """Load the ENTIRE R5 training dataset. This is our proven backbone."""
    r5_path = SPLITS / "r5_research_train.jsonl"
    if not r5_path.exists():
        raise FileNotFoundError(f"R5 data not found at {r5_path}")

    results = []
    with r5_path.open() as f:
        for line in f:
            results.append(json.loads(line))

    print(f"[r5-base] Loaded {len(results)} R5 training samples (FULL, UNTOUCHED)", flush=True)

    # Show category breakdown
    cats = Counter(r.get("category", "unknown") for r in results)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Layer 1: PrimeIntellect SYNTHETIC-1-SFT-Data (repetition-filtered)
# ---------------------------------------------------------------------------

def load_primeintellect(n: int) -> list[dict]:
    """Load and sample from PrimeIntellect SYNTHETIC-1-SFT-Data.

    894k verified DeepSeek-R1 reasoning traces. Already has aggressive
    repetition filtering (>5% hitting 12k token limit are wrong).
    We additionally filter for repetition ourselves.
    """
    from datasets import load_dataset

    print(f"[pi] Loading PrimeIntellect/SYNTHETIC-1-SFT-Data...", flush=True)
    ds = load_dataset("PrimeIntellect/SYNTHETIC-1-SFT-Data", split="train",
                      streaming=True)

    rng = random.Random(SEED + 7)  # different seed from R5
    results = []
    scanned = 0
    skipped_rep = 0
    skipped_long = 0
    skipped_format = 0
    task_types = Counter()

    for row in ds:
        scanned += 1
        if len(results) >= n:
            break
        if scanned > n * 20:
            break

        # Random subsample
        if rng.random() > 0.05:
            continue

        msgs = row.get("messages", [])
        if not msgs or len(msgs) < 2:
            skipped_format += 1
            continue

        # Ensure proper role/content format
        valid = True
        for m in msgs:
            if "role" not in m or "content" not in m:
                valid = False
                break
        if not valid:
            skipped_format += 1
            continue

        asst = next((m for m in msgs if m["role"] == "assistant"), None)
        if not asst:
            skipped_format += 1
            continue

        content = asst["content"]

        # Filter: skip responses with repetition
        if has_repetition(content):
            skipped_rep += 1
            continue

        # Filter: skip extremely long (>4000 chars)
        if len(content) > 4000:
            skipped_long += 1
            continue

        task_type = row.get("task_type", "unknown")
        task_types[task_type] += 1

        results.append({
            "messages": msgs,
            "category": f"reasoning_pi_{task_type}",
        })

    print(f"[pi] Sampled {len(results)} from {scanned} scanned", flush=True)
    print(f"  Skipped: {skipped_rep} repetitive, {skipped_long} too long, {skipped_format} bad format", flush=True)
    print(f"  Task types: {dict(task_types)}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Layer 2: Anti-loop corrected pairs
# ---------------------------------------------------------------------------

def load_antiloop() -> list[dict]:
    """Load anti-loop data from R6."""
    path = SPLITS / "antiloop_pairs.jsonl"
    if not path.exists():
        print(f"[antiloop] No anti-loop data at {path}", flush=True)
        return []

    results = []
    with path.open() as f:
        for line in f:
            results.append(json.loads(line))

    print(f"[antiloop] Loaded {len(results)} anti-loop samples", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(SEED)

    print("=" * 70)
    print("R7 DATA CURATION — Additive (R5 base + layers)")
    print("PRINCIPLE: Never replace R5. Only add on top.")
    print("=" * 70)

    all_data = []

    # Layer 0: FULL R5 (the proven backbone — DO NOT MODIFY)
    r5 = load_r5_full()
    all_data.extend(r5)

    # Layer 1: PrimeIntellect SYNTHETIC-1 (repetition-filtered reasoning)
    pi = load_primeintellect(1000)
    all_data.extend(pi)

    # Layer 2: Anti-loop corrections
    antiloop = load_antiloop()
    all_data.extend(antiloop)

    # Shuffle
    rng.shuffle(all_data)

    n_total = len(all_data)
    n_train = int(n_total * 0.90)
    n_val = int(n_total * 0.05)

    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]

    prefix = "r7_additive"
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        path = SPLITS / f"{prefix}_{name}.jsonl"
        with path.open("w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
        print(f"[write] {path.name}: {len(data)} samples")

    manifest = {
        "round": "r7-additive",
        "seed": SEED,
        "total": n_total,
        "train": n_train, "val": n_val, "test": len(test_data),
        "strategy": "R5 full base + PrimeIntellect layer + anti-loop layer",
        "category_counts": {},
        "sha256_train": _sha(SPLITS / f"{prefix}_train.jsonl"),
        "sha256_val": _sha(SPLITS / f"{prefix}_val.jsonl"),
        "sha256_test": _sha(SPLITS / f"{prefix}_test.jsonl"),
    }
    for row in all_data:
        cat = row.get("category", "unknown")
        manifest["category_counts"][cat] = manifest["category_counts"].get(cat, 0) + 1

    (SPLITS / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{'='*70}")
    print(f"R7 DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {n_total}  (train={n_train}, val={n_val}, test={len(test_data)})")
    print(f"\nCategory distribution:")
    for cat, count in sorted(manifest["category_counts"].items()):
        pct = count / n_total * 100
        print(f"  {cat:<35} {count:>5} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
