#!/usr/bin/env python3
"""Generate anti-loop training data from the model's own repetition failures.

Strategy:
  1. Run the current model on reasoning prompts with greedy/low-temp decoding
  2. Detect traces where repetition loops occur
  3. For each looping trace, create a corrected version:
     - Truncate at the point where looping begins
     - Add a clean conclusion
  4. Output paired data: the prompt + corrected (non-looping) response

This creates training signal that teaches the model to:
  - Reach conclusions instead of looping
  - Stop reasoning when the answer is found
  - Not repeat the same reasoning step multiple times

Usage:
  python scripts/generate_antiloop.py --model qwen3.5-9b-r5-research:q4km
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
OLLAMA = "http://localhost:11434/api/chat"


# Prompts known to trigger repetition
TRIGGER_PROMPTS = [
    # Math with multiple approaches
    "Solve 3x^2 - 12x + 9 = 0 using two different methods. Show all work.",
    "Find the derivative of f(x) = x^3 * sin(x) using the product rule. Verify by expanding.",
    "Calculate the probability of drawing exactly 2 aces from a standard deck when drawing 5 cards.",
    "Prove that the sum of first n odd numbers equals n^2.",
    "Find all solutions to x^4 - 5x^2 + 4 = 0.",
    "A train leaves station A at 60 mph. Another leaves station B (300 miles away) at 80 mph toward A. When and where do they meet?",
    "Simplify: (x^2 - 4)/(x^2 - 4x + 4) and state domain restrictions.",
    "Find the area enclosed between y = x^2 and y = 2x + 3.",

    # Open-ended reasoning
    "What are the key differences between classical and quantum computing? Analyze from multiple angles.",
    "Discuss the trolley problem and its variations. What do different ethical frameworks say?",
    "Explain why the sky is blue, and why sunsets are red. Use physics to reason through this.",
    "Compare the economic systems of capitalism and socialism. Consider historical evidence.",

    # Ambiguous / paradoxical
    "Is this statement true: 'I always lie'? Think through all implications.",
    "Can an omnipotent being create a rock so heavy it cannot lift it? Analyze logically.",
    "If you go back in time and prevent your own birth, what happens? Reason carefully.",

    # Diversity / creative (repeat curse prone)
    "Name 15 unique animals and describe one interesting fact about each.",
    "Write 10 different metaphors for the concept of 'time'.",
    "List 12 creative business ideas for a small town. Each must be completely different.",
    "Describe 8 different ways to explain 'gravity' to different audiences.",

    # Long chains
    "A farmer has chickens and rabbits. There are 35 heads and 94 legs. How many of each? Show every step.",
    "Solve the system: 2x + 3y - z = 7, x - y + 2z = 3, 3x + y + z = 12. Show all steps.",
    "Find the first 10 prime numbers greater than 100 and verify each one.",
    "Convert the decimal 0.142857142857... to a fraction. Show all reasoning.",
]


def detect_loop(text: str) -> dict:
    """Detect repetition and find where it starts."""
    sentences = re.split(r'[.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if len(sentences) < 3:
        return {"has_loop": False, "onset_idx": None}

    # Find first repeated sentence
    seen = {}
    for i, s in enumerate(sentences):
        if s in seen:
            return {
                "has_loop": True,
                "onset_idx": seen[s],
                "onset_char": sum(len(sentences[j]) + 2 for j in range(seen[s])),
                "repeated_sentence": s[:100],
            }
        seen[s] = i

    # Check n-gram repetition
    words = text.lower().split()
    if len(words) > 20:
        ngrams = [" ".join(words[i:i+6]) for i in range(len(words) - 5)]
        counts = Counter(ngrams)
        for ng, c in counts.most_common(5):
            if c >= 3:
                first_pos = text.lower().find(ng)
                return {
                    "has_loop": True,
                    "onset_idx": None,
                    "onset_char": first_pos,
                    "repeated_ngram": ng[:100],
                }

    return {"has_loop": False, "onset_idx": None}


def truncate_at_loop(content: str, onset_char: int | None) -> str:
    """Truncate content at the point where looping starts, add clean ending."""
    if onset_char is None or onset_char < 100:
        # Can't find a good cut point, take first 60%
        onset_char = int(len(content) * 0.6)

    truncated = content[:onset_char].rstrip()

    # Find a good sentence boundary
    for end_marker in ['. ', '.\n', '!\n', '?\n']:
        last = truncated.rfind(end_marker)
        if last > onset_char * 0.5:  # don't go back too far
            truncated = truncated[:last + 1]
            break

    # If it's in a think block, close it properly
    if "<think>" in truncated and "</think>" not in truncated:
        truncated += "\n</think>"

    return truncated


def generate_antiloop_data(model: str, max_prompts: int = None) -> list[dict]:
    """Generate anti-loop training pairs from model's own failures."""
    prompts = TRIGGER_PROMPTS[:max_prompts] if max_prompts else TRIGGER_PROMPTS
    pairs = []

    print(f"[antiloop] Probing {model} with {len(prompts)} prompts...", flush=True)

    for i, prompt in enumerate(prompts):
        t0 = time.time()
        try:
            r = httpx.post(OLLAMA, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Low temp to trigger greedy loops
                    "num_predict": 4096,
                },
            }, timeout=300)
            r.raise_for_status()
            data = r.json()
            raw = data.get("message", {}).get("content", "")
            tokens = data.get("eval_count", 0)
        except Exception as e:
            print(f"  [{i+1}/{len(prompts)}] ERROR: {e}", flush=True)
            continue

        elapsed = round(time.time() - t0, 1)
        loop = detect_loop(raw)

        if loop["has_loop"]:
            # Create corrected version by truncating at loop onset
            corrected = truncate_at_loop(raw, loop.get("onset_char"))

            pairs.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": corrected},
                ],
                "category": "antiloop_corrected",
                "meta": {
                    "original_len": len(raw),
                    "corrected_len": len(corrected),
                    "loop_onset": loop.get("onset_char"),
                },
            })

            print(f"  [{i+1}/{len(prompts)}] LOOP → corrected "
                  f"({len(raw)}→{len(corrected)} chars) {elapsed}s", flush=True)
        else:
            # No loop — this is a clean trace, use as positive example
            # but only if not too long
            if len(raw) < 3000:
                pairs.append({
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": raw},
                    ],
                    "category": "antiloop_clean",
                })
            print(f"  [{i+1}/{len(prompts)}] CLEAN ({len(raw)} chars) {elapsed}s",
                  flush=True)

    print(f"[antiloop] Generated {len(pairs)} anti-loop samples "
          f"({sum(1 for p in pairs if p['category'] == 'antiloop_corrected')} corrected, "
          f"{sum(1 for p in pairs if p['category'] == 'antiloop_clean')} clean)",
          flush=True)
    return pairs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.5-9b-r5-research:q4km")
    ap.add_argument("--out", type=Path, default=SPLITS / "antiloop_pairs.jsonl")
    ap.add_argument("--max-prompts", type=int, default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("ANTI-LOOP DATA GENERATION")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(TRIGGER_PROMPTS)}")
    print("=" * 70)

    # Warmup
    print(f"\n[warmup] {args.model}...", flush=True)
    httpx.post(OLLAMA, json={
        "model": args.model,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"num_predict": 2},
    }, timeout=120)

    pairs = generate_antiloop_data(args.model, args.max_prompts)

    # Write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"\n[saved] {args.out} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
