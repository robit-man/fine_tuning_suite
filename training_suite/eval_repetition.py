#!/usr/bin/env python3
"""Repetition stress test for reasoning models.

Detects and quantifies thinking-token repetition loops:
  - N-gram repetition (sentence-level and paragraph-level)
  - Cyclic pattern detection (same reasoning step repeated)
  - Token-level repetition ratio
  - Loop onset detection (where in the trace does looping start)

Prompts designed to trigger repetition-prone reasoning:
  - Multi-step math (long chains)
  - Open-ended analysis (no clear stopping point)
  - Ambiguous problems (model may loop searching for certainty)
  - Recursive/self-referential queries

Usage:
  python scripts/eval_repetition.py <model> [--out <path>]
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import httpx

OLLAMA = "http://localhost:11434/api/chat"


# ---------------------------------------------------------------------------
# Repetition detection
# ---------------------------------------------------------------------------

def detect_repetition(text: str) -> dict:
    """Analyze text for repetition patterns. Returns metrics."""
    if not text or len(text) < 50:
        return {"has_loop": False, "loop_score": 0.0, "sentence_repeat_ratio": 0.0,
                "ngram_repeat_ratio": 0.0, "cycle_score": 0.0,
                "chunk_repeat_ratio": 0.0, "loop_onset": None,
                "repeated_sentences": 0, "total_sentences": 0,
                "total_chars": len(text) if text else 0}

    sentences = re.split(r'[.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    # 1. Exact sentence repetition
    sentence_counts = Counter(sentences)
    repeated_sentences = {s: c for s, c in sentence_counts.items() if c >= 2}
    sentence_repeat_ratio = (
        sum(c - 1 for c in repeated_sentences.values()) / max(len(sentences), 1)
    )

    # 2. N-gram repetition (4-gram on words)
    words = text.lower().split()
    if len(words) >= 8:
        ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
        ngram_counts = Counter(ngrams)
        repeated_ngrams = sum(1 for c in ngram_counts.values() if c >= 3)
        ngram_repeat_ratio = repeated_ngrams / max(len(ngrams), 1)
    else:
        ngram_repeat_ratio = 0.0

    # 3. Paragraph-level repetition (chunks of ~100 chars)
    chunk_size = 100
    chunks = [text[i:i+chunk_size] for i in range(0, len(text) - chunk_size + 1, chunk_size // 2)]
    chunk_counts = Counter(chunks)
    repeated_chunks = sum(1 for c in chunk_counts.values() if c >= 2)
    chunk_repeat_ratio = repeated_chunks / max(len(chunks), 1)

    # 4. Sliding window similarity (detect cyclic patterns)
    window = 200
    cycle_score = 0.0
    if len(text) > window * 3:
        for i in range(window, len(text) - window, window):
            seg = text[i:i+window]
            # Check if this segment appeared before
            prev = text[:i]
            if seg in prev:
                cycle_score += 1.0
        cycle_score = cycle_score / max((len(text) - window * 2) // window, 1)

    # 5. Where does looping start? (first repeated sentence position)
    loop_onset = None
    seen = {}
    for i, s in enumerate(sentences):
        if s in seen and len(s) > 30:
            loop_onset = seen[s] / max(len(sentences), 1)
            break
        seen[s] = i

    # Composite loop score
    loop_score = (
        sentence_repeat_ratio * 0.4 +
        ngram_repeat_ratio * 100 * 0.3 +  # scale up ngram ratio
        cycle_score * 0.2 +
        chunk_repeat_ratio * 0.1
    )

    has_loop = loop_score > 0.05 or sentence_repeat_ratio > 0.1

    return {
        "has_loop": has_loop,
        "loop_score": round(loop_score, 4),
        "sentence_repeat_ratio": round(sentence_repeat_ratio, 4),
        "ngram_repeat_ratio": round(ngram_repeat_ratio, 6),
        "cycle_score": round(cycle_score, 4),
        "chunk_repeat_ratio": round(chunk_repeat_ratio, 4),
        "loop_onset": round(loop_onset, 3) if loop_onset else None,
        "repeated_sentences": len(repeated_sentences),
        "total_sentences": len(sentences),
        "total_chars": len(text),
    }


# ---------------------------------------------------------------------------
# Stress test prompts
# ---------------------------------------------------------------------------

STRESS_PROMPTS = [
    # Multi-step math (long reasoning chains)
    {
        "name": "multi_step_algebra",
        "category": "math",
        "messages": [{"role": "user", "content":
            "Solve step by step: If 3x + 7 = 2(x - 4) + 15, find x. "
            "Then verify your answer by substituting back."}],
    },
    {
        "name": "combinatorics",
        "category": "math",
        "messages": [{"role": "user", "content":
            "How many ways can you arrange the letters in the word MISSISSIPPI? "
            "Show all your reasoning steps."}],
    },
    {
        "name": "probability_chains",
        "category": "math",
        "messages": [{"role": "user", "content":
            "A bag has 5 red, 3 blue, and 2 green balls. You draw 3 balls "
            "without replacement. What is the probability that all 3 are different "
            "colors? Show detailed step-by-step reasoning."}],
    },
    # Open-ended analysis (no clear stopping point)
    {
        "name": "open_analysis",
        "category": "open_ended",
        "messages": [{"role": "user", "content":
            "Analyze the pros and cons of artificial intelligence in education. "
            "Consider multiple perspectives."}],
    },
    {
        "name": "philosophical",
        "category": "open_ended",
        "messages": [{"role": "user", "content":
            "Is consciousness a fundamental property of the universe or an "
            "emergent phenomenon? Reason through this carefully."}],
    },
    {
        "name": "compare_contrast",
        "category": "open_ended",
        "messages": [{"role": "user", "content":
            "Compare and contrast democracy and authoritarianism as systems of "
            "governance. Consider historical examples and theoretical frameworks."}],
    },
    # Ambiguous/tricky (may loop searching for certainty)
    {
        "name": "ambiguous_logic",
        "category": "ambiguous",
        "messages": [{"role": "user", "content":
            "This statement is false. Is it true or false? Think through "
            "this carefully, considering all logical implications."}],
    },
    {
        "name": "impossible_task",
        "category": "ambiguous",
        "messages": [{"role": "user", "content":
            "Find a number that is both greater than 10 and less than 5. "
            "Explain your reasoning process."}],
    },
    {
        "name": "recursive_question",
        "category": "ambiguous",
        "messages": [{"role": "user", "content":
            "What would happen if every rule had an exception, including "
            "this rule? Think through all the implications step by step."}],
    },
    # Complex reasoning that might trigger backtracking loops
    {
        "name": "optimization",
        "category": "complex",
        "messages": [{"role": "user", "content":
            "You have a 4x4 grid. Place the numbers 1-16 such that every "
            "row, column, and diagonal sums to 34. Show your reasoning."}],
    },
    {
        "name": "proof_attempt",
        "category": "complex",
        "messages": [{"role": "user", "content":
            "Prove that the square root of 2 is irrational. Show every step "
            "of the proof with detailed justification."}],
    },
    {
        "name": "constraint_satisfaction",
        "category": "complex",
        "messages": [{"role": "user", "content":
            "Schedule 5 meetings (A, B, C, D, E) across 3 time slots with "
            "constraints: A and B cannot be in the same slot, C must be before "
            "D, E must be alone in its slot. Find all valid schedules."}],
    },
    # Diversity challenge - specifically designed to trigger repeat curse
    {
        "name": "list_generation",
        "category": "diversity",
        "messages": [{"role": "user", "content":
            "List 20 unique and creative uses for a paperclip. Each use "
            "must be genuinely different from all others."}],
    },
    {
        "name": "story_continuation",
        "category": "diversity",
        "messages": [{"role": "user", "content":
            "Write a story that has exactly 5 plot twists. Each twist must "
            "be surprising and different from the others."}],
    },
    {
        "name": "unique_descriptions",
        "category": "diversity",
        "messages": [{"role": "user", "content":
            "Describe the color blue in 10 completely different ways. Each "
            "description must use a different metaphor or approach."}],
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_stress_test(model: str, temperature: float = 0.6,
                    max_tokens: int = 4096) -> list[dict]:
    """Run all stress prompts and analyze repetition."""
    results = []
    total = len(STRESS_PROMPTS)

    for i, prompt in enumerate(STRESS_PROMPTS, 1):
        t0 = time.time()
        try:
            r = httpx.post(OLLAMA, json={
                "model": model,
                "messages": prompt["messages"],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }, timeout=300)
            r.raise_for_status()
            data = r.json()
            raw_content = data.get("message", {}).get("content", "")
            tokens = data.get("eval_count", 0)
        except Exception as e:
            results.append({
                "name": prompt["name"],
                "category": prompt["category"],
                "error": str(e),
                "has_loop": False,
            })
            print(f"  [{i:2d}/{total}] [ERR] {prompt['name']}: {e}", flush=True)
            continue

        elapsed = round(time.time() - t0, 1)

        # Analyze the FULL response (including thinking) for repetition
        rep = detect_repetition(raw_content)

        # Also check just the think block if present
        think_rep = {"has_loop": False}
        if "<think>" in raw_content and "</think>" in raw_content:
            think_block = raw_content.split("<think>", 1)[1].split("</think>", 1)[0]
            think_rep = detect_repetition(think_block)

        status = "LOOP" if rep["has_loop"] or think_rep["has_loop"] else "OK"
        print(
            f"  [{i:2d}/{total}] [{status:4s}] {prompt['name']:<25s} "
            f"score={max(rep['loop_score'], think_rep.get('loop_score', 0)):.3f} "
            f"sent_rep={rep['sentence_repeat_ratio']:.2f} "
            f"think_rep={think_rep.get('sentence_repeat_ratio', 0):.2f} "
            f"{tokens}tok {elapsed}s",
            flush=True,
        )

        results.append({
            "name": prompt["name"],
            "category": prompt["category"],
            "has_loop": rep["has_loop"] or think_rep.get("has_loop", False),
            "full_response_rep": rep,
            "think_block_rep": think_rep if think_rep["has_loop"] else None,
            "response_preview": raw_content[:500],
            "tokens": tokens,
            "elapsed_s": elapsed,
            "total_chars": len(raw_content),
        })

    return results


def generate_report(model: str, results: list[dict]) -> dict:
    """Generate repetition analysis report."""
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    valid = [r for r in results if "error" not in r]
    loops = sum(1 for r in valid if r["has_loop"])

    by_category = {}
    for r in valid:
        cat = r["category"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "loops": 0, "tests": []}
        by_category[cat]["total"] += 1
        if r["has_loop"]:
            by_category[cat]["loops"] += 1
        by_category[cat]["tests"].append(r["name"])

    return {
        "model": model,
        "total_tests": total,
        "errors": errors,
        "valid": len(valid),
        "loops_detected": loops,
        "loop_rate": round(loops / max(len(valid), 1), 4),
        "by_category": {
            cat: {
                "total": v["total"],
                "loops": v["loops"],
                "loop_rate": round(v["loops"] / v["total"], 4),
            }
            for cat, v in by_category.items()
        },
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--base", default=None)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 70)
    print(f"REPETITION STRESS TEST")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Tests: {len(STRESS_PROMPTS)}")
    print("=" * 70)

    # Warmup
    print(f"\n[warmup] {args.model}...", flush=True)
    try:
        httpx.post(OLLAMA, json={
            "model": args.model,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
            "options": {"num_predict": 2},
        }, timeout=120)
        print("  ok", flush=True)
    except Exception as e:
        print(f"  FAILED: {e}")
        return

    print(f"\n[test] {args.model}...", flush=True)
    results = run_stress_test(args.model, args.temperature, args.max_tokens)
    report = generate_report(args.model, results)

    base_report = None
    if args.base:
        print(f"\n[warmup] {args.base}...", flush=True)
        try:
            httpx.post(OLLAMA, json={
                "model": args.base,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 2},
            }, timeout=120)
            print("  ok", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}")

        print(f"\n[test] {args.base}...", flush=True)
        base_results = run_stress_test(args.base, args.temperature, args.max_tokens)
        base_report = generate_report(args.base, base_results)

    out_path = args.out or f"logs/eval_repetition_{args.model.replace('/', '_').replace(':', '_')}.json"
    combined = {"model": report}
    if base_report:
        combined["base"] = base_report
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(combined, indent=2, default=str))

    # Summary
    print(f"\n{'='*70}")
    print("REPETITION STRESS TEST SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Model':<45} {'Loops':>6} {'Total':>6} {'Rate':>8}")
    print("-" * 70)
    print(f"{args.model:<45} {report['loops_detected']:>6} {report['valid']:>6} {report['loop_rate']:>8.1%}")
    if base_report:
        print(f"{args.base:<45} {base_report['loops_detected']:>6} {base_report['valid']:>6} {base_report['loop_rate']:>8.1%}")

    print(f"\n{'Category':<20} {'Loops':>6} {'Total':>6} {'Rate':>8}")
    print("-" * 45)
    for cat, v in sorted(report["by_category"].items()):
        print(f"  {cat:<18} {v['loops']:>6} {v['total']:>6} {v['loop_rate']:>8.1%}")

    print(f"\n[saved] {out_path}\n")


if __name__ == "__main__":
    main()
