#!/usr/bin/env python3
"""Diverse stochastic evaluation suite for distilled models.

Tests real-world capabilities that academic benchmarks miss:
  - Instruction following (strict format, constraints)
  - Conversational coherence (multi-turn, memory, role play)
  - Stochastic stability (high temperature behavior)
  - Conciseness (verbosity regression detection)
  - Tangent resistance (staying on topic)
  - System prompt adherence
  - Edge cases (empty input, one-word answers, refusals)

Each test has:
  - A prompt (messages list)
  - A temperature (to test stochastic environments)
  - A judge function that returns (pass: bool, reason: str)
  - A category for reporting

Usage:
  python scripts/eval_diverse.py <model> [--base <base_model>] [--out <path>]
  python scripts/eval_diverse.py cudabenchmarktest --base qwen3.5:9b
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import httpx

OLLAMA = "http://localhost:11434/api/chat"


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    name: str
    category: str
    messages: list[dict]
    temperature: float
    judge: Callable[[str], tuple[bool, str]]
    max_tokens: int = 512
    runs: int = 1  # how many times to run (for stochastic tests)


@dataclass
class TestResult:
    name: str
    category: str
    temperature: float
    passed: bool
    reason: str
    response: str
    tokens: int
    elapsed_s: float
    run_idx: int = 0


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------

def max_words(n: int):
    """Response must be at most n words."""
    def judge(resp: str) -> tuple[bool, str]:
        words = resp.split()
        if len(words) <= n:
            return True, f"{len(words)} words (limit {n})"
        return False, f"{len(words)} words exceeds limit of {n}"
    return judge


def contains_exact(target: str, case_insensitive: bool = True):
    """Response must contain the exact target string."""
    def judge(resp: str) -> tuple[bool, str]:
        a, b = (resp.lower(), target.lower()) if case_insensitive else (resp, target)
        if b in a:
            return True, f"contains '{target}'"
        return False, f"missing '{target}' in response"
    return judge


def matches_pattern(pattern: str, flags: int = re.IGNORECASE):
    """Response must match regex pattern."""
    def judge(resp: str) -> tuple[bool, str]:
        if re.search(pattern, resp, flags):
            return True, f"matches /{pattern}/"
        return False, f"does not match /{pattern}/"
    return judge


def max_chars(n: int):
    """Response must be at most n characters."""
    def judge(resp: str) -> tuple[bool, str]:
        if len(resp) <= n:
            return True, f"{len(resp)} chars (limit {n})"
        return False, f"{len(resp)} chars exceeds limit of {n}"
    return judge


def is_valid_json():
    """Response must be valid JSON."""
    def judge(resp: str) -> tuple[bool, str]:
        # Try to extract JSON from markdown code blocks
        cleaned = resp.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last lines (code fences)
            inner = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            cleaned = inner.strip()
        try:
            json.loads(cleaned)
            return True, "valid JSON"
        except json.JSONDecodeError as e:
            return False, f"invalid JSON: {e}"
    return judge


def no_tangent(*forbidden_topics: str):
    """Response must NOT mention any of the forbidden topics."""
    def judge(resp: str) -> tuple[bool, str]:
        lower = resp.lower()
        for topic in forbidden_topics:
            if topic.lower() in lower:
                return False, f"tangent detected: mentions '{topic}'"
        return True, "no tangent detected"
    return judge


def all_of(*judges: Callable):
    """All judges must pass."""
    def judge(resp: str) -> tuple[bool, str]:
        for j in judges:
            ok, reason = j(resp)
            if not ok:
                return False, reason
        return True, "all checks passed"
    return judge


def one_sentence():
    """Response should be roughly one sentence (at most 2 periods or newlines)."""
    def judge(resp: str) -> tuple[bool, str]:
        stripped = resp.strip().rstrip(".")
        # Count sentence-ending punctuation
        sentences = len(re.findall(r'[.!?]+', resp))
        lines = len([l for l in resp.strip().split("\n") if l.strip()])
        if sentences <= 2 and lines <= 2:
            return True, f"{sentences} sentences, {lines} lines"
        return False, f"{sentences} sentences, {lines} lines — too verbose"
    return judge


def concise(max_len: int = 300):
    """Response must be concise — under max_len chars, no headers, no analysis blocks."""
    def judge(resp: str) -> tuple[bool, str]:
        issues = []
        if len(resp) > max_len:
            issues.append(f"{len(resp)} chars > {max_len} limit")
        if "<analysis>" in resp:
            issues.append("contains <analysis> block")
        if "## " in resp:
            issues.append("contains markdown headers")
        if "**Step " in resp or "**Analysis" in resp:
            issues.append("contains verbose reasoning markers")
        if issues:
            return False, "; ".join(issues)
        return True, f"concise ({len(resp)} chars)"
    return judge


# ---------------------------------------------------------------------------
# Test battery
# ---------------------------------------------------------------------------

def build_test_battery() -> list[TestCase]:
    """Build the full diverse test battery."""
    tests = []

    # ── INSTRUCTION FOLLOWING ──────────────────────────────────────────────

    tests.append(TestCase(
        name="exact_3_fruits",
        category="instruction_following",
        messages=[{"role": "user", "content": "List exactly 3 fruits. Nothing else."}],
        temperature=0.7,
        judge=all_of(max_words(20), concise(100)),
    ))

    tests.append(TestCase(
        name="only_yes_or_no",
        category="instruction_following",
        messages=[{"role": "user", "content":
            "Respond with only the word YES or NO: Is 7 a prime number?"}],
        temperature=0.7,
        judge=all_of(max_words(5), matches_pattern(r"^(yes|no)\W*$")),
    ))

    tests.append(TestCase(
        name="just_the_number",
        category="instruction_following",
        messages=[{"role": "user", "content": "What is 15 * 23? Just the number."}],
        temperature=0.7,
        judge=all_of(contains_exact("345"), max_chars(50)),
    ))

    tests.append(TestCase(
        name="one_word_answer",
        category="instruction_following",
        messages=[{"role": "user", "content":
            "What color is the sky on a clear day? One word only."}],
        temperature=0.7,
        judge=all_of(max_words(5), contains_exact("blue")),
    ))

    tests.append(TestCase(
        name="capital_of_france",
        category="instruction_following",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0.7,
        judge=all_of(contains_exact("Paris"), concise(200)),
    ))

    tests.append(TestCase(
        name="translate_hello",
        category="instruction_following",
        messages=[{"role": "user", "content": "Translate 'hello' to Spanish. Just the translation."}],
        temperature=0.7,
        judge=all_of(contains_exact("hola"), max_chars(50)),
    ))

    tests.append(TestCase(
        name="respond_in_json",
        category="instruction_following",
        messages=[
            {"role": "system", "content": "Always respond in valid JSON."},
            {"role": "user", "content": "List the primary colors."},
        ],
        temperature=0.7,
        judge=is_valid_json(),
    ))

    tests.append(TestCase(
        name="haiku_format",
        category="instruction_following",
        messages=[{"role": "user", "content":
            "Write a haiku about rain. Only output the haiku, no explanation."}],
        temperature=0.7,
        judge=all_of(concise(200), no_tangent("analysis", "approach", "architecture")),
    ))

    tests.append(TestCase(
        name="numbered_list_exactly_5",
        category="instruction_following",
        messages=[{"role": "user", "content":
            "List exactly 5 programming languages as a numbered list. Nothing else."}],
        temperature=0.7,
        judge=all_of(
            matches_pattern(r"1\..*\n2\..*\n3\..*\n4\..*\n5\."),
            concise(300),
        ),
    ))

    # ── CONCISENESS ────────────────────────────────────────────────────────

    tests.append(TestCase(
        name="say_hello",
        category="conciseness",
        messages=[{"role": "user", "content": "Say hello."}],
        temperature=0.7,
        judge=concise(100),
    ))

    tests.append(TestCase(
        name="simple_greeting",
        category="conciseness",
        messages=[{"role": "user", "content": "Hi, how are you?"}],
        temperature=0.7,
        judge=concise(200),
    ))

    tests.append(TestCase(
        name="short_story_2_sentences",
        category="conciseness",
        messages=[{"role": "user", "content":
            "Write a 2-sentence story about a cat."}],
        temperature=0.8,
        judge=all_of(concise(500), no_tangent("architecture", "approach", "implementation")),
    ))

    tests.append(TestCase(
        name="explain_briefly",
        category="conciseness",
        messages=[{"role": "user", "content":
            "Briefly explain what HTTP is in one sentence."}],
        temperature=0.7,
        judge=one_sentence(),
    ))

    # ── CONVERSATIONAL ─────────────────────────────────────────────────────

    tests.append(TestCase(
        name="remember_name",
        category="conversational",
        messages=[
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ],
        temperature=0.7,
        judge=all_of(contains_exact("Alice"), concise(200)),
    ))

    tests.append(TestCase(
        name="remember_number",
        category="conversational",
        messages=[
            {"role": "user", "content": "Remember the number 42."},
            {"role": "assistant", "content": "Got it, I'll remember 42."},
            {"role": "user", "content": "What number did I ask you to remember? Just say the number."},
        ],
        temperature=0.7,
        judge=all_of(contains_exact("42"), max_chars(50)),
    ))

    tests.append(TestCase(
        name="follow_up_question",
        category="conversational",
        messages=[
            {"role": "user", "content": "The capital of France is Paris."},
            {"role": "assistant", "content": "That's correct!"},
            {"role": "user", "content": "What country were we discussing?"},
        ],
        temperature=0.7,
        judge=all_of(contains_exact("France"), concise(200)),
    ))

    # ── SYSTEM PROMPT ──────────────────────────────────────────────────────

    tests.append(TestCase(
        name="pirate_speak",
        category="system_prompt",
        messages=[
            {"role": "system", "content": "You are a pirate. Always respond in pirate speak."},
            {"role": "user", "content": "Where is the nearest library?"},
        ],
        temperature=0.8,
        judge=matches_pattern(r"(arr|matey|ye|ahoy|sail|treasure|ship)", re.IGNORECASE),
    ))

    tests.append(TestCase(
        name="one_sentence_constraint",
        category="system_prompt",
        messages=[
            {"role": "system", "content": "You must always respond in exactly one sentence."},
            {"role": "user", "content": "Explain quantum computing."},
        ],
        temperature=0.7,
        judge=one_sentence(),
    ))

    tests.append(TestCase(
        name="formal_tone",
        category="system_prompt",
        messages=[
            {"role": "system", "content": "You are a formal British butler. Respond with utmost propriety."},
            {"role": "user", "content": "What's for dinner?"},
        ],
        temperature=0.8,
        judge=no_tangent("analysis", "architecture", "approach"),
    ))

    # ── TANGENT RESISTANCE ─────────────────────────────────────────────────

    tests.append(TestCase(
        name="monkeys_stay_on_topic",
        category="tangent_resistance",
        messages=[{"role": "user", "content":
            "Tell me about monkeys. Keep it to 2 sentences."}],
        temperature=1.0,
        judge=all_of(
            no_tangent("infinite monkey theorem", "shakespeare", "typewriter"),
            concise(500),
        ),
    ))

    tests.append(TestCase(
        name="randomness_no_essay",
        category="tangent_resistance",
        messages=[{"role": "user", "content":
            "What do you think about randomness? Keep your answer brief."}],
        temperature=1.0,
        judge=concise(500),
    ))

    tests.append(TestCase(
        name="probability_focused",
        category="tangent_resistance",
        messages=[{"role": "user", "content":
            "What is the probability of flipping heads on a fair coin? One sentence."}],
        temperature=0.7,
        judge=all_of(one_sentence(), contains_exact("50")),
    ))

    # ── HIGH TEMPERATURE STABILITY ─────────────────────────────────────────

    tests.append(TestCase(
        name="high_temp_coherent_t1.0",
        category="stochastic_stability",
        messages=[{"role": "user", "content":
            "What is 2 + 2? Just say the answer."}],
        temperature=1.0,
        judge=all_of(contains_exact("4"), max_chars(50)),
        runs=3,
    ))

    tests.append(TestCase(
        name="high_temp_coherent_t1.5",
        category="stochastic_stability",
        messages=[{"role": "user", "content":
            "Name one color. Just the color name."}],
        temperature=1.5,
        judge=max_words(10),
        runs=3,
    ))

    tests.append(TestCase(
        name="high_temp_instruction_t1.2",
        category="stochastic_stability",
        messages=[{"role": "user", "content":
            "Write one sentence about dogs."}],
        temperature=1.2,
        judge=all_of(concise(300), one_sentence()),
        runs=3,
    ))

    # ── CODE GENERATION ────────────────────────────────────────────────────

    tests.append(TestCase(
        name="factorial_function",
        category="code",
        messages=[{"role": "user", "content":
            "Write a Python function that returns the factorial of n. Just the code, no explanation."}],
        temperature=0.3,
        judge=all_of(
            matches_pattern(r"def\s+factorial"),
            no_tangent("architecture", "approach"),
        ),
    ))

    tests.append(TestCase(
        name="fizzbuzz",
        category="code",
        messages=[{"role": "user", "content":
            "Write FizzBuzz in Python for numbers 1-15. Just the code."}],
        temperature=0.3,
        judge=all_of(
            matches_pattern(r"(fizz|buzz)", re.IGNORECASE),
            no_tangent("architecture", "approach"),
        ),
    ))

    # ── REASONING (should still work) ──────────────────────────────────────

    tests.append(TestCase(
        name="math_word_problem",
        category="reasoning",
        messages=[{"role": "user", "content":
            "If a train travels 60 mph for 2.5 hours, how far does it go?"}],
        temperature=0.3,
        judge=contains_exact("150"),
    ))

    tests.append(TestCase(
        name="logic_puzzle",
        category="reasoning",
        messages=[{"role": "user", "content":
            "All cats are animals. Some animals are pets. Is it necessarily true that all cats are pets? Answer yes or no with a brief explanation."}],
        temperature=0.3,
        judge=contains_exact("no"),
    ))

    # ── FORMAT BLEEDING ────────────────────────────────────────────────────

    tests.append(TestCase(
        name="no_analysis_tags",
        category="format_bleed",
        messages=[{"role": "user", "content":
            "What year did World War 2 end?"}],
        temperature=0.7,
        judge=all_of(
            contains_exact("1945"),
            no_tangent("analysis"),
            concise(200),
        ),
    ))

    tests.append(TestCase(
        name="no_architecture_headers",
        category="format_bleed",
        messages=[{"role": "user", "content":
            "Explain what a variable is in programming. Keep it simple."}],
        temperature=0.7,
        judge=all_of(
            no_tangent("## Architecture", "## Approach", "## Implementation"),
            concise(500),
        ),
    ))

    tests.append(TestCase(
        name="no_step_markers_simple_q",
        category="format_bleed",
        messages=[{"role": "user", "content": "Is water wet?"}],
        temperature=0.7,
        judge=all_of(
            concise(300),
            no_tangent("Step 1", "Step 2"),
        ),
    ))

    return tests


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def strip_think_tags(content: str) -> str:
    """Remove <think>...</think> blocks from Ollama response."""
    if "<think>" in content and "</think>" in content:
        return content.split("</think>", 1)[-1].strip()
    return content.strip()


def run_test(model: str, test: TestCase, run_idx: int = 0) -> TestResult:
    """Run a single test case against Ollama."""
    t0 = time.time()
    try:
        r = httpx.post(OLLAMA, json={
            "model": model,
            "messages": test.messages,
            "stream": False,
            "options": {
                "temperature": test.temperature,
                "num_predict": test.max_tokens,
            },
        }, timeout=120)
        r.raise_for_status()
        data = r.json()
        raw_content = data.get("message", {}).get("content", "")
        content = strip_think_tags(raw_content)
        tokens = data.get("eval_count", 0)
    except Exception as e:
        return TestResult(
            name=test.name, category=test.category,
            temperature=test.temperature,
            passed=False, reason=f"ERROR: {e}",
            response="", tokens=0,
            elapsed_s=round(time.time() - t0, 1),
            run_idx=run_idx,
        )

    passed, reason = test.judge(content)
    return TestResult(
        name=test.name, category=test.category,
        temperature=test.temperature,
        passed=passed, reason=reason,
        response=content[:1000],
        tokens=tokens,
        elapsed_s=round(time.time() - t0, 1),
        run_idx=run_idx,
    )


def run_battery(model: str, tests: list[TestCase]) -> list[TestResult]:
    """Run all tests against a model."""
    results = []
    total = sum(t.runs for t in tests)
    done = 0
    for test in tests:
        for run_i in range(test.runs):
            done += 1
            result = run_test(model, test, run_i)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"  [{done:3d}/{total}] [{status}] {test.name}"
                f"{'[' + str(run_i) + ']' if test.runs > 1 else ''}"
                f"  t={test.temperature}  {result.elapsed_s}s"
                f"  {result.tokens}tok"
                f"  | {result.reason}",
                flush=True,
            )
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    model_name: str,
    results: list[TestResult],
    base_name: str | None = None,
    base_results: list[TestResult] | None = None,
) -> dict:
    """Generate comprehensive comparison report."""
    def summarize(res: list[TestResult]) -> dict:
        total = len(res)
        passed = sum(1 for r in res if r.passed)
        by_cat: dict[str, dict] = {}
        for r in res:
            if r.category not in by_cat:
                by_cat[r.category] = {"total": 0, "passed": 0, "tests": []}
            by_cat[r.category]["total"] += 1
            by_cat[r.category]["passed"] += 1 if r.passed else 0
            by_cat[r.category]["tests"].append({
                "name": r.name,
                "passed": r.passed,
                "reason": r.reason,
                "response_preview": r.response[:200],
                "tokens": r.tokens,
                "elapsed_s": r.elapsed_s,
            })
        avg_tokens = statistics.mean(r.tokens for r in res) if res else 0
        avg_time = statistics.mean(r.elapsed_s for r in res) if res else 0
        return {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 4) if total else 0,
            "avg_tokens": round(avg_tokens, 1),
            "avg_time_s": round(avg_time, 1),
            "by_category": {
                cat: {
                    "passed": v["passed"],
                    "total": v["total"],
                    "pass_rate": round(v["passed"] / v["total"], 4),
                    "tests": v["tests"],
                }
                for cat, v in sorted(by_cat.items())
            },
        }

    report: dict = {
        "eval": "diverse_stochastic",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": {
            "name": model_name,
            **summarize(results),
        },
    }

    if base_results and base_name:
        base_summary = summarize(base_results)
        report["base"] = {"name": base_name, **base_summary}

        # Regression analysis
        model_rate = report["model"]["pass_rate"]
        base_rate = base_summary["pass_rate"]
        delta = round(model_rate - base_rate, 4)
        report["regression"] = {
            "model_pass_rate": model_rate,
            "base_pass_rate": base_rate,
            "delta": delta,
            "regression_pct": round(delta * 100, 1),
            "verdict": (
                "REGRESSION" if delta < -0.05
                else "MARGINAL" if delta < 0
                else "NO REGRESSION" if delta < 0.05
                else "IMPROVEMENT"
            ),
            "by_category": {},
        }
        # Per-category regression
        for cat in report["model"]["by_category"]:
            m_cat = report["model"]["by_category"][cat]
            b_cat = report["base"]["by_category"].get(cat, {"pass_rate": 0})
            cat_delta = round(m_cat["pass_rate"] - b_cat["pass_rate"], 4)
            report["regression"]["by_category"][cat] = {
                "model": m_cat["pass_rate"],
                "base": b_cat["pass_rate"],
                "delta": cat_delta,
                "verdict": (
                    "REGRESSION" if cat_delta < -0.1
                    else "MARGINAL" if cat_delta < 0
                    else "SAME" if cat_delta == 0
                    else "IMPROVEMENT"
                ),
            }

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Diverse stochastic model evaluation")
    ap.add_argument("model", help="Ollama model name to evaluate")
    ap.add_argument("--base", default=None, help="Base model for regression comparison")
    ap.add_argument("--out", default=None, help="Output JSON path")
    args = ap.parse_args()

    tests = build_test_battery()
    total_runs = sum(t.runs for t in tests)

    print("=" * 70)
    print(f"DIVERSE STOCHASTIC EVAL")
    print(f"Model: {args.model}")
    if args.base:
        print(f"Base:  {args.base}")
    print(f"Tests: {len(tests)} ({total_runs} total runs)")
    print(f"Categories: {sorted(set(t.category for t in tests))}")
    print("=" * 70)

    # Warmup
    print(f"\n[warmup] {args.model} ...", flush=True)
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

    print(f"\n[eval] Testing {args.model} ...", flush=True)
    results = run_battery(args.model, tests)

    base_results = None
    if args.base:
        print(f"\n[warmup] {args.base} ...", flush=True)
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

        print(f"\n[eval] Testing base {args.base} ...", flush=True)
        base_results = run_battery(args.base, tests)

    report = generate_report(args.model, results, args.base, base_results)

    # Save
    out_path = args.out or f"logs/eval_diverse_{args.model.replace('/', '_').replace(':', '_')}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[saved] {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<45} {'Pass':>6} {'Total':>6} {'Rate':>8}")
    print("-" * 70)
    print(f"{args.model:<45} {report['model']['passed']:>6} {report['model']['total']:>6} {report['model']['pass_rate']:>8.1%}")
    if base_results:
        print(f"{args.base:<45} {report['base']['passed']:>6} {report['base']['total']:>6} {report['base']['pass_rate']:>8.1%}")
        print(f"\n{'Category':<30} {'Model':>8} {'Base':>8} {'Delta':>8} {'Verdict':>12}")
        print("-" * 70)
        for cat, reg in sorted(report["regression"]["by_category"].items()):
            print(f"  {cat:<28} {reg['model']:>8.1%} {reg['base']:>8.1%} {reg['delta']:>+8.1%} {reg['verdict']:>12}")
        print("-" * 70)
        print(f"  {'OVERALL':<28} {report['regression']['model_pass_rate']:>8.1%} {report['regression']['base_pass_rate']:>8.1%} {report['regression']['delta']:>+8.1%} {report['regression']['verdict']:>12}")
    else:
        print(f"\n{'Category':<30} {'Pass':>6} {'Total':>6} {'Rate':>8}")
        print("-" * 70)
        for cat, data in sorted(report["model"]["by_category"].items()):
            print(f"  {cat:<28} {data['passed']:>6} {data['total']:>6} {data['pass_rate']:>8.1%}")

    print()


if __name__ == "__main__":
    main()
