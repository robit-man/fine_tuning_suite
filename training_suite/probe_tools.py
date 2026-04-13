#!/usr/bin/env python3
"""Probe Ollama models for tool-calling capability.

Tests each model with a single-tool weather prompt via /api/chat and records:
  - HTTP status
  - structured message.tool_calls presence
  - correct tool name + argument
  - wall-clock time (warmup separately, so cold-load is not counted in tool_sec)

Writes results incrementally to logs/ollama_tool_probe.json so partial runs
are recoverable. Flushes stdout on every line so tail -f works.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
LOG_JSON = ROOT / "logs" / "ollama_tool_probe.json"
OLLAMA = "http://localhost:11434/api/chat"
WARMUP_TIMEOUT = 1200
TOOL_TIMEOUT = 600

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


def flush(msg: str) -> None:
    print(msg, flush=True)


def probe(model: str) -> dict:
    entry: dict = {"model": model}
    # 1) Warmup (cold-load)
    t0 = time.time()
    try:
        rw = httpx.post(
            OLLAMA,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 2},
            },
            timeout=WARMUP_TIMEOUT,
        )
    except Exception as e:
        entry["error"] = f"warmup {type(e).__name__}: {str(e)[:200]}"
        entry["pass"] = False
        entry["status"] = "WARMUP_EXC"
        flush(f"[warmup-exc] {model}: {e}")
        return entry
    entry["warmup_status"] = rw.status_code
    entry["warmup_sec"] = round(time.time() - t0, 1)
    if rw.status_code != 200:
        entry["error"] = rw.text[:200]
        entry["pass"] = False
        if "not found" in rw.text.lower() or rw.status_code == 404:
            entry["status"] = "NOT_FOUND"
            flush(f"[404] {model}")
        else:
            entry["status"] = "WARMUP_FAIL"
            flush(f"[warmup-fail] {model}: HTTP {rw.status_code}")
        return entry

    # 2) Tool call
    t0 = time.time()
    try:
        r = httpx.post(
            OLLAMA,
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
                "stream": False,
                "tools": TOOLS,
            },
            timeout=TOOL_TIMEOUT,
        )
    except Exception as e:
        entry["error"] = f"tool {type(e).__name__}: {str(e)[:200]}"
        entry["pass"] = False
        entry["status"] = "TOOL_EXC"
        flush(f"[tool-exc] {model}: {e}")
        return entry
    entry["tool_status"] = r.status_code
    entry["tool_sec"] = round(time.time() - t0, 1)
    if r.status_code != 200:
        entry["error"] = r.text[:200]
        entry["pass"] = False
        entry["status"] = "NO_TOOL_SUPPORT"
        flush(f"[no-tool] {model}: HTTP {r.status_code} — {r.text[:80]}")
        return entry
    d = r.json()
    msg = d.get("message", {})
    tcs = msg.get("tool_calls") or []
    entry["tool_calls_count"] = len(tcs)
    if not tcs:
        entry["pass"] = False
        entry["status"] = "TEXT_ONLY"
        entry["content_snippet"] = (msg.get("content") or "")[:150]
        flush(f"[text-only] {model}: {entry['content_snippet']!r}")
        return entry
    fn = tcs[0].get("function", {})
    entry["called_name"] = fn.get("name")
    entry["called_args"] = fn.get("arguments")
    correct = (
        fn.get("name") == "get_weather"
        and isinstance(fn.get("arguments"), dict)
        and "paris" in str(fn["arguments"].get("city", "")).lower()
    )
    entry["pass"] = bool(correct)
    entry["status"] = "PASS" if correct else "WRONG_CALL"
    flush(
        f"[{entry['status']}] {model}: {fn.get('name')}({fn.get('arguments')})  "
        f"tool_sec={entry['tool_sec']}"
    )
    return entry


def main() -> None:
    candidates = sys.argv[1:]
    if not candidates:
        print("Usage: probe_tools.py <model> [<model> ...]", file=sys.stderr)
        sys.exit(2)
    results = []
    for m in candidates:
        results.append(probe(m))
        LOG_JSON.write_text(json.dumps(results, indent=2, default=str))

    flush("\n================ CAPABILITY MATRIX ================\n")
    flush(f"{'STATUS':<16} {'MODEL':<72} {'warmup':>8} {'tool':>8}")
    flush("-" * 108)
    for e in results:
        st = e.get("status") or ("PASS" if e.get("pass") else "FAIL")
        flush(
            f"{st:<16} {e['model']:<72} "
            f"{str(e.get('warmup_sec', '?')):>8} {str(e.get('tool_sec', '?')):>8}"
        )
    passed = [e["model"] for e in results if e.get("pass")]
    failed = [e["model"] for e in results if not e.get("pass") and e.get("status") != "NOT_FOUND"]
    not_found = [e["model"] for e in results if e.get("status") == "NOT_FOUND"]
    flush(f"\nPASS ({len(passed)}):")
    for m in passed:
        flush(f"  + {m}")
    flush(f"\nFAIL / NO-TOOL ({len(failed)}):")
    for m in failed:
        flush(f"  - {m}")
    if not_found:
        flush(f"\nNOT_FOUND ({len(not_found)}):")
        for m in not_found:
            flush(f"  ? {m}")


if __name__ == "__main__":
    main()
