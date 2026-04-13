#!/usr/bin/env python3
"""Non-standard / adversarial tool-calling tests via the Open Agents gateway (11435).

These tests deliberately go beyond the vanilla "weather in city X" contract
and stress the reasoning side of tool use:

  H1  many-tool distraction      : 15 tools, user asks for exactly one
  H2  nested object argument     : tool parameter is an object-of-objects
  H3  array argument             : user provides a list, tool wants an array
  H4  enum normalization         : user says a synonym not in the enum
  H5  numeric extraction         : numbers written in words ("fifty-two")
  H6  negative / expression      : negative numbers + parentheses in math
  H7  chained sequential         : result of call A needed to call B
  H8  same tool, different args  : 3 calls of the same tool, distinct args
  H9  tool non-existence         : no tool can answer, must decline cleanly
  H10 disambiguation             : two similar tools, only one fits
  H11 enum strictness refusal    : user asks for out-of-enum value
  H12 reasoning before call      : chain-of-thought followed by precise call

Each test is scored against a small rubric in its `check` callable. The
scoring is strict: either the tool contract is respected (right tool, valid
args, schema-clean) or it fails.

Runs against the OA gateway at http://localhost:11435/v1/chat/completions
(OpenAI-compatible). Defaults to testing both the base (`local/qwen3.5:9b`)
and our distilled model (`local/qwen3.5-9b-qwen3.6-distilled:q4km`) so we
get a side-by-side.

Usage:
    python scripts/hard_tool_tests.py [model1 model2 ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import httpx

ROOT = Path(__file__).resolve().parent.parent
OUT_JSON = ROOT / "logs" / "hard_tool_tests.json"
OA_URL = "http://localhost:11435/v1/chat/completions"
TIMEOUT = 600

DEFAULT_MODELS = [
    "local/qwen3.5:9b",
    "local/qwen3.5-9b-qwen3.6-distilled:q4km",
]

# ---- tool universe (used across tests) ----
DISTRACTOR_TOOLS = [
    {"type":"function","function":{"name":"get_weather","description":"Get current weather for a city.",
     "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
    {"type":"function","function":{"name":"get_forecast","description":"Multi-day weather forecast for a city.",
     "parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city","days"]}}},
    {"type":"function","function":{"name":"calculator","description":"Evaluate a numeric math expression.",
     "parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
    {"type":"function","function":{"name":"search_wikipedia","description":"Search Wikipedia.",
     "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"search_web","description":"Search the public web.",
     "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"get_current_time","description":"Current local time in a timezone.",
     "parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"]}}},
    {"type":"function","function":{"name":"convert_currency","description":"Currency conversion.",
     "parameters":{"type":"object","properties":{"amount":{"type":"number"},"from_currency":{"type":"string"},"to_currency":{"type":"string"}},"required":["amount","from_currency","to_currency"]}}},
    {"type":"function","function":{"name":"translate_text","description":"Translate text between languages.",
     "parameters":{"type":"object","properties":{"text":{"type":"string"},"target_lang":{"type":"string"}},"required":["text","target_lang"]}}},
    {"type":"function","function":{"name":"stock_price","description":"Current stock price by ticker.",
     "parameters":{"type":"object","properties":{"ticker":{"type":"string"}},"required":["ticker"]}}},
    {"type":"function","function":{"name":"air_quality","description":"Air quality index for a city.",
     "parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}},
    {"type":"function","function":{"name":"dns_lookup","description":"DNS lookup for a hostname.",
     "parameters":{"type":"object","properties":{"hostname":{"type":"string"}},"required":["hostname"]}}},
    {"type":"function","function":{"name":"send_email","description":"Send an email.",
     "parameters":{"type":"object","properties":{
         "to":{"type":"object","properties":{"name":{"type":"string"},"address":{"type":"string"}},"required":["address"]},
         "subject":{"type":"string"},"body":{"type":"string"}},"required":["to","subject","body"]}}},
    {"type":"function","function":{"name":"run_sql","description":"Run a SQL query on the analytics warehouse.",
     "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{"name":"restart_service","description":"Restart a systemd service by name.",
     "parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}},
    {"type":"function","function":{"name":"roll_dice","description":"Roll n-sided dice.",
     "parameters":{"type":"object","properties":{"sides":{"type":"integer"},"count":{"type":"integer"}},"required":["sides","count"]}}},
]

# Enum tool for H4/H11
ENUM_TOOL = [{
    "type":"function","function":{
        "name":"set_thermostat",
        "description":"Set the HVAC thermostat mode.",
        "parameters":{
            "type":"object",
            "properties":{
                "mode":{"type":"string","enum":["off","cool","heat","auto","eco"]},
                "target_f":{"type":"number"},
            },
            "required":["mode","target_f"],
        },
    },
}]

# Array tool for H3
ARRAY_TOOL = [{
    "type":"function","function":{
        "name":"batch_lookup",
        "description":"Look up multiple user IDs at once.",
        "parameters":{
            "type":"object",
            "properties":{
                "user_ids":{"type":"array","items":{"type":"string"}},
            },
            "required":["user_ids"],
        },
    },
}]

# Nested object tool for H2
NESTED_TOOL = [{
    "type":"function","function":{
        "name":"send_email",
        "description":"Send an email with optional attachments.",
        "parameters":{
            "type":"object",
            "properties":{
                "recipient":{"type":"object","properties":{
                    "name":{"type":"string"},
                    "address":{"type":"string"},
                },"required":["address"]},
                "subject":{"type":"string"},
                "body":{"type":"string"},
            },
            "required":["recipient","subject","body"],
        },
    },
}]

# Two similar tools for H10 disambiguation
DISAMBIG_TOOLS = [
    {"type":"function","function":{
        "name":"search_wikipedia",
        "description":"Search the encyclopedia for factual, long-form information.",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
    {"type":"function","function":{
        "name":"search_news",
        "description":"Search recent breaking news articles from the last 48 hours.",
        "parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}},
]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def _get_calls(message: dict) -> list:
    return message.get("tool_calls") or []


def _call_name(tc: dict) -> str:
    return (tc.get("function") or {}).get("name", "")


def _call_args(tc: dict) -> dict:
    raw = (tc.get("function") or {}).get("arguments", {})
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return raw or {}


TESTS: list[dict] = [
    {
        "id": "H1",
        "label": "many-tool distraction (15 tools, 1 right answer)",
        "tools": DISTRACTOR_TOOLS,
        "user": "What's the current time in the Europe/London timezone?",
        "check": lambda tcs, content: (
            len(tcs) == 1
            and _call_name(tcs[0]) == "get_current_time"
            and "london" in str(_call_args(tcs[0]).get("timezone", "")).lower()
        ),
    },
    {
        "id": "H2",
        "label": "nested object argument (recipient={name,address})",
        "tools": NESTED_TOOL,
        "user": "Send an email to Dr. Alice Chen at alice.chen@example.com with subject 'Tuesday standup' and body 'See you at 10am'.",
        "check": lambda tcs, content: (
            len(tcs) >= 1
            and _call_name(tcs[0]) == "send_email"
            and isinstance(_call_args(tcs[0]).get("recipient"), dict)
            and "alice.chen@example.com" in str(_call_args(tcs[0])["recipient"].get("address", "")).lower()
            and "tuesday" in str(_call_args(tcs[0]).get("subject", "")).lower()
        ),
    },
    {
        "id": "H3",
        "label": "array argument (list of user_ids)",
        "tools": ARRAY_TOOL,
        "user": "Please look up these three users: u-1001, u-1002, and u-1003.",
        "check": lambda tcs, content: (
            len(tcs) >= 1
            and _call_name(tcs[0]) == "batch_lookup"
            and isinstance(_call_args(tcs[0]).get("user_ids"), list)
            and set(str(x) for x in _call_args(tcs[0])["user_ids"]) == {"u-1001", "u-1002", "u-1003"}
        ),
    },
    {
        "id": "H4",
        "label": "enum normalization (user says 'cooling', enum is 'cool')",
        "tools": ENUM_TOOL,
        "user": "Put the thermostat into cooling mode at 72 degrees Fahrenheit.",
        "check": lambda tcs, content: (
            len(tcs) >= 1
            and _call_name(tcs[0]) == "set_thermostat"
            and _call_args(tcs[0]).get("mode") == "cool"
            and float(_call_args(tcs[0]).get("target_f", 0)) == 72.0
        ),
    },
    {
        "id": "H5",
        "label": "numeric extraction from words ('fifty-two')",
        "tools": DISTRACTOR_TOOLS,
        "user": "Roll fifty-two six-sided dice for me.",
        "check": lambda tcs, content: (
            len(tcs) >= 1
            and _call_name(tcs[0]) == "roll_dice"
            and int(_call_args(tcs[0]).get("sides", 0)) == 6
            and int(_call_args(tcs[0]).get("count", 0)) == 52
        ),
    },
    {
        "id": "H6",
        "label": "negative + expression (calculator preserves parens)",
        "tools": DISTRACTOR_TOOLS,
        "user": "Compute -42 times (3 plus 4) divided by 7 using the calculator.",
        "check": lambda tcs, content: (
            len(tcs) >= 1
            and _call_name(tcs[0]) == "calculator"
            and "-42" in str(_call_args(tcs[0]).get("expression", ""))
            and any(tok in str(_call_args(tcs[0]).get("expression", "")) for tok in ["3+4", "3 + 4", "(3+4)", "(3 + 4)"])
        ),
    },
    {
        "id": "H7",
        "label": "chained sequential (convert + compute)",
        "tools": DISTRACTOR_TOOLS,
        # Model should call convert_currency FIRST (it can't know the rate otherwise)
        "user": "I need to know what 450 USD is worth in EUR. Use a tool.",
        "check": lambda tcs, content: (
            len(tcs) >= 1
            and _call_name(tcs[0]) == "convert_currency"
            and float(_call_args(tcs[0]).get("amount", 0)) == 450
            and str(_call_args(tcs[0]).get("from_currency", "")).upper() == "USD"
            and str(_call_args(tcs[0]).get("to_currency", "")).upper() == "EUR"
        ),
    },
    {
        "id": "H8",
        "label": "same tool, THREE different args (air_quality x 3)",
        "tools": DISTRACTOR_TOOLS,
        "user": "What is the air quality right now in London, Tokyo, and São Paulo? Call the tool separately for each city.",
        "check": lambda tcs, content: (
            len(tcs) >= 3
            and all(_call_name(tc) == "air_quality" for tc in tcs)
            and {
                str(_call_args(tc).get("city", "")).lower().replace("ã", "a")
                for tc in tcs
            }
            >= {"london", "tokyo", "sao paulo"}
        ),
    },
    {
        "id": "H9",
        "label": "tool non-existence (no tool fits — should NOT hallucinate)",
        "tools": DISTRACTOR_TOOLS,
        "user": "Please generate a hand-drawn portrait of my cat in watercolor style.",
        "check": lambda tcs, content: (
            # Either no tool call at all, OR only tools in the defined set
            all(_call_name(tc) in {t["function"]["name"] for t in DISTRACTOR_TOOLS} for tc in tcs)
            # and ideally length 0 — we score PASS if it declines cleanly (no call)
            and len(tcs) == 0
        ),
    },
    {
        "id": "H10",
        "label": "disambiguation (wiki vs news — user wants breaking news)",
        "tools": DISAMBIG_TOOLS,
        "user": "Get me the latest news from the last few hours about the Mars Curiosity rover.",
        "check": lambda tcs, content: (
            len(tcs) >= 1
            and _call_name(tcs[0]) == "search_news"
        ),
    },
    {
        "id": "H11",
        "label": "enum strictness (user asks 'freezing' — not in enum)",
        "tools": ENUM_TOOL,
        "user": "Set the thermostat to freezing mode at 65F.",
        # Correct behavior: either refuse (no call) or normalize to an allowed enum
        # ('cool' is the closest). We fail on literal 'freezing'.
        "check": lambda tcs, content: (
            len(tcs) == 0
            or (
                len(tcs) == 1
                and _call_name(tcs[0]) == "set_thermostat"
                and _call_args(tcs[0]).get("mode") in {"cool", "off", "eco", "auto", "heat"}
            )
        ),
    },
    {
        "id": "H12",
        "label": "reasoning-before-call (multi-step arithmetic → single call)",
        "tools": DISTRACTOR_TOOLS,
        # Model should figure out 7*23 - 45 = 116 and call calculator
        # with that FULL expression (or equivalent)
        "user": "Use the calculator to compute seven times twenty-three, then subtract forty-five from that result.",
        "check": lambda tcs, content: (
            len(tcs) >= 1
            and _call_name(tcs[0]) == "calculator"
            and any(
                tok in str(_call_args(tcs[0]).get("expression", "")).replace(" ", "")
                for tok in ["7*23-45", "(7*23)-45", "7×23-45", "7*23-45.0", "161-45"]
            )
        ),
    },

    # ======================================================================
    # HX series — ultra-hard, designed to maximize base-vs-tuned divergence
    # ======================================================================

    {
        "id": "HX1",
        "label": "HARD: four-op precedence (12² − (5+3)) / 4",
        "tools": DISTRACTOR_TOOLS,
        "user": (
            "Use the calculator exactly once to compute the following: take twelve squared, "
            "subtract the sum of 5 and 3, then divide the whole thing by 4."
        ),
        "check": lambda tcs, content: (
            len(tcs) == 1
            and _call_name(tcs[0]) == "calculator"
            and (lambda e: (
                # Accept any algebraically equivalent form
                any(tok in e for tok in [
                    "(12**2-(5+3))/4", "(12*12-(5+3))/4", "(144-(5+3))/4",
                    "(144-8)/4", "(12^2-(5+3))/4", "(12²-(5+3))/4",
                    "(12**2-8)/4", "(12*12-8)/4",
                ])
                or e in {"34", "34.0"}  # literal answer also acceptable
            ))(str(_call_args(tcs[0]).get("expression", "")).replace(" ", ""))
        ),
    },
    {
        "id": "HX2",
        "label": "HARD: C→F conversion inside enum-constrained call",
        "tools": ENUM_TOOL,
        "user": (
            "Please put the thermostat into cooling mode with a target of 22 degrees Celsius. "
            "The tool only accepts Fahrenheit in target_f, so convert correctly."
        ),
        "check": lambda tcs, content: (
            len(tcs) == 1
            and _call_name(tcs[0]) == "set_thermostat"
            and _call_args(tcs[0]).get("mode") == "cool"
            # 22°C = 71.6°F ; accept 71, 71.6, 72 (rounded) — NOT 22
            and (lambda v: isinstance(v, (int, float)) and 71.0 <= float(v) <= 72.1)(
                _call_args(tcs[0]).get("target_f")
            )
        ),
    },
    {
        "id": "HX3",
        "label": "HARD: Gauss formula vs brute force (sum 1..100)",
        "tools": DISTRACTOR_TOOLS,
        "user": (
            "Use the calculator to compute the sum of all integers from 1 to 100. Please use a "
            "compact closed-form expression, not a long 1+2+3+... chain."
        ),
        "check": lambda tcs, content: (
            len(tcs) == 1
            and _call_name(tcs[0]) == "calculator"
            and (lambda e: (
                # Accept any algebraic form of Gauss's formula n*(n+1)/2 with n=100
                any(
                    tok in e
                    for tok in [
                        "100*101/2",
                        "101*100/2",
                        "100*(101)/2",
                        "(100*101)/2",
                        "(100)(101)/2",
                        "100*(100+1)/2",
                        "(100+1)*100/2",
                        "100*(100+1)/2.0",
                        "100/2*(100+1)",
                        "n*(n+1)/2",
                    ]
                )
                or e in {"5050", "5050.0"}
            ))(str(_call_args(tcs[0]).get("expression", "")).replace(" ", ""))
            # Explicit reject: brute-force 1+2+3+... chain
            and "1+2+3+4+5" not in str(_call_args(tcs[0]).get("expression", "")).replace(" ", "")
        ),
    },
    {
        "id": "HX4",
        "label": "HARD: nested-object + array (multi-attendee meeting)",
        "tools": [{
            "type": "function",
            "function": {
                "name": "schedule_meeting",
                "description": "Schedule a meeting with attendees.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "attendees": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "email": {"type": "string"},
                                },
                                "required": ["name", "email"],
                            },
                        },
                        "subject": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "normal", "urgent"]},
                    },
                    "required": ["attendees", "subject", "priority"],
                },
            },
        }],
        "user": (
            "Schedule an URGENT meeting titled 'Q3 planning review' with Alice Chen "
            "(alice.chen@acme.io), Bob Patel (bob.patel@acme.io), and their manager Carol Yu "
            "(carol.yu@acme.io)."
        ),
        "check": lambda tcs, content: (
            len(tcs) == 1
            and _call_name(tcs[0]) == "schedule_meeting"
            and (lambda a: (
                isinstance(a.get("attendees"), list)
                and len(a["attendees"]) == 3
                and {att.get("email", "").lower() for att in a["attendees"]} == {
                    "alice.chen@acme.io", "bob.patel@acme.io", "carol.yu@acme.io",
                }
                and a.get("priority") == "urgent"
                and "q3" in a.get("subject", "").lower()
                and "planning" in a.get("subject", "").lower()
            ))(_call_args(tcs[0]))
        ),
    },
    {
        "id": "HX5",
        "label": "HARD: world knowledge + calculator (c − v_sound)",
        "tools": DISTRACTOR_TOOLS,
        "user": (
            "Use the calculator to compute the difference, in meters per second, between the "
            "speed of light in vacuum and the speed of sound in dry air at sea level. Use "
            "standard textbook values and put them directly into the expression."
        ),
        "check": lambda tcs, content: (
            len(tcs) == 1
            and _call_name(tcs[0]) == "calculator"
            and (lambda e: (
                # speed of light = 299792458, speed of sound ~343
                "299792458" in e
                and any(tok in e for tok in ["-343", "- 343", "-340", "-344", "-331"])
            ))(str(_call_args(tcs[0]).get("expression", "")))
        ),
    },
    {
        "id": "HX6",
        "label": "HARD: 5-element array from varied prose",
        "tools": ARRAY_TOOL,
        "user": (
            "Look up these users for me: u-42, u-199, and u-7. Oh, and don't forget u-301 — "
            "and please also include u-55 in the same batch."
        ),
        "check": lambda tcs, content: (
            len(tcs) == 1
            and _call_name(tcs[0]) == "batch_lookup"
            and set(str(x) for x in _call_args(tcs[0]).get("user_ids", [])) == {
                "u-42", "u-199", "u-7", "u-301", "u-55",
            }
        ),
    },
    {
        "id": "HX7",
        "label": "HARD: schema min/max refusal or clamp (roll 5000 d20)",
        "tools": [{
            "type": "function",
            "function": {
                "name": "roll_dice",
                "description": "Roll dice. Count must be between 1 and 1000.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sides": {"type": "integer", "minimum": 2, "maximum": 100},
                        "count": {"type": "integer", "minimum": 1, "maximum": 1000},
                    },
                    "required": ["sides", "count"],
                },
            },
        }],
        "user": "Please roll five thousand twenty-sided dice.",
        # Correct behavior: decline (no call) OR clamp count to 1000. Wrong: call with 5000.
        "check": lambda tcs, content: (
            len(tcs) == 0
            or (
                len(tcs) == 1
                and _call_name(tcs[0]) == "roll_dice"
                and int(_call_args(tcs[0]).get("sides", 0)) == 20
                and 1 <= int(_call_args(tcs[0]).get("count", 0)) <= 1000
            )
        ),
    },
    {
        "id": "HX8",
        "label": "HARD: domain knowledge → 7 parallel calls (G7 capitals)",
        "tools": DISTRACTOR_TOOLS,
        "user": (
            "Get the current weather in the capital city of every G7 country. Call the weather "
            "tool once per city in parallel — not sequentially."
        ),
        "check": lambda tcs, content: (
            len(tcs) == 7
            and all(_call_name(tc) == "get_weather" for tc in tcs)
            and (lambda cities: (
                # G7 capitals: Washington, Ottawa, London, Paris, Berlin, Rome, Tokyo
                {"washington", "ottawa", "london", "paris", "berlin", "rome", "tokyo"}.issubset(
                    {str(c).lower().replace(",", "").split()[0] for c in cities if c}
                )
            ))([_call_args(tc).get("city", "") for tc in tcs])
        ),
    },
    {
        "id": "HX9",
        "label": "HARD: compound convert + chained compute (tax on EUR)",
        "tools": DISTRACTOR_TOOLS,
        "user": (
            "I need to convert 450 US dollars to euros using the currency tool. Use it with the "
            "correct ISO codes."
        ),
        "check": lambda tcs, content: (
            len(tcs) == 1
            and _call_name(tcs[0]) == "convert_currency"
            and float(_call_args(tcs[0]).get("amount", 0)) == 450
            and str(_call_args(tcs[0]).get("from_currency", "")).upper() == "USD"
            and str(_call_args(tcs[0]).get("to_currency", "")).upper() == "EUR"
        ),
    },
    {
        "id": "HX10",
        "label": "HARD: deep-nested email with enum priority + body synthesis",
        "tools": [{
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Send a corporate email with priority.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "address": {"type": "string"},
                            },
                            "required": ["name", "address"],
                        },
                        "cc": {"type": "array", "items": {"type": "string"}},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                        "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]},
                    },
                    "required": ["recipient", "subject", "body", "priority"],
                },
            },
        }],
        "user": (
            "Email Dr. Alice Chen (alice.chen@acme.io) with subject 'Q3 review' and mark it "
            "urgent. CC her manager carol.yu@acme.io and the audit team at audit@acme.io. "
            "Body should ask her to confirm availability for Tuesday 10am."
        ),
        "check": lambda tcs, content: (
            len(tcs) == 1
            and _call_name(tcs[0]) == "send_email"
            and (lambda a: (
                isinstance(a.get("recipient"), dict)
                and a["recipient"].get("address", "").lower() == "alice.chen@acme.io"
                and "alice" in a["recipient"].get("name", "").lower()
                and isinstance(a.get("cc"), list)
                and {str(x).lower() for x in a["cc"]} == {"carol.yu@acme.io", "audit@acme.io"}
                and "q3" in a.get("subject", "").lower()
                and a.get("priority") == "urgent"
                and any(tok in a.get("body", "").lower() for tok in ["tuesday", "10"])
            ))(_call_args(tcs[0]))
        ),
    },
]


def run_test(model: str, test: dict) -> dict:
    print(f"  [{test['id']}] {test['label']}", flush=True)
    t0 = time.time()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": test["user"]}],
        "tools": test["tools"],
        "stream": False,
        # Deterministic decoding so base vs tuned is comparable run-to-run
        "options": {"temperature": 0, "seed": 42, "top_p": 1.0, "top_k": 1},
    }
    try:
        r = httpx.post(OA_URL, json=payload, timeout=TIMEOUT)
    except Exception as e:
        return {"id": test["id"], "label": test["label"], "pass": False,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
                "sec": round(time.time() - t0, 1)}
    sec = round(time.time() - t0, 1)
    if r.status_code != 200:
        return {"id": test["id"], "label": test["label"], "pass": False,
                "error": f"HTTP {r.status_code}: {r.text[:200]}", "sec": sec}
    data = r.json()
    msg = data["choices"][0]["message"]
    tcs = _get_calls(msg)
    content = msg.get("content") or ""
    try:
        ok = bool(test["check"](tcs, content))
    except Exception as e:
        ok = False
        check_err = f"check raised {type(e).__name__}: {e}"
    else:
        check_err = None
    return {
        "id": test["id"],
        "label": test["label"],
        "pass": ok,
        "sec": sec,
        "tool_calls": [
            {"name": _call_name(tc), "args": _call_args(tc)} for tc in tcs
        ],
        "content_snippet": content[:300],
        "check_error": check_err,
    }


def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS
    all_results: dict = {}
    for m in models:
        print(f"\n=== model: {m} ===", flush=True)
        rows = []
        for t in TESTS:
            row = run_test(m, t)
            status = "PASS" if row["pass"] else "FAIL"
            print(f"    [{status}] {row['id']}  sec={row['sec']}", flush=True)
            rows.append(row)
        all_results[m] = rows
        OUT_JSON.write_text(json.dumps(all_results, indent=2, default=str))

    # Summary
    print("\n================ HARD TOOL-CALL MATRIX ================\n")
    headers = [m.split("/")[-1] for m in models]
    print(f"{'TEST':<55} " + " ".join(f"{h:<30}" for h in headers))
    print("-" * (55 + 31 * len(headers)))
    for t in TESTS:
        row = [t["id"] + " " + t["label"][:50]]
        for m in models:
            r = next(x for x in all_results[m] if x["id"] == t["id"])
            row.append(f"{'PASS' if r['pass'] else 'FAIL':<6} {r['sec']:>5}s")
        print(f"{row[0]:<55} " + " ".join(f"{cell:<30}" for cell in row[1:]))
    print()
    for m in models:
        passed = sum(1 for r in all_results[m] if r["pass"])
        total = len(all_results[m])
        print(f"  {m}: {passed}/{total} passed")


if __name__ == "__main__":
    main()
