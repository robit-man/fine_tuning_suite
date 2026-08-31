from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from training_suite.core.config import DEFAULT_OLLAMA_URL, PACKAGE_ROOT, PATHS, slugify
from training_suite.models.audio import AudioContractError, decode_wav_payload, validate_audio_input
from training_suite.models.ollama import show_model


PYTHON = sys.executable or "python3"


@dataclass(frozen=True)
class EvalSpec:
    key: str
    label: str
    command: list[str]
    requires_model: bool = True


def _script(name: str) -> str:
    return str(PACKAGE_ROOT / name)


def eval_specs(model_name: str) -> dict[str, EvalSpec]:
    safe = slugify(model_name, "model")
    PATHS.ensure()
    return {
        "tool-smoke": EvalSpec(
            "tool-smoke",
            "Tool smoke probe",
            [PYTHON, _script("probe_tools.py"), model_name],
        ),
        "hard-tools": EvalSpec(
            "hard-tools",
            "Hard tool tests",
            [PYTHON, _script("hard_tool_tests.py"), model_name],
        ),
        "image-probe": EvalSpec(
            "image-probe",
            "Image probe",
            [PYTHON, _script("image_probe.py")],
            requires_model=False,
        ),
        "diverse": EvalSpec(
            "diverse",
            "Diverse stochastic eval",
            [PYTHON, _script("eval_diverse.py"), model_name, "--out", str(PATHS.logs / f"eval_diverse_{safe}.json")],
        ),
        "repetition": EvalSpec(
            "repetition",
            "Repetition stress eval",
            [PYTHON, _script("eval_repetition.py"), model_name, "--out", str(PATHS.logs / f"eval_repetition_{safe}.json")],
        ),
        "ollama-suite": EvalSpec(
            "ollama-suite",
            "GSM8K/MMLU/Vision via Ollama",
            [PYTHON, _script("eval_via_ollama.py"), model_name, "--out", str(PATHS.logs / f"eval_ollama_{safe}.json")],
        ),
    }


def get_eval(key: str, model_name: str) -> EvalSpec:
    specs = eval_specs(model_name)
    if key not in specs:
        raise KeyError(f"unknown eval: {key}")
    return specs[key]


def tool_smoke(model_name: str, ollama_url: str = DEFAULT_OLLAMA_URL, timeout: float = 120) -> dict[str, Any]:
    tools = [
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
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": tools,
        "stream": False,
    }
    t0 = time.time()
    try:
        response = httpx.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "elapsed_s": round(time.time() - t0, 2)}
    out = {"status_code": response.status_code, "elapsed_s": round(time.time() - t0, 2)}
    if response.status_code != 200:
        out.update({"ok": False, "error": response.text[:500]})
        return out
    data = response.json()
    calls = (data.get("message") or {}).get("tool_calls") or []
    out.update(
        {
            "ok": bool(calls),
            "tool_calls": calls,
            "raw": data,
        }
    )
    return out


def capability_gate(model_name: str, target: list[str] | None = None) -> dict[str, Any]:
    target = target or ["vision", "tools", "thinking"]
    shown = show_model(model_name, verbose=False)
    present = set(shown.capabilities)
    missing = sorted(set(target) - present)
    return {
        "ok": shown.exists and not missing,
        "model": shown.to_dict(),
        "missing": missing,
    }


def omni_audio_smoke(
    audio_path: Path,
    *,
    endpoint: str = "http://127.0.0.1:7860/api/omni/cascade",
    prompt: str = "Transcribe this audio and answer naturally.",
    timeout: float = 900,
) -> dict[str, Any]:
    """Exercise the configured cascade and validate its returned WAV envelope."""
    started = time.time()
    try:
        import base64

        encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        source = validate_audio_input(encoded)
        response = httpx.post(
            endpoint,
            json={
                "prompt": prompt,
                "audio": {
                    "mime_type": "audio/wav",
                    "encoding": "base64",
                    "data": encoded,
                },
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            return {
                "ok": False,
                "status_code": response.status_code,
                "error": response.text[:1000],
                "elapsed_s": round(time.time() - started, 2),
            }
        data = response.json()
        output = decode_wav_payload(data.get("audio") or {})
        output_ok = (
            output.sample_rate_hz == 24000
            and output.channels == 1
            and output.sample_width_bits == 16
        )
        return {
            "ok": output_ok,
            "status_code": response.status_code,
            "input": source.metadata(),
            "output": output.metadata(),
            "audio_understanding": data.get("audio_understanding"),
            "message": data.get("message"),
            "elapsed_s": round(time.time() - started, 2),
        }
    except (OSError, ValueError, httpx.HTTPError, AudioContractError) as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.time() - started, 2),
        }


def write_eval_report(path: Path, report: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
