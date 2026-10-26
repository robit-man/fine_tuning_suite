"""Behavioral probe for official Ollama's native ASR endpoint."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import httpx

from training_suite.core.config import DEFAULT_OLLAMA_URL


def word_error_rate(expected: str, actual: str) -> float:
    reference = re.findall(r"\w+", expected.casefold())
    hypothesis = re.findall(r"\w+", actual.casefold())
    if not reference:
        raise ValueError("expected transcript must contain words")
    previous = list(range(len(hypothesis) + 1))
    for index, word in enumerate(reference, 1):
        current = [index]
        for position, candidate in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[position] + 1,
                               previous[position - 1] + (word != candidate)))
        previous = current
    return previous[-1] / len(reference)


def ollama_transcription_smoke(model: str, *, audio: Path, expected_text: str,
                               ollama_url=DEFAULT_OLLAMA_URL, max_wer=0.1,
                               timeout=180, client: httpx.Client | None = None) -> dict[str, Any]:
    """Send actual multipart audio, and score the returned words against a fixture."""
    started = time.monotonic()
    report: dict[str, Any] = {
        "ok": False, "model": model, "endpoint": ollama_url.rstrip("/") + "/v1/audio/transcriptions",
        "execution_owner": "ollama", "expected_text": expected_text, "max_word_error_rate": max_wer,
    }
    session = client or httpx.Client(timeout=timeout)
    try:
        if not 0 <= max_wer <= 1:
            raise ValueError("max_wer must be between 0 and 1")
        word_error_rate(expected_text, expected_text)
        with audio.open("rb") as handle:
            response = session.post(report["endpoint"],
                                    data={"model": model, "response_format": "json"},
                                    files={"file": (audio.name, handle, "audio/wav")})
        report["status_code"] = response.status_code
        if response.is_error:
            raise ValueError(f"native ASR returned HTTP {response.status_code}: {response.text[:500]}")
        body = response.json()
        text = body.get("text") if isinstance(body, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("native ASR must return nonempty transcription text")
        report["text"] = text
        report["word_error_rate"] = word_error_rate(expected_text, text)
        report["ok"] = report["word_error_rate"] <= max_wer
        if not report["ok"]:
            report["error"] = "native ASR did not meet the fixture transcription threshold"
    except (httpx.HTTPError, OSError, TypeError, ValueError) as exc:
        report["error"] = str(exc)
    finally:
        if client is None:
            session.close()
        report["elapsed_s"] = round(time.monotonic() - started, 2)
    return report
