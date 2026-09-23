from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import re
import time
import wave
from pathlib import Path
from typing import Any

import httpx

from training_suite.evals.audio_bridge import (
    MEDIA_SUFFIX_PROMPT,
    MEDIA_SYSTEM_PROMPT,
    comprehension_prompt_sha256,
    parse_tagged_audio_evidence,
    word_error_rate,
)

TTS_ALIGNMENT_SCHEMA = "robit.qwen3a-tts-alignment-evaluation.v1"
DEFAULT_SEQUENCE = (
    "Copper lighthouse seven.",
    "Blue river eight.",
    "Copper lighthouse seven.",
)
NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def compact_speech_similarity(reference: str, hypothesis: str) -> float:
    def compact(value: str) -> str:
        tokens = re.findall(r"[a-z0-9]+", value.casefold())
        return "".join(NUMBER_WORDS.get(token, token) for token in tokens)

    return difflib.SequenceMatcher(
        None, compact(reference), compact(hypothesis)
    ).ratio()


def _audio_payload(*, model: str, wav: bytes, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": MEDIA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": base64.b64encode(wav).decode("ascii")},
                    },
                    {
                        "type": "text",
                        "text": MEDIA_SUFFIX_PROMPT.split("<|im_end|>", 1)[0].strip(),
                    },
                ],
            },
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "repeat_penalty": 1.1,
        "max_tokens": max_tokens,
        "cache_prompt": False,
    }


def _wav_contract(payload: bytes) -> dict[str, Any]:
    with wave.open(io.BytesIO(payload), "rb") as audio:
        return {
            "sample_rate_hz": audio.getframerate(),
            "channels": audio.getnchannels(),
            "sample_width_bits": audio.getsampwidth() * 8,
            "frames": audio.getnframes(),
            "valid": (
                audio.getframerate() == 24_000
                and audio.getnchannels() == 1
                and audio.getsampwidth() == 2
                and audio.getnframes() > 0
            ),
        }


def evaluate_tts_alignment(
    *,
    tts_endpoint: str,
    comprehension_endpoint: str,
    model: str,
    output_path: Path,
    speaker_file: Path | None = None,
    timeout_seconds: float = 300.0,
    max_tokens: int = 512,
    maximum_wer: float = 0.34,
    minimum_compact_similarity: float = 0.90,
) -> dict[str, Any]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= maximum_wer <= 1:
        raise ValueError("maximum_wer must be between zero and one")
    if not 0 <= minimum_compact_similarity <= 1:
        raise ValueError("minimum compact similarity must be between zero and one")
    speaker = str(speaker_file.expanduser().resolve()) if speaker_file else None
    results = []
    started = time.time()
    with httpx.Client(timeout=timeout_seconds) as client:
        for turn, expected in enumerate(DEFAULT_SEQUENCE, 1):
            tts_body: dict[str, Any] = {
                "text": expected,
                "language": "en",
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "seed": 42,
            }
            if speaker:
                tts_body["speaker_file"] = speaker
            synthesis = client.post(tts_endpoint, json=tts_body)
            synthesis.raise_for_status()
            wav = synthesis.content
            wav_contract = _wav_contract(wav)
            transcription = client.post(
                comprehension_endpoint,
                json=_audio_payload(model=model, wav=wav, max_tokens=max_tokens),
            )
            transcription.raise_for_status()
            content = str(
                transcription.json()["choices"][0]["message"].get("content") or ""
            )
            parsed = parse_tagged_audio_evidence(content)
            predicted = str(parsed["speech_transcript"])
            wer = word_error_rate(expected, predicted)
            compact_similarity = compact_speech_similarity(expected, predicted)
            results.append(
                {
                    "turn": turn,
                    "expected": expected,
                    "predicted": predicted,
                    "word_error_rate": wer,
                    "compact_speech_similarity": compact_similarity,
                    "correct": (
                        wer <= maximum_wer
                        or compact_similarity >= minimum_compact_similarity
                    ),
                    "tagged_output_valid": parsed["valid"],
                    "wav": wav_contract,
                    "raw_content": content,
                }
            )

    gates = {
        "a_b_a_sequence": all(row["correct"] for row in results),
        "exact_tagged_output": all(row["tagged_output_valid"] for row in results),
        "wav_contract": all(row["wav"]["valid"] for row in results),
        "no_one_turn_lag": (
            results[1]["predicted"].casefold() != results[0]["predicted"].casefold()
            and results[2]["predicted"].casefold() != results[1]["predicted"].casefold()
        ),
    }
    report = {
        "schema": TTS_ALIGNMENT_SCHEMA,
        "tts_endpoint": tts_endpoint,
        "comprehension_endpoint": comprehension_endpoint,
        "model": model,
        "comprehension_prompt_sha256": comprehension_prompt_sha256(),
        "sequence": list(DEFAULT_SEQUENCE),
        "maximum_wer": maximum_wer,
        "minimum_compact_similarity": minimum_compact_similarity,
        "metrics": {
            "samples": len(results),
            "mean_word_error_rate": (
                sum(row["word_error_rate"] for row in results) / len(results)
            ),
            "valid_wavs": sum(row["wav"]["valid"] for row in results),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    partial.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesize and transcribe an A-B-A sequence to detect TTS state lag"
    )
    parser.add_argument("--tts-endpoint", required=True)
    parser.add_argument("--comprehension-endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--speaker-file", type=Path)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--maximum-wer", type=float, default=0.34)
    parser.add_argument("--minimum-compact-similarity", type=float, default=0.90)
    args = parser.parse_args()
    report = evaluate_tts_alignment(
        tts_endpoint=args.tts_endpoint,
        comprehension_endpoint=args.comprehension_endpoint,
        model=args.model,
        output_path=Path(args.out),
        speaker_file=args.speaker_file,
        timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
        maximum_wer=args.maximum_wer,
        minimum_compact_similarity=args.minimum_compact_similarity,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
