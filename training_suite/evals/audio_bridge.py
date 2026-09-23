from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from training_suite.training.audio_bridge_training import (
    MEDIA_SUFFIX_PROMPT,
    MEDIA_SYSTEM_PROMPT,
    load_audio_bridge_records,
)

EVALUATION_SCHEMA = "robit.qwen3a-audio-bridge-evaluation.v1"
TAGGED_OUTPUT = re.compile(
    r"<speech_transcript>(.*?)</speech_transcript>\s*"
    r"<audio_observation>(.*?)</audio_observation>",
    re.DOTALL,
)


def comprehension_prompt_sha256() -> str:
    """Identify the exact audio-comprehension policy used for live evaluation."""
    return hashlib.sha256(
        (MEDIA_SYSTEM_PROMPT + "\0" + MEDIA_SUFFIX_PROMPT).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class AudioBridgeEvaluationThresholds:
    maximum_speech_wer: float = 0.35
    maximum_empty_audio_false_transcript_rate: float = 0.05
    minimum_tagged_output_rate: float = 0.98
    maximum_visual_claims: int = 0


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.casefold())


def word_error_rate(reference: str, hypothesis: str) -> float:
    expected = _words(reference)
    actual = _words(hypothesis)
    if not expected:
        return 0.0 if not actual else 1.0
    previous = list(range(len(actual) + 1))
    for row, expected_word in enumerate(expected, 1):
        current = [row]
        for column, actual_word in enumerate(actual, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_word != actual_word),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def parse_tagged_audio_evidence(text: str) -> dict[str, Any]:
    match = TAGGED_OUTPUT.fullmatch(text.strip())
    if not match or any(
        marker in value for value in match.groups() for marker in ("<", ">", "```")
    ):
        return {
            "valid": False,
            "speech_transcript": "",
            "audio_observation": "",
        }
    return {
        "valid": True,
        "speech_transcript": match.group(1).strip(),
        "audio_observation": match.group(2).strip(),
    }


def _request_payload(
    record: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    repeat_penalty: float,
) -> dict[str, Any]:
    audio = base64.b64encode(Path(record["audio_path"]).read_bytes()).decode("ascii")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": MEDIA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": audio}},
                    {
                        "type": "text",
                        "text": MEDIA_SUFFIX_PROMPT.split("<|im_end|>", 1)[0].strip(),
                    },
                ],
            },
        ],
        # Match the no-thinking prefill used during bridge training. Media
        # extraction needs bounded evidence, not a reasoning preamble.
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "repeat_penalty": repeat_penalty,
        # Match the deployed adapter's comprehension budget. Reasoning-capable
        # target trunks may consume an internal reasoning channel before the
        # tagged answer even when the caller did not explicitly enable it.
        "max_tokens": max_tokens,
        "cache_prompt": False,
    }


def audio_observation_claim_scope(content: str, parsed: dict[str, Any]) -> str:
    """Exclude attributed speech before checking for visual claims.

    A speaker is allowed to say words such as "camera", "image", or "seen".
    Those words become an unsupported visual claim only if the model emits
    them as its own observation. The fallback also handles a response cut off
    after an opening transcript tag without reclassifying speech as perception.
    """
    if parsed["valid"]:
        return str(parsed["audio_observation"])
    return re.sub(
        r"<speech_transcript>.*?(?:</speech_transcript>|$)",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )


def evaluate_audio_bridge(
    *,
    endpoint: str,
    model: str,
    manifest: Path,
    output_path: Path,
    limit: int = 0,
    timeout_seconds: float = 180.0,
    max_tokens: int = 2_048,
    repeat_penalty: float = 1.1,
    record_ids: tuple[str, ...] = (),
    thresholds: AudioBridgeEvaluationThresholds | None = None,
) -> dict[str, Any]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if repeat_penalty <= 0:
        raise ValueError("repeat_penalty must be positive")
    selected_thresholds = thresholds or AudioBridgeEvaluationThresholds()
    records = [
        record
        for record in load_audio_bridge_records(manifest)
        if record["split"] == "test"
    ]
    if record_ids:
        selected_ids = set(record_ids)
        records = [record for record in records if record["id"] in selected_ids]
        missing_ids = selected_ids - {record["id"] for record in records}
        if missing_ids:
            raise ValueError(
                "audio bridge evaluation IDs are absent from the test split: "
                + ", ".join(sorted(missing_ids))
            )
    if limit > 0:
        records = records[:limit]
    if not records:
        raise ValueError("audio bridge evaluation has no test records")

    results = []
    started = time.time()
    with httpx.Client(timeout=timeout_seconds) as client:
        for record in records:
            response = client.post(
                endpoint,
                json=_request_payload(
                    record,
                    model=model,
                    max_tokens=max_tokens,
                    repeat_penalty=repeat_penalty,
                ),
            )
            response.raise_for_status()
            payload = response.json()
            content = str(payload["choices"][0]["message"].get("content") or "")
            parsed = parse_tagged_audio_evidence(content)
            reference = str(record.get("transcript") or "")
            hypothesis = str(parsed["speech_transcript"])
            claim_scope = audio_observation_claim_scope(content, parsed)
            results.append(
                {
                    "id": record["id"],
                    "source": record.get("source"),
                    "reference_transcript": reference,
                    "predicted_transcript": hypothesis,
                    "predicted_audio_observation": parsed["audio_observation"],
                    "tagged_output_valid": parsed["valid"],
                    "word_error_rate": word_error_rate(reference, hypothesis),
                    "visual_claim": bool(
                        re.search(
                            r"<visual_observation|\b(?:see|seen|camera|image|video|visual)\b",
                            claim_scope,
                            re.IGNORECASE,
                        )
                    ),
                    "raw_content": content,
                }
            )

    speech = [row for row in results if row["reference_transcript"]]
    empty_audio = [row for row in results if not row["reference_transcript"]]
    metrics = {
        "samples": len(results),
        "speech_samples": len(speech),
        "empty_audio_samples": len(empty_audio),
        "speech_wer": (
            sum(row["word_error_rate"] for row in speech) / len(speech)
            if speech
            else 0.0
        ),
        "empty_audio_false_transcript_rate": (
            sum(bool(row["predicted_transcript"]) for row in empty_audio)
            / len(empty_audio)
            if empty_audio
            else 0.0
        ),
        "tagged_output_rate": (
            sum(row["tagged_output_valid"] for row in results) / len(results)
        ),
        "visual_claims": sum(row["visual_claim"] for row in results),
    }
    gates = {
        "speech_wer": metrics["speech_wer"] <= selected_thresholds.maximum_speech_wer,
        "empty_audio_false_transcript_rate": (
            metrics["empty_audio_false_transcript_rate"]
            <= selected_thresholds.maximum_empty_audio_false_transcript_rate
        ),
        "tagged_output_rate": (
            metrics["tagged_output_rate"]
            >= selected_thresholds.minimum_tagged_output_rate
        ),
        "visual_claims": (
            metrics["visual_claims"] <= selected_thresholds.maximum_visual_claims
        ),
    }
    report = {
        "schema": EVALUATION_SCHEMA,
        "endpoint": endpoint,
        "model": model,
        "manifest": str(manifest.expanduser().resolve()),
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty,
        "record_ids": list(record_ids),
        "comprehension_prompt_sha256": comprehension_prompt_sha256(),
        "thresholds": asdict(selected_thresholds),
        "metrics": metrics,
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
    parser = argparse.ArgumentParser(description="Evaluate a live audio-bridge server")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=2_048)
    parser.add_argument("--repeat-penalty", type=float, default=1.1)
    parser.add_argument("--record-id", action="append", default=[])
    args = parser.parse_args()
    report = evaluate_audio_bridge(
        endpoint=args.endpoint,
        model=args.model,
        manifest=Path(args.manifest),
        output_path=Path(args.out),
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        max_tokens=args.max_tokens,
        repeat_penalty=args.repeat_penalty,
        record_ids=tuple(args.record_id),
    )
    print(
        json.dumps(
            {key: report[key] for key in ("passed", "metrics", "gates")}, indent=2
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
