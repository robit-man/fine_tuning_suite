from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import time
import wave
from pathlib import Path
from typing import Any

import httpx

from training_suite.evals.audio_bridge import (
    comprehension_prompt_sha256,
    load_audio_bridge_records,
    parse_tagged_audio_evidence,
    word_error_rate,
)
from training_suite.evals.tts_alignment import (
    NUMBER_WORDS,
    _audio_payload,
    _wav_contract,
)

TTS_VOCABULARY_SCHEMA = "robit.qwen3a-tts-vocabulary-evaluation.v1"
VOCABULARY_PROMPT_PREFIX = "Please pronounce these words: "
TTS_WAV_CACHE_SCHEMA = "robit.qwen3a-tts-vocabulary-wav-cache.v1"


def normalized_words(text: str) -> list[str]:
    return [
        NUMBER_WORDS.get(token, token)
        for token in re.findall(r"[a-z0-9']+", text.casefold())
    ]


def vocabulary_prompt(words: list[str]) -> str:
    if not words:
        raise ValueError("vocabulary prompt requires at least one word")
    return VOCABULARY_PROMPT_PREFIX + ", ".join(words) + "."


def build_vocabulary_batches(
    manifest: Path,
    *,
    chunk_words: int,
    seed: int,
) -> tuple[list[list[str]], collections.Counter[str]]:
    if chunk_words <= 0:
        raise ValueError("chunk words must be positive")
    frequencies: collections.Counter[str] = collections.Counter()
    for record in load_audio_bridge_records(manifest):
        frequencies.update(normalized_words(str(record.get("transcript") or "")))
    vocabulary = sorted(word for word in frequencies if len(word) > 1)
    random.Random(seed).shuffle(vocabulary)
    return (
        [
            vocabulary[index : index + chunk_words]
            for index in range(0, len(vocabulary), chunk_words)
        ],
        frequencies,
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_wav_cache(
    cache_dir: Path,
    *,
    tts_endpoint: str,
    manifest: Path,
    speaker_file: Path | None,
    chunk_words: int,
    seed: int,
    limit_batches: int,
    batches: list[list[str]],
) -> Path:
    destination = cache_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    speaker = speaker_file.expanduser().resolve() if speaker_file else None
    identity = {
        "schema": TTS_WAV_CACHE_SCHEMA,
        "tts_endpoint": tts_endpoint,
        "manifest": str(manifest.expanduser().resolve()),
        "manifest_sha256": _sha256_file(manifest.expanduser().resolve()),
        "speaker_file": str(speaker) if speaker else None,
        "speaker_sha256": _sha256_file(speaker) if speaker else None,
        "chunk_words": chunk_words,
        "seed": seed,
        "limit_batches": limit_batches,
        "total_batches": len(batches),
        "prompt_prefix": VOCABULARY_PROMPT_PREFIX,
        "batch_sha256": hashlib.sha256(
            json.dumps(batches, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    metadata = destination / "cache-identity.json"
    if metadata.is_file():
        if json.loads(metadata.read_text(encoding="utf-8")) != identity:
            raise ValueError("TTS WAV cache identity does not match this run")
    else:
        _write_report(metadata, identity)
    return destination


def _cached_wav(path: Path) -> bytes | None:
    if not path.is_file():
        return None
    payload = path.read_bytes()
    try:
        contract = _wav_contract(payload)
    except (EOFError, wave.Error):
        return None
    if not contract["valid"]:
        return None
    return payload


def _write_cached_wav(path: Path, payload: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload)
    partial.replace(path)


def _resume_results(
    partial: Path,
    *,
    model: str,
    manifest: Path,
    chunk_words: int,
    seed: int,
    limit_batches: int,
    batches: list[list[str]],
) -> tuple[list[dict[str, Any]], float]:
    if not partial.is_file():
        return [], 0.0
    checkpoint = json.loads(partial.read_text(encoding="utf-8"))
    identity = {
        "schema": TTS_VOCABULARY_SCHEMA,
        "model": model,
        "manifest": str(manifest.expanduser().resolve()),
        "chunk_words": chunk_words,
        "seed": seed,
        "limit_batches": limit_batches,
        "total_batches": len(batches),
        "prompt_prefix": VOCABULARY_PROMPT_PREFIX,
        "comprehension_prompt_sha256": comprehension_prompt_sha256(),
    }
    mismatched = [
        key for key, value in identity.items() if checkpoint.get(key) != value
    ]
    if mismatched:
        raise ValueError(
            "vocabulary checkpoint does not match this run: " + ", ".join(mismatched)
        )
    results = checkpoint.get("results")
    if not isinstance(results, list) or len(results) > len(batches):
        raise ValueError("vocabulary checkpoint has invalid result count")
    for index, result in enumerate(results):
        if (
            result.get("batch") != index + 1
            or result.get("expected_words") != batches[index]
        ):
            raise ValueError("vocabulary checkpoint batch sequence is invalid")
    return results, float(checkpoint.get("elapsed_seconds") or 0.0)


def _checkpoint_report(
    *,
    tts_endpoint: str,
    comprehension_endpoint: str,
    model: str,
    manifest: Path,
    chunk_words: int,
    seed: int,
    limit_batches: int,
    total_batches: int,
    elapsed_seconds: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": TTS_VOCABULARY_SCHEMA,
        "status": "incomplete",
        "tts_endpoint": tts_endpoint,
        "comprehension_endpoint": comprehension_endpoint,
        "model": model,
        "manifest": str(manifest.expanduser().resolve()),
        "chunk_words": chunk_words,
        "seed": seed,
        "limit_batches": limit_batches,
        "total_batches": total_batches,
        "prompt_prefix": VOCABULARY_PROMPT_PREFIX,
        "comprehension_prompt_sha256": comprehension_prompt_sha256(),
        "completed_batches": len(results),
        "elapsed_seconds": elapsed_seconds,
        "results": results,
    }


def evaluate_tts_vocabulary(
    *,
    tts_endpoint: str,
    comprehension_endpoint: str,
    model: str,
    manifest: Path,
    output_path: Path,
    speaker_file: Path | None = None,
    chunk_words: int = 10,
    seed: int = 42,
    limit_batches: int = 0,
    timeout_seconds: float = 300.0,
    max_tokens: int = 1_024,
    minimum_vocabulary_size: int = 4_000,
    minimum_vocabulary_recall: float = 0.70,
    minimum_rare_word_recall: float = 0.60,
    maximum_mean_wer: float = 0.45,
    minimum_tagged_output_rate: float = 0.99,
    resume: bool = True,
    wav_cache_dir: Path | None = None,
) -> dict[str, Any]:
    if limit_batches < 0:
        raise ValueError("limit batches cannot be negative")
    if max_tokens <= 0:
        raise ValueError("max tokens must be positive")
    if minimum_vocabulary_size <= 0:
        raise ValueError("minimum vocabulary size must be positive")
    for label, value in (
        ("minimum vocabulary recall", minimum_vocabulary_recall),
        ("minimum rare-word recall", minimum_rare_word_recall),
        ("minimum tagged-output rate", minimum_tagged_output_rate),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{label} must be between zero and one")
    if maximum_mean_wer < 0:
        raise ValueError("maximum mean WER cannot be negative")
    batches, frequencies = build_vocabulary_batches(
        manifest,
        chunk_words=chunk_words,
        seed=seed,
    )
    if limit_batches > 0:
        batches = batches[:limit_batches]
    if not batches:
        raise ValueError("vocabulary manifest produced no evaluation batches")
    expected_vocabulary = {word for batch in batches for word in batch}
    rare_vocabulary = {
        word for word in expected_vocabulary if frequencies.get(word, 0) <= 2
    }
    speaker = str(speaker_file.expanduser().resolve()) if speaker_file else None
    wav_cache = (
        _prepare_wav_cache(
            wav_cache_dir,
            tts_endpoint=tts_endpoint,
            manifest=manifest,
            speaker_file=speaker_file,
            chunk_words=chunk_words,
            seed=seed,
            limit_batches=limit_batches,
            batches=batches,
        )
        if wav_cache_dir
        else None
    )
    destination = output_path.expanduser().resolve()
    partial = destination.with_name(destination.name + ".partial")
    results: list[dict[str, Any]] = []
    prior_elapsed = 0.0
    if resume:
        results, prior_elapsed = _resume_results(
            partial,
            model=model,
            manifest=manifest,
            chunk_words=chunk_words,
            seed=seed,
            limit_batches=limit_batches,
            batches=batches,
        )
    observed_vocabulary = {
        word
        for result in results
        for word in set(result["expected_words"])
        & set(normalized_words(str(result["predicted_transcript"])))
    }
    started = time.time() - prior_elapsed
    with httpx.Client(timeout=timeout_seconds) as client:
        for index, words in enumerate(batches[len(results) :], len(results) + 1):
            prompt = vocabulary_prompt(words)
            cached_path = wav_cache / f"batch-{index:05d}.wav" if wav_cache else None
            wav = _cached_wav(cached_path) if cached_path else None
            if wav is None:
                tts_body: dict[str, Any] = {
                    "text": prompt,
                    "language": "en",
                    "temperature": 0.7,
                    "top_k": 40,
                    "top_p": 0.9,
                    "seed": seed,
                }
                if speaker:
                    tts_body["speaker_file"] = speaker
                synthesis = client.post(tts_endpoint, json=tts_body)
                synthesis.raise_for_status()
                wav = synthesis.content
                if cached_path:
                    if not _wav_contract(wav)["valid"]:
                        raise ValueError("refusing to cache invalid TTS WAV")
                    _write_cached_wav(cached_path, wav)
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
            predicted_words = normalized_words(predicted)
            predicted_set = set(predicted_words)
            observed_vocabulary.update(predicted_set & set(words))
            results.append(
                {
                    "batch": index,
                    "expected_words": words,
                    "predicted_transcript": predicted,
                    "word_error_rate": word_error_rate(prompt, predicted),
                    "vocabulary_recall": len(set(words) & predicted_set)
                    / len(set(words)),
                    "tagged_output_valid": parsed["valid"],
                    "wav": wav_contract,
                    "raw_content": content,
                }
            )
            elapsed = time.time() - started
            _write_report(
                partial,
                _checkpoint_report(
                    tts_endpoint=tts_endpoint,
                    comprehension_endpoint=comprehension_endpoint,
                    model=model,
                    manifest=manifest,
                    chunk_words=chunk_words,
                    seed=seed,
                    limit_batches=limit_batches,
                    total_batches=len(batches),
                    elapsed_seconds=elapsed,
                    results=results,
                ),
            )
            print(
                json.dumps(
                    {
                        "completed_batches": len(results),
                        "total_batches": len(batches),
                        "latest_vocabulary_recall": results[-1]["vocabulary_recall"],
                    }
                ),
                flush=True,
            )

    vocabulary_recall = len(observed_vocabulary) / max(len(expected_vocabulary), 1)
    rare_observed = observed_vocabulary & rare_vocabulary
    rare_word_recall = len(rare_observed) / max(len(rare_vocabulary), 1)
    mean_wer = sum(row["word_error_rate"] for row in results) / len(results)
    tagged_output_rate = sum(row["tagged_output_valid"] for row in results) / len(
        results
    )
    metrics = {
        "batches": len(results),
        "expected_unique_words": len(expected_vocabulary),
        "observed_unique_words": len(observed_vocabulary),
        "vocabulary_recall": vocabulary_recall,
        "rare_unique_words": len(rare_vocabulary),
        "observed_rare_words": len(rare_observed),
        "rare_word_recall": rare_word_recall,
        "mean_word_error_rate": mean_wer,
        "tagged_output_rate": tagged_output_rate,
        "valid_wavs": sum(row["wav"]["valid"] for row in results),
    }
    thresholds = {
        "minimum_vocabulary_size": minimum_vocabulary_size,
        "minimum_vocabulary_recall": minimum_vocabulary_recall,
        "minimum_rare_word_recall": minimum_rare_word_recall,
        "maximum_mean_wer": maximum_mean_wer,
        "minimum_tagged_output_rate": minimum_tagged_output_rate,
    }
    gates = {
        "vocabulary_size": len(expected_vocabulary) >= minimum_vocabulary_size,
        "vocabulary_recall": vocabulary_recall >= minimum_vocabulary_recall,
        "rare_word_recall": rare_word_recall >= minimum_rare_word_recall,
        "mean_word_error_rate": mean_wer <= maximum_mean_wer,
        "exact_tagged_output": tagged_output_rate >= minimum_tagged_output_rate,
        "wav_contract": all(row["wav"]["valid"] for row in results),
    }
    report = {
        "schema": TTS_VOCABULARY_SCHEMA,
        "tts_endpoint": tts_endpoint,
        "comprehension_endpoint": comprehension_endpoint,
        "model": model,
        "manifest": str(manifest.expanduser().resolve()),
        "chunk_words": chunk_words,
        "seed": seed,
        "limit_batches": limit_batches,
        "total_batches": len(batches),
        "prompt_prefix": VOCABULARY_PROMPT_PREFIX,
        "comprehension_prompt_sha256": comprehension_prompt_sha256(),
        "wav_cache_dir": str(wav_cache) if wav_cache else None,
        "thresholds": thresholds,
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    _write_report(partial, report)
    partial.replace(destination)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a large-vocabulary Qwen3-TTS to audio-bridge ASR gate"
    )
    parser.add_argument("--tts-endpoint", required=True)
    parser.add_argument("--comprehension-endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--speaker-file", type=Path)
    parser.add_argument(
        "--wav-cache-dir",
        type=Path,
        help="Atomically cache validated TTS WAVs for reuse across ASR candidates",
    )
    parser.add_argument("--chunk-words", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-tokens", type=int, default=1_024)
    parser.add_argument("--minimum-vocabulary-size", type=int, default=4_000)
    parser.add_argument("--minimum-vocabulary-recall", type=float, default=0.70)
    parser.add_argument("--minimum-rare-word-recall", type=float, default=0.60)
    parser.add_argument("--maximum-mean-wer", type=float, default=0.45)
    parser.add_argument("--minimum-tagged-output-rate", type=float, default=0.99)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any matching .partial checkpoint and restart from batch one",
    )
    args = parser.parse_args()
    report = evaluate_tts_vocabulary(
        tts_endpoint=args.tts_endpoint,
        comprehension_endpoint=args.comprehension_endpoint,
        model=args.model,
        manifest=Path(args.manifest),
        output_path=Path(args.out),
        speaker_file=args.speaker_file,
        chunk_words=args.chunk_words,
        seed=args.seed,
        limit_batches=args.limit_batches,
        timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
        minimum_vocabulary_size=args.minimum_vocabulary_size,
        minimum_vocabulary_recall=args.minimum_vocabulary_recall,
        minimum_rare_word_recall=args.minimum_rare_word_recall,
        maximum_mean_wer=args.maximum_mean_wer,
        minimum_tagged_output_rate=args.minimum_tagged_output_rate,
        resume=not args.no_resume,
        wav_cache_dir=args.wav_cache_dir,
    )
    print(json.dumps({"passed": report["passed"], **report["metrics"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
