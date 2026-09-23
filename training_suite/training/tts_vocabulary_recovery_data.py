from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import time
import wave
from pathlib import Path
from typing import Any

import httpx

from training_suite.training.audio_bridge_data import (
    AUDIO_BRIDGE_DATASET_SCHEMA,
    SAMPLE_RATE,
    _write_wav,
)

RECOVERY_DATASET_SCHEMA = "robit.qwen3a-tts-vocabulary-recovery.v1"
RECOVERY_PROMPT_PREFIX = "Please repeat this training vocabulary clearly: "
SHORT_UTTERANCES = (
    "Copper lighthouse seven.",
    "Blue river eight.",
    "Amber mountain nine.",
    "Green harbor six.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transcript_vocabulary(transcript_root: Path) -> list[str]:
    root = transcript_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    vocabulary: set[str] = set()
    for path in sorted(root.rglob("*.trans.txt")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"invalid transcript at {path}:{line_number}")
            vocabulary.update(
                token
                for token in re.findall(r"[a-z']+", fields[1].casefold())
                if len(token) > 1
            )
    if not vocabulary:
        raise ValueError(f"no transcript vocabulary found below {root}")
    return sorted(vocabulary)


def build_recovery_prompts(
    vocabulary: list[str],
    *,
    chunk_words: int,
    seed: int,
    maximum_numeral: int,
) -> list[str]:
    if chunk_words <= 0:
        raise ValueError("chunk words must be positive")
    if maximum_numeral < 10:
        raise ValueError("maximum numeral must be at least 10")
    words = sorted(set(vocabulary) | {str(value) for value in range(10, maximum_numeral + 1)})
    random.Random(seed).shuffle(words)
    return [
        RECOVERY_PROMPT_PREFIX + ", ".join(words[index : index + chunk_words]) + "."
        for index in range(0, len(words), chunk_words)
    ]


def _decode_tts_wav(payload: bytes):
    import numpy as np
    from scipy.signal import resample_poly

    with wave.open(io.BytesIO(payload), "rb") as audio:
        if (
            audio.getframerate() != 24_000
            or audio.getnchannels() != 1
            or audio.getsampwidth() != 2
            or audio.getnframes() <= 0
        ):
            raise ValueError("TTS recovery input is not non-empty 24 kHz mono PCM16 WAV")
        values = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2").astype(
            np.float32
        ) / 32767.0
    return resample_poly(values, 2, 3).astype(np.float32)


def _valid_cached_wav(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            return (
                audio.getframerate() == SAMPLE_RATE
                and audio.getnchannels() == 1
                and audio.getsampwidth() == 2
                and frames > 0
                and len(audio.readframes(frames)) == frames * 2
            )
    except (EOFError, wave.Error):
        return False


def _write_atomic_wav(path: Path, samples: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    _write_wav(partial, samples)
    partial.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def build_tts_vocabulary_recovery_dataset(
    output_dir: Path,
    *,
    base_manifest: Path,
    transcript_root: Path,
    tts_endpoint: str,
    speaker_file: Path,
    chunk_words: int = 20,
    maximum_numeral: int = 1_000,
    training_repeats: int = 3,
    short_variants: int = 4,
    seed: int = 7_173,
    timeout_seconds: float = 300.0,
) -> dict[str, Any]:
    if training_repeats <= 0 or short_variants <= 0:
        raise ValueError("training repeats and short variants must be positive")
    destination = output_dir.expanduser().resolve()
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = base_manifest.expanduser().resolve()
    source_transcripts = transcript_root.expanduser().resolve()
    source_speaker = speaker_file.expanduser().resolve()
    for source in (source_manifest, source_speaker):
        if not source.is_file():
            raise FileNotFoundError(source)

    vocabulary = transcript_vocabulary(source_transcripts)
    prompts = build_recovery_prompts(
        vocabulary,
        chunk_words=chunk_words,
        seed=seed,
        maximum_numeral=maximum_numeral,
    )
    short_prompts = [
        phrase
        for _variant in range(short_variants)
        for phrase in SHORT_UTTERANCES
    ]
    synthesis_plan = prompts + short_prompts
    plan = {
        "schema": RECOVERY_DATASET_SCHEMA,
        "base_manifest": str(source_manifest),
        "base_manifest_sha256": _sha256(source_manifest),
        "transcript_root": str(source_transcripts),
        "vocabulary_size": len(vocabulary),
        "vocabulary_sha256": hashlib.sha256(
            ("\n".join(vocabulary) + "\n").encode("utf-8")
        ).hexdigest(),
        "speaker_file": str(source_speaker),
        "speaker_sha256": _sha256(source_speaker),
        "tts_endpoint": tts_endpoint,
        "prompt_prefix": RECOVERY_PROMPT_PREFIX,
        "chunk_words": chunk_words,
        "maximum_numeral": maximum_numeral,
        "training_repeats": training_repeats,
        "short_variants": short_variants,
        "seed": seed,
        "synthesis_count": len(synthesis_plan),
    }
    plan_path = destination / "recovery_plan.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise ValueError("existing TTS recovery plan does not match this run")
    else:
        _atomic_json(plan_path, plan)

    started = time.time()
    with httpx.Client(timeout=timeout_seconds) as client:
        for index, transcript in enumerate(synthesis_plan):
            wav = audio_dir / f"tts-recovery-{index:05d}.wav"
            if not _valid_cached_wav(wav):
                response = client.post(
                    tts_endpoint,
                    json={
                        "text": transcript,
                        "language": "en",
                        "temperature": 0.7,
                        "top_k": 40,
                        "top_p": 0.9,
                        "seed": seed + index,
                        "speaker_file": str(source_speaker),
                    },
                )
                response.raise_for_status()
                _write_atomic_wav(wav, _decode_tts_wav(response.content))
            print(
                json.dumps(
                    {
                        "completed_syntheses": index + 1,
                        "total_syntheses": len(synthesis_plan),
                    }
                ),
                flush=True,
            )

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source_manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema") != AUDIO_BRIDGE_DATASET_SCHEMA:
            raise ValueError(f"base manifest line {line_number} has an invalid schema")
        source_audio = (source_manifest.parent / str(record["audio"])).resolve()
        if not source_audio.is_file():
            raise FileNotFoundError(source_audio)
        copied = dict(record)
        copied["audio"] = str(source_audio)
        records.append(copied)

    recovery_records: list[dict[str, Any]] = []
    for index, transcript in enumerate(synthesis_plan):
        is_short = index >= len(prompts)
        split = "train" if is_short or index % 10 else "validation"
        repetitions = training_repeats if split == "train" and not is_short else 1
        for repetition in range(repetitions):
            record_id = f"tts-recovery-{index:05d}-r{repetition}"
            recovery_records.append(
                {
                    "schema": AUDIO_BRIDGE_DATASET_SCHEMA,
                    "id": record_id,
                    "audio": str(Path("audio") / f"tts-recovery-{index:05d}.wav"),
                    "split": split,
                    "transcript": transcript,
                    "audio_observation": "One synthetic speaker with no notable background sound.",
                    "assistant_target": (
                        f"<speech_transcript>{transcript}</speech_transcript>"
                        "<audio_observation>One synthetic speaker with no notable "
                        "background sound.</audio_observation>"
                    ),
                    "current_visual_input": False,
                    "augmentation": "qwen3-tts-reference-voice",
                    "source": "Qwen3-TTS large-vocabulary recovery corpus",
                    "license": "generated for this release; source vocabulary is LibriSpeech CC BY 4.0",
                }
            )
    records.extend(recovery_records)
    manifest_path = destination / "manifest.jsonl"
    partial = manifest_path.with_name(manifest_path.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    partial.replace(manifest_path)
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    report = {
        **plan,
        "manifest": str(manifest_path),
        "base_samples": len(records) - len(recovery_records),
        "recovery_samples": len(recovery_records),
        "samples": len(records),
        "split_counts": split_counts,
        "sample_rate_hz": SAMPLE_RATE,
        "audio_only_current_visual_input": False,
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(destination / "dataset_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a crash-resumable Qwen3-TTS vocabulary recovery corpus"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--transcript-root", required=True, type=Path)
    parser.add_argument("--tts-endpoint", required=True)
    parser.add_argument("--speaker-file", required=True, type=Path)
    parser.add_argument("--chunk-words", type=int, default=20)
    parser.add_argument("--maximum-numeral", type=int, default=1_000)
    parser.add_argument("--training-repeats", type=int, default=3)
    parser.add_argument("--short-variants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7_173)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    report = build_tts_vocabulary_recovery_dataset(
        args.out,
        base_manifest=args.base_manifest,
        transcript_root=args.transcript_root,
        tts_endpoint=args.tts_endpoint,
        speaker_file=args.speaker_file,
        chunk_words=args.chunk_words,
        maximum_numeral=args.maximum_numeral,
        training_repeats=args.training_repeats,
        short_variants=args.short_variants,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
