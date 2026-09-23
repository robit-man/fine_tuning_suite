from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any

AUDIO_BRIDGE_DATASET_SCHEMA = "robit.qwen3a-audio-bridge-dataset.v1"
SAMPLE_RATE = 16_000


def training_phrases(limit: int) -> list[str]:
    topics = [
        "the difference between RAM and storage",
        "why the sky changes color at sunset",
        "how speculative decoding works",
        "the safest way to back up a project",
        "what makes a background task reliable",
        "how a heat pump moves energy",
        "the next prime number after ninety seven",
        "why leaves change color",
    ]
    actions = [
        "summarize the latest project notes",
        "compare the two deployment reports",
        "write a short checklist for the release",
        "explain the result in plain language",
        "remember that the preferred color is cobalt",
        "check the calculation before answering",
        "list the unresolved work items",
        "prepare a concise status update",
    ]
    tasks = [
        "inspect the build logs",
        "validate every generated artifact",
        "benchmark the inference path",
        "review the test failures",
        "verify the published checksums",
        "organize the research notes",
        "check memory usage after each stage",
        "compare the baseline and candidate outputs",
    ]
    things = [
        "the weather forecast",
        "my saved project note",
        "the package version",
        "the arithmetic in the report",
        "whether the upload completed",
        "the service health status",
        "the release checklist",
        "the current background task",
    ]
    phrases = [
        "The verification phrase is copper lighthouse seven.",
        "Please answer my question without describing the camera.",
        "I am asking about software, not the room around me.",
        "Do not infer visual evidence from this audio recording.",
        "What did I just say?",
        "Can you hear the alarm behind my voice?",
        "Pause the background job after its next checkpoint.",
        "Resume the task and verify the result before continuing.",
    ]
    phrases.extend(f"Explain {topic}." for topic in topics)
    phrases.extend(f"Please {action}." for action in actions)
    phrases.extend(
        f"Start a background task to {task}, sanity check each intermediate result, "
        "and report only when it is complete."
        for task in tasks
    )
    phrases.extend(
        f"Check {thing}, and do not mention the camera unless I ask about the scene."
        for thing in things
    )
    phrases.extend(
        f"First {left}; then {right}. Confirm the first step before starting the second."
        for left in actions
        for right in tasks
    )
    phrases.extend(
        f"Question {index}: {topic}. Give a direct answer."
        for index, topic in enumerate(topics, 1)
    )
    unique = list(dict.fromkeys(phrases))
    if limit <= len(unique):
        return unique[:limit]
    expanded = list(unique)
    index = 0
    while len(expanded) < limit:
        base = unique[index % len(unique)]
        expanded.append(f"Request {index + 1}. {base}")
        index += 1
    return expanded


def _split_for(key: str) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _write_wav(path: Path, samples: Any) -> None:
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _read_resampled(path: Path):
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)
    if int(rate) != SAMPLE_RATE:
        divisor = gcd(int(rate), SAMPLE_RATE)
        samples = resample_poly(
            samples,
            SAMPLE_RATE // divisor,
            int(rate) // divisor,
        )
    return samples


def _synthesize(text: str, *, voice: str, speed: int, pitch: int):
    executable = shutil.which("espeak-ng")
    if not executable:
        raise RuntimeError("espeak-ng is required to generate the bridge corpus")
    with tempfile.TemporaryDirectory(prefix="audio-bridge-espeak-") as tmp:
        wav = Path(tmp) / "speech.wav"
        subprocess.run(
            [
                executable,
                "-v",
                voice,
                "-s",
                str(speed),
                "-p",
                str(pitch),
                "-w",
                str(wav),
                text,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return _read_resampled(wav)


def _event(kind: str, length: int, rng: Any):
    import numpy as np

    time_axis = np.arange(length, dtype=np.float32) / SAMPLE_RATE
    if kind == "silence":
        return np.zeros(length, dtype=np.float32)
    if kind == "room-tone":
        return rng.normal(0, 0.004, length).astype(np.float32)
    if kind == "fan":
        tone = 0.035 * np.sin(2 * math.pi * 92 * time_axis)
        return (tone + rng.normal(0, 0.012, length)).astype(np.float32)
    if kind == "alarm":
        gate = (np.mod(time_axis, 0.8) < 0.38).astype(np.float32)
        return (0.16 * gate * np.sin(2 * math.pi * 880 * time_axis)).astype(np.float32)
    if kind == "clicks":
        values = np.zeros(length, dtype=np.float32)
        for position in range(SAMPLE_RATE // 2, length, SAMPLE_RATE):
            values[position : min(length, position + 48)] = np.hanning(
                min(48, length - position)
            ).astype(np.float32) * 0.3
        return values
    raise ValueError(f"unknown event kind: {kind}")


def _mix(speech: Any, noise: Any, snr_db: float):
    import numpy as np

    if len(noise) < len(speech):
        noise = np.resize(noise, len(speech))
    noise = noise[: len(speech)]
    speech_rms = float(np.sqrt(np.mean(speech**2) + 1e-9))
    noise_rms = float(np.sqrt(np.mean(noise**2) + 1e-9))
    wanted_noise_rms = speech_rms / (10 ** (snr_db / 20))
    return speech + noise * (wanted_noise_rms / max(noise_rms, 1e-6))


def _speech_observation(augmentation: str) -> str:
    return {
        "clean": "One synthetic speaker with no notable background sound.",
        "room-tone": "One speaker over faint steady room tone.",
        "fan": "One speaker with a steady fan-like background sound.",
        "alarm": "One speaker with an intermittent electronic alarm in the background.",
    }[augmentation]


def build_synthetic_audio_bridge_dataset(
    output_dir: Path,
    *,
    phrase_count: int = 320,
    variants_per_phrase: int = 2,
    nonspeech_count: int = 80,
    seed: int = 42,
) -> dict[str, Any]:
    import numpy as np

    if phrase_count <= 0 or variants_per_phrase <= 0 or nonspeech_count <= 0:
        raise ValueError("dataset counts must be positive")
    destination = output_dir.expanduser().resolve()
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.jsonl"
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    voices = ("en-us", "en-gb", "en-sc", "en")
    augmentations = ("clean", "room-tone", "fan", "alarm")
    records: list[dict[str, Any]] = []

    for phrase_index, phrase in enumerate(training_phrases(phrase_count)):
        split = _split_for(phrase)
        for variant in range(variants_per_phrase):
            voice = voices[(phrase_index + variant) % len(voices)]
            speed = py_rng.randint(135, 195)
            pitch = py_rng.randint(35, 65)
            speech = _synthesize(phrase, voice=voice, speed=speed, pitch=pitch)
            augmentation = augmentations[(phrase_index + variant) % len(augmentations)]
            if augmentation != "clean":
                noise = _event(augmentation, len(speech), rng)
                speech = _mix(speech, noise, snr_db=py_rng.choice((8.0, 12.0, 18.0)))
            name = f"speech-{phrase_index:04d}-{variant}.wav"
            path = audio_dir / name
            _write_wav(path, speech)
            records.append(
                {
                    "schema": AUDIO_BRIDGE_DATASET_SCHEMA,
                    "id": path.stem,
                    "audio": str(path.relative_to(destination)),
                    "split": split,
                    "transcript": phrase,
                    "audio_observation": _speech_observation(augmentation),
                    "assistant_target": (
                        f"<speech_transcript>{phrase}</speech_transcript>"
                        f"<audio_observation>{_speech_observation(augmentation)}</audio_observation>"
                    ),
                    "current_visual_input": False,
                    "augmentation": augmentation,
                    "source": "espeak-ng plus deterministic procedural ambience",
                    "license": "generated for this release; Apache-2.0 script",
                }
            )

    event_descriptions = {
        "silence": "Silence; no intelligible speech or distinct sound event.",
        "room-tone": "Faint steady room tone; no intelligible speech.",
        "fan": "A steady fan-like mechanical hum; no intelligible speech.",
        "alarm": "An intermittent electronic alarm; no intelligible speech.",
        "clicks": "A sequence of brief clicks; no intelligible speech.",
    }
    kinds = tuple(event_descriptions)
    for index in range(nonspeech_count):
        kind = kinds[index % len(kinds)]
        duration = py_rng.uniform(1.0, 4.0)
        values = _event(kind, int(duration * SAMPLE_RATE), rng)
        path = audio_dir / f"nonspeech-{index:04d}-{kind}.wav"
        _write_wav(path, values)
        description = event_descriptions[kind]
        records.append(
            {
                "schema": AUDIO_BRIDGE_DATASET_SCHEMA,
                "id": path.stem,
                "audio": str(path.relative_to(destination)),
                "split": _split_for(path.stem),
                "transcript": "",
                "audio_observation": description,
                "assistant_target": (
                    "<speech_transcript></speech_transcript>"
                    f"<audio_observation>{description}</audio_observation>"
                ),
                "current_visual_input": False,
                "augmentation": kind,
                "source": "deterministic procedural audio",
                "license": "Apache-2.0",
            }
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    report = {
        "schema": AUDIO_BRIDGE_DATASET_SCHEMA,
        "manifest": str(manifest_path),
        "audio_dir": str(audio_dir),
        "samples": len(records),
        "split_counts": split_counts,
        "speech_samples": phrase_count * variants_per_phrase,
        "nonspeech_samples": nonspeech_count,
        "sample_rate_hz": SAMPLE_RATE,
        "seed": seed,
        "audio_only_current_visual_input": False,
    }
    (destination / "dataset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _natural_transcript(text: str) -> str:
    normalized = " ".join(text.strip().split()).lower()
    if not normalized:
        raise ValueError("LibriSpeech transcript must not be empty")
    sentence = normalized[0].upper() + normalized[1:]
    if sentence[-1] not in ".?!":
        sentence += "."
    return sentence


def _librispeech_utterances(source_dir: Path) -> list[dict[str, str]]:
    root = source_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    utterances: list[dict[str, str]] = []
    for transcript_path in sorted(root.rglob("*.trans.txt")):
        for line_number, line in enumerate(
            transcript_path.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if not line.strip():
                continue
            try:
                utterance_id, transcript = line.split(maxsplit=1)
            except ValueError as exc:
                raise ValueError(
                    f"invalid LibriSpeech transcript at {transcript_path}:{line_number}"
                ) from exc
            audio_path = transcript_path.parent / f"{utterance_id}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            speaker = utterance_id.split("-", 1)[0]
            utterances.append(
                {
                    "id": utterance_id,
                    "speaker": speaker,
                    "transcript": _natural_transcript(transcript),
                    "audio_path": str(audio_path),
                }
            )
    if not utterances:
        raise ValueError(f"no LibriSpeech transcripts found below {root}")
    return utterances


def build_librispeech_augmented_audio_bridge_dataset(
    output_dir: Path,
    *,
    base_manifest: Path,
    librispeech_dir: Path,
    max_utterances: int = 1_200,
    max_duration_seconds: float = 12.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Combine the focused synthetic corpus with licensed natural speech.

    LibriSpeech records are split by speaker so a voice never occurs in more
    than one split. Long recordings are skipped instead of truncating speech
    while retaining an impossible full-transcript target.
    """
    import numpy as np
    import soundfile as sf

    if max_utterances <= 0 or max_duration_seconds <= 0:
        raise ValueError("LibriSpeech limits must be positive")
    source_manifest = base_manifest.expanduser().resolve()
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)
    destination = output_dir.expanduser().resolve()
    manifest_path = destination / "manifest.jsonl"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to replace existing manifest: {manifest_path}")
    audio_dir = destination / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        source_manifest.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema") != AUDIO_BRIDGE_DATASET_SCHEMA:
            raise ValueError(f"base manifest line {line_number} has an invalid schema")
        source_audio = (source_manifest.parent / str(record["audio"])).resolve()
        if not source_audio.is_file():
            raise FileNotFoundError(source_audio)
        suffix = source_audio.suffix.lower() or ".wav"
        output_audio = audio_dir / f"base-{record['id']}{suffix}"
        shutil.copy2(source_audio, output_audio)
        copied = dict(record)
        copied["id"] = f"base-{record['id']}"
        copied["audio"] = str(output_audio.relative_to(destination))
        copied["current_visual_input"] = False
        records.append(copied)

    candidates = _librispeech_utterances(librispeech_dir)
    py_rng = random.Random(seed)
    py_rng.shuffle(candidates)
    rng = np.random.default_rng(seed)
    augmentations = ("clean", "clean", "room-tone", "fan", "alarm")
    accepted = 0
    skipped_long = 0
    for candidate in candidates:
        if accepted >= max_utterances:
            break
        source_audio = Path(candidate["audio_path"])
        info = sf.info(source_audio)
        if info.duration > max_duration_seconds:
            skipped_long += 1
            continue
        speech = _read_resampled(source_audio)
        augmentation = augmentations[accepted % len(augmentations)]
        if augmentation != "clean":
            speech = _mix(
                speech,
                _event(augmentation, len(speech), rng),
                snr_db=py_rng.choice((10.0, 15.0, 20.0)),
            )
        output_audio = audio_dir / f"librispeech-{candidate['id']}.wav"
        _write_wav(output_audio, speech)
        observation = {
            "clean": "One speaker with no notable background sound.",
            "room-tone": "One speaker over faint steady room tone.",
            "fan": "One speaker with a steady fan-like background sound.",
            "alarm": "One speaker with an intermittent electronic alarm in the background.",
        }[augmentation]
        transcript = candidate["transcript"]
        records.append(
            {
                "schema": AUDIO_BRIDGE_DATASET_SCHEMA,
                "id": f"librispeech-{candidate['id']}",
                "audio": str(output_audio.relative_to(destination)),
                "split": _split_for(f"librispeech-speaker:{candidate['speaker']}"),
                "transcript": transcript,
                "audio_observation": observation,
                "assistant_target": (
                    f"<speech_transcript>{transcript}</speech_transcript>"
                    f"<audio_observation>{observation}</audio_observation>"
                ),
                "current_visual_input": False,
                "augmentation": augmentation,
                "source": "LibriSpeech dev-clean with deterministic local augmentation",
                "license": "CC BY 4.0",
            }
        )
        accepted += 1
    if accepted < max_utterances:
        raise ValueError(
            f"only {accepted} LibriSpeech utterances met the duration limit; "
            f"requested {max_utterances}"
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    report = {
        "schema": AUDIO_BRIDGE_DATASET_SCHEMA,
        "manifest": str(manifest_path),
        "audio_dir": str(audio_dir),
        "samples": len(records),
        "base_samples": len(records) - accepted,
        "librispeech_samples": accepted,
        "librispeech_source": str(librispeech_dir.expanduser().resolve()),
        "librispeech_license": "CC BY 4.0",
        "max_duration_seconds": max_duration_seconds,
        "skipped_long": skipped_long,
        "split_counts": split_counts,
        "sample_rate_hz": SAMPLE_RATE,
        "split_policy": "synthetic content hash plus LibriSpeech speaker-group hash",
        "seed": seed,
        "audio_only_current_visual_input": False,
    }
    (destination / "dataset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio-bridge supervision data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--phrases", type=int, default=320)
    parser.add_argument("--variants", type=int, default=2)
    parser.add_argument("--nonspeech", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-manifest")
    parser.add_argument("--librispeech-dir")
    parser.add_argument("--librispeech-samples", type=int, default=1_200)
    parser.add_argument("--max-duration-seconds", type=float, default=12.0)
    args = parser.parse_args()
    if bool(args.base_manifest) != bool(args.librispeech_dir):
        parser.error("--base-manifest and --librispeech-dir must be provided together")
    if args.base_manifest:
        report = build_librispeech_augmented_audio_bridge_dataset(
            Path(args.out),
            base_manifest=Path(args.base_manifest),
            librispeech_dir=Path(args.librispeech_dir),
            max_utterances=args.librispeech_samples,
            max_duration_seconds=args.max_duration_seconds,
            seed=args.seed,
        )
    else:
        report = build_synthetic_audio_bridge_dataset(
            Path(args.out),
            phrase_count=args.phrases,
            variants_per_phrase=args.variants,
            nonspeech_count=args.nonspeech,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
