from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training_suite.training.audio_bridge_data import AUDIO_BRIDGE_DATASET_SCHEMA

REBALANCE_DATASET_SCHEMA = "robit.qwen3a-audio-bridge-nonspeech-rebalance.v1"
VALID_SPLITS = {"train", "validation", "test"}
SOURCE_REPORT_FIELDS = (
    "vocabulary_size",
    "vocabulary_sha256",
    "maximum_numeral",
    "synthesis_count",
    "base_samples",
    "recovery_samples",
    "training_repeats",
    "short_variants",
    "sample_rate_hz",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(path)


def _load_source_records(manifest: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema") != AUDIO_BRIDGE_DATASET_SCHEMA:
            raise ValueError(
                f"source manifest line {line_number} has an invalid schema"
            )
        identifier = str(record.get("id") or "")
        if not identifier or identifier in identifiers:
            raise ValueError(
                f"source manifest line {line_number} has a missing or duplicate id"
            )
        identifiers.add(identifier)
        if record.get("split") not in VALID_SPLITS:
            raise ValueError(f"source manifest line {line_number} has an invalid split")
        if record.get("current_visual_input") is not False:
            raise ValueError(
                f"source manifest line {line_number} violates audio-only visual provenance"
            )
        target = str(record.get("assistant_target") or "")
        if "<visual_observation" in target:
            raise ValueError(
                f"source manifest line {line_number} contains visual target evidence"
            )
        audio = (manifest.parent / str(record.get("audio") or "")).resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        copied = dict(record)
        copied["audio"] = str(audio)
        records.append(copied)
    if not records:
        raise ValueError("source manifest is empty")
    return records


def build_nonspeech_rebalanced_dataset(
    output_dir: Path,
    *,
    source_manifest: Path,
    additional_repeats: int = 6,
) -> dict[str, Any]:
    if additional_repeats <= 0:
        raise ValueError("additional repeats must be positive")
    source = source_manifest.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    source_records = _load_source_records(source)
    train_nonspeech = [
        record
        for record in source_records
        if record["split"] == "train"
        and not str(record.get("transcript") or "").strip()
    ]
    if not train_nonspeech:
        raise ValueError("source manifest has no train-split no-speech records")

    added: list[dict[str, Any]] = []
    for record in train_nonspeech:
        for repeat in range(1, additional_repeats + 1):
            duplicate = dict(record)
            duplicate["id"] = f"{record['id']}-nonspeech-repeat-{repeat:02d}"
            duplicate["rebalance_parent_id"] = record["id"]
            duplicate["rebalance_repeat"] = repeat
            added.append(duplicate)
    records = source_records + added

    manifest = destination / "manifest.jsonl"
    partial = manifest.with_name(manifest.name + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    partial.replace(manifest)

    source_report_path = source.parent / "dataset_report.json"
    source_report = (
        json.loads(source_report_path.read_text(encoding="utf-8"))
        if source_report_path.is_file()
        else {}
    )
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "validation", "test")
    }
    report = {
        "schema": REBALANCE_DATASET_SCHEMA,
        "manifest": str(manifest),
        "source_manifest": str(source),
        "source_manifest_sha256": _sha256(source),
        "source_dataset_schema": source_report.get("schema"),
        **{
            key: source_report[key]
            for key in SOURCE_REPORT_FIELDS
            if key in source_report
        },
        "source_samples": len(source_records),
        "samples": len(records),
        "split_counts": split_counts,
        "nonspeech_additional_repeats": additional_repeats,
        "nonspeech_source_samples": len(train_nonspeech),
        "nonspeech_added_samples": len(added),
        "train_nonspeech_samples": len(train_nonspeech) + len(added),
        "validation_test_unchanged": True,
        "audio_only_current_visual_input": False,
    }
    _atomic_json(destination / "dataset_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebalance train-only no-speech audio-bridge evidence"
    )
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--additional-repeats", type=int, default=6)
    args = parser.parse_args()
    report = build_nonspeech_rebalanced_dataset(
        args.out,
        source_manifest=args.source_manifest,
        additional_repeats=args.additional_repeats,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
