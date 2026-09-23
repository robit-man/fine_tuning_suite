from __future__ import annotations

import json
import wave

import pytest

from training_suite.training.audio_bridge_data import AUDIO_BRIDGE_DATASET_SCHEMA
from training_suite.training.audio_bridge_rebalance import (
    REBALANCE_DATASET_SCHEMA,
    build_nonspeech_rebalanced_dataset,
)


def _wav(path) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)


def _record(identifier: str, split: str, transcript: str) -> dict[str, object]:
    return {
        "schema": AUDIO_BRIDGE_DATASET_SCHEMA,
        "id": identifier,
        "audio": "sample.wav",
        "split": split,
        "source": "fixture",
        "transcript": transcript,
        "assistant_target": (
            f"<speech_transcript>{transcript}</speech_transcript>"
            "<audio_observation>Quiet room tone.</audio_observation>"
        ),
        "current_visual_input": False,
    }


def _source_manifest(tmp_path, records) -> object:
    _wav(tmp_path / "sample.wav")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (tmp_path / "dataset_report.json").write_text(
        json.dumps(
            {
                "schema": "recovery.v1",
                "vocabulary_size": 8321,
                "vocabulary_sha256": "abc123",
                "synthesis_count": 482,
                "speaker_file": "/private/speaker.wav",
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_rebalance_duplicates_only_train_nonspeech_and_preserves_provenance(
    tmp_path,
) -> None:
    manifest = _source_manifest(
        tmp_path,
        [
            _record("train-noise", "train", ""),
            _record("train-speech", "train", "hello"),
            _record("validation-noise", "validation", ""),
            _record("test-noise", "test", ""),
        ],
    )

    report = build_nonspeech_rebalanced_dataset(
        tmp_path / "rebalanced",
        source_manifest=manifest,
        additional_repeats=2,
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "rebalanced" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert report["schema"] == REBALANCE_DATASET_SCHEMA
    assert report["source_samples"] == 4
    assert report["nonspeech_source_samples"] == 1
    assert report["nonspeech_added_samples"] == 2
    assert report["train_nonspeech_samples"] == 3
    assert report["split_counts"] == {"train": 4, "validation": 1, "test": 1}
    assert report["vocabulary_size"] == 8321
    assert "speaker_file" not in report
    assert len(records) == 6
    assert len({record["id"] for record in records}) == 6
    assert all(record["current_visual_input"] is False for record in records)
    assert all(str(record["audio"]).startswith("/") for record in records)
    assert sum(record["id"].startswith("validation-noise") for record in records) == 1
    assert sum(record["id"].startswith("test-noise") for record in records) == 1


def test_rebalance_rejects_nonpositive_repeats(tmp_path) -> None:
    with pytest.raises(ValueError, match="positive"):
        build_nonspeech_rebalanced_dataset(
            tmp_path / "out",
            source_manifest=tmp_path / "missing.jsonl",
            additional_repeats=0,
        )


def test_rebalance_rejects_visual_or_duplicate_source_records(tmp_path) -> None:
    visual = _record("visual", "train", "")
    visual["current_visual_input"] = True
    manifest = _source_manifest(tmp_path, [visual])
    with pytest.raises(ValueError, match="visual provenance"):
        build_nonspeech_rebalanced_dataset(
            tmp_path / "visual-out", source_manifest=manifest
        )

    duplicate = _record("same", "train", "")
    manifest.write_text(
        json.dumps(duplicate) + "\n" + json.dumps(duplicate) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate id"):
        build_nonspeech_rebalanced_dataset(
            tmp_path / "duplicate-out", source_manifest=manifest
        )
