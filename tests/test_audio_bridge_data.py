from __future__ import annotations

import json
import wave

import numpy as np
import pytest

from training_suite.training.audio_bridge_data import (
    AUDIO_BRIDGE_DATASET_SCHEMA,
    build_librispeech_augmented_audio_bridge_dataset,
    build_synthetic_audio_bridge_dataset,
    training_phrases,
)


def test_training_phrases_cover_provenance_and_background_work() -> None:
    phrases = training_phrases(96)

    assert len(phrases) == len(set(phrases)) == 96
    assert any("camera" in phrase for phrase in phrases)
    assert any("background task" in phrase for phrase in phrases)
    assert any("sanity check" in phrase for phrase in phrases)


@pytest.mark.skipif(not __import__("shutil").which("espeak-ng"), reason="needs espeak-ng")
def test_tiny_dataset_has_audio_only_evidence_and_all_splits(tmp_path) -> None:
    pytest.importorskip("soundfile")
    pytest.importorskip("scipy")
    report = build_synthetic_audio_bridge_dataset(
        tmp_path,
        phrase_count=20,
        variants_per_phrase=1,
        nonspeech_count=15,
        seed=9,
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report["schema"] == AUDIO_BRIDGE_DATASET_SCHEMA
    assert report["samples"] == 35
    assert set(report["split_counts"]) == {"train", "validation", "test"}
    assert all(record["current_visual_input"] is False for record in records)
    assert all("<visual_observation>" not in record["assistant_target"] for record in records)
    assert any(record["transcript"] == "" for record in records)
    with wave.open(str(tmp_path / records[0]["audio"]), "rb") as handle:
        assert handle.getframerate() == 16_000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2


def test_librispeech_augmentation_groups_speakers_and_preserves_provenance(
    tmp_path,
) -> None:
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("scipy")

    base = tmp_path / "base"
    base_audio = base / "audio"
    base_audio.mkdir(parents=True)
    samples = np.zeros(8_000, dtype=np.float32)
    sf.write(base_audio / "quiet.wav", samples, 16_000)
    base_record = {
        "schema": AUDIO_BRIDGE_DATASET_SCHEMA,
        "id": "quiet",
        "audio": "audio/quiet.wav",
        "split": "train",
        "transcript": "",
        "audio_observation": "Silence.",
        "assistant_target": (
            "<speech_transcript></speech_transcript>"
            "<audio_observation>Silence.</audio_observation>"
        ),
        "current_visual_input": False,
    }
    (base / "manifest.jsonl").write_text(json.dumps(base_record) + "\n")

    libri = tmp_path / "LibriSpeech" / "dev-clean" / "100" / "1"
    libri.mkdir(parents=True)
    sf.write(libri / "100-1-0001.flac", samples, 16_000)
    sf.write(libri / "100-1-0002.flac", samples, 16_000)
    (libri / "100-1.trans.txt").write_text(
        "100-1-0001 HELLO FROM THE FIRST RECORDING\n"
        "100-1-0002 THIS IS THE SECOND RECORDING\n"
    )

    report = build_librispeech_augmented_audio_bridge_dataset(
        tmp_path / "combined",
        base_manifest=base / "manifest.jsonl",
        librispeech_dir=tmp_path / "LibriSpeech" / "dev-clean",
        max_utterances=2,
        seed=7,
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "combined" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    real = [record for record in records if record["id"].startswith("librispeech-")]

    assert report["base_samples"] == 1
    assert report["librispeech_samples"] == 2
    assert len({record["split"] for record in real}) == 1
    assert all(record["current_visual_input"] is False for record in records)
    assert all("<visual_observation>" not in record["assistant_target"] for record in records)
    assert real[0]["transcript"].endswith(".")
