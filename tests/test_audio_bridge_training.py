from __future__ import annotations

import json
import wave

import pytest

from training_suite.training.audio_bridge_data import AUDIO_BRIDGE_DATASET_SCHEMA
from training_suite.training.audio_bridge_training import (
    MEDIA_SUFFIX_PROMPT,
    MEDIA_SYSTEM_PROMPT,
    AudioBridgeCollator,
    AudioBridgeTrainingConfig,
    _dataset_provenance,
    _select_validation_records,
    _signal_ready,
    load_audio_bridge_records,
)


def test_training_prompt_matches_no_thinking_generation_prefill() -> None:
    assert MEDIA_SUFFIX_PROMPT.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert "plain text inside both elements" in MEDIA_SYSTEM_PROMPT
    assert "never emit nested markup" in MEDIA_SUFFIX_PROMPT
    assert "never call a click a camera shutter" in MEDIA_SYSTEM_PROMPT
    assert "never call a click a camera shutter" in MEDIA_SUFFIX_PROMPT


def _wav(path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 320)


def test_manifest_loader_rejects_visual_provenance(tmp_path) -> None:
    wav = tmp_path / "sample.wav"
    _wav(wav)
    record = {
        "schema": AUDIO_BRIDGE_DATASET_SCHEMA,
        "id": "bad",
        "audio": "sample.wav",
        "split": "train",
        "current_visual_input": True,
        "assistant_target": "<visual_observation>invented</visual_observation>",
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="visual provenance"):
        load_audio_bridge_records(manifest)


def test_training_config_rejects_missing_self_check_interval() -> None:
    with pytest.raises(ValueError, match="positive"):
        AudioBridgeTrainingConfig(
            target_source="target",
            omni_projector="projector",
            initial_checkpoint="checkpoint",
            dataset_manifest="manifest",
            output_dir="out",
            check_interval=0,
        )


def test_validation_sample_balances_sources_and_no_speech() -> None:
    records = [
        {"id": f"synthetic-{index}", "source": "synthetic", "transcript": "speech"}
        for index in range(20)
    ]
    records += [
        {"id": f"natural-{index}", "source": "natural", "transcript": "speech"}
        for index in range(8)
    ]
    records += [
        {"id": f"noise-{index}", "source": "procedural", "transcript": ""}
        for index in range(4)
    ]

    selected = _select_validation_records(records, limit=9, seed=7)

    assert len(selected) == 9
    assert {record["source"] for record in selected} == {
        "synthetic",
        "natural",
        "procedural",
    }
    assert sum(not record["transcript"] for record in selected) == 3
    assert [record["id"] for record in selected] == [
        record["id"]
        for record in _select_validation_records(records, limit=9, seed=7)
    ]


def test_collator_supervises_only_the_assistant_target(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    wav = tmp_path / "sample.wav"
    _wav(wav)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(char) % 97 + 3 for char in text]

    class FeatureExtractor:
        def __call__(self, values, **_kwargs):
            return {
                "input_features": torch.zeros(len(values), 128, 4),
                "attention_mask": torch.ones(len(values), 4, dtype=torch.long),
            }

    collator = AudioBridgeCollator(FeatureExtractor(), Tokenizer())
    batch = collator(
        [
            {
                "audio_path": str(wav),
                "assistant_target": (
                    "<speech_transcript>hello</speech_transcript>"
                    "<audio_observation>quiet</audio_observation>"
                ),
            }
        ]
    )

    labels = batch["suffix_labels"][0]
    assert torch.all(labels[: len(collator.suffix_prompt_ids)] == -100)
    assert torch.all(labels[len(collator.suffix_prompt_ids) :] >= 0)


def test_ready_signal_is_atomic_machine_readable_evidence(tmp_path) -> None:
    path = tmp_path / "ready.json"

    _signal_ready(str(path), model="standard-ornith", device="cuda")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "robit.qwen3a-audio-bridge-training.v1"
    assert payload["status"] == "models-resident"
    assert payload["model"] == "standard-ornith"
    assert not path.with_name("ready.json.partial").exists()


def test_dataset_provenance_is_public_allowlisted(tmp_path) -> None:
    (tmp_path / "dataset_report.json").write_text(
        json.dumps(
            {
                "schema": "recovery.v1",
                "vocabulary_size": 8321,
                "maximum_numeral": 1000,
                "speaker_file": "/private/speaker.wav",
                "tts_endpoint": "http://127.0.0.1:18892/synthesize",
            }
        ),
        encoding="utf-8",
    )

    provenance = _dataset_provenance(tmp_path / "manifest.jsonl")

    assert provenance == {
        "schema": "recovery.v1",
        "vocabulary_size": 8321,
        "maximum_numeral": 1000,
    }
