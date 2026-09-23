from __future__ import annotations

import json
import wave

import pytest

from training_suite.evals.audio_bridge import comprehension_prompt_sha256
from training_suite.evals.tts_vocabulary import (
    TTS_VOCABULARY_SCHEMA,
    VOCABULARY_PROMPT_PREFIX,
    _cached_wav,
    _prepare_wav_cache,
    _resume_results,
    build_vocabulary_batches,
    evaluate_tts_vocabulary,
    normalized_words,
    vocabulary_prompt,
)
from training_suite.training.audio_bridge_data import AUDIO_BRIDGE_DATASET_SCHEMA


def _wav(path) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 160)


def test_vocabulary_batches_are_deterministic_unique_and_normalized(tmp_path) -> None:
    wav = tmp_path / "sample.wav"
    _wav(wav)
    manifest = tmp_path / "manifest.jsonl"
    records = []
    for index, transcript in enumerate(
        ("Blue river 8", "Copper lighthouse seven", "Blue meadow nine")
    ):
        records.append(
            {
                "schema": AUDIO_BRIDGE_DATASET_SCHEMA,
                "id": str(index),
                "audio": wav.name,
                "split": "test",
                "source": "test",
                "transcript": transcript,
                "assistant_target": (
                    f"<speech_transcript>{transcript}</speech_transcript>"
                    "<audio_observation></audio_observation>"
                ),
                "current_visual_input": False,
            }
        )
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    batches, frequencies = build_vocabulary_batches(manifest, chunk_words=3, seed=7)

    flattened = [word for batch in batches for word in batch]
    assert len(flattened) == len(set(flattened))
    assert set(flattened) == {
        "blue",
        "river",
        "eight",
        "copper",
        "lighthouse",
        "seven",
        "meadow",
        "nine",
    }
    assert frequencies["blue"] == 2
    assert normalized_words("River 8!") == ["river", "eight"]
    assert batches == build_vocabulary_batches(manifest, chunk_words=3, seed=7)[0]
    assert vocabulary_prompt(["copper", "river"]) == (
        "Please pronounce these words: copper, river."
    )


def test_vocabulary_checkpoint_resumes_only_an_exact_batch_prefix(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    batches = [["alpha", "beta"], ["gamma"]]
    partial = tmp_path / "report.json.partial"
    partial.write_text(
        json.dumps(
            {
                "schema": TTS_VOCABULARY_SCHEMA,
                "model": "candidate",
                "manifest": str(manifest.resolve()),
                "chunk_words": 2,
                "seed": 42,
                "limit_batches": 0,
                "total_batches": 2,
                "prompt_prefix": VOCABULARY_PROMPT_PREFIX,
                "comprehension_prompt_sha256": comprehension_prompt_sha256(),
                "elapsed_seconds": 12.5,
                "results": [
                    {
                        "batch": 1,
                        "expected_words": ["alpha", "beta"],
                        "predicted_transcript": "alpha beta",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    results, elapsed = _resume_results(
        partial,
        model="candidate",
        manifest=manifest,
        chunk_words=2,
        seed=42,
        limit_batches=0,
        batches=batches,
    )

    assert len(results) == 1
    assert elapsed == 12.5

    checkpoint = json.loads(partial.read_text(encoding="utf-8"))
    checkpoint["comprehension_prompt_sha256"] = "stale-prompt"
    partial.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ValueError, match="comprehension_prompt_sha256"):
        _resume_results(
            partial,
            model="candidate",
            manifest=manifest,
            chunk_words=2,
            seed=42,
            limit_batches=0,
            batches=batches,
        )
    checkpoint["comprehension_prompt_sha256"] = comprehension_prompt_sha256()
    partial.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(ValueError, match="seed"):
        _resume_results(
            partial,
            model="candidate",
            manifest=manifest,
            chunk_words=2,
            seed=7,
            limit_batches=0,
            batches=batches,
        )


def test_vocabulary_checkpoint_rejects_non_prefix_results(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    partial = tmp_path / "report.json.partial"
    partial.write_text(
        json.dumps(
            {
                "schema": TTS_VOCABULARY_SCHEMA,
                "model": "candidate",
                "manifest": str(manifest.resolve()),
                "chunk_words": 2,
                "seed": 42,
                "limit_batches": 0,
                "total_batches": 2,
                "prompt_prefix": VOCABULARY_PROMPT_PREFIX,
                "comprehension_prompt_sha256": comprehension_prompt_sha256(),
                "results": [
                    {
                        "batch": 2,
                        "expected_words": ["gamma"],
                        "predicted_transcript": "gamma",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="batch sequence"):
        _resume_results(
            partial,
            model="candidate",
            manifest=manifest,
            chunk_words=2,
            seed=42,
            limit_batches=0,
            batches=[["alpha", "beta"], ["gamma"]],
        )


def test_wav_cache_identity_is_exact_and_rejects_corrupt_audio(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    speaker = tmp_path / "speaker.wav"
    _wav(speaker)
    cache = _prepare_wav_cache(
        tmp_path / "cache",
        tts_endpoint="http://127.0.0.1:8892/synthesize",
        manifest=manifest,
        speaker_file=speaker,
        chunk_words=2,
        seed=42,
        limit_batches=0,
        batches=[["alpha", "beta"]],
    )
    assert cache.is_dir()
    assert (
        _prepare_wav_cache(
            cache,
            tts_endpoint="http://127.0.0.1:8892/synthesize",
            manifest=manifest,
            speaker_file=speaker,
            chunk_words=2,
            seed=42,
            limit_batches=0,
            batches=[["alpha", "beta"]],
        )
        == cache
    )

    corrupt = cache / "batch-00001.wav"
    corrupt.write_bytes(b"not a wav")
    assert _cached_wav(corrupt) is None

    with pytest.raises(ValueError, match="identity"):
        _prepare_wav_cache(
            cache,
            tts_endpoint="http://127.0.0.1:8892/synthesize",
            manifest=manifest,
            speaker_file=speaker,
            chunk_words=2,
            seed=7,
            limit_batches=0,
            batches=[["alpha", "beta"]],
        )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("limit_batches", -1, "limit batches"),
        ("max_tokens", 0, "max tokens"),
        ("minimum_vocabulary_size", 0, "vocabulary size"),
        ("minimum_vocabulary_recall", 1.1, "vocabulary recall"),
        ("minimum_rare_word_recall", -0.1, "rare-word recall"),
        ("maximum_mean_wer", -0.1, "mean WER"),
        ("minimum_tagged_output_rate", 2.0, "tagged-output"),
    ],
)
def test_vocabulary_evaluation_rejects_invalid_thresholds(
    tmp_path, argument, value, message
) -> None:
    kwargs = {
        "tts_endpoint": "http://127.0.0.1:1/synthesize",
        "comprehension_endpoint": "http://127.0.0.1:1/chat",
        "model": "candidate",
        "manifest": tmp_path / "unused.jsonl",
        "output_path": tmp_path / "report.json",
        argument: value,
    }

    with pytest.raises(ValueError, match=message):
        evaluate_tts_vocabulary(**kwargs)
