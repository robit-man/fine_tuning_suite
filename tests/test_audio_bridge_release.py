from __future__ import annotations

import hashlib
import json

import pytest

from training_suite.models.audio_bridge_release import (
    AudioBridgeReleaseError,
    AudioBridgeReleaseSpec,
    build_audio_bridge_release,
)


def _json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _passed_release_reports(tmp_path) -> dict[str, object]:
    values = {
        "training_report": {
            "schema": "training",
            "status": "complete",
            "best_step": 25,
            "initial_validation_loss": 3.0,
            "best_validation_loss": 1.5,
            "trainable_parameters": 10,
            "frozen_language_parameters": 100,
            "frozen_audio_parameters": 50,
        },
        "evaluation_report": {
            "schema": "evaluation",
            "passed": True,
            "thresholds": {},
            "metrics": {
                "samples": 3,
                "speech_wer": 0.1,
                "empty_audio_false_transcript_rate": 0.0,
                "tagged_output_rate": 1.0,
                "visual_claims": 0,
            },
            "gates": {"speech": True, "visual": True},
        },
        "vision_evaluation_report": {
            "schema": "vision-evaluation",
            "passed": True,
            "sequence": ["red", "blue", "red"],
            "metrics": {
                "samples": 3,
                "correct": 3,
                "tagged_output_rate": 1.0,
                "stale_media_failures": 0,
            },
            "gates": {
                "red_blue_red_sequence": True,
                "exact_tagged_output": True,
                "stale_media_failures": True,
                "cache_prompt_disabled": True,
            },
        },
        "capability_report": {"ok": True, "missing": []},
        "tool_smoke_report": {
            "ok": True,
            "tool_calls": [{"function": {"name": "get_weather"}}],
            "raw": {"message": {"thinking": "I should call the weather tool."}},
        },
        "tts_alignment_report": {
            "schema": "tts-alignment",
            "passed": True,
            "maximum_wer": 0.34,
            "minimum_compact_similarity": 0.9,
            "metrics": {
                "samples": 3,
                "mean_word_error_rate": 0.1,
                "valid_wavs": 3,
            },
            "gates": {
                "a_b_a_sequence": True,
                "exact_tagged_output": True,
                "wav_contract": True,
                "no_one_turn_lag": True,
            },
        },
        "tts_vocabulary_report": {
            "schema": "tts-vocabulary",
            "passed": True,
            "thresholds": {"minimum_vocabulary_size": 4_000},
            "metrics": {
                "batches": 150,
                "expected_unique_words": 4_500,
                "observed_unique_words": 3_600,
                "vocabulary_recall": 0.8,
                "rare_unique_words": 1_000,
                "observed_rare_words": 700,
                "rare_word_recall": 0.7,
                "mean_word_error_rate": 0.2,
                "tagged_output_rate": 1.0,
                "valid_wavs": 150,
            },
            "gates": {
                "vocabulary_size": True,
                "vocabulary_recall": True,
                "rare_word_recall": True,
                "mean_word_error_rate": True,
                "exact_tagged_output": True,
                "wav_contract": True,
            },
        },
    }
    paths = {}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        _json(path, value)
        paths[name] = path
    vocabulary_bytes = paths["tts_vocabulary_report"].read_bytes()
    audit = {
        "schema": "strict-audit",
        "passed": True,
        "batches": 150,
        "expected_unique_words": 4_500,
        "empty_transcript_batches": [],
        "invalid_tag_batches": [],
        "invalid_wav_batches": [],
        "comprehension_prompt_sha256": "policy-digest",
        "source_report_sha256": hashlib.sha256(vocabulary_bytes).hexdigest(),
    }
    path = tmp_path / "tts_vocabulary_audit_report.json"
    _json(path, audit)
    paths["tts_vocabulary_audit_report"] = path
    return paths


def _release_kwargs(tmp_path, **overrides):
    reports = _passed_release_reports(tmp_path)
    values = {
        "spec": AudioBridgeReleaseSpec(
            display_name="Test Omni Audio Bridge",
            base_model="owner/base",
            ollama_tag="owner/base-omni-audio-bridge:q4km",
            prior_bundle_bytes=100,
        ),
        "language_model_gguf": tmp_path / "model.gguf",
        "language_filename": "model-q4_k_m.gguf",
        "combined_projector_gguf": tmp_path / "projector.gguf",
        "projector_filename": "mmproj-audio-bridge-bf16.gguf",
        "tts_sidecar_gguf": tmp_path / "sidecar.gguf",
        "sidecar_filename": "tts-sidecar.gguf",
        **reports,
        "output_dir": tmp_path / "release",
    }
    values.update(overrides)
    return values


def test_release_records_reduced_weight_set_and_honest_memory_scope(tmp_path) -> None:
    kwargs = _release_kwargs(tmp_path)
    artifacts = [
        kwargs["language_model_gguf"],
        kwargs["combined_projector_gguf"],
        kwargs["tts_sidecar_gguf"],
    ]
    for path, payload in zip(artifacts, (b"model", b"mmproj", b"tts"), strict=True):
        path.write_bytes(payload)

    report = build_audio_bridge_release(**kwargs)

    manifest = json.loads((tmp_path / "release/release-manifest.json").read_text())
    card = (tmp_path / "release/README.md").read_text()
    assert report["weight_set_bytes"] == sum(path.stat().st_size for path in artifacts)
    assert manifest["memory"]["jetson_peak_measured"] is False
    assert manifest["architecture"]["language_trunk_copies"] == 1
    assert manifest["tts_vocabulary_evaluation"]["metrics"]["expected_unique_words"] == 4_500
    assert manifest["tts_vocabulary_strict_audit"]["empty_transcript_batches"] == []
    assert manifest["training"]["selected_step"] == 25
    assert manifest["tool_evaluation"]["thinking_channel_present"] is True
    assert "audio-bridge" in card
    assert "not a process-peak claim" in card
    assert "Unique words exercised: 4,500" in card
    assert "Empty transcripts in strict audit: 0" in card


def test_release_binds_behavior_selected_checkpoint(tmp_path) -> None:
    kwargs = _release_kwargs(tmp_path)
    artifacts = [
        kwargs["language_model_gguf"],
        kwargs["combined_projector_gguf"],
        kwargs["tts_sidecar_gguf"],
    ]
    for path, payload in zip(artifacts, (b"model", b"mmproj", b"tts"), strict=True):
        path.write_bytes(payload)
    projector_sha = hashlib.sha256(b"mmproj").hexdigest()
    selection = tmp_path / "checkpoint-selection.json"
    _json(
        selection,
        {
            "schema": "selection",
            "selected_step": 20,
            "selected_projector_sha256": projector_sha,
            "byte_identical_to_tested_projector": True,
            "comprehension_prompt_sha256": "policy-digest",
            "candidates": {"20": {"selected": True}},
        },
    )
    kwargs["checkpoint_selection_report"] = selection

    build_audio_bridge_release(**kwargs)

    manifest = json.loads((tmp_path / "release/release-manifest.json").read_text())
    assert manifest["training"]["best_step"] == 25
    assert manifest["training"]["selected_step"] == 20
    assert manifest["checkpoint_selection"]["byte_identical_to_tested_projector"]


def test_release_refuses_checkpoint_selection_for_another_projector(tmp_path) -> None:
    kwargs = _release_kwargs(tmp_path)
    artifacts = [
        kwargs["language_model_gguf"],
        kwargs["combined_projector_gguf"],
        kwargs["tts_sidecar_gguf"],
    ]
    for path, payload in zip(artifacts, (b"model", b"mmproj", b"tts"), strict=True):
        path.write_bytes(payload)
    selection = tmp_path / "checkpoint-selection.json"
    _json(
        selection,
        {
            "selected_step": 20,
            "selected_projector_sha256": "0" * 64,
            "byte_identical_to_tested_projector": True,
            "comprehension_prompt_sha256": "policy-digest",
            "candidates": {"20": {"selected": True}},
        },
    )
    kwargs["checkpoint_selection_report"] = selection

    with pytest.raises(AudioBridgeReleaseError, match="release projector"):
        build_audio_bridge_release(**kwargs)


def test_release_refuses_strict_vocabulary_audit_with_empty_transcript(
    tmp_path,
) -> None:
    kwargs = _release_kwargs(tmp_path)
    audit_path = kwargs["tts_vocabulary_audit_report"]
    audit = json.loads(audit_path.read_text())
    audit["empty_transcript_batches"] = [81]
    _json(audit_path, audit)

    with pytest.raises(AudioBridgeReleaseError, match="contains failures"):
        build_audio_bridge_release(**kwargs)


def test_release_refuses_strict_audit_for_another_vocabulary_report(tmp_path) -> None:
    kwargs = _release_kwargs(tmp_path)
    audit_path = kwargs["tts_vocabulary_audit_report"]
    audit = json.loads(audit_path.read_text())
    audit["source_report_sha256"] = "0" * 64
    _json(audit_path, audit)

    with pytest.raises(AudioBridgeReleaseError, match="does not bind"):
        build_audio_bridge_release(**kwargs)


def test_release_refuses_failed_evaluation(tmp_path) -> None:
    kwargs = _release_kwargs(tmp_path)
    evaluation = kwargs["evaluation_report"]
    _json(evaluation, {"passed": False, "gates": {"audio": False}})

    with pytest.raises(AudioBridgeReleaseError, match="did not pass"):
        build_audio_bridge_release(**kwargs)


def test_release_refuses_failed_native_vision_gate(tmp_path) -> None:
    kwargs = _release_kwargs(tmp_path)
    vision_evaluation = kwargs["vision_evaluation_report"]
    _json(
        vision_evaluation,
        {
            "passed": False,
            "gates": {"red_blue_red_sequence": False},
        },
    )

    with pytest.raises(AudioBridgeReleaseError, match="native vision"):
        build_audio_bridge_release(**kwargs)


@pytest.mark.parametrize(
    ("report_name", "mutate", "message"),
    [
        ("capability_report", {"ok": False, "missing": ["tools"]}, "capability"),
        (
            "tool_smoke_report",
            {"ok": True, "tool_calls": [], "raw": {"message": {"thinking": "x"}}},
            "tool smoke",
        ),
        (
            "tool_smoke_report",
            {"ok": True, "tool_calls": [{}], "raw": {"message": {}}},
            "thinking-channel",
        ),
        ("tts_alignment_report", {"passed": False}, "A-B-A"),
        ("tts_vocabulary_report", {"passed": False}, "vocabulary"),
    ],
)
def test_release_refuses_failed_extended_gate(
    tmp_path, report_name, mutate, message
) -> None:
    kwargs = _release_kwargs(tmp_path)
    _json(kwargs[report_name], mutate)

    with pytest.raises(AudioBridgeReleaseError, match=message):
        build_audio_bridge_release(**kwargs)
