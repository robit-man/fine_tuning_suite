from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

RELEASE_SCHEMA = "robit.ollama-audio-bridge-release.v1"


class AudioBridgeReleaseError(RuntimeError):
    """Raised when release evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class AudioBridgeReleaseSpec:
    display_name: str
    base_model: str
    ollama_tag: str
    classifier: str = "audio-bridge"
    quantization: str = "Q4_K_M language; BF16 final audio projection"
    prior_bundle_bytes: int = 0
    license_id: str = "other"
    license_name: str | None = None
    component_models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.display_name or not self.base_model or not self.ollama_tag:
            raise ValueError("release name, base model, and Ollama tag are required")
        if self.classifier != "audio-bridge":
            raise ValueError("release classifier must be audio-bridge")
        if self.prior_bundle_bytes <= 0:
            raise ValueError("prior bundle size must be positive")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def _artifact(path: Path, filename: str) -> dict[str, Any]:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not filename or Path(filename).name != filename:
        raise ValueError("release artifact filename must be a plain basename")
    return {
        "filename": filename,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _gib(value: int) -> str:
    return f"{value / (1024**3):.2f} GiB"


def _render_card(manifest: dict[str, Any]) -> str:
    spec = manifest["release"]
    evaluation = manifest["evaluation"]
    memory = manifest["memory"]
    artifacts = manifest["artifacts"]
    rows = "\n".join(
        f"| `{artifact['filename']}` | {_gib(artifact['size_bytes'])} | "
        f"`{artifact['sha256']}` |"
        for artifact in artifacts.values()
    )
    metrics = evaluation["metrics"]
    vision = manifest["vision_evaluation"]
    vision_metrics = vision["metrics"]
    capability = manifest["capability_evaluation"]
    tools = manifest["tool_evaluation"]
    alignment = manifest["tts_alignment_evaluation"]
    vocabulary = manifest["tts_vocabulary_evaluation"]
    vocabulary_audit = manifest["tts_vocabulary_strict_audit"]
    vocabulary_metrics = vocabulary["metrics"]
    training = manifest["training"]
    training_dataset = training.get("dataset") or {}
    training_provenance = training.get("dataset_provenance") or {}
    recovery_summary = ""
    if training_provenance.get("vocabulary_size"):
        recovery_summary = (
            f"- Independent recovery vocabulary: "
            f"{training_provenance['vocabulary_size']:,} words, plus systematic "
            f"numerals through {training_provenance.get('maximum_numeral', 0):,}\n"
            f"- Synthesized recovery utterances: "
            f"{training_provenance.get('synthesis_count', 0):,}\n"
        )
    component_models = spec["component_models"] or [spec["base_model"]]
    base_models = "\n".join(f"  - {model}" for model in component_models)
    license_name = (
        f"license_name: {spec['license_name']}\n" if spec["license_name"] else ""
    )
    return f"""---
license: {spec["license_id"]}
{license_name}base_model:
{base_models}
tags:
- gguf
- ollama
- multimodal
- audio
- automatic-speech-recognition
- text-to-speech
- tool-use
- audio-bridge
---

# {spec["display_name"]}

This is an `{spec["classifier"]}` release for the
[`qwen-omni-adapters`](https://github.com/robit-man/qwen-omni-adapters)
runtime. It keeps `{spec["base_model"]}` as the sole language trunk, retains
the target's native vision path, reuses the frozen Qwen3-Omni audio tower, and
replaces only the final 1,280-wide audio projection with the trained bridge.
Qwen3-TTS remains an independently executed output graph in the lightweight
sidecar.

The release combines multiple upstream components. Review each component's
license and acceptable-use terms before redistribution or deployment.

## Reduced operational weight set

The three deployable weight artifacts total **{memory["weight_set_bytes"]:,}
bytes ({memory["weight_set_gib"]:.2f} GiB)**. The previous full-router Ollama
bundle occupied {memory["prior_bundle_bytes"]:,} bytes
({_gib(memory["prior_bundle_bytes"])}); the release weight set is
**{memory["reduction_percent"]:.1f}% smaller**.

This is an exact artifact/resident-weight comparison, not a process-peak claim.
KV cache, graph workspaces, CUDA allocations, the OS, and the portal also use
memory. A 32 GB Jetson no-eviction claim requires a measured production run on
that device and is intentionally not inferred from file sizes.

| Artifact | Size | SHA-256 |
|---|---:|---|
{rows}

## Use

The preferred distribution is the logical Ollama tag:

```bash
ollama pull {spec["ollama_tag"]}
git clone https://github.com/robit-man/qwen-omni-adapters.git
cd qwen-omni-adapters
./scripts/bootstrap.sh
.venv/bin/qwen-omni doctor --deployment
```

Stock Ollama executes the language, native image-vision, tool, and thinking
paths. Audio/video comprehension and speech output require the linked adapter
runtime. The Hugging Face repository contains a standard language GGUF, a
standard combined native-vision/Omni-audio projector, and a custom TTS sidecar;
the three files are a coordinated release, not one directly loadable
single-architecture GGUF.

## Held-out audio gate

- Training samples: {training_dataset.get("total", 0):,}
{recovery_summary}- Trainable bridge parameters: {training.get("trainable_parameters", 0):,}
- Best validation-loss checkpoint: step {training.get("best_step")}
- Released behavior-gated checkpoint: step {training.get("selected_step")}
- Samples: {metrics["samples"]}
- Speech WER: {metrics["speech_wer"]:.4f}
- No-speech false-transcript rate: {metrics["empty_audio_false_transcript_rate"]:.4f}
- Exact tagged-output rate: {metrics["tagged_output_rate"]:.4f}
- Visual claims from audio-only input: {metrics["visual_claims"]}
- All configured gates passed: **{str(evaluation["passed"]).lower()}**

Audio evidence is emitted separately as `<speech_transcript>` and
`<audio_observation>`. Audio-only input never sets current visual provenance.

## Native vision gate

- Red -> blue -> red correct: **{str(vision["gates"]["red_blue_red_sequence"]).lower()}**
- Exact visual tagged-output rate: {vision_metrics["tagged_output_rate"]:.4f}
- Stale-media failures: {vision_metrics["stale_media_failures"]}
- `cache_prompt:false` exercised: **{str(vision["gates"]["cache_prompt_disabled"]).lower()}**
- All configured gates passed: **{str(vision["passed"]).lower()}**

## Language capability and tool gate

- Required advertised capabilities present: **{str(capability["passed"]).lower()}**
- Structured tool calls returned: {tools["tool_calls"]}
- Separate thinking channel returned: **{str(tools["thinking_channel_present"]).lower()}**
- All configured gates passed: **{str(tools["passed"]).lower()}**

## TTS state and large-vocabulary gates

- A -> B -> A state-reset samples: {alignment["metrics"]["samples"]}
- Mean A -> B -> A transcription WER: {alignment["metrics"]["mean_word_error_rate"]:.4f}
- One-turn lag failures: {0 if alignment["gates"]["no_one_turn_lag"] else 1}
- Unique words exercised: {vocabulary_metrics["expected_unique_words"]:,}
- Unique-word recall: {vocabulary_metrics["vocabulary_recall"]:.4f}
- Rare-word recall: {vocabulary_metrics["rare_word_recall"]:.4f}
- Mean vocabulary-batch WER: {vocabulary_metrics["mean_word_error_rate"]:.4f}
- Valid 24 kHz mono PCM16 WAVs: {vocabulary_metrics["valid_wavs"]}/{vocabulary_metrics["batches"]}
- Empty transcripts in strict audit: {len(vocabulary_audit["empty_transcript_batches"])}
- Comprehension-policy SHA-256: `{vocabulary_audit["comprehension_prompt_sha256"]}`
- All configured gates passed: **{str(vocabulary["passed"]).lower()}**

## Runtime notes

- Ollama tag: `{spec["ollama_tag"]}`
- Quantization: {spec["quantization"]}
- Language-trunk copies: 1
- Omni Thinker included: no
- Weight-free `ngram-simple` speculative decoding is supported by the runtime,
  but should remain benchmark-controlled on the target device.
- The TTS sidecar is a valid namespaced GGUF container, not a stock Ollama
  `FROM` target. Use the linked runtime to materialize its executable views.

See `release-manifest.json` for exact evidence, thresholds, and full digests.
"""


def build_audio_bridge_release(
    *,
    spec: AudioBridgeReleaseSpec,
    language_model_gguf: Path,
    language_filename: str,
    combined_projector_gguf: Path,
    projector_filename: str,
    tts_sidecar_gguf: Path,
    sidecar_filename: str,
    training_report: Path,
    evaluation_report: Path,
    vision_evaluation_report: Path,
    capability_report: Path,
    tool_smoke_report: Path,
    tts_alignment_report: Path,
    tts_vocabulary_report: Path,
    tts_vocabulary_audit_report: Path,
    output_dir: Path,
    checkpoint_selection_report: Path | None = None,
) -> dict[str, Any]:
    """Write publishable release evidence only after training and eval pass."""
    training = _read_json(training_report)
    evaluation = _read_json(evaluation_report)
    vision_evaluation = _read_json(vision_evaluation_report)
    capability = _read_json(capability_report)
    tools = _read_json(tool_smoke_report)
    tts_alignment = _read_json(tts_alignment_report)
    tts_vocabulary = _read_json(tts_vocabulary_report)
    tts_vocabulary_audit = _read_json(tts_vocabulary_audit_report)
    checkpoint_selection = (
        _read_json(checkpoint_selection_report)
        if checkpoint_selection_report is not None
        else None
    )
    if training.get("status") != "complete":
        raise AudioBridgeReleaseError("training report is not complete")
    if evaluation.get("passed") is not True:
        raise AudioBridgeReleaseError("audio evaluation gates did not pass")
    if not evaluation.get("gates") or not all(evaluation["gates"].values()):
        raise AudioBridgeReleaseError("audio evaluation gate evidence is incomplete")
    if vision_evaluation.get("passed") is not True:
        raise AudioBridgeReleaseError("native vision evaluation gates did not pass")
    if not vision_evaluation.get("gates") or not all(
        vision_evaluation["gates"].values()
    ):
        raise AudioBridgeReleaseError("native vision gate evidence is incomplete")
    if capability.get("ok") is not True or capability.get("missing"):
        raise AudioBridgeReleaseError("required model capability gates did not pass")
    tool_calls = tools.get("tool_calls") or []
    thinking = str(((tools.get("raw") or {}).get("message") or {}).get("thinking") or "")
    if tools.get("ok") is not True or not tool_calls:
        raise AudioBridgeReleaseError("structured tool smoke gate did not pass")
    if not thinking.strip():
        raise AudioBridgeReleaseError("separate thinking-channel gate did not pass")
    if tts_alignment.get("passed") is not True:
        raise AudioBridgeReleaseError("TTS A-B-A alignment gates did not pass")
    if not tts_alignment.get("gates") or not all(tts_alignment["gates"].values()):
        raise AudioBridgeReleaseError("TTS A-B-A gate evidence is incomplete")
    if tts_vocabulary.get("passed") is not True:
        raise AudioBridgeReleaseError("TTS vocabulary gates did not pass")
    if not tts_vocabulary.get("gates") or not all(tts_vocabulary["gates"].values()):
        raise AudioBridgeReleaseError("TTS vocabulary gate evidence is incomplete")
    if tts_vocabulary_audit.get("passed") is not True:
        raise AudioBridgeReleaseError("strict TTS vocabulary audit did not pass")
    for field in (
        "empty_transcript_batches",
        "invalid_tag_batches",
        "invalid_wav_batches",
    ):
        if tts_vocabulary_audit.get(field):
            raise AudioBridgeReleaseError(
                f"strict TTS vocabulary audit contains failures: {field}"
            )
    vocabulary_sha256 = _sha256(tts_vocabulary_report)
    if tts_vocabulary_audit.get("source_report_sha256") != vocabulary_sha256:
        raise AudioBridgeReleaseError(
            "strict TTS vocabulary audit does not bind the vocabulary report"
        )
    vocabulary_metrics = tts_vocabulary["metrics"]
    if tts_vocabulary_audit.get("batches") != vocabulary_metrics.get("batches"):
        raise AudioBridgeReleaseError(
            "strict TTS vocabulary audit batch count does not match"
        )
    if (
        tts_vocabulary_audit.get("expected_unique_words")
        != vocabulary_metrics.get("expected_unique_words")
    ):
        raise AudioBridgeReleaseError(
            "strict TTS vocabulary audit vocabulary size does not match"
        )
    prompt_digests = {
        report.get("comprehension_prompt_sha256")
        for report in (
            evaluation,
            tts_alignment,
            tts_vocabulary,
            tts_vocabulary_audit,
        )
        if report.get("comprehension_prompt_sha256")
    }
    if len(prompt_digests) != 1:
        raise AudioBridgeReleaseError(
            "release evaluations do not share one comprehension-policy digest"
        )

    artifacts = {
        "language_model": _artifact(language_model_gguf, language_filename),
        "combined_projector": _artifact(
            combined_projector_gguf,
            projector_filename,
        ),
        "tts_sidecar": _artifact(tts_sidecar_gguf, sidecar_filename),
    }
    selected_step = training.get("best_step")
    checkpoint_selection_evidence = None
    if checkpoint_selection is not None:
        if checkpoint_selection.get("byte_identical_to_tested_projector") is not True:
            raise AudioBridgeReleaseError(
                "selected checkpoint was not proven byte-identical to the tested projector"
            )
        if (
            checkpoint_selection.get("selected_projector_sha256")
            != artifacts["combined_projector"]["sha256"]
        ):
            raise AudioBridgeReleaseError(
                "checkpoint selection does not bind the release projector"
            )
        selected_step = checkpoint_selection.get("selected_step")
        if not isinstance(selected_step, int) or selected_step <= 0:
            raise AudioBridgeReleaseError("checkpoint selection has no valid selected step")
        selected_candidate = (checkpoint_selection.get("candidates") or {}).get(
            str(selected_step)
        )
        if not selected_candidate or selected_candidate.get("selected") is not True:
            raise AudioBridgeReleaseError(
                "checkpoint selection does not mark the released step as selected"
            )
        if (
            checkpoint_selection.get("comprehension_prompt_sha256")
            not in prompt_digests
        ):
            raise AudioBridgeReleaseError(
                "checkpoint selection policy digest does not match release evaluations"
            )
        checkpoint_selection_evidence = {
            "schema": checkpoint_selection.get("schema"),
            "selected_step": selected_step,
            "byte_identical_to_tested_projector": True,
            "evidence_sha256": _sha256(checkpoint_selection_report),
        }
    weight_set_bytes = sum(item["size_bytes"] for item in artifacts.values())
    if weight_set_bytes >= spec.prior_bundle_bytes:
        raise AudioBridgeReleaseError("release does not reduce the prior bundle size")
    manifest = {
        "schema": RELEASE_SCHEMA,
        "release": asdict(spec),
        "architecture": {
            "language_trunk_copies": 1,
            "omni_thinker_included": False,
            "native_target_vision": True,
            "audio_bridge_input_width": 1280,
            "tts_is_independent": True,
        },
        "artifacts": artifacts,
        "memory": {
            "weight_set_bytes": weight_set_bytes,
            "weight_set_gib": weight_set_bytes / (1024**3),
            "prior_bundle_bytes": spec.prior_bundle_bytes,
            "reduction_bytes": spec.prior_bundle_bytes - weight_set_bytes,
            "reduction_percent": (
                (spec.prior_bundle_bytes - weight_set_bytes)
                / spec.prior_bundle_bytes
                * 100
            ),
            "measurement_scope": "artifact/resident weights only",
            "jetson_peak_measured": False,
        },
        "training": {
            "schema": training.get("schema"),
            "best_step": training.get("best_step"),
            "selected_step": selected_step,
            "initial_validation_loss": training.get("initial_validation_loss"),
            "best_validation_loss": training.get("best_validation_loss"),
            "validation_sample": training.get("validation_sample"),
            "trainable_parameters": training.get("trainable_parameters"),
            "frozen_language_parameters": training.get("frozen_language_parameters"),
            "frozen_audio_parameters": training.get("frozen_audio_parameters"),
            "dataset": training.get("dataset"),
            "dataset_provenance": training.get("dataset_provenance"),
            "evidence_sha256": _sha256(training_report),
        },
        "checkpoint_selection": checkpoint_selection_evidence,
        "evaluation": {
            "schema": evaluation.get("schema"),
            "passed": evaluation["passed"],
            "thresholds": evaluation.get("thresholds"),
            "metrics": evaluation["metrics"],
            "gates": evaluation["gates"],
            "comprehension_prompt_sha256": evaluation.get(
                "comprehension_prompt_sha256"
            ),
            "evidence_sha256": _sha256(evaluation_report),
        },
        "vision_evaluation": {
            "schema": vision_evaluation.get("schema"),
            "passed": vision_evaluation["passed"],
            "sequence": vision_evaluation.get("sequence"),
            "metrics": vision_evaluation["metrics"],
            "gates": vision_evaluation["gates"],
            "evidence_sha256": _sha256(vision_evaluation_report),
        },
        "capability_evaluation": {
            "passed": capability["ok"],
            "missing": capability.get("missing", []),
            "evidence_sha256": _sha256(capability_report),
        },
        "tool_evaluation": {
            "passed": tools["ok"],
            "tool_calls": len(tool_calls),
            "thinking_channel_present": bool(thinking.strip()),
            "evidence_sha256": _sha256(tool_smoke_report),
        },
        "tts_alignment_evaluation": {
            "schema": tts_alignment.get("schema"),
            "passed": tts_alignment["passed"],
            "maximum_wer": tts_alignment.get("maximum_wer"),
            "minimum_compact_similarity": tts_alignment.get(
                "minimum_compact_similarity"
            ),
            "metrics": tts_alignment["metrics"],
            "gates": tts_alignment["gates"],
            "evidence_sha256": _sha256(tts_alignment_report),
        },
        "tts_vocabulary_evaluation": {
            "schema": tts_vocabulary.get("schema"),
            "passed": tts_vocabulary["passed"],
            "thresholds": tts_vocabulary["thresholds"],
            "metrics": tts_vocabulary["metrics"],
            "gates": tts_vocabulary["gates"],
            "comprehension_prompt_sha256": tts_vocabulary.get(
                "comprehension_prompt_sha256"
            ),
            "evidence_sha256": vocabulary_sha256,
        },
        "tts_vocabulary_strict_audit": {
            "schema": tts_vocabulary_audit.get("schema"),
            "passed": tts_vocabulary_audit["passed"],
            "batches": tts_vocabulary_audit["batches"],
            "expected_unique_words": tts_vocabulary_audit[
                "expected_unique_words"
            ],
            "empty_transcript_batches": tts_vocabulary_audit[
                "empty_transcript_batches"
            ],
            "invalid_tag_batches": tts_vocabulary_audit["invalid_tag_batches"],
            "invalid_wav_batches": tts_vocabulary_audit["invalid_wav_batches"],
            "comprehension_prompt_sha256": tts_vocabulary_audit[
                "comprehension_prompt_sha256"
            ],
            "source_report_sha256": vocabulary_sha256,
            "evidence_sha256": _sha256(tts_vocabulary_audit_report),
        },
    }
    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "release-manifest.json"
    card_path = destination / "README.md"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    card_path.write_text(_render_card(manifest), encoding="utf-8")
    return {
        "schema": RELEASE_SCHEMA,
        "manifest": str(manifest_path),
        "model_card": str(card_path),
        "weight_set_bytes": weight_set_bytes,
        "weight_set_gib": weight_set_bytes / (1024**3),
        "reduction_percent": manifest["memory"]["reduction_percent"],
        "artifacts": artifacts,
    }
