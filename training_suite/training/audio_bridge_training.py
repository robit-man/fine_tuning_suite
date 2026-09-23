from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training_suite.training.audio_bridge_data import AUDIO_BRIDGE_DATASET_SCHEMA
from training_suite.training.omni_encoder_bridge import (
    FrozenOmniAudioLanguageAdapter,
    load_bridge_checkpoint,
    load_omni_audio_encoder_from_gguf,
    save_bridge_checkpoint,
)

TRAINING_SCHEMA = "robit.qwen3a-audio-bridge-training.v1"
MEDIA_SYSTEM_PROMPT = (
    "You are a media perception encoder, not a conversational assistant. "
    "Analyze only the supplied audio. Return exactly a speech_transcript XML "
    "element followed by an audio_observation XML element. The transcript is "
    "verbatim speech, or empty when no speech is intelligible. The observation "
    "contains only non-speech sound evidence and must never claim visual evidence. "
    "Write plain text inside both elements; never use nested tags, spans, timestamps, "
    "coordinates, Markdown, or code fences. Keep audio_observation strictly acoustic; "
    "never call a click a camera shutter."
)
MEDIA_SUFFIX_PROMPT = (
    "\nAnalyze this audio. Output exactly <speech_transcript>verbatim speech, or "
    "empty if none</speech_transcript><audio_observation>objective non-speech "
    "evidence</audio_observation>, and nothing else. Use plain text inside both "
    "elements; never emit nested markup or timestamps. Keep audio_observation strictly "
    "acoustic; never call a click a camera shutter."
    "<|im_end|>\n"
    # Both supported target trunks use this exact generation prefill when
    # enable_thinking=false. Training against the live tokens prevents private
    # reasoning from consuming the bounded perception response budget.
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


@dataclass(frozen=True)
class AudioBridgeTrainingConfig:
    target_source: str
    omni_projector: str
    initial_checkpoint: str
    dataset_manifest: str
    output_dir: str
    omni_source: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_steps: int = 800
    gradient_accumulation_steps: int = 4
    check_interval: int = 25
    validation_batches: int = 16
    gradient_clip_norm: float = 1.0
    seed: int = 42
    ready_file: str | None = None

    def __post_init__(self) -> None:
        positive = {
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "check_interval": self.check_interval,
            "validation_batches": self.validation_batches,
            "gradient_clip_norm": self.gradient_clip_norm,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError("training values must be positive: " + ", ".join(invalid))


def load_audio_bridge_records(manifest: Path) -> list[dict[str, Any]]:
    path = manifest.expanduser().resolve()
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema") != AUDIO_BRIDGE_DATASET_SCHEMA:
            raise ValueError(f"line {line_number} has an unsupported dataset schema")
        if record.get("split") not in {"train", "validation", "test"}:
            raise ValueError(f"line {line_number} has an invalid split")
        if record.get("current_visual_input") is not False:
            raise ValueError(f"line {line_number} violates audio-only visual provenance")
        if "<visual_observation" in str(record.get("assistant_target") or ""):
            raise ValueError(f"line {line_number} contains a visual observation")
        audio = (path.parent / str(record.get("audio") or "")).resolve()
        if not audio.is_file():
            raise FileNotFoundError(audio)
        record["audio_path"] = str(audio)
        records.append(record)
    if not records:
        raise ValueError("audio bridge manifest is empty")
    return records


def _pad_token_rows(rows: list[list[int]], *, pad_id: int, torch: Any):
    width = max(len(row) for row in rows)
    values = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.bool)
    for index, row in enumerate(rows):
        values[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row)] = True
    return values, mask


class AudioBridgeCollator:
    def __init__(self, feature_extractor: Any, tokenizer: Any) -> None:
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.pad_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
        self.prefix_ids = tokenizer.encode(
            f"<|im_start|>system\n{MEDIA_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n",
            add_special_tokens=False,
        )
        self.suffix_prompt_ids = tokenizer.encode(
            MEDIA_SUFFIX_PROMPT,
            add_special_tokens=False,
        )
        self.end_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

    def __call__(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        import soundfile as sf
        import torch

        audio_values = []
        prefix_rows = []
        suffix_rows = []
        suffix_label_rows = []
        for record in records:
            samples, sample_rate = sf.read(
                record["audio_path"],
                dtype="float32",
                always_2d=False,
            )
            if int(sample_rate) != 16_000 or getattr(samples, "ndim", 1) != 1:
                raise ValueError(f"audio is not 16 kHz mono: {record['audio_path']}")
            audio_values.append(samples)
            answer_ids = self.tokenizer.encode(
                str(record["assistant_target"]),
                add_special_tokens=False,
            ) + self.end_ids
            suffix = self.suffix_prompt_ids + answer_ids
            prefix_rows.append(self.prefix_ids)
            suffix_rows.append(suffix)
            suffix_label_rows.append([-100] * len(self.suffix_prompt_ids) + answer_ids)

        extracted = self.feature_extractor(
            audio_values,
            sampling_rate=16_000,
            padding="longest",
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prefix_ids, prefix_mask = _pad_token_rows(
            prefix_rows,
            pad_id=self.pad_id,
            torch=torch,
        )
        suffix_ids, suffix_mask = _pad_token_rows(
            suffix_rows,
            pad_id=self.pad_id,
            torch=torch,
        )
        suffix_labels, _ = _pad_token_rows(
            suffix_label_rows,
            pad_id=-100,
            torch=torch,
        )
        return {
            "input_features": extracted["input_features"],
            "feature_attention_mask": extracted["attention_mask"].bool(),
            "prefix_input_ids": prefix_ids,
            "prefix_attention_mask": prefix_mask,
            "suffix_input_ids": suffix_ids,
            "suffix_attention_mask": suffix_mask,
            "suffix_labels": suffix_labels,
        }


def _move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _select_validation_records(
    records: list[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic source- and speech-balanced validation slice."""
    if limit <= 0:
        raise ValueError("validation limit must be positive")
    if limit >= len(records):
        return list(records)

    buckets: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record.get("source") or "unknown"), bool(record.get("transcript")))
        buckets.setdefault(key, []).append(record)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    ordered_keys = sorted(buckets)
    while len(selected) < limit:
        added = False
        for key in ordered_keys:
            bucket = buckets[key]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
                added = True
        if not added:
            break
    return selected


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_provenance(manifest: Path) -> dict[str, Any]:
    report_path = manifest.expanduser().resolve().parent / "dataset_report.json"
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    allowed = (
        "schema",
        "vocabulary_size",
        "vocabulary_sha256",
        "maximum_numeral",
        "synthesis_count",
        "base_samples",
        "recovery_samples",
        "training_repeats",
        "short_variants",
        "sample_rate_hz",
        "split_counts",
        "nonspeech_additional_repeats",
        "nonspeech_source_samples",
        "nonspeech_added_samples",
        "train_nonspeech_samples",
    )
    return {key: report[key] for key in allowed if key in report}


def _signal_ready(path: str | None, *, model: str, device: str) -> None:
    if not path:
        return
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        destination,
        {
            "schema": TRAINING_SCHEMA,
            "status": "models-resident",
            "model": model,
            "device": device,
            "timestamp": time.time(),
        },
    )


def _validation_loss(
    adapter: Any,
    records: list[dict[str, Any]],
    collator: AudioBridgeCollator,
    *,
    device: str,
    batches: int,
) -> float:
    import torch

    if not records:
        raise ValueError("validation split is empty")
    losses = []
    adapter.bridge.eval()
    for record in records[:batches]:
        with torch.no_grad():
            result = adapter(**_move_batch(collator([record]), device))
        losses.append(float(result.loss.detach().cpu()))
    adapter.bridge.train()
    return sum(losses) / len(losses)


def train_audio_bridge(config: AudioBridgeTrainingConfig) -> dict[str, Any]:
    """Fine-tune only the deployable projector with periodic sanity gates."""
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    destination = Path(config.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    dataset_manifest = Path(config.dataset_manifest).expanduser().resolve()
    records = load_audio_bridge_records(dataset_manifest)
    train_records = [record for record in records if record["split"] == "train"]
    all_validation_records = [
        record for record in records if record["split"] == "validation"
    ]
    if not train_records or not all_validation_records:
        raise ValueError("training and validation splits must both be non-empty")
    validation_records = _select_validation_records(
        all_validation_records,
        limit=config.validation_batches,
        seed=config.seed,
    )
    validation_sources: dict[str, int] = {}
    for record in validation_records:
        source = str(record.get("source") or "unknown")
        validation_sources[source] = validation_sources.get(source, 0) + 1

    dtype = getattr(torch, config.dtype)
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.omni_source)
    tokenizer = AutoTokenizer.from_pretrained(config.target_source)
    collator = AudioBridgeCollator(feature_extractor, tokenizer)
    language = AutoModelForCausalLM.from_pretrained(
        config.target_source,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": config.device},
        attn_implementation="sdpa",
    )
    language.requires_grad_(False)
    language.config.use_cache = False
    language.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    audio_encoder, encoder_report = load_omni_audio_encoder_from_gguf(
        Path(config.omni_projector),
        omni_source=config.omni_source,
        dtype=dtype,
        device=config.device,
    )
    bridge = load_bridge_checkpoint(
        Path(config.initial_checkpoint),
        device=config.device,
        dtype=dtype,
    )
    adapter = FrozenOmniAudioLanguageAdapter(audio_encoder, language, bridge)
    adapter.bridge.train()
    # Gradient checkpointing is active only in train mode. The weights remain
    # frozen, and these target architectures use zero dropout in the text trunk.
    adapter.language_model.train()
    _signal_ready(
        config.ready_file,
        model=config.target_source,
        device=config.device,
    )
    optimizer = torch.optim.AdamW(
        adapter.bridge.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    initial_validation = _validation_loss(
        adapter,
        validation_records,
        collator,
        device=config.device,
        batches=config.validation_batches,
    )
    best_validation = initial_validation
    best_step = 0
    initial_event = {
        "step": 0,
        "validation_loss": initial_validation,
        "sanity_check": "pass",
        "checkpoint_role": "initialized-rollback",
    }
    save_bridge_checkpoint(
        destination / "best",
        adapter.bridge,
        manifest={"schema": TRAINING_SCHEMA, "event": initial_event},
    )
    progress_path = destination / "progress.jsonl"
    started = time.time()
    order = list(range(len(train_records)))
    cursor = 0
    optimizer.zero_grad(set_to_none=True)

    for step in range(1, config.max_steps + 1):
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation_steps):
            if cursor == 0:
                random.shuffle(order)
            record = train_records[order[cursor]]
            cursor = (cursor + 1) % len(order)
            batch = _move_batch(collator([record]), config.device)
            result = adapter(**batch)
            loss = result.loss / config.gradient_accumulation_steps
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite training loss at step {step}")
            loss.backward()
            accumulated_loss += float(loss.detach().cpu())

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            adapter.bridge.parameters(),
            config.gradient_clip_norm,
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite bridge gradient at step {step}")
        if float(gradient_norm.detach().cpu()) <= 0:
            raise RuntimeError(f"zero bridge gradient at step {step}")
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all())
            for parameter in adapter.bridge.parameters()
        ):
            raise RuntimeError(f"non-finite bridge parameter at step {step}")
        optimizer.zero_grad(set_to_none=True)

        event = {
            "step": step,
            "training_loss": accumulated_loss,
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "elapsed_seconds": time.time() - started,
        }
        check_due = step % config.check_interval == 0 or step == config.max_steps
        if check_due:
            validation = _validation_loss(
                adapter,
                validation_records,
                collator,
                device=config.device,
                batches=config.validation_batches,
            )
            if not math.isfinite(validation):
                raise RuntimeError(f"non-finite validation loss at step {step}")
            event["validation_loss"] = validation
            event["sanity_check"] = "pass"
            checkpoint_dir = destination / f"checkpoint-{step:06d}"
            save_bridge_checkpoint(
                checkpoint_dir,
                adapter.bridge,
                manifest={"schema": TRAINING_SCHEMA, "event": event},
            )
            if validation < best_validation:
                best_validation = validation
                best_step = step
                save_bridge_checkpoint(
                    destination / "best",
                    adapter.bridge,
                    manifest={"schema": TRAINING_SCHEMA, "event": event},
                )
            _atomic_json(
                destination / "state.json",
                {
                    "schema": TRAINING_SCHEMA,
                    "status": "running",
                    "step": step,
                    "best_step": best_step,
                    "initial_validation_loss": initial_validation,
                    "best_validation_loss": best_validation,
                    "last_sanity_check": event,
                },
            )
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    report = {
        "schema": TRAINING_SCHEMA,
        "status": "complete",
        "config": asdict(config),
        "encoder": encoder_report,
        "dataset": {
            "manifest_sha256": _sha256(dataset_manifest),
            "total": len(records),
            "train": len(train_records),
            "validation": len(all_validation_records),
            "test": sum(record["split"] == "test" for record in records),
        },
        "dataset_provenance": _dataset_provenance(dataset_manifest),
        "validation_sample": {
            "count": len(validation_records),
            "speech": sum(bool(record.get("transcript")) for record in validation_records),
            "no_speech": sum(
                not bool(record.get("transcript")) for record in validation_records
            ),
            "sources": validation_sources,
            "record_ids": [record["id"] for record in validation_records],
        },
        "initial_validation_loss": initial_validation,
        "best_validation_loss": best_validation,
        "best_step": best_step,
        "elapsed_seconds": time.time() - started,
        "trainable_parameters": sum(
            parameter.numel() for parameter in adapter.bridge.parameters()
        ),
        "frozen_language_parameters": sum(
            parameter.numel() for parameter in adapter.language_model.parameters()
        ),
        "frozen_audio_parameters": sum(
            parameter.numel() for parameter in adapter.omni_audio_encoder.parameters()
        ),
    }
    _atomic_json(destination / "training_report.json", report)
    _atomic_json(destination / "state.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a deployable Qwen3A audio bridge")
    parser.add_argument("--target-source", required=True)
    parser.add_argument("--omni-projector", required=True)
    parser.add_argument("--initial-checkpoint", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--omni-source", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--check-interval", type=int, default=25)
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ready-file",
        help="Atomically write a broker readiness record after CUDA models are resident",
    )
    args = parser.parse_args()
    report = train_audio_bridge(
        AudioBridgeTrainingConfig(
            target_source=args.target_source,
            omni_projector=args.omni_projector,
            initial_checkpoint=args.initial_checkpoint,
            dataset_manifest=args.dataset_manifest,
            output_dir=args.out,
            omni_source=args.omni_source,
            device=args.device,
            dtype=args.dtype,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_steps=args.max_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            check_interval=args.check_interval,
            validation_batches=args.validation_batches,
            gradient_clip_norm=args.gradient_clip_norm,
            seed=args.seed,
            ready_file=args.ready_file,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
