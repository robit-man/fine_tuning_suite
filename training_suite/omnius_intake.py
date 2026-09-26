from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training_suite.core.config import PATHS, slugify


BUNDLE_SCHEMA = "omnius.self-improvement.bundle.v1"


@dataclass(frozen=True)
class OmniusIntakeResult:
    bundle_path: Path
    bundle_sha256: str
    split_dir: Path
    split_names: dict[str, str]
    split_counts: dict[str, int]
    split_sha256: dict[str, str]
    examples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "fine-tuning-suite.omnius-intake.v1",
            "bundle_path": str(self.bundle_path),
            "bundle_sha256": self.bundle_sha256,
            "split_dir": str(self.split_dir),
            "split_names": self.split_names,
            "split_counts": self.split_counts,
            "split_sha256": self.split_sha256,
            "examples": self.examples,
            "training_environment": {
                "DISTILL_TRAIN_FILE": self.split_names["train"],
                "DISTILL_VAL_FILE": self.split_names["val"],
            },
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _validate_ref(ref: Any) -> bool:
    return (
        isinstance(ref, dict)
        and isinstance(ref.get("id"), str)
        and bool(ref["id"])
        and isinstance(ref.get("uri"), str)
        and bool(ref["uri"])
        and isinstance(ref.get("sha256"), str)
        and len(ref["sha256"]) == 64
        and all(char in "0123456789abcdefABCDEF" for char in ref["sha256"])
    )


def _validate_example(example: Any, index: int) -> None:
    if not isinstance(example, dict) or not isinstance(example.get("id"), str):
        raise ValueError(f"example {index} requires an id")
    messages = example.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"example {example['id']} requires messages")
    if not any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and isinstance(message.get("content"), str)
        and bool(message["content"].strip())
        for message in messages
    ):
        raise ValueError(f"example {example['id']} has no assistant transcript")
    refs = example.get("sourceRefs")
    if not isinstance(refs, list) or not refs or not all(_validate_ref(ref) for ref in refs):
        raise ValueError(f"example {example['id']} has invalid sourceRefs")
    if not isinstance(example.get("reviewEventId"), str) or not example["reviewEventId"]:
        raise ValueError(f"example {example['id']} requires reviewEventId")


def _provenance_group(example: dict[str, Any]) -> str:
    hashes = sorted(ref["sha256"] for ref in example["sourceRefs"])
    return hashlib.sha256("|".join(hashes).encode("utf-8")).hexdigest()


def _assign_groups(
    examples: list[dict[str, Any]],
    split_config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    test_ratio = float(split_config.get("test", split_config.get("holdout", 0.2)))
    val_ratio = float(split_config.get("val", 0.1))
    if test_ratio < 0 or val_ratio < 0 or test_ratio + val_ratio >= 1:
        raise ValueError("split ratios must be non-negative and leave a positive train split")
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(_provenance_group(example), []).append(example)
    ordered_groups = sorted(groups.items())
    if len(ordered_groups) < 3:
        raise ValueError("provenance-isolated training requires at least three independent source groups")
    test_groups = max(1, round(len(ordered_groups) * test_ratio)) if test_ratio else 0
    val_groups = max(1, round(len(ordered_groups) * val_ratio)) if val_ratio else 0
    while test_groups + val_groups >= len(ordered_groups):
        if test_groups >= val_groups and test_groups > 1:
            test_groups -= 1
        elif val_groups > 1:
            val_groups -= 1
        else:
            raise ValueError("split ratios do not leave an independent training provenance group")
    result = {"train": [], "val": [], "test": []}
    for index, (_group_id, members) in enumerate(ordered_groups):
        split = "test" if index < test_groups else "val" if index < test_groups + val_groups else "train"
        result[split].extend(members)
    return result


def ingest_omnius_bundle(
    bundle_path: str | Path,
    *,
    split_config: dict[str, Any] | None = None,
) -> OmniusIntakeResult:
    PATHS.ensure()
    source = Path(bundle_path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    bundle = json.loads(raw)
    if not isinstance(bundle, dict) or bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"bundle schema must be {BUNDLE_SCHEMA}")
    examples = bundle.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("bundle requires at least one example")
    for index, example in enumerate(examples):
        _validate_example(example, index)
    manifest = bundle.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("exampleCount") != len(examples):
        raise ValueError("bundle manifest exampleCount does not match examples")
    review_ids = set(manifest.get("reviewEventIds") or [])
    if any(example["reviewEventId"] not in review_ids for example in examples):
        raise ValueError("bundle manifest does not cover every reviewEventId")

    digest = _sha256_bytes(raw)
    name = slugify(f"omnius-{digest[:16]}")
    # app.py resolves DISTILL_{TRAIN,VAL,TEST}_FILE relative to data/splits.
    # Keep the frozen provenance groups in that exact tree so an admitted job
    # can consume the intake result without a second copy or path translation.
    split_dir = PATHS.data / "splits" / "omnius" / name
    split_dir.mkdir(parents=True, exist_ok=True)
    splits = _assign_groups(examples, split_config or {})
    counts: dict[str, int] = {}
    hashes: dict[str, str] = {}
    names: dict[str, str] = {}
    for split, rows in splits.items():
        path = split_dir / f"{split}.jsonl"
        body = "".join(
            json.dumps(
                {
                    "messages": row["messages"],
                    "omnius_example_id": row["id"],
                    "review_event_id": row["reviewEventId"],
                    "source_refs": row["sourceRefs"],
                    "synthetic": bool(row.get("synthetic", False)),
                },
                sort_keys=True,
            )
            + "\n"
            for row in rows
        )
        path.write_text(body, encoding="utf-8")
        counts[split] = len(rows)
        hashes[split] = _sha256_file(path)
        names[split] = f"omnius/{name}/{split}"
    frozen_manifest = {
        "schema": "fine-tuning-suite.omnius-splits.v1",
        "bundle_path": str(source),
        "bundle_sha256": digest,
        "split_config": split_config or {},
        "split_counts": counts,
        "split_sha256": hashes,
        "provenance_grouped": True,
        "held_out_training_forbidden": True,
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(frozen_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return OmniusIntakeResult(
        bundle_path=source,
        bundle_sha256=digest,
        split_dir=split_dir,
        split_names=names,
        split_counts=counts,
        split_sha256=hashes,
        examples=len(examples),
    )
