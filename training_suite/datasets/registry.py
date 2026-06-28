from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CURATION_RECIPES = {
    "curate-r5": {
        "label": "R5 research mix",
        "script": "curate_r5.py",
        "description": "Bespoke-Stratos, Tulu 3, SlimOrca, and format examples.",
    },
    "curate-r6": {
        "label": "R6 anti-loop mix",
        "script": "curate_r6.py",
        "description": "Mixture-of-Thoughts, OpenThoughts, and anti-loop data.",
    },
    "curate-r7": {
        "label": "R7 additive mix",
        "script": "curate_r7.py",
        "description": "R5 plus PrimeIntellect SYNTHETIC-1 and anti-loop data.",
    },
}


def detect_dataset_source(source: str) -> str:
    source = (source or "").strip()
    if source.startswith(("http://", "https://")) or "/" in source and not Path(source).exists():
        return "huggingface"
    if source.endswith(".jsonl") or Path(source).exists():
        return "local-jsonl"
    return "unknown"


def preview_jsonl(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_raw": line.strip()[:500]})
            if len(rows) >= limit:
                break
    return rows


def dataset_record(
    *,
    name: str,
    source: str,
    schema_mapping: dict[str, Any] | None = None,
    split_config: dict[str, Any] | None = None,
    license_note: str | None = None,
) -> dict[str, Any]:
    source_type = detect_dataset_source(source)
    preview = preview_jsonl(Path(source)) if source_type == "local-jsonl" else []
    return {
        "name": name or source,
        "source": source,
        "source_type": source_type,
        "schema_mapping": schema_mapping or {},
        "split_config": split_config or {"train": 0.85, "val": 0.075, "test": 0.075, "seed": 42},
        "license_note": license_note,
        "preview": preview,
    }
