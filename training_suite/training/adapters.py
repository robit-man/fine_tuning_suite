from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from training_suite.core.config import PACKAGE_ROOT
from training_suite.datasets.registry import CURATION_RECIPES


PYTHON = sys.executable or "python3"


@dataclass(frozen=True)
class ActionSpec:
    key: str
    label: str
    kind: str
    command: list[str]
    requires_model: bool = False
    requires_dataset: bool = False


def _script(name: str) -> str:
    return str(PACKAGE_ROOT / name)


def action_specs() -> dict[str, ActionSpec]:
    specs = {
        "bootstrap": ActionSpec("bootstrap", "Bootstrap environment", "bootstrap", [PYTHON, _script("app.py"), "bootstrap"]),
        "prepare": ActionSpec("prepare", "Prepare default split", "prepare", [PYTHON, _script("app.py"), "prepare"]),
        "train": ActionSpec("train", "Train LoRA", "train", [PYTHON, _script("app.py"), "train"]),
        "merge-export": ActionSpec("merge-export", "Merge and export GGUF", "export", [PYTHON, _script("app.py"), "export"], requires_model=True),
        "prepare-tools": ActionSpec("prepare-tools", "Prepare tool benchmark", "tools", [PYTHON, _script("app.py"), "prepare-tools"]),
        "baseline-tools": ActionSpec("baseline-tools", "Run base tool benchmark", "tools", [PYTHON, _script("app.py"), "baseline-tools"]),
        "eval-tools": ActionSpec("eval-tools", "Run tuned tool benchmark", "tools", [PYTHON, _script("app.py"), "eval-tools"], requires_model=True),
        "verify-ollama": ActionSpec("verify-ollama", "Verify Ollama gate", "verify", [PYTHON, _script("app.py"), "verify-ollama"], requires_model=True),
    }
    for key, recipe in CURATION_RECIPES.items():
        specs[key] = ActionSpec(
            key=key,
            label=recipe["label"],
            kind="dataset-curation",
            command=[PYTHON, _script(recipe["script"])],
            requires_dataset=False,
        )
    return specs


def get_action(key: str) -> ActionSpec:
    specs = action_specs()
    if key not in specs:
        raise KeyError(f"unknown action: {key}")
    return specs[key]


def local_script_path(name: str) -> Path:
    return PACKAGE_ROOT / name
