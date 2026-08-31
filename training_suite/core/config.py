from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class SuitePaths:
    project_root: Path = PROJECT_ROOT
    package_root: Path = PACKAGE_ROOT
    data: Path = PACKAGE_ROOT / "data"
    outputs: Path = PACKAGE_ROOT / "outputs"
    logs: Path = PACKAGE_ROOT / "logs"
    state: Path = PACKAGE_ROOT / "state"
    vendor: Path = PACKAGE_ROOT / "vendor"
    llama_cpp: Path = PACKAGE_ROOT / "vendor" / "llama.cpp"
    db: Path = _env_path("TRAINING_SUITE_DB", PACKAGE_ROOT / "state" / "suite.sqlite3")

    def ensure(self) -> None:
        for path in (
            self.data,
            self.outputs,
            self.logs,
            self.state,
            self.outputs / "ollama",
            self.logs / "jobs",
        ):
            path.mkdir(parents=True, exist_ok=True)


PATHS = SuitePaths()

DEFAULT_TARGET_CAPABILITIES = ("completion", "vision", "tools", "thinking")
TARGET_CAPABILITIES = (
    "completion",
    "vision",
    "audio-input",
    "audio-output",
    "video-input",
    "tools",
    "thinking",
)
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_RENDERER = os.environ.get("TRAINING_SUITE_OLLAMA_RENDERER", "qwen3.5")
DEFAULT_PARSER = os.environ.get("TRAINING_SUITE_OLLAMA_PARSER", "qwen3.5")


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str, fallback: str = "item") -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-._")
    return value or fallback


def safe_model_tag(value: str) -> str:
    value = (value or "").strip()
    if ":" in value:
        name, tag = value.rsplit(":", 1)
    else:
        name, tag = value, "latest"
    name = re.sub(r"[^A-Za-z0-9._/-]+", "-", name).strip("/-")
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", tag).strip("-") or "latest"
    return f"{name}:{tag}" if name else f"model:{tag}"
