from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .config import PATHS, utc_now


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True)


def _loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


class StateStore:
    """Small SQLite state store shared by CLI, jobs, and Flask routes."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or PATHS.db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    raw_source TEXT,
                    donor_model TEXT,
                    local_ollama_model TEXT,
                    architecture TEXT,
                    quantization TEXT,
                    context_length INTEGER,
                    tensor_count INTEGER,
                    detected_capabilities TEXT NOT NULL DEFAULT '[]',
                    target_capabilities TEXT NOT NULL DEFAULT '[]',
                    repair_plan TEXT NOT NULL DEFAULT '{}',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'intake',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    schema_mapping TEXT NOT NULL DEFAULT '{}',
                    split_config TEXT NOT NULL DEFAULT '{}',
                    license_note TEXT,
                    preview TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    command TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    status TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    model_id INTEGER,
                    dataset_id INTEGER,
                    returncode INTEGER,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    job_id INTEGER,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS eval_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id INTEGER,
                    model_name TEXT NOT NULL,
                    eval_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    report_path TEXT,
                    metrics TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        out = dict(row)
        for key, default in (
            ("detected_capabilities", []),
            ("target_capabilities", []),
            ("repair_plan", {}),
            ("metadata", {}),
            ("schema_mapping", {}),
            ("split_config", {}),
            ("preview", []),
            ("command", []),
            ("metrics", {}),
        ):
            if key in out:
                out[key] = _loads(out[key], default)
        return out

    def list_models(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM models ORDER BY updated_at DESC, id DESC").fetchall()
        return [self._row(r) for r in rows if r is not None]

    def get_model(self, model_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
        return self._row(row)

    def upsert_model(self, data: dict[str, Any]) -> int:
        now = utc_now()
        existing_id = data.get("id")
        fields = {
            "name": data.get("name") or data.get("source") or "model",
            "source": data.get("source") or "",
            "source_type": data.get("source_type") or "unknown",
            "raw_source": data.get("raw_source"),
            "donor_model": data.get("donor_model"),
            "local_ollama_model": data.get("local_ollama_model"),
            "architecture": data.get("architecture"),
            "quantization": data.get("quantization"),
            "context_length": data.get("context_length"),
            "tensor_count": data.get("tensor_count"),
            "detected_capabilities": _json(data.get("detected_capabilities", [])),
            "target_capabilities": _json(data.get("target_capabilities", [])),
            "repair_plan": _json(data.get("repair_plan", {})),
            "metadata": _json(data.get("metadata", {})),
            "status": data.get("status") or "intake",
        }
        with self.connect() as conn:
            if existing_id:
                assignments = ", ".join(f"{k} = ?" for k in fields)
                conn.execute(
                    f"UPDATE models SET {assignments}, updated_at = ? WHERE id = ?",
                    (*fields.values(), now, existing_id),
                )
                return int(existing_id)
            cur = conn.execute(
                """
                INSERT INTO models (
                    name, source, source_type, raw_source, donor_model, local_ollama_model,
                    architecture, quantization, context_length, tensor_count,
                    detected_capabilities, target_capabilities, repair_plan, metadata,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*fields.values(), now, now),
            )
            return int(cur.lastrowid)

    def list_datasets(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM datasets ORDER BY updated_at DESC, id DESC").fetchall()
        return [self._row(r) for r in rows if r is not None]

    def add_dataset(self, data: dict[str, Any]) -> int:
        now = utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO datasets (
                    name, source, source_type, schema_mapping, split_config,
                    license_note, preview, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("name") or data.get("source") or "dataset",
                    data.get("source") or "",
                    data.get("source_type") or "unknown",
                    _json(data.get("schema_mapping", {})),
                    _json(data.get("split_config", {})),
                    data.get("license_note"),
                    _json(data.get("preview", [])),
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def create_job(
        self,
        *,
        kind: str,
        command: Iterable[str],
        cwd: Path,
        log_path: Path,
        model_id: int | None = None,
        dataset_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        now = utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO jobs (
                    kind, command, cwd, status, log_path, model_id, dataset_id,
                    created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    _json(list(command)),
                    str(cwd),
                    "queued",
                    str(log_path),
                    model_id,
                    dataset_id,
                    now,
                    _json(metadata or {}),
                ),
            )
            return int(cur.lastrowid)

    def update_job(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        normalized = {}
        for key, value in fields.items():
            normalized[key] = _json(value) if key in {"command", "metadata"} else value
        assignments = ", ".join(f"{k} = ?" for k in normalized)
        with self.connect() as conn:
            conn.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*normalized.values(), job_id),
            )

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row(row)

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row(r) for r in rows if r is not None]

    def add_eval_run(self, data: dict[str, Any]) -> int:
        now = utc_now()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO eval_runs (
                    model_id, model_name, eval_type, status, report_path,
                    metrics, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("model_id"),
                    data.get("model_name") or "",
                    data.get("eval_type") or "unknown",
                    data.get("status") or "queued",
                    data.get("report_path"),
                    _json(data.get("metrics", {})),
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def list_eval_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM eval_runs ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row(r) for r in rows if r is not None]
