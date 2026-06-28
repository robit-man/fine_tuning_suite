from __future__ import annotations

import sys
import time

from training_suite.core.jobs import JobRunner
from training_suite.core.state import StateStore


def test_state_store_round_trips_model(tmp_path) -> None:
    store = StateStore(tmp_path / "suite.sqlite3")
    model_id = store.upsert_model(
        {
            "name": "ornith",
            "source": "hf.co/repo",
            "source_type": "huggingface",
            "detected_capabilities": ["completion"],
            "target_capabilities": ["completion", "tools"],
            "repair_plan": {"mode": "package-only"},
            "metadata": {"arch": "qwen35"},
        }
    )

    row = store.get_model(model_id)
    assert row is not None
    assert row["detected_capabilities"] == ["completion"]
    assert row["repair_plan"]["mode"] == "package-only"


def test_job_runner_records_success(tmp_path) -> None:
    store = StateStore(tmp_path / "suite.sqlite3")
    runner = JobRunner(store)
    job_id = runner.start(
        kind="unit",
        command=[sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
    )

    deadline = time.time() + 10
    job = store.get_job(job_id)
    while job and job["status"] not in {"succeeded", "failed"} and time.time() < deadline:
        time.sleep(0.1)
        job = store.get_job(job_id)

    assert job is not None
    assert job["status"] == "succeeded"
    assert "ok" in runner.read_log(job_id)
