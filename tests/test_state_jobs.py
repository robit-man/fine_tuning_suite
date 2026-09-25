from __future__ import annotations

import sys
import time
from types import SimpleNamespace

from training_suite.core.config import SuitePaths
from training_suite.core.jobs import GpuLeaseSpec, JobRunner
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


def test_job_runner_passes_explicit_environment(tmp_path) -> None:
    store = StateStore(tmp_path / "suite.sqlite3")
    runner = JobRunner(store)
    job_id = runner.start(
        kind="unit-env",
        command=[sys.executable, "-c", "import os; print(os.environ['DISTILL_TEST_VALUE'])"],
        cwd=tmp_path,
        environment={"DISTILL_TEST_VALUE": "bound-to-job"},
    )

    deadline = time.time() + 10
    job = store.get_job(job_id)
    while job and job["status"] not in {"succeeded", "failed"} and time.time() < deadline:
        time.sleep(0.1)
        job = store.get_job(job_id)

    assert job is not None
    assert job["status"] == "succeeded"
    assert "bound-to-job" in runner.read_log(job_id)
    assert job["metadata"]["environment_keys"] == ["DISTILL_TEST_VALUE"]


def test_job_runner_builds_scoped_broker_command_and_exact_uuid_environment(tmp_path, monkeypatch) -> None:
    package = tmp_path / "training_suite"
    paths = SuitePaths(
        project_root=tmp_path,
        package_root=package,
        data=package / "data",
        outputs=package / "outputs",
        logs=package / "logs",
        state=package / "state",
        vendor=package / "vendor",
        llama_cpp=package / "vendor" / "llama.cpp",
        db=package / "state" / "suite.sqlite3",
    )
    monkeypatch.setattr("training_suite.core.jobs.PATHS", paths)
    monkeypatch.setattr(
        "training_suite.core.jobs.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    captured = {}

    class DeferredThread:
        def __init__(self, *, target, args, daemon):
            captured["target"] = target
            captured["args"] = args
            captured["daemon"] = daemon

        def start(self):
            captured["started"] = True

    monkeypatch.setattr("training_suite.core.jobs.threading.Thread", DeferredThread)
    store = StateStore(tmp_path / "suite.sqlite3")
    runner = JobRunner(store)
    job_id = runner.start(
        kind="train",
        command=[sys.executable, "app.py", "train"],
        cwd=tmp_path,
        gpu_lease=GpuLeaseSpec(
            owner="omnius/candidate-1",
            justification="held-out adapter qualification",
            expected_duration=3600,
            gpu_uuids=("GPU-one",),
            ready_command="auto",
            vram_mib=24000,
        ),
    )

    job = store.get_job(job_id)
    assert job is not None
    assert job["command"][:3] == ["docker", "gpu", "run"]
    assert "--" not in job["command"]
    assert job["command"][-3:] == [sys.executable, "app.py", "train"]
    environment = captured["args"][4]
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-one"
    assert environment["DISTILL_GPUS"] == "GPU-one"
    assert environment["DISTILL_READY_FILE"].endswith(".json")
