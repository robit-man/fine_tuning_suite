from __future__ import annotations

import json

import pytest

flask = pytest.importorskip("flask")

from training_suite.core.config import SuitePaths
from training_suite.core.jobs import JobRunner
from training_suite.core.state import StateStore
from training_suite.web import create_app


def _paths(tmp_path):
    package = tmp_path / "training_suite"
    return SuitePaths(
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


def test_completed_experiment_exposes_hashed_promotion_artifacts(tmp_path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    monkeypatch.setattr("training_suite.omnius_results.PATHS", paths)
    adapter = paths.outputs / "checkpoints" / "candidate" / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"rank": 8}\n', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    result_path = paths.outputs / "omnius" / "candidate" / "7" / "train-result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "schema": "fine-tuning-suite.omnius-experiment-result.v1",
                "candidate_id": "candidate",
                "stage": "train",
                "seed": 7,
                "metrics": [{"name": "training_loss", "value": 1.2, "direction": "minimize"}],
                "artifacts": [{"kind": "adapter", "path": str(adapter)}],
            }
        ),
        encoding="utf-8",
    )
    store = StateStore(paths.db)
    job_id = store.create_job(
        kind="train",
        command=["unused"],
        cwd=paths.package_root,
        log_path=paths.logs / "unused.log",
        metadata={
            "omnius_experiment": {
                "candidate_id": "candidate",
                "stage": "train",
                "seed": 7,
                "result_path": str(result_path),
            }
        },
    )
    store.update_job(job_id, status="succeeded", returncode=0)
    app = create_app(store=store, runner=JobRunner(store))
    app.testing = True

    response = app.test_client().get(f"/api/jobs/{job_id}/artifacts")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["candidate_id"] == "candidate"
    assert payload["artifactRefs"][0]["kind"] == "adapter"
    assert len(payload["artifactRefs"][0]["sha256"]) == 64
    assert {entry["path"] for entry in payload["artifactRefs"][0]["entries"]} == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }


def test_artifact_endpoint_rejects_unfinished_jobs(tmp_path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    monkeypatch.setattr("training_suite.omnius_results.PATHS", paths)
    store = StateStore(paths.db)
    job_id = store.create_job(
        kind="train",
        command=["unused"],
        cwd=paths.package_root,
        log_path=paths.logs / "unused.log",
        metadata={"omnius_experiment": {"result_path": str(paths.outputs / "missing.json")}},
    )
    app = create_app(store=store, runner=JobRunner(store))
    app.testing = True

    response = app.test_client().get(f"/api/jobs/{job_id}/artifacts")

    assert response.status_code == 409
    assert "has not succeeded" in response.get_json()["error"]
