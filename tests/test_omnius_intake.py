from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")

from training_suite.core.config import SuitePaths
from training_suite.core.jobs import GpuLeaseSpec, JobRunner
from training_suite.core.state import StateStore
from training_suite.omnius_intake import ingest_omnius_bundle
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


def _bundle(tmp_path):
    source_hash = hashlib.sha256(b"trace").hexdigest()
    examples = []
    review_ids = []
    for index in range(12):
        review_id = f"review-{index}"
        review_ids.append(review_id)
        examples.append(
            {
                "id": f"example-{index}",
                "messages": [
                    {"role": "user", "content": f"question {index}"},
                    {"role": "assistant", "content": f"answer {index}"},
                ],
                "sourceRefs": [
                    {
                        "id": f"source-{index}",
                        "uri": f"memory://source-{index}",
                        "sha256": hashlib.sha256(f"trace-{index}".encode()).hexdigest(),
                    }
                ],
                "reviewEventId": review_id,
                "synthetic": False,
            }
        )
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "schema": "omnius.self-improvement.bundle.v1",
                "createdAt": "2026-09-25T00:00:00Z",
                "sourceArchive": "/evidence/events.jsonl",
                "examples": examples,
                "manifest": {
                    "exampleCount": len(examples),
                    "sourceHashes": [source_hash],
                    "reviewEventIds": review_ids,
                    "humanOrOriginalExamples": len(examples),
                    "syntheticExamples": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_omnius_bundle_intake_materializes_frozen_provenance_splits(tmp_path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr("training_suite.omnius_intake.PATHS", paths)
    result = ingest_omnius_bundle(_bundle(tmp_path), split_config={"test": 0.2, "val": 0.2})

    assert result.examples == 12
    assert sum(result.split_counts.values()) == 12
    assert result.split_names["train"].startswith("omnius/omnius-")
    manifest = json.loads((result.split_dir / "manifest.json").read_text())
    assert manifest["provenance_grouped"] is True
    assert manifest["held_out_training_forbidden"] is True
    assert set(manifest["split_sha256"]) == {"train", "val", "test"}


def test_web_intake_registers_dataset_and_cuda_jobs_require_broker_lease(tmp_path, monkeypatch) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr("training_suite.omnius_intake.PATHS", paths)
    store = StateStore(tmp_path / "suite.sqlite3")
    app = create_app(store=store, runner=JobRunner(store))
    app.testing = True
    client = app.test_client()

    response = client.post(
        "/api/omnius/intake",
        json={"bundle_path": str(_bundle(tmp_path)), "name": "omnius-reviewed"},
    )
    assert response.status_code == 201
    payload = response.get_json()
    assert payload["intake"]["training_environment"]["DISTILL_TRAIN_FILE"].endswith("/train")
    assert store.list_datasets()[0]["name"] == "omnius-reviewed"

    denied = client.post("/api/jobs", json={"action": "train", "dataset_id": payload["id"]})
    assert denied.status_code == 400
    assert "requires gpu_lease" in denied.get_json()["error"]


def test_train_job_binds_ingested_splits_and_explicit_gpu_lease(tmp_path, monkeypatch) -> None:
    class CapturingRunner:
        def __init__(self):
            self.kwargs = None

        def start(self, **kwargs):
            self.kwargs = kwargs
            return 42

        def cancel(self, _job_id):
            return False

    paths = _paths(tmp_path)
    monkeypatch.setattr("training_suite.omnius_intake.PATHS", paths)
    store = StateStore(tmp_path / "suite.sqlite3")
    runner = CapturingRunner()
    app = create_app(store=store, runner=runner)
    app.testing = True
    client = app.test_client()
    intake = client.post(
        "/api/omnius/intake",
        json={"bundle_path": str(_bundle(tmp_path)), "name": "omnius-reviewed"},
    ).get_json()

    response = client.post(
        "/api/jobs",
        json={
            "action": "train",
            "dataset_id": intake["id"],
            "gpu_lease": {
                "owner": "omnius/adapter-trial",
                "justification": "held-out Omnius adapter qualification",
                "expected_duration": 3600,
                "gpu_uuids": ["GPU-example"],
                "vram_mib": 24000,
                "ready_command": "auto",
            },
        },
    )

    assert response.status_code == 201
    assert runner.kwargs["environment"]["DISTILL_TRAIN_FILE"].endswith("/train")
    assert runner.kwargs["environment"]["DISTILL_VAL_FILE"].endswith("/val")
    assert "DISTILL_TEST_FILE" not in runner.kwargs["environment"]
    assert runner.kwargs["gpu_lease"].gpu_uuids == ("GPU-example",)


def test_gpu_lease_requires_a_visible_owner_scope_and_readiness_probe() -> None:
    with pytest.raises(ValueError, match="GPU UUID"):
        GpuLeaseSpec(
            owner="omnius/test",
            justification="adapter trial for held-out evaluation",
            expected_duration=600,
            gpu_uuids=(),
            ready_command="auto",
        ).validate()
    with pytest.raises(ValueError, match="ready_command"):
        GpuLeaseSpec(
            owner="omnius/test",
            justification="adapter trial for held-out evaluation",
            expected_duration=600,
            gpu_uuids=("GPU-example",),
            ready_command="",
        ).validate()


def test_direct_cuda_command_refuses_before_bootstrap(tmp_path) -> None:
    env = os.environ.copy()
    env.pop("OLLAMA_UNIFY_GPU_LEASE", None)
    env.pop("DISTILL_GPUS", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    app_path = Path(__file__).parents[1] / "training_suite" / "app.py"
    result = subprocess.run(
        [sys.executable, str(app_path), "train"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "requires an ollama-unify broker lease" in result.stderr
    assert not (app_path.parent / ".venv").exists()
