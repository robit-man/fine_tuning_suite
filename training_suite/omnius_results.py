from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from training_suite.core.config import PATHS


RESULT_SCHEMA = "fine-tuning-suite.omnius-experiment-result.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_manifest(path: Path) -> tuple[str, int, list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    total_size = 0
    children = sorted(path.rglob("*"))
    if any(child.is_symlink() for child in children):
        raise ValueError(f"experiment artifact directory contains a symlink: {path}")
    for child in (item for item in children if item.is_file()):
        size = child.stat().st_size
        total_size += size
        entries.append(
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": _sha256_file(child),
                "size_bytes": size,
            }
        )
    encoded = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), total_size, entries


def _inside_outputs(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    root = PATHS.outputs.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"experiment artifact is outside suite outputs: {resolved}")
    return resolved


def resolve_omnius_job_artifacts(job: dict[str, Any]) -> dict[str, Any]:
    """Resolve a completed experiment manifest into immutable artifact references."""

    if job.get("status") != "succeeded":
        raise ValueError(f"job {job.get('id', '?')} has not succeeded")
    experiment = (job.get("metadata") or {}).get("omnius_experiment")
    if not isinstance(experiment, dict) or not experiment.get("result_path"):
        raise ValueError(f"job {job.get('id', '?')} has no Omnius experiment result contract")
    result_path = _inside_outputs(Path(str(experiment["result_path"])))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"unsupported Omnius experiment result at {result_path}")
    if result.get("candidate_id") != experiment.get("candidate_id"):
        raise ValueError("experiment result candidate does not match the job contract")
    if result.get("stage") != experiment.get("stage"):
        raise ValueError("experiment result stage does not match the job contract")
    if result.get("seed") != experiment.get("seed"):
        raise ValueError("experiment result seed does not match the job contract")

    refs: list[dict[str, Any]] = []
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("experiment result has no artifacts")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or not artifact.get("path") or not artifact.get("kind"):
            raise ValueError(f"invalid experiment artifact {index}")
        path = _inside_outputs(Path(str(artifact["path"])))
        if path.is_dir():
            digest, size, entries = _directory_manifest(path)
            media_type = "application/vnd.omnius.directory-manifest+json"
        else:
            digest = _sha256_file(path)
            size = path.stat().st_size
            entries = None
            media_type = str(artifact.get("media_type") or "application/octet-stream")
        ref = {
            "id": f"{artifact['kind']}:{digest}",
            "kind": str(artifact["kind"]),
            "uri": path.as_uri(),
            "sha256": digest,
            "mediaType": media_type,
            "sizeBytes": size,
        }
        if entries is not None:
            ref["entries"] = entries
        refs.append(ref)

    return {
        "schema": "fine-tuning-suite.omnius-job-artifacts.v1",
        "job_id": job.get("id"),
        "candidate_id": result["candidate_id"],
        "stage": result["stage"],
        "seed": result["seed"],
        "metrics": result.get("metrics") or [],
        "artifactRefs": refs,
        "resultManifest": {
            "uri": result_path.as_uri(),
            "sha256": _sha256_file(result_path),
        },
    }
