# Omnius recurrent-improvement intake

The training suite accepts an Omnius evidence bundle at `POST /api/omnius/intake`. Intake validates the bundle, writes frozen provenance-grouped train/validation/test JSONL splits under `training_suite/outputs/splits/omnius/`, registers the dataset, and returns the exact `DISTILL_TRAIN_FILE` and `DISTILL_VAL_FILE` values.

The bundle contract is `omnius.self-improvement.bundle.v1`. Every example must contain a real assistant transcript, immutable source references with SHA-256 values, an inference review event ID, and a synthetic-data flag. Omnius performs the review/archive validation before bundle creation; this service validates transport and provenance structure again.

When `POST /api/jobs` starts `baseline`, `train`, or `evaluate-adapter` with that `dataset_id`, the split names are bound automatically. Training reads only train and validation; baseline/evaluation can address the frozen test split.

## CUDA lease contract

The `train` REST action refuses to start without `gpu_lease`:

```json
{
  "action": "train",
  "dataset_id": 12,
  "gpu_lease": {
    "owner": "omnius/adapter-candidate-42",
    "justification": "held-out qualification of adapter candidate 42",
    "expected_duration": 7200,
    "gpu_uuids": ["GPU-..."],
    "vram_mib": 24000,
    "ready_command": "auto"
  }
}
```

The job runner executes `docker gpu discover`, wraps the training command with `docker gpu run`, and binds both `CUDA_VISIBLE_DEVICES` and `DISTILL_GPUS` to exactly the leased UUIDs. `ready_command=auto` creates a per-job marker path, and the training process writes the marker only after the model (and adapter where applicable) is resident. The marker is removed when the child exits. Arbitrary readiness commands are rejected for suite CUDA actions.

Only `DISTILL_*` environment overrides are accepted through the REST API. Lease tokens are never accepted or stored in the API payload.

Direct CUDA commands refuse to run unless `OLLAMA_UNIFY_GPU_LEASE` is present and the visible device list contains exact GPU UUIDs. Use the REST job path or an explicit `docker gpu run` wrapper.
