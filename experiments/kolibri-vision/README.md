# Kolibri vision experiment — paused

Source and handoff saved on 2026-10-07 UTC (October 6 evening PDT). Development
is paused at the user's request. There is no scheduled training or automatic
resume. **No `robit/kolibri-1-vision` model has been published.**

The existing text releases, `robit/kolibri-1:bf16` and
`robit/kolibri-1:q4_k_m`, were published with completion, tools, and thinking.
They run through stock Ollama without an external runtime adapter. The final
tool template and October 6 registry verification receipts are preserved here;
those receipts are historical observations, not a fresh registry check.

## What is saved

This is an experimental source snapshot, separate from the suite's supported
CLI. `runtime-source/` preserves the frozen decoder, sparse expert operations,
pretrained projector transfer, alignment/checkpoint code, GGUF export and
verification, bounded sampling, image split fencing, and stock evaluation.
`source-manifest.json` records each source hash and the observed Python package
versions. `receipts/` contains compact provenance, pilot, and shutdown records.
There are no weights, images, feature arrays, native binaries, credentials, or
GPU lease tokens in this snapshot.

The scripts retain their original run-directory assumptions. **Do not execute
them from this snapshot directory.** They resolve `ROOT` as the parent of
`runtime-source`, expect generated artifacts there, and locate the text release
in a sibling directory. Some native imports also depend on the original
workspace depth. The authoritative local workspace is:

```text
/srv/fine_tuning_suite/training_suite/outputs/
  kolibri-1-20261004/           # Published text GGUFs, tokenizer, template, receipts
  kolibri-1-vision-20261006/    # Vision checkpoints, data, caches, proofs, runtime-source
```

Those outputs are intentionally ignored by Git. Restoring this snapshot on
another host requires rebuilding the native helpers and restoring the pinned
inputs/proofs; copying these files alone does not reproduce a trained model.
The installed PyTorch version recorded in the manifest differs from the
suite's general requirements; this experiment uses BF16 GEMM `out_dtype` APIs
and cannot assume compatibility with the general training environment.

Host profile installers and the old broader-training/recovery launchers were
excluded. They change shared services or encode a superseded GPU scope. They
remain in the ignored historical run directory and are not resume entry points.

## Method and measured result

The experiment preserves Kolibri's released language weights and the existing
Ornith/Qwen vision tower. It transfers the pretrained final projection from
4,096 to 2,560 output dimensions through shared-token ridge alignment, then
trains only that final projection. The merged projection is exported as a
normal vision-projector GGUF alongside the normal language-model layer. Stock
Ollama image input and ordinary CLI execution have been demonstrated for this
packaging; that demonstrates runtime compatibility only.

The preserved text release provenance is in `receipts/build-source.json`.
The donor language and projector hashes, transfer measurements, calibration,
and pilot parity measurements are in
`receipts/single-gpu-1-alignment-report.json`. Vocabulary alignment scores are
not vision benchmark scores. The native feature tap is a training helper; it
is not an extra service required by the intended stock Ollama release.

| Evidence | Result | Meaning |
|---|---|---|
| Saved small adapter | Best selected at 144 optimizer updates | Current durable candidate |
| Fresh stock engineering checks | BF16 2/8, Q4_K_M 2/8; donor 8/8 | Vision quality gate failed |
| Small natural DocVQA subset | Candidate 1/10; donor 8/10 | OCR remains inadequate |
| Small natural TextVQA subset | Candidate 4/10; donor 10/10 | OCR remains inadequate |
| October 6 broader attempt | 388 attempted new updates; 0 durable new updates | Interrupted by host GPU fault before a new checkpoint |
| October 7 independent-GPU pilot | Initial validation completed; 0 new updates | User paused during the first training forward |

The two-example initial validation reports 144 total updates because it loaded
the existing adapter. Its `stage_optimizer_updates` is zero. Any
`single-gpu-1-best-projector.pt` is therefore an initial-validation copy, not a
newly trained improvement. The broader run's initial best checkpoint likewise
does not contain its 388 attempted updates. Do not promote either as a new
trained result.

The October 7 parity probe passed against native BF16/CPU references and failed
against the saved GPU capture. All three measurements are retained. This small
text probe does not establish universal numerical equivalence, successful
training, or preserved capability on every task. Neither BF16/Q4 vision
equivalence nor excellent/best-in-class vision has been established.

## Checkpoints and cached data

Retain these local artifacts for the next session:

| Artifact under the vision run | Identity / scope |
|---|---|
| `aligned-best-projector.pt` | SHA256 `bce863a21f7c656c206fcca179908d8ac6c05393958366d3a70c146a9b3b6afb`; 144 updates |
| `Kolibri-1-vision-aligned-mmproj.gguf` | SHA256 `e04280d08056feb4197db272d2625b8bc18706961dc64dfce2ae85f98d4b1c0d`; unqualified |
| `broader-alignment-data/examples.jsonl` | 1,261 images: 761 real, 500 generated; 969 training, 292 holdout |
| `feature-cache-broader-2048/manifest.json` | 1,261 cached frozen-tower features, about 11.2 GiB |
| `ocrbench-text-evaluation/examples.jsonl` | 70 evaluation-only English short-answer images, fenced against all 1,261 images |
| `counterfactual-checks/manifest.json` | 16 paired checks on 8 fresh images; never trained |
| `vision-status.json` | Current local status and references to full results |

The OCR subset is a bounded convenience sample across six task groups, not the
official full OCRBench score. Position questions score the answer text only;
bounding boxes are not assessed. Natural captions are retained for review.
Image hashes and near-duplicate checks fence evaluation images before training.
No million-example training campaign was started.

Periodic checkpoint code saves optimizer state, epoch order/cursor, RNG state,
and run/data/cache identities, with atomic replacement and explicit disk limits.
Real interruption/resumption has **not** been validated end to end. The latest
per-image validation lease transitions and macro loss selection were edited
after the running pilot loaded its source; they have only been syntax checked.

## Shutdown and resume constraints

SIGINT stopped only the project's own Python worker. The supervising wrapper
exited and released its lease after the model allocations disappeared. Broker
discovery then showed no external leases and only 19 MiB used on GPU82. The
record is `receipts/safe-resume-state-20261007.json`. No peer process was stopped,
and no host, driver, display, or shared service restart/reset was performed.

Read `/home/roko/Desktop/AGENTS.md`, `/srv/AGENTS.md`, and
`/usr/local/share/ollama-unify/AGENTS.md` again when the user resumes. Historical
GPU UUID defaults in these source files do not authorize using those devices.
GPU84 remains quarantined, and GPU01/OBLITERATUS/Omnius and other protected peer
workloads must be preserved. Do not remove incident holds or restart/reset the
host or GPUs. The October 6 fault's initiating cause remains unresolved; an
idle healthy discovery result does not certify hardware stability.

The stopped pilot used only independent GPU82
(`GPU-170a99ee-850f-2182-1050-4e8d3c87b6b0`), 32 frozen decoder layers on CPU,
and 8 CPU threads. No CUDA peer transfers were used. Its padded CPU expert path
was too slow for a useful bounded pilot. The next implementation decision is
whether the existing sparse per-expert CPU path can preserve parity while
meeting the broker's nonrenewable pending-transition deadline. That change was
not made or tested before the pause.

After the user resumes, inspect live discovery warnings and leases, retain an
exact UUID-scoped reservation for any CUDA work, and qualify a bounded pilot
before continuing alignment. Resume/checkpoint behavior and image-grounded
quality need validation before export/promotion. Publish BF16 first and Q4_K_M
afterward only if fresh OCR, screenshots, general images, tools, thinking, and
ordinary stock CLI checks pass, followed by registry verification. Keep raw
answers and comparison results; capability tags alone cannot satisfy that gate.

## CPU-only source check

This checks Python syntax without importing modules, allocating CUDA memory,
starting services, downloading data, or writing bytecode caches:

```bash
python3 - <<'PY'
from pathlib import Path
root = Path('experiments/kolibri-vision/runtime-source')
for path in sorted(root.glob('*.py')):
    compile(path.read_bytes(), str(path), 'exec')
print('Python source syntax passed')
PY
```

Native C++ sources were preserved without a new build during shutdown. Existing
qualification results live under the ignored run's `compatibility/` directory.
The snapshot is not integrated into dashboard jobs or enabled by installation.

## Licensing

The text model originates from Aleph Alpha under Apache-2.0. The pretrained
Ornith/Qwen donor retains its own model and base-component license terms, which
must accompany any eventual release. Native GGUF/llama.cpp interfaces remain
subject to MIT attribution; Aleph Alpha reference-derived additions retain
Apache-2.0 terms. The corresponding license texts are included. The final
Ollama template is preserved byte-for-byte from the corrected text release.
