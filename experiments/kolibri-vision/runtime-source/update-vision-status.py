"""Create an honest, compact handoff from immutable qualification artifacts."""
from datetime import datetime, timezone
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
def read(relative):
    path = ROOT / relative
    return json.loads(path.read_text()) if path.exists() else None

reports = {name: read(name + "/sampling-report.json") for name in ["data-pilot-counted", "alignment-pilot"]}
qualified = (read("compatibility/full-bf16-differentiable-training-qualification.json")
    or read("compatibility/full-bf16-compensated-training-qualification.json")
    or read("compatibility/full-bf16-ragged-training-qualification.json")
    or read("compatibility/full-bf16-native-envelope-training-qualification.json")
    or read("compatibility/full-bf16-native-matched-training-qualification.json"))
recipe = read("data-mixture.json")
alignment = read("compatibility/targeted-alignment-report.json")
broader = read("compatibility/broader-1-alignment-report.json")
broader_data = read("broader-alignment-data/preparation-report.json")
q4_quality = read("compatibility/aligned-independent-stock-evaluation.json")
bf16_quality = read("compatibility/aligned-bf16-independent-evaluation.json")
safe_resume = read('compatibility/safe-resume-state-20261007.json')
single_gpu_pilot = read('compatibility/single-gpu-1-alignment-report.json')
benchmarks = []
for path in (ROOT / "data-sources").glob("*-benchmark-info.json"):
    data = json.loads(path.read_text())
    benchmarks.append({"repo": data["id"], "revision": data["sha"],
        "status": "evaluation-only image-fenced subset sampled; inference pending" if
        data['id']=='lmms-lab/OCRBench-v2' and read('ocrbench-evaluation/sampling-report.json') else
        "inventory pinned; benchmark image exclusion/evaluation not yet run"})
status = {
    "utc": datetime.now(timezone.utc).isoformat(), "destination": "robit/kolibri-1-vision",
    "phase": ("broader pretrained projection alignment completed; stock quality evaluation pending" if
        broader and broader.get('completed') else "broader pretrained projection alignment running" if
        broader else "existing pretrained projector transfer and targeted alignment"),
    "safe_resume":safe_resume,
    "single_gpu_cpu_pilot":single_gpu_pilot,
    "latest_user_direction": "reuse existing weights and adapt cleanly to save time; million-example campaign superseded",
    "pretrained_projection_transfer": read("compatibility/pretrained-projection-transfer.json"),
    "pretrained_projection_stock_answers": read("compatibility/pretrained-transfer-stock-evaluation.json"),
    "targeted_alignment": alignment,
    "broader_alignment_data":broader_data,
    "broader_alignment":broader,
    "broader_stage_updates_completed":broader.get('stage_optimizer_updates',0) if broader else 0,
    "independent_stock_q4_answers":q4_quality,
    "independent_stock_bf16_answers":bf16_quality,
    "ocrbench_evaluation_subset":read('ocrbench-evaluation/sampling-report.json'),
    "ocrbench_text_evaluation_subset":read('ocrbench-text-evaluation/sampling-report.json'),
    "fresh_counterfactual_checks":read('counterfactual-checks/manifest.json'),
    "current_small_adapter_qualified":False,
    "current_small_adapter_bf16_checks":{
        'correct':sum(e.get('normalized_exact',False) for e in bf16_quality.get('examples',[])),
        'total':len(bf16_quality.get('examples',[]))} if bf16_quality else None,
    "own_experimental_bf16_profile":read('compatibility/bf16-experiment-profile-install.json'),
    "alignment_optimizer_updates_completed": alignment.get('optimizer_updates',0) if alignment else 0,
    "published": False, "trained_vision_quality_claim": False, "best_in_class_claim": False,
    "original_text_release": {"bf16": "robit/kolibri-1:bf16", "q4_k_m": "robit/kolibri-1:q4_k_m", "published": True},
    "stock_ollama": {"qwen_tower_api_interface": read("compatibility/qwen-api-interface-proof.json"),
        "feature_tap": read("compatibility/vision-feature-tap-proof.json"),
        "small_decoder_parity": read("compatibility/head128/training-decoder-parity.json")},
    "full_stock_image_and_cli_interface": read("compatibility/full-stock-pilot-proof.json"),
    "native_projection_export": read("compatibility/pilot-native-projection-proof.json"),
    "sparse_expert_kernel": read("compatibility/ragged-experts-dual-actual-proof.json"),
    "data": {"accepted_pilot_examples": sum(r["accepted_examples"] for r in reports.values() if r),
        "pilot_training_examples": sum(r["training_examples"] for r in reports.values() if r),
        "development_holdout_examples": sum(r["development_holdout_examples"] for r in reports.values() if r),
        "range_transfer_bytes_upper_bound": sum(r["transfer_bytes_upper_bound"] for r in reports.values() if r),
        "finevision_pinned_revision": recipe["sources"]["finevision"]["revision"],
        "onevision85m_pinned_revision": recipe["sources"]["onevision85m"]["revision"],
        "onevision85m_available_pool": 85000000,
        "curriculum_target_examples": sum(s["target_examples"] for s in recipe["stages"]),
        "curriculum_status": "superseded by user request to adapt existing pretrained weights; no bulk campaign started", "pilot_reports": reports},
    "full_training_qualification": qualified,
    "optimizer_updates_completed": qualified.get("optimizer_updates", 0) if qualified else 0,
    "precision_reference": read("compatibility/stock-cpu-gpu-precision-comparison.json"),
    "benchmarks": benchmarks,
    "corrections": ["Initial transfer counter missed fsspec's cached fetch method; corrected and resampled with bounded range accounting.",
        "CPU-versus-GPU native arithmetic variation is recorded separately; training is compared against the observed legitimate native CPU/GPU reference envelope using unchanged numeric thresholds.",
        "A first gradient attempt retained its 20-token diagnostic capture during a 1032-token image example; this diagnostic hook is now removed before training.",
        "The padded expert implementation exhausted VRAM on a 1024-token image. The corrected ragged BF16 implementation passes both-GPU numerical checks and two full-model image gradient updates.",
        "A two-update exported pilot passes stock image IO and ordinary CLI but failed its document question; capability badges and image IO are not a vision-quality gate.",
        "The 144-update small adapter answers only 2/8 fresh checks in both actual stock BF16 and Q4; it is unqualified and unpublished.",
        "Generated image hashes initially used a different format. The broader partition was rebuilt with the canonical identity helper before any broader gradient updates.",
        "The first BF16 experiment requested a GPU budget larger than any single GPU and was denied. The corrected 64-GiB GPU/128-GiB host profile completed real stock BF16 requests.",
        "The first OCRBench subset has 30 images from only APP-agent and ASCII tasks; it is not a representative OCR benchmark or official score."],
    "remaining": ["complete bounded alignment on the cached broader cohort",
        "export and verify the selected trained projection against native stock preprocessing",
        "OCR, screenshots and general-image quality comparisons against the pretrained donor",
        "stock image/tools/thinking tests, Q4 comparison, publish and registry roundtrip"],
}
if safe_resume:
    status['phase']=safe_resume['stage']
    if single_gpu_pilot and single_gpu_pilot.get('completed') and not safe_resume.get('paused'):
        status['phase']='single-GPU/CPU pilot complete; bounded alignment and stock quality evaluation pending'
    status['prior_broader_attempt_interrupted']=True
    status['prior_broader_updates_durable']=0
    status['broader_stage_updates_attempted']=status['broader_stage_updates_completed']
    status['broader_stage_updates_completed']=0
    status['remaining']=['qualify bounded single-independent-GPU/CPU training pilot',
        'continue bounded pretrained-projection alignment using qualified independent GPU scope',
        'export projection and verify native stock preprocessing',
        'fresh OCR, screenshot, natural-image comparisons and stock tools/thinking/CLI checks',
        'publish both precision variants only after quality checks and registry verification']
    status['corrections']+=['The 388-update broader run failed on October 6 before a new checkpoint was saved; these are attempted updates, not a durable trained artifact.',
        'October 7 work follows the Desktop incident notes: quarantined GPU84 and unregistered GPU01 are preserved; the new pilot uses independent GPU82 plus CPU offload.',
        'Periodic checkpoints now include optimizer state, epoch order/cursor, RNG state and data/cache identities; real checkpoint resumption still needs validation.']
    if safe_resume.get('paused'):
        status['latest_user_direction']='Push the repository to main, gracefully unload own weights, and pause development until the user returns.'
        status['remaining'].insert(0,'wait for the user to resume development; do not automatically restart jobs')
        status['corrections'].append('The October 7 pilot was interrupted at the user request after initial validation and before its first optimizer update; its own process and lease exited and GPU82 returned to 19 MiB used.')
(ROOT / "vision-status.json").write_text(json.dumps(status, indent=2) + "\n")
handoff = {"utc": status["utc"], "host": "basilisk", "run_root": str(ROOT), "destination": status["destination"],
    "vision_published": False, "phase": status["phase"], "accepted_real_pilot_examples": status["data"]["accepted_pilot_examples"],
    "optimizer_updates_completed": status["optimizer_updates_completed"], "status_artifact": str(ROOT / "vision-status.json"),
    "alignment_optimizer_updates_completed": status['alignment_optimizer_updates_completed'],
    "broader_stage_updates_completed":status['broader_stage_updates_completed'],
    "broader_stage_updates_attempted":status.get('broader_stage_updates_attempted'),
    "small_adapter_bf16_checks":status['current_small_adapter_bf16_checks'],
    "safe_resume":safe_resume,
    "prior_broader_updates_durable":status.get('prior_broader_updates_durable'),
    "confirmed": ["Stock Ollama 0.35 accepts normal AFMoE model plus Qwen vision-projector registry layers and completes image API requests.",
        "The pretrained native vision feature tap changes with image content; tiny F32 decoder/gradient/grouped routing checks pass.",
        "The full released Kolibri Q4 plus a learned vision GGUF completes public image API and ordinary CLI; packaging is supported without a runtime adapter.",
        "The existing trained donor projection is transferred through shared vocabulary alignment, with Kolibri language weights untouched; initial stock answers fail quality gates.",
        "Large public training inventories are pinned and real image/answer rows are sampled through bounded Parquet range reads."],
    "unknowns": ["The small adapter fails OCR quality checks. No updated broader checkpoint survived the failed run; the saved 144-update adapter remains the current candidate.",
        "Best-in-class performance is a requested target, not a measured result."],
    "actions": ["All new work is in the unique run directory; existing text release and peer weights are untouched.",
        "Historical October 6 pilots used broker-scoped GPU0/GPU2 leases and preserved the then-existing Voryn GPU1 lease; that scope is superseded by the Desktop incident containment.",
        "Historical October 6 cleanup required one forced own-lease release after an unrelated new CUDA PID prevented baseline settling; own allocations were verified gone first.",
        "October 7 used only independent GPU82 plus CPU offload. On user pause, SIGINT ended the own worker, its wrapper released the lease, and discovery showed GPU82 at 19 MiB with no active external leases. No peer processes were stopped."],
    "next": status["remaining"]}
(ROOT / "debug-handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
print(json.dumps({"phase": status["phase"], "accepted_examples": status["data"]["accepted_pilot_examples"],
    "broader_stage_updates_completed":status['broader_stage_updates_completed'],
    "optimizer_updates_completed": status["optimizer_updates_completed"], "vision_published": False}))
