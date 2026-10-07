"""Pin public dataset inventories and create an executable vision curriculum.

Upstream example counts describe available pools, never completed training.
The training driver must record actual accepted rows and image/answer tokens.
"""
from datetime import datetime, timezone
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data-sources"
fine = json.loads((SOURCES / "HuggingFaceM4--FineVision-info.json").read_text())
one = json.loads((SOURCES / "mvp-lab--LLaVA-OneVision-1.5-Mid-Training-85M-info.json").read_text())
groups = {
    "ocr_documents": {"weight": .40, "configs": ["DoclingMatix", "olmOCR-mix-0225-documents",
        "olmOCR-mix-0225-books", "docvqa", "sroie", "textocr(gpt4v)", "ocrvqa", "textvqa",
        "rendered_text", "iam", "invoices_receipts", "mathwriting-google", "ureader_cap", "ureader_qa_processed"]},
    "charts_tables": {"weight": .20, "configs": ["CoSyn_400k_chart", "CoSyn_400k_table",
        "chartqa", "SynthChartNet", "Unichart", "dvqa", "plotqa", "hitab", "tat_qa", "tabmwp"]},
    "screenshots_ui": {"weight": .15, "configs": ["screenqa", "screen2words", "groundui",
        "aguvis-stage-1", "websight"]},
    "general_images": {"weight": .25, "configs": ["sharegpt4v(coco)", "sharegpt4v(knowledge)",
        "sharegpt4o", "densefusion_1m", "objects365_qa", "vqav2", "lvis_instruct4v",
        "localized_narratives", "spatialsense", "vsr", "visual7w"]},
}
inventory = {s["rfilename"] for s in fine["siblings"]}
known = {c["config_name"] for c in fine["cardData"]["configs"]}
stats = {d["config_name"]: d for d in fine["cardData"]["dataset_info"]}
for group in groups.values():
    missing = [n for n in group["configs"] if n not in known]
    if missing:
        raise ValueError(f"Unknown dataset configurations: {missing}")
    group["configurations"] = []
    for name in group.pop("configs"):
        files = sorted(p for p in inventory if p.startswith(name + "/train-") and p.endswith(".parquet"))
        if not files:
            raise ValueError(f"No pinned training shards for {name}")
        group["configurations"].append({"name": name, "shards": files,
            "available_examples": sum(s["num_examples"] for s in stats[name]["splits"] if s["name"] == "train")})
recipe = {
    "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
    "destination": "robit/kolibri-1-vision", "status": "planned; no training count implied",
    "sources": {"finevision": {"repo": fine["id"], "revision": fine["sha"], "split": "train",
        "upstream_url": "https://huggingface.co/datasets/HuggingFaceM4/FineVision"},
        "onevision85m": {"repo": one["id"], "revision": one["sha"], "split": "train",
        "advertised_examples": 85000000, "all_training_shards": sorted(s["rfilename"] for s in one["siblings"] if s["rfilename"].endswith(".parquet")),
        "upstream_url": "https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M"}},
    "instruction_groups": groups,
    "stages": [
        {"name": "projector_alignment", "target_examples": 1500000, "pool": "onevision85m",
         "image_tokens": [256, 1024], "trainable": ["final_native_vision_projection"],
         "start_gate": "full decoder numerical qualification, finite gradients, measured throughput"},
        {"name": "multimodal_instruction", "target_examples": 6000000, "pool": "finevision",
         "image_tokens": [1024, 2048], "trainable": ["vision_projection", "merged_decoder_lora"],
         "start_gate": "projector alignment and implemented/verified LoRA export"},
        {"name": "high_resolution_ocr_ui", "target_examples": 2000000, "pool": "finevision",
         "image_tokens": [1024, 4096], "trainable": ["vision_projection", "merged_decoder_lora"],
         "start_gate": "bounded high-resolution memory and clean held-out document evaluation"}],
    "quality": {"per_turn_minimum": {"relevance": 4, "visual_dependency": 3,
        "image_correspondence": 4, "formatting": 4}, "images_required": True,
        "decoded_image_pixel_limit": 32000000, "example_dedup": "decoded RGB SHA256 + normalized dialogue",
        "holdout_exclusion": "SHA256 and perceptual image hashes; benchmark inventories required before scaling",
        "training_accounting": ["raw_rows", "rejected_rows_by_reason", "unique_images", "accepted_examples",
            "answer_tokens", "image_tokens", "source_revision", "source_shard", "source_row"]},
    "evaluation_gates": {"vision": ["OCRBench-v2", "OmniDocBench", "DocVQA", "TextVQA", "ChartQA",
        "ScreenSpot-Pro", "MMMU", "RealWorldQA"], "counterfactuals": ["blank image", "swapped image", "same text different image"],
        "language": "existing 117-test text/tools/thinking suite + image/tool-result conversations",
        "release": "stock Ollama CLI and API, BF16/Q4 comparison, registry roundtrip; no best-in-class claim without measured comparisons"},
    "supplementary_sources": [
        {"repo": "nvidia/Nemotron-VLM-Dataset-v2", "revision": "51f4f4d219315c3283950994d4eb3d7fc30aa87b",
         "status": "media resolver required for label-only subsets; not counted as image training"},
        {"repo": "lightonai/LightOnOCR-mix-0126", "revision": "af0218b88fc337468d91f9c107ae33453f65cf30",
         "status": "source PDF/page resolver required; not counted as image training"}],
    "storage": {"max_pilot_download_bytes": 2 * 2**30, "max_pilot_artifact_bytes": 128 * 2**20,
        "streaming_required": True, "shard_cache_max_bytes": 32 * 2**30, "minimum_disk_free_bytes": 64 * 2**30},
}
(ROOT / "data-mixture.json").write_text(json.dumps(recipe, indent=2) + "\n")
print(json.dumps({"groups": {k: {"configs": len(v["configurations"]), "available_examples": sum(c["available_examples"] for c in v["configurations"])} for k,v in groups.items()},
    "onevision85m_shards": len(recipe["sources"]["onevision85m"]["all_training_shards"]), "trained_examples": 0}))
