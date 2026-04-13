#!/usr/bin/env python3
"""Vision splice for R7 — copies vision tensors from base Qwen3.5-9B multimodal
into our R7 trained text weights, producing a full vision+text model.

Based on splice_vision_v2.py (proven in R3).
"""
from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_ID = "Qwen/Qwen3.5-9B"
TEXT_MERGED_DIR = ROOT / "outputs" / "merged" / "qwen3.5-9b-r7-additive"
OUT_DIR = ROOT / "outputs" / "merged_vision" / "qwen3.5-9b-r7-additive"
SHARD_SIZE = 5 * 1024 * 1024 * 1024


def log(msg: str) -> None:
    print(f"[splice-r7] {msg}", flush=True)


def remap_text_to_full(k: str) -> str:
    if k.startswith("model.") and not k.startswith("model.language_model."):
        return "model.language_model." + k[len("model."):]
    return k


def shard_state_dict(state: dict, shard_size: int):
    shards: list[dict] = [{}]
    current_size = 0
    for k, v in state.items():
        sz = v.numel() * v.element_size()
        if current_size + sz > shard_size and shards[-1]:
            shards.append({})
            current_size = 0
        shards[-1][k] = v
        current_size += sz
    return shards


def main() -> None:
    assert TEXT_MERGED_DIR.exists(), f"missing {TEXT_MERGED_DIR}"
    if OUT_DIR.exists():
        log(f"Cleaning {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    log(f"Loading base multimodal {BASE_MODEL_ID}")
    t0 = time.time()
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    full_model.eval()
    state = full_model.state_dict()
    log(f"  loaded in {time.time() - t0:.1f}s, {len(state)} tensors")

    text_count = sum(1 for k in state if k.startswith("model.language_model"))
    visual_count = sum(1 for k in state if "visual" in k)
    log(f"  text: {text_count}, visual: {visual_count}")

    # Load R7 trained text weights
    text_st_path = TEXT_MERGED_DIR / "model.safetensors"
    log(f"Loading R7 text weights from {text_st_path}")
    trained_text = load_safetensors(str(text_st_path), device="cpu")
    log(f"  loaded {len(trained_text)} tensors")

    # Splice: overwrite text keys
    with torch.no_grad():
        overwritten = 0
        for k, v in trained_text.items():
            target = remap_text_to_full(k)
            if target not in state:
                raise RuntimeError(f"key {k!r} -> {target!r} missing from base")
            dst = state[target]
            if dst.shape != v.shape:
                raise RuntimeError(f"shape mismatch {target}: {tuple(dst.shape)} vs {tuple(v.shape)}")
            state[target] = v.to(dst.dtype).contiguous()
            overwritten += 1
    log(f"Overwrote {overwritten} text tensors; {visual_count} vision tensors preserved")

    del full_model, trained_text
    gc.collect()

    # Detach + clone
    log("Detaching tensors")
    clean_state = {k: v.detach().contiguous().clone() for k, v in state.items()}
    del state
    gc.collect()

    # Save config, tokenizer, processor
    log("Saving config + tokenizer + processor")
    cfg = AutoConfig.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    cfg.save_pretrained(str(OUT_DIR))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tok.save_pretrained(str(OUT_DIR))
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        proc.save_pretrained(str(OUT_DIR))
        log("  processor saved")
    except Exception as e:
        log(f"  processor: {e}")

    # Shard and save
    log(f"Sharding at {SHARD_SIZE // (1024**3)} GB")
    shards = shard_state_dict(clean_state, SHARD_SIZE)
    log(f"  {len(shards)} shards")

    weight_map: dict[str, str] = {}
    total_bytes = 0
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        out = OUT_DIR / fname
        save_safetensors(shard, str(out), metadata={"format": "pt"})
        sz = out.stat().st_size
        total_bytes += sz
        log(f"  {fname} ({sz / 1e9:.2f} GB, {len(shard)} tensors)")
        for k in shard:
            weight_map[k] = fname

    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    log(f"Total: {total_bytes / 1e9:.2f} GB")

    # Verify
    keys = list(weight_map.keys())
    vis = [k for k in keys if "visual" in k]
    txt = [k for k in keys if k.startswith("model.language_model.") and "visual" not in k]
    bad = [k for k in keys if k.count("language_model") >= 2]
    log(f"\nVerify: total={len(keys)} visual={len(vis)} text={len(txt)} bad_nesting={len(bad)}")
    assert not bad, "double nesting detected!"
    assert len(vis) >= 300, f"expected 300+ visual keys, got {len(vis)}"
    log("DONE — vision splice complete!")


if __name__ == "__main__":
    main()
