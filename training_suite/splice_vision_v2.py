#!/usr/bin/env python3
"""Vision-preserving splice — v2.

v1 used `full_model.save_pretrained()` which on transformers 5.5 +
Qwen3_5ForConditionalGeneration emits state_dicts with extra wrappers,
producing keys like `model.language_model.language_model.language_model.*`
and `model.language_model.visual.*`. That breaks the GGUF converter.

v2 takes full control of serialization: load → splice text weights →
save the raw state_dict with safetensors + manual sharding + hand-written
index.json + config.json copied from base.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Qwen3_5ForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_ID = "Qwen/Qwen3.5-9B"
OUTPUT_NAME = "qwen3.5-9b-qwen3.6-reasoning-distilled"
TEXT_MERGED_DIR = ROOT / "outputs" / "merged" / OUTPUT_NAME
OUT_DIR = ROOT / "outputs" / "merged_vision" / OUTPUT_NAME

SHARD_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB per shard (HF convention)


def log(msg: str) -> None:
    print(f"[splice-v2] {msg}", flush=True)


def remap_text_to_full(k: str) -> str:
    if k.startswith("model.") and not k.startswith("model.language_model."):
        return "model.language_model." + k[len("model.") :]
    return k


def shard_state_dict(state: dict, shard_size: int):
    """Greedy bin-packing of tensors into shards under shard_size bytes."""
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
        log(f"Cleaning previous output: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    log(f"Loading base multimodal {BASE_MODEL_ID}")
    t0 = time.time()
    full_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    full_model.eval()
    state = full_model.state_dict()
    log(f"  loaded in {time.time() - t0:.1f}s, {len(state)} tensors")

    # Sanity: exactly 426 language_model text keys + 333 visual + 1 lm_head
    text_count = sum(1 for k in state if k.startswith("model.language_model"))
    visual_count = sum(1 for k in state if "visual" in k)
    log(f"  text tensors: {text_count}, visual tensors: {visual_count}, lm_head: "
        f"{'yes' if 'lm_head.weight' in state else 'no'}")
    assert text_count == 426 and visual_count == 333, "unexpected tensor layout"

    # Load our trained text weights
    text_st_path = TEXT_MERGED_DIR / "model.safetensors"
    log(f"Loading trained text weights from {text_st_path}")
    trained_text = load_safetensors(str(text_st_path), device="cpu")
    log(f"  loaded {len(trained_text)} tensors")

    # Splice: overwrite text keys in the in-memory state dict
    with torch.no_grad():
        overwritten = 0
        for k, v in trained_text.items():
            target = remap_text_to_full(k)
            if target not in state:
                raise RuntimeError(f"trained key {k!r} → {target!r} missing from base")
            dst = state[target]
            if dst.shape != v.shape:
                raise RuntimeError(
                    f"shape mismatch for {target}: full={tuple(dst.shape)} trained={tuple(v.shape)}"
                )
            state[target] = v.to(dst.dtype).contiguous()
            overwritten += 1
    log(f"Overwrote {overwritten} text-subnet tensors; {visual_count} vision tensors preserved")

    # Free the HF model so sharding has RAM
    del full_model
    import gc
    gc.collect()

    # Detach + contiguous + CPU — some tensors may share memory with safetensors mmaps
    log("Detaching and making tensors contiguous")
    t0 = time.time()
    clean_state: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        clean_state[k] = v.detach().contiguous().clone()
    del state
    gc.collect()
    log(f"  done in {time.time() - t0:.1f}s")

    # Build config.json from the base config (already correct)
    log("Copying config.json from base")
    cfg = AutoConfig.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    cfg.save_pretrained(str(OUT_DIR))

    # Chat template + tokenizer + processor files
    log("Copying tokenizer + processor from base")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tok.save_pretrained(str(OUT_DIR))
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        proc.save_pretrained(str(OUT_DIR))
        log("  AutoProcessor saved (image preprocessor included)")
    except Exception as e:
        log(f"  AutoProcessor save failed: {e}")

    # Generation config
    try:
        from transformers import GenerationConfig
        gen = GenerationConfig.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        gen.save_pretrained(str(OUT_DIR))
        log("  generation_config saved")
    except Exception:
        log("  (no generation_config)")

    # Shard and save
    log(f"Sharding state dict at {SHARD_SIZE//(1024**3)} GB per shard")
    shards = shard_state_dict(clean_state, SHARD_SIZE)
    log(f"  -> {len(shards)} shards")

    weight_map: dict[str, str] = {}
    total_bytes = 0
    t0 = time.time()
    for i, shard in enumerate(shards, 1):
        fname = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        out = OUT_DIR / fname
        save_safetensors(shard, str(out), metadata={"format": "pt"})
        sz = out.stat().st_size
        total_bytes += sz
        log(f"  wrote {fname} ({sz/1e9:.2f} GB, {len(shard)} tensors)")
        for k in shard.keys():
            weight_map[k] = fname

    # Write index.json in the exact format HF expects
    index = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    (OUT_DIR / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    log(f"Wrote model.safetensors.index.json (total_size={total_bytes/1e9:.2f} GB)")
    log(f"Total save time: {time.time() - t0:.1f}s")

    # Final listing
    log("Output files:")
    for p in sorted(OUT_DIR.iterdir()):
        log(f"  {p.stat().st_size/1e6:>10.2f} MB  {p.name}")

    # Verify: reload the index and confirm key structure
    log("\nVerification — reloading index.json and checking key structure:")
    idx = json.loads((OUT_DIR / "model.safetensors.index.json").read_text())
    keys = list(idx["weight_map"].keys())
    visual = [k for k in keys if "visual" in k]
    text = [k for k in keys if k.startswith("model.language_model.") and "visual" not in k]
    bad = [k for k in keys if k.count("language_model") >= 2]
    log(f"  total: {len(keys)}")
    log(f"  model.visual.*: {len(visual)}")
    log(f"  model.language_model.* (text): {len(text)}")
    log(f"  keys with language_model x2+ (BAD): {len(bad)}")
    assert not bad, "triple-nesting still present!"
    assert len(visual) == 333, f"expected 333 visual keys, got {len(visual)}"


if __name__ == "__main__":
    main()
