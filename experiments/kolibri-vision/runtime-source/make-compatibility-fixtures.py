"""Small synthetic fixtures for stock-runtime interface qualification only.

These random weights are not a trained vision model and must never be published.
The language fixture uses the same AFMoE architecture and tokenizer as our
Kolibri release. The image fixture uses Ollama's standard CLIP + MLP interface.
"""
from pathlib import Path
import json
import sys

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / "training_suite/vendor/kolibri-port/llama.cpp/gguf-py"))
import gguf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = WORKSPACE / "training_suite/outputs/kolibri-1-20261004/Kolibri-1-stock-BF16-release.gguf"


def main():
    destination = ROOT / "compatibility/head128"
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(6100626)
    language = destination / "afmoe-interface-fixture.gguf"
    projector = destination / "clip-interface-fixture.gguf"
    if language.exists() or projector.exists():
        raise FileExistsError("Fixtures already exist; inspect the saved proof before replacing.")
    source = gguf.GGUFReader(SOURCE)
    overrides = {
        "general.name": "Synthetic Kolibri vision interface fixture; NOT TRAINED",
        "general.file_type": 0,
        "afmoe.block_count": 8,
        "afmoe.context_length": 2048,
        "afmoe.embedding_length": 32,
        "afmoe.feed_forward_length": 32,
        "afmoe.attention.head_count": 4,
        "afmoe.attention.head_count_kv": 2,
        "afmoe.attention.key_length": 128,
        "afmoe.attention.value_length": 128,
        "afmoe.rope.dimension_count": 128,
        "afmoe.expert_count": 8,
        "afmoe.expert_used_count": 2,
        "afmoe.expert_feed_forward_length": 32,
        "afmoe.expert_shared_feed_forward_length": 32,
        "afmoe.attention.sliding_window": 33,
        "afmoe.attention.sliding_window_pattern": [True] * 7 + [False],
    }
    writer = gguf.GGUFWriter(language, "afmoe")
    for key, field in source.fields.items():
        if key.startswith(("GGUF.", "kolibri.")) or key == "general.architecture":
            continue
        writer.add_key_value(key, overrides.get(key, field.contents()), field.types[0],
                             field.types[-1] if len(field.types) > 1 else None)
    shapes = {
        "attn_q.weight": (512, 32), "attn_k.weight": (256, 32),
        "attn_v.weight": (256, 32), "attn_output.weight": (32, 512),
        "attn_q_norm.weight": (128,), "attn_k_norm.weight": (128,),
        "attn_norm.weight": (32,), "post_attention_norm.weight": (32,),
        "ffn_norm.weight": (32,), "post_ffw_norm.weight": (32,),
        "ffn_gate_inp.weight": (8, 32), "exp_probs_b.bias": (8,),
        "ffn_gate_shexp.weight": (32, 32), "ffn_up_shexp.weight": (32, 32),
        "ffn_down_shexp.weight": (32, 32),
        "ffn_gate_exps.weight": (8, 32, 32),
        "ffn_up_exps.weight": (8, 32, 32),
        "ffn_down_exps.weight": (8, 32, 32), "attn_gate.weight": (512, 32),
    }
    writer.add_tensor("token_embd.weight", rng.normal(0, .01, (128000, 32)).astype(np.float32))
    writer.add_tensor("output.weight", rng.normal(0, .1, (128000, 32)).astype(np.float32))
    writer.add_tensor("output_norm.weight", np.ones(32, np.float32))
    for layer in range(8):
        for name, shape in shapes.items():
            if "norm.weight" in name:
                value = np.ones(shape, np.float32)
            elif name in {"attn_gate.weight", "exp_probs_b.bias"}:
                value = np.zeros(shape, np.float32)
            else:
                value = rng.normal(0, .035, shape).astype(np.float32)
            writer.add_tensor(f"blk.{layer}.{name}", value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    writer = gguf.GGUFWriter(projector, "clip")
    writer.add_name("Synthetic image interface fixture; NOT TRAINED")
    writer.add_string("general.type", "mmproj")
    writer.add_bool("clip.has_vision_encoder", True)
    writer.add_bool("clip.has_text_encoder", False)
    writer.add_bool("clip.has_llava_projector", True)
    writer.add_string("clip.projector_type", "mlp")
    writer.add_bool("clip.use_gelu", True)
    for key, value in {
        "image_size": 28, "patch_size": 14, "embedding_length": 32,
        "feed_forward_length": 64, "projection_dim": 32,
        "attention.head_count": 4, "block_count": 2,
    }.items():
        writer.add_uint32("clip.vision." + key, value)
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-6)
    writer.add_array("clip.vision.image_mean", [0.5] * 3)
    writer.add_array("clip.vision.image_std", [0.5] * 3)
    encoder_shapes = {
        "v.patch_embd.weight": (32, 3, 14, 14),
        "v.position_embd.weight": (4, 32),
        "mm.0.weight": (64, 32), "mm.0.bias": (64,),
        "mm.2.weight": (32, 64), "mm.2.bias": (32,),
    }
    for layer in range(2):
        for name in ["attn_q", "attn_k", "attn_v", "attn_out"]:
            encoder_shapes[f"v.blk.{layer}.{name}.weight"] = (32, 32)
            encoder_shapes[f"v.blk.{layer}.{name}.bias"] = (32,)
        for name in ["ln1", "ln2"]:
            encoder_shapes[f"v.blk.{layer}.{name}.weight"] = (32,)
            encoder_shapes[f"v.blk.{layer}.{name}.bias"] = (32,)
        encoder_shapes[f"v.blk.{layer}.ffn_up.weight"] = (64, 32)
        encoder_shapes[f"v.blk.{layer}.ffn_up.bias"] = (64,)
        encoder_shapes[f"v.blk.{layer}.ffn_down.weight"] = (32, 64)
        encoder_shapes[f"v.blk.{layer}.ffn_down.bias"] = (32,)
    for name, shape in encoder_shapes.items():
        if ".ln" in name and name.endswith("weight"):
            value = np.ones(shape, np.float32)
        elif name.endswith("bias"):
            value = np.zeros(shape, np.float32)
        else:
            value = rng.normal(0, .03, shape).astype(np.float32)
        writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(json.dumps({"scope": "synthetic interface test; no trained capability claim",
                      "files": [{"path": str(p), "bytes": p.stat().st_size}
                                for p in [language, projector]]}))


if __name__ == "__main__":
    main()
