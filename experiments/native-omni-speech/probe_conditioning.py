"""Check a weights-preparation shortcut against Qwen's actual tensor assembly.

CPU only, synthetic weights. This proves tensor-preparation parity, not audio
quality or support in an unmodified Ollama release.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeTalkerResizeMLP,
)


def probe() -> dict:
    torch.manual_seed(17)
    config = SimpleNamespace(
        thinker_hidden_size=16,
        text_config=SimpleNamespace(
            intermediate_size=20, hidden_size=12, hidden_act="silu"
        ),
        codec_nothink_id=1,
        codec_think_bos_id=2,
        codec_think_eos_id=3,
        codec_pad_id=4,
        codec_bos_id=5,
    )
    text = torch.nn.Embedding(64, 16, device="cpu")
    codec = torch.nn.Embedding(24, 12, device="cpu")
    projection = Qwen3OmniMoeTalkerResizeMLP(config).to("cpu").eval()
    full_config = SimpleNamespace(talker_config=config, tts_pad_token_id=33)
    reference = Qwen3OmniMoeForConditionalGeneration._get_talker_assistant_parts

    def model(projector):
        return SimpleNamespace(
            config=full_config,
            talker=SimpleNamespace(
                text_projection=projector,
                get_input_embeddings=lambda: codec,
                dtype=torch.float32,
            ),
        )

    cases = []
    with torch.inference_mode():
        # Compile a positionwise projection into the original vocabulary table.
        folded = projection(text.weight)
        specials = torch.tensor([[30, 31, 33]], device="cpu")
        original_specials = projection(text(specials)).chunk(3, 1)
        folded_specials = torch.nn.functional.embedding(specials, folded).chunk(3, 1)
        for length in [4, 9, 41]:
            tokens = torch.randint(0, 30, (1, length), device="cpu")
            for speaker in [7, 8, 9]:
                original = reference(
                    model(projection),
                    0,
                    length,
                    speaker,
                    text(tokens),
                    original_specials[2],
                    original_specials[0],
                    original_specials[1],
                )
                prepared = reference(
                    model(torch.nn.Identity()),
                    0,
                    length,
                    speaker,
                    torch.nn.functional.embedding(tokens, folded),
                    folded_specials[2],
                    folded_specials[0],
                    folded_specials[1],
                )
                errors = [
                    float((a - b).abs().max()) for a, b in zip(original, prepared)
                ]
                assert all(
                    torch.allclose(a, b, atol=1e-6, rtol=1e-5)
                    for a, b in zip(original, prepared)
                )
                cases.append(
                    {
                        "tokens": length,
                        "speaker": speaker,
                        "max_absolute_errors": errors,
                    }
                )
    source = inspect.getsource(reference)
    return {
        "ok": True,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "reference_function": reference.__qualname__,
        "reference_function_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "synthetic_weights": True,
        "device": "cpu",
        "cuda_initialized": torch.cuda.is_initialized(),
        "cases": cases,
        "prepared_embedding_shape": list(folded.shape),
        "thinker_transformer_executed": False,
        "audio_generation_tested": False,
        "stock_ollama_compatibility_proven": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = probe()
    serialized = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
