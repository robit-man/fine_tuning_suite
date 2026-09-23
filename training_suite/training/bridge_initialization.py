from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Any

from training_suite.training.omni_encoder_bridge import (
    BRIDGE_SCHEMA,
    Qwen3AFinalProjectorBridge,
    Qwen3AFinalProjectorConfig,
    save_bridge_checkpoint,
)

TOKEN_ALIGNMENT_SCHEMA = "robit.qwen3a-audio-bridge-token-alignment.v1"
NORMAL_TOKEN_TYPES = {1, 6}


class BridgeInitializationError(RuntimeError):
    """Raised when token-space initialization cannot produce a valid bridge."""


def _require_dependencies():
    try:
        import numpy as np  # type: ignore
        import torch
        from gguf import GGUFReader, dequantize  # type: ignore
    except Exception as exc:  # pragma: no cover - environment guard
        raise BridgeInitializationError(
            "token-space initialization requires torch, numpy, and gguf"
        ) from exc
    return np, torch, GGUFReader, dequantize


def _find_tensor(reader: Any, name: str) -> Any:
    for tensor in reader.tensors:
        if tensor.name == name:
            return tensor
    raise BridgeInitializationError(f"GGUF has no tensor named {name!r}")


def _tokens_and_types(reader: Any) -> tuple[list[str], list[int]]:
    try:
        tokens = list(reader.fields["tokenizer.ggml.tokens"].contents())
    except KeyError as exc:
        raise BridgeInitializationError("GGUF has no tokenizer token list") from exc
    field = reader.fields.get("tokenizer.ggml.token_type")
    types = list(field.contents()) if field is not None else [1] * len(tokens)
    if len(tokens) != len(types):
        raise BridgeInitializationError("token and token-type lists have different lengths")
    return tokens, [int(value) for value in types]


def aligned_token_ids(
    source_reader: Any,
    target_reader: Any,
    *,
    max_tokens: int,
    holdout_tokens: int,
    seed: int,
) -> tuple[list[int], list[int], int]:
    source_tokens, source_types = _tokens_and_types(source_reader)
    target_tokens, target_types = _tokens_and_types(target_reader)
    source_by_token = {
        token: index
        for index, (token, token_type) in enumerate(zip(source_tokens, source_types))
        if token_type in NORMAL_TOKEN_TYPES
    }
    target_by_token = {
        token: index
        for index, (token, token_type) in enumerate(zip(target_tokens, target_types))
        if token_type in NORMAL_TOKEN_TYPES
    }
    shared = sorted(set(source_by_token).intersection(target_by_token))
    minimum = holdout_tokens + 2
    if len(shared) < minimum:
        raise BridgeInitializationError(
            f"only {len(shared)} normal/byte tokens overlap; need at least {minimum}"
        )
    rng = random.Random(seed)
    rng.shuffle(shared)
    selected = shared[: min(len(shared), max_tokens + holdout_tokens)]
    return (
        [source_by_token[token] for token in selected],
        [target_by_token[token] for token in selected],
        len(shared),
    )


def _selected_embeddings(reader: Any, token_ids: list[int], dequantize: Any, np: Any):
    tensor = _find_tensor(reader, "token_embd.weight")
    values = dequantize(tensor.data, tensor.tensor_type)
    if values.ndim != 2 or values.shape[0] <= max(token_ids):
        raise BridgeInitializationError(
            f"unexpected token embedding storage shape: {tuple(values.shape)}"
        )
    selected = np.asarray(values[token_ids], dtype=np.float32).copy()
    del values
    gc.collect()
    return selected


def _fit_affine_ridge(source, target, *, ridge: float, torch: Any):
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        raise BridgeInitializationError("aligned embedding matrices have invalid shapes")
    ones = torch.ones((source.shape[0], 1), device=source.device, dtype=source.dtype)
    design = torch.cat((source, ones), dim=1)
    gram = design.T @ design / design.shape[0]
    cross = design.T @ target / design.shape[0]
    scale = torch.diagonal(gram[:-1, :-1]).mean()
    regularizer = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    regularizer[-1, -1] = 0
    coefficients = torch.linalg.solve(gram + regularizer * (ridge * scale), cross)
    return coefficients[:-1], coefficients[-1]


def _score(source, target, mapping, offset, *, torch: Any) -> dict[str, float]:
    predicted = source @ mapping + offset
    cosine = torch.nn.functional.cosine_similarity(predicted, target, dim=-1).mean()
    rmse = torch.sqrt(torch.mean((predicted - target) ** 2))
    target_rms = torch.sqrt(torch.mean(target**2))
    return {
        "mean_cosine_similarity": float(cosine.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "normalized_rmse": float((rmse / target_rms).detach().cpu()),
    }


def initialize_bridge_from_token_spaces(
    *,
    omni_text_gguf: Path,
    target_text_gguf: Path,
    omni_projector_gguf: Path,
    output_dir: Path,
    device: str = "cpu",
    max_tokens: int = 32_768,
    holdout_tokens: int = 2_048,
    ridge: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit an affine text-space map and fuse it with Omni's final audio layer."""
    if max_tokens <= 1 or holdout_tokens <= 0 or ridge <= 0:
        raise ValueError("max_tokens, holdout_tokens, and ridge must be positive")
    np, torch, GGUFReader, dequantize = _require_dependencies()
    source_path = omni_text_gguf.expanduser().resolve()
    target_path = target_text_gguf.expanduser().resolve()
    projector_path = omni_projector_gguf.expanduser().resolve()
    for path in (source_path, target_path, projector_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_reader = GGUFReader(str(source_path), "r")
    target_reader = GGUFReader(str(target_path), "r")
    source_ids, target_ids, shared_count = aligned_token_ids(
        source_reader,
        target_reader,
        max_tokens=max_tokens,
        holdout_tokens=holdout_tokens,
        seed=seed,
    )
    source_embeddings = _selected_embeddings(source_reader, source_ids, dequantize, np)
    target_embeddings = _selected_embeddings(target_reader, target_ids, dequantize, np)
    sample_count = source_embeddings.shape[0]
    holdout_count = min(holdout_tokens, sample_count - 2)
    train_count = sample_count - holdout_count

    fit_device = torch.device(device)
    source_tensor = torch.from_numpy(source_embeddings).to(fit_device)
    target_tensor = torch.from_numpy(target_embeddings).to(fit_device)
    mapping, offset = _fit_affine_ridge(
        source_tensor[:train_count],
        target_tensor[:train_count],
        ridge=ridge,
        torch=torch,
    )
    train_metrics = _score(
        source_tensor[:train_count],
        target_tensor[:train_count],
        mapping,
        offset,
        torch=torch,
    )
    holdout_metrics = _score(
        source_tensor[train_count:],
        target_tensor[train_count:],
        mapping,
        offset,
        torch=torch,
    )

    projector_reader = GGUFReader(str(projector_path), "r")
    donor_weight_tensor = _find_tensor(projector_reader, "mm.a.mlp.2.weight")
    donor_bias_tensor = _find_tensor(projector_reader, "mm.a.mlp.2.bias")
    donor_weight = torch.from_numpy(
        dequantize(donor_weight_tensor.data, donor_weight_tensor.tensor_type).copy()
    ).to(device=fit_device, dtype=torch.float32)
    donor_bias = torch.from_numpy(
        dequantize(donor_bias_tensor.data, donor_bias_tensor.tensor_type).copy()
    ).to(device=fit_device, dtype=torch.float32)
    if donor_weight.shape[0] != mapping.shape[0]:
        raise BridgeInitializationError(
            "Omni final projector output does not match source text embedding width"
        )
    fused_weight = mapping.T @ donor_weight
    fused_bias = donor_bias @ mapping + offset

    config = Qwen3AFinalProjectorConfig(
        source_width=int(fused_weight.shape[1]),
        target_width=int(fused_weight.shape[0]),
    )
    bridge = Qwen3AFinalProjectorBridge(config)
    with torch.no_grad():
        bridge.projection.weight.copy_(fused_weight.cpu())
        bridge.projection.bias.copy_(fused_bias.cpu())
    report = {
        "schema": TOKEN_ALIGNMENT_SCHEMA,
        "bridge_schema": BRIDGE_SCHEMA,
        "method": "shared-token affine ridge followed by exact final-projector fusion",
        "sources": {
            "omni_text_gguf": str(source_path),
            "target_text_gguf": str(target_path),
            "omni_projector_gguf": str(projector_path),
        },
        "shared_normal_or_byte_tokens": shared_count,
        "training_tokens": train_count,
        "holdout_tokens": holdout_count,
        "ridge": ridge,
        "seed": seed,
        "device": str(fit_device),
        "train_metrics": train_metrics,
        "holdout_metrics": holdout_metrics,
        "fused_projection_shape": list(fused_weight.shape),
        "language_weights_changed": False,
        "omni_thinker_required_after_export": False,
    }
    destination = output_dir.expanduser().resolve()
    checkpoint_report = save_bridge_checkpoint(
        destination,
        bridge,
        manifest=report,
    )
    report["checkpoint"] = checkpoint_report
    report_path = destination / "initialization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["report"] = str(report_path)
    return report
