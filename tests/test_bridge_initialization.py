from __future__ import annotations

import pytest

from training_suite.training.bridge_initialization import (
    TOKEN_ALIGNMENT_SCHEMA,
    initialize_bridge_from_token_spaces,
)


def _write_text_gguf(path, embeddings, tokens) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    writer = gguf.GGUFWriter(str(path), arch="llama")
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types([1] * len(tokens))
    writer.add_tensor("token_embd.weight", np.asarray(embeddings, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _write_projector(path, weight, bias) -> None:
    gguf = pytest.importorskip("gguf")
    np = pytest.importorskip("numpy")
    writer = gguf.GGUFWriter(str(path), arch="clip")
    writer.add_tensor("mm.a.mlp.2.weight", np.asarray(weight, dtype=np.float32))
    writer.add_tensor("mm.a.mlp.2.bias", np.asarray(bias, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_token_space_initializer_recovers_and_fuses_an_affine_map(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from training_suite.training.omni_encoder_bridge import load_bridge_checkpoint

    rng = np.random.default_rng(7)
    source = rng.normal(size=(24, 3)).astype(np.float32)
    mapping = rng.normal(size=(3, 5)).astype(np.float32)
    offset = rng.normal(size=(5,)).astype(np.float32)
    target = source @ mapping + offset
    tokens = [f"token-{index}" for index in range(len(source))]
    source_path = tmp_path / "source.gguf"
    target_path = tmp_path / "target.gguf"
    projector_path = tmp_path / "projector.gguf"
    checkpoint = tmp_path / "checkpoint"
    _write_text_gguf(source_path, source, tokens)
    _write_text_gguf(target_path, target, tokens)
    donor_weight = rng.normal(size=(3, 2)).astype(np.float32)
    donor_bias = rng.normal(size=(3,)).astype(np.float32)
    _write_projector(projector_path, donor_weight, donor_bias)

    report = initialize_bridge_from_token_spaces(
        omni_text_gguf=source_path,
        target_text_gguf=target_path,
        omni_projector_gguf=projector_path,
        output_dir=checkpoint,
        max_tokens=18,
        holdout_tokens=6,
        ridge=1e-7,
        seed=3,
    )

    bridge = load_bridge_checkpoint(checkpoint)
    expected_weight = mapping.T @ donor_weight
    expected_bias = donor_bias @ mapping + offset
    assert report["schema"] == TOKEN_ALIGNMENT_SCHEMA
    assert report["holdout_metrics"]["mean_cosine_similarity"] > 0.999
    assert torch.allclose(
        bridge.projection.weight,
        torch.from_numpy(expected_weight),
        atol=2e-3,
    )
    assert torch.allclose(
        bridge.projection.bias,
        torch.from_numpy(expected_bias),
        atol=2e-3,
    )
