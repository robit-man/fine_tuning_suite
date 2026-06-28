from __future__ import annotations

from training_suite.models.gguf import compatible_for_splice, infer_capabilities, inspect_metadata


def test_infer_qwen35_vision_tools_thinking_from_metadata_and_tensors() -> None:
    metadata = {
        "general.architecture": "qwen35",
        "qwen35.vision.block_count": 27,
        "qwen35.vision_start_token_id": 248053,
        "tokenizer.chat_template": "<think>{% if tools %}<tool_call>{% endif %}",
    }
    caps, flags = infer_capabilities(
        architecture="qwen35",
        metadata=metadata,
        tensor_names=["blk.0.attn_q.weight", "v.blk.0.attn_q.weight"],
    )

    assert caps == ["completion", "thinking", "tools", "vision"]
    assert flags == []


def test_inspect_metadata_counts_tensor_prefixes() -> None:
    inspection = inspect_metadata(
        architecture="qwen35moe",
        metadata={"general.architecture": "qwen35moe", "qwen35moe.context_length": 262144},
        tensor_names=["blk.0.weight", "blk.1.weight", "v.blk.0.weight"],
    )

    assert inspection.architecture == "qwen35moe"
    assert inspection.context_length == 262144
    assert inspection.tensor_prefix_counts["blk"] == 2
    assert inspection.tensor_prefix_counts["v"] == 1


def test_splice_compatibility_blocks_architecture_mismatch() -> None:
    ok, errors = compatible_for_splice(
        {"architecture": "qwen3_5_moe", "text_hidden_size": 2048},
        {"architecture": "qwen35", "text_hidden_size": 4096},
    )

    assert ok is False
    assert "architecture" in errors[0]
    assert any("text_hidden_size" in item for item in errors)
