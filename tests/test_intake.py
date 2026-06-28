from __future__ import annotations

from training_suite.models.intake import build_repair_plan, parse_hf_repo


def test_parse_hf_repo_from_url_and_repo_id() -> None:
    assert parse_hf_repo("https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF") == "deepreinforce-ai/Ornith-1.0-9B-GGUF"
    assert parse_hf_repo("deepreinforce-ai/Ornith-1.0-9B") == "deepreinforce-ai/Ornith-1.0-9B"
    assert parse_hf_repo("hf.co/deepreinforce-ai/Ornith-1.0-35B") == "deepreinforce-ai/Ornith-1.0-35B"


def test_ornith_35b_raw_is_blocked_for_9b_gguf() -> None:
    plan = build_repair_plan(
        detected=["completion"],
        target=["completion", "vision", "tools", "thinking"],
        source_architecture="qwen35",
        gguf_has_vision=False,
        raw_signature={"architecture": "qwen3_5_moe", "text_hidden_size": 2048},
        gguf_signature={"architecture": "qwen35", "text_hidden_size": 4096},
        raw_source="https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B",
        source="https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF",
    )

    assert plan["mode"] == "blocked"
    assert any("not compatible" in item for item in plan["blockers"])
    assert any("text_hidden_size" in item for item in plan["blockers"])


def test_package_only_plan_for_missing_tools() -> None:
    plan = build_repair_plan(
        detected=["completion", "vision"],
        target=["completion", "vision", "tools", "thinking"],
        source_architecture="qwen35",
        gguf_has_vision=True,
        raw_signature={},
        gguf_signature={},
        raw_source=None,
        source="model.gguf",
    )

    assert plan["mode"] == "package-only"
    assert any("RENDERER" in action for action in plan["actions"])
