from __future__ import annotations

import pytest

from training_suite.training.tts_vocabulary_recovery_data import (
    RECOVERY_PROMPT_PREFIX,
    build_recovery_prompts,
    transcript_vocabulary,
)


def test_transcript_vocabulary_is_unique_normalized_and_ignores_ids(tmp_path) -> None:
    source = tmp_path / "dev-clean" / "1" / "2"
    source.mkdir(parents=True)
    (source / "1-2.trans.txt").write_text(
        "1-2-0001 HELLO, copper lighthouse!\n"
        "1-2-0002 Hello O'CLOCK world.\n",
        encoding="utf-8",
    )

    assert transcript_vocabulary(tmp_path / "dev-clean") == [
        "copper",
        "hello",
        "lighthouse",
        "o'clock",
        "world",
    ]


def test_recovery_prompts_are_deterministic_broad_and_not_eval_carriers() -> None:
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon"]

    prompts = build_recovery_prompts(
        vocabulary,
        chunk_words=4,
        seed=17,
        maximum_numeral=12,
    )

    assert prompts == build_recovery_prompts(
        vocabulary,
        chunk_words=4,
        seed=17,
        maximum_numeral=12,
    )
    assert all(prompt.startswith(RECOVERY_PROMPT_PREFIX) for prompt in prompts)
    assert all(not prompt.startswith("Please pronounce these words: ") for prompt in prompts)
    assert {"10", "11", "12"} <= {
        token.strip(".,")
        for prompt in prompts
        for token in prompt.split()
    }


def test_recovery_prompt_validation() -> None:
    with pytest.raises(ValueError, match="chunk words"):
        build_recovery_prompts(["alpha"], chunk_words=0, seed=1, maximum_numeral=10)
    with pytest.raises(ValueError, match="maximum numeral"):
        build_recovery_prompts(["alpha"], chunk_words=1, seed=1, maximum_numeral=9)
