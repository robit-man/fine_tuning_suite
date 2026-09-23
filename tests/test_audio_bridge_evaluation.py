from __future__ import annotations

from training_suite.evals.audio_bridge import (
    _request_payload,
    audio_observation_claim_scope,
    comprehension_prompt_sha256,
    parse_tagged_audio_evidence,
    word_error_rate,
)


def test_evaluation_uses_bounded_no_thinking_media_prompt(tmp_path) -> None:
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")

    payload = _request_payload(
        {"audio_path": str(audio)},
        model="candidate",
        max_tokens=2_048,
        repeat_penalty=1.1,
    )

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] == 2_048
    assert payload["repeat_penalty"] == 1.1
    assert len(comprehension_prompt_sha256()) == 64


def test_word_error_rate_normalizes_case_and_punctuation() -> None:
    assert word_error_rate("Hello, copper seven!", "hello copper seven") == 0
    assert word_error_rate("one two three", "one four three") == 1 / 3
    assert word_error_rate("", "invented") == 1


def test_tagged_audio_parser_requires_only_the_two_audio_tags() -> None:
    parsed = parse_tagged_audio_evidence(
        "<speech_transcript>Hello.</speech_transcript>"
        "<audio_observation>Quiet room tone.</audio_observation>"
    )
    invalid = parse_tagged_audio_evidence(
        "<visual_observation>A camera.</visual_observation>"
    )

    assert parsed == {
        "valid": True,
        "speech_transcript": "Hello.",
        "audio_observation": "Quiet room tone.",
    }
    assert invalid["valid"] is False


def test_tagged_audio_parser_rejects_nested_tags_and_markdown_fences() -> None:
    for content in (
        (
            "<speech_transcript><visual_observation>red</visual_observation>"
            "</speech_transcript><audio_observation></audio_observation>"
        ),
        (
            "<speech_transcript>```text\nhello\n```</speech_transcript>"
            "<audio_observation>quiet</audio_observation>"
        ),
    ):
        assert parse_tagged_audio_evidence(content)["valid"] is False


def test_visual_claim_scope_excludes_attributed_speech() -> None:
    complete = (
        "<speech_transcript>Do not mention the camera.</speech_transcript>"
        "<audio_observation>Quiet room tone.</audio_observation>"
    )
    partial = "<speech_transcript>They are seen by everyone."

    parsed = parse_tagged_audio_evidence(complete)

    assert audio_observation_claim_scope(complete, parsed) == "Quiet room tone."
    assert (
        audio_observation_claim_scope(
            partial,
            parse_tagged_audio_evidence(partial),
        )
        == ""
    )
