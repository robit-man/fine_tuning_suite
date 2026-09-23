from __future__ import annotations

import base64
import io
import wave

from training_suite.evals.tts_alignment import (
    _audio_payload,
    _wav_contract,
    compact_speech_similarity,
)


def _wav() -> bytes:
    target = io.BytesIO()
    with wave.open(target, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x00\x00" * 240)
    return target.getvalue()


def test_alignment_audio_request_matches_bridge_runtime_contract() -> None:
    wav = _wav()
    payload = _audio_payload(model="candidate", wav=wav, max_tokens=512)

    assert payload["cache_prompt"] is False
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["temperature"] == 0
    assert payload["repeat_penalty"] == 1.1
    encoded = payload["messages"][1]["content"][0]["input_audio"]["data"]
    assert base64.b64decode(encoded) == wav


def test_alignment_requires_24khz_mono_pcm16_wav() -> None:
    contract = _wav_contract(_wav())

    assert contract == {
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_width_bits": 16,
        "frames": 240,
        "valid": True,
    }


def test_compact_similarity_accepts_asr_spacing_and_digit_aliases() -> None:
    assert compact_speech_similarity("Blue river eight.", "Blue River 8.") == 1
    assert (
        compact_speech_similarity(
            "Copper lighthouse seven.", "Copper Light House Seven."
        )
        == 1
    )
    assert (
        compact_speech_similarity("Silver meadow nine.", "Silver medallion.")
        < 0.9
    )
