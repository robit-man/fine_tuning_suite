from __future__ import annotations

import httpx
import pytest

from training_suite.evals.transcription import ollama_transcription_smoke, word_error_rate


@pytest.mark.parametrize("expected,actual,wer", [
    ("Blue river eight.", "BLUE river eight!", 0),
    ("blue river eight", "blue mountain eight", 1 / 3),
    ("blue river eight", "blue eight", 1 / 3),
    ("blue river eight", "blue blue river eight", 1 / 3),
    ("blue", "red green orange", 3),
])
def test_word_error_rate_preserves_order_and_counts_edits(expected, actual, wer):
    assert word_error_rate(expected, actual) == pytest.approx(wer)


def test_native_asr_sends_recorded_audio_as_multipart_and_checks_words(tmp_path):
    path = tmp_path / "reference.wav"
    path.write_bytes(b"fixture-recording")

    def handler(request):
        assert request.url.path == "/v1/audio/transcriptions"
        assert request.headers["content-type"].startswith("multipart/form-data")
        assert b'filename="reference.wav"' in request.content
        assert b"fixture-recording" in request.content
        assert b"native-hybrid" in request.content
        return httpx.Response(200, json={"text": "Blue river eight."})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = ollama_transcription_smoke("native-hybrid", audio=path,
                                            expected_text="Blue river eight.", client=client)
    assert report["ok"]
    assert report["word_error_rate"] == 0


@pytest.mark.parametrize("body", [{"text": "Unrelated words"}, {"audio": "base64"}, {"text": ""}])
def test_native_asr_rejects_missing_or_inaccurate_transcripts(tmp_path, body):
    path = tmp_path / "reference.wav"
    path.write_bytes(b"fixture")
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as client:
        report = ollama_transcription_smoke("native", audio=path, expected_text="Blue river eight.", client=client)
    assert not report["ok"]


def test_native_asr_reports_unsupported_model(tmp_path):
    path = tmp_path / "reference.wav"
    path.write_bytes(b"fixture")
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(400, json={"error": "model does not support completion"})
    )) as client:
        report = ollama_transcription_smoke("phonon", audio=path, expected_text="Blue river eight.", client=client)
    assert not report["ok"]
    assert report["status_code"] == 400
