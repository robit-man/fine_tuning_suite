import hashlib
import io
import wave

import httpx
import pytest

from training_suite import phonon_native_tts as native


def mock_http(monkeypatch, handler):
    client = httpx.Client
    monkeypatch.setattr(native.httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs))


def test_prepare_rejects_corrupt_download_without_replacing_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "FILES", {"model.safetensors": hashlib.sha256(b"expected").hexdigest()})
    target = tmp_path / "source/model.safetensors"
    target.parent.mkdir()
    target.write_bytes(b"previous")
    mock_http(monkeypatch, lambda request: httpx.Response(200, content=b"corrupt"))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        native.prepare(tmp_path)
    assert target.read_bytes() == b"previous"
    assert not list(target.parent.glob("*.partial"))
    assert not (tmp_path / "Modelfile").exists()


def test_native_readiness_rejects_extension_version(monkeypatch):
    mock_http(monkeypatch, lambda request: httpx.Response(200, json={"version": "0.40.11+phonon-hybrid.1"}))
    with pytest.raises(ValueError, match="expected official"):
        native.ready("http://localhost", "speech")


def test_native_readiness_consumes_streaming_wav_actual_samples(monkeypatch):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * 24000)
    data = bytearray(output.getvalue())
    data[4:8] = data[40:44] = b"\xff" * 4
    def handle(request):
        return httpx.Response(200, json={"version": "0.40.11"}) if request.method == "GET" else httpx.Response(200, content=bytes(data))
    mock_http(monkeypatch, handle)
    assert native.ready("http://localhost", "speech")["pcm_bytes"] == 48000
