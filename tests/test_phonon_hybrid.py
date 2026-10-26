from __future__ import annotations

import base64
import io
import json
import wave

import numpy as np
import pytest

from training_suite.phonon_hybrid import VoiceStore, create_app
from training_suite.models.phonon_hybrid import SCHEMA, complete_tokenizer_assets, digest, verify_bundle


def recording(frequency=220, *, duration=1, rate=16000):
    audio = (np.sin(np.arange(int(rate * duration)) * frequency * 2 * np.pi / rate) * 8000).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(audio.tobytes())
    return output.getvalue()


class Engine:
    device = "cpu"

    def __init__(self):
        self.speakers = []
        self.asr_inputs = []

    def speaker_embedding(self, audio, rate):
        return np.full((1, 256, 1), np.mean(np.abs(audio)), dtype=np.float32)

    def synthesize(self, text, speaker=None):
        value = float(np.mean(speaker)) if speaker is not None else 0.1
        self.speakers.append(None if speaker is None else np.array(speaker).copy())
        return np.sin(np.arange(2400) * 0.1).astype(np.float32) * value

    def transcribe(self, audio, rate):
        self.asr_inputs.append((audio.copy(), rate))
        return "Blue river eight."


@pytest.fixture
def runtime(tmp_path):
    engine = Engine()
    app = create_app(engine, voices_dir=tmp_path / "voices")
    app.config["TESTING"] = True
    return engine, app.test_client()


def test_runtime_identifies_itself_and_serves_standard_speech_fields(runtime):
    engine, client = runtime
    health = client.get("/health").json
    assert health["official_ollama"] is False
    assert health["capabilities"] == ["speech", "voice-cloning", "transcription"]
    result = client.post("/v1/audio/speech", json={"model": "phonon-hybrid-v1", "input": "Read this.", "voice": "default"})
    assert result.status_code == 200
    assert result.content_type == "audio/wav"
    with wave.open(io.BytesIO(result.data)) as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (24000, 1, 2)
        assert len(handle.readframes(handle.getnframes())) == 4800
    assert engine.speakers == [None]


def test_enrolled_voices_persist_and_a_b_a_requests_keep_their_own_speaker(tmp_path):
    engine = Engine()
    app = create_app(engine, voices_dir=tmp_path / "voices")
    client = app.test_client()
    for name, amplitude in [("alice", 1), ("bob", 0.4)]:
        data = recording()
        with wave.open(io.BytesIO(data)) as handle:
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        out = io.BytesIO()
        with wave.open(out, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes((samples * amplitude).astype("<i2").tobytes())
        result = client.post("/v1/audio/voices", data={"name": name, "file": (io.BytesIO(out.getvalue()), "ref.wav")})
        assert result.status_code == 201
    # Recreate the app to prove profiles are loaded from disk.
    client = create_app(engine, voices_dir=tmp_path / "voices").test_client()
    outputs = [client.post("/v1/audio/speech", json={"model": "phonon-hybrid-v1", "input": "Same words.", "voice": name,
                                                     "response_format": "pcm"}).data for name in ["alice", "bob", "alice"]]
    assert outputs[0] == outputs[2]
    assert outputs[0] != outputs[1]
    np.testing.assert_array_equal(engine.speakers[0], engine.speakers[2])
    assert not np.array_equal(engine.speakers[0], engine.speakers[1])


def test_inline_reference_is_consumed_without_creating_a_profile(runtime):
    engine, client = runtime
    result = client.post("/v1/audio/speech", json={"model": "phonon-hybrid-v1", "input": "Hello.",
                         "reference_audio": {"format": "wav", "data": base64.b64encode(recording()).decode()}})
    assert result.status_code == 200
    assert engine.speakers[-1] is not None
    assert client.get("/v1/audio/voices").json["voices"] == ["default"]


def test_asr_consumes_uploaded_waveform_and_returns_text(runtime):
    engine, client = runtime
    result = client.post("/v1/audio/transcriptions", data={"model": "phonon-hybrid-v1", "file": (io.BytesIO(recording()), "ref.wav")})
    assert result.status_code == 200
    assert result.json["text"] == "Blue river eight."
    assert engine.asr_inputs[0][1] == 16000
    assert len(engine.asr_inputs[0][0]) == 16000


def test_silence_transcribes_to_empty_text_without_hallucinating(runtime):
    engine, client = runtime
    payload = io.BytesIO()
    with wave.open(payload, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes(32000))
    result = client.post("/v1/audio/transcriptions", data={"model": "phonon-hybrid-v1",
                         "file": (io.BytesIO(payload.getvalue()), "silence.wav")})
    assert result.status_code == 200
    assert result.json["text"] == ""
    assert not engine.asr_inputs


@pytest.mark.parametrize("extra", [
    {"model": "unknown"}, {"voice": "unknown"}, {"response_format": "mp3"}, {"speed": 2},
    {"instructions": "different voice"}, {"input": ""}, {"input": "a" * 4097},
    {"reference_audio": {"data": "not-base64!"}}, {"reference_audio": {"path": "/etc/passwd"}},
    {"reference_audio": {"data": "YQ=="}, "voice": "alice"}, {"ref_audio": "ignored"},
])
def test_bad_speech_requests_fail_before_model_inference(runtime, extra):
    engine, client = runtime
    result = client.post("/v1/audio/speech", json={"model": "phonon-hybrid-v1", "input": "Hello.", **extra})
    assert result.status_code == 400
    assert "error" in result.json
    assert not engine.speakers


def test_duplicate_voice_name_does_not_replace_saved_embedding(runtime):
    _, client = runtime
    for expected in [201, 409]:
        result = client.post("/v1/audio/voices", data={"name": "reference", "file": (io.BytesIO(recording()), "ref.wav")})
        assert result.status_code == expected


@pytest.mark.parametrize("name", ["../escape", "a/b", "default", "", "a" * 65])
def test_profile_names_cannot_escape_store(tmp_path, name):
    with pytest.raises(ValueError):
        VoiceStore(tmp_path).put(name, np.zeros((1, 256, 1), dtype=np.float32))


def test_versioned_tokenizer_filename_survives_export(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"fast_tokenizer_files": ["tokenizer.5.0.0.json"]}))
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer fixture")
    complete_tokenizer_assets(tmp_path)
    assert (tmp_path / "tokenizer.5.0.0.json").read_bytes() == b"tokenizer fixture"


def test_hybrid_integrity_rejects_changed_weights_before_loading_any_model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"schema": SCHEMA}))
    (tmp_path / "model.safetensors").write_bytes(b"original weights")
    manifest = {"schema": SCHEMA, "files": {name: digest(tmp_path / name) for name in ["config.json", "model.safetensors"]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert verify_bundle(tmp_path)["schema"] == SCHEMA
    (tmp_path / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="integrity"):
        verify_bundle(tmp_path)


def test_streaming_stops_future_generation_when_client_closes(tmp_path):
    class StreamingEngine(Engine):
        def __init__(self):
            super().__init__()
            self.generated = 0
            self.closed = False

        def iter_speech(self, text, speaker=None):
            try:
                for _ in range(3):
                    self.generated += 1
                    yield np.ones(2400, dtype=np.float32) * 0.1
            finally:
                self.closed = True

    engine = StreamingEngine()
    client = create_app(engine, voices_dir=tmp_path).test_client()
    response = client.post("/v1/audio/speech", json={"model": "phonon-hybrid-v1", "input": "First. Second. Third."}, buffered=False)
    assert next(response.response).startswith(b"RIFF")
    next(response.response)
    response.close()
    assert engine.generated == 1
    assert engine.closed


def test_streamed_sentence_pause_is_valid_pcm(tmp_path):
    class StreamingEngine(Engine):
        def iter_speech(self, text, speaker=None):
            yield np.ones(2400, dtype=np.float32) * 0.1
            yield np.zeros(1200, dtype=np.float32)
            yield np.ones(2400, dtype=np.float32) * 0.1

    client = create_app(StreamingEngine(), voices_dir=tmp_path).test_client()
    response = client.post("/v1/audio/speech", json={"model": "phonon-hybrid-v1", "input": "First. Second."})
    assert response.status_code == 200
    with wave.open(io.BytesIO(response.data)) as handle:
        assert len(handle.readframes(handle.getnframes())) == 12000


def test_pinned_source_download_records_exact_bytes_and_revision(tmp_path, monkeypatch):
    import hashlib
    import httpx
    from training_suite.models import phonon_hybrid_sources as sources

    source = {"repo": "publisher/model", "revision": "fixed-revision", "files": ["config.json"]}
    monkeypatch.setattr(sources, "SOURCES", {"phonon": source})

    def handler(request):
        assert request.url.path == "/publisher/model/resolve/fixed-revision/config.json"
        return httpx.Response(200, content=b'{"model":"fixture"}')

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(sources.httpx, "Client", lambda **kwargs: client)
    report = sources.download_components(tmp_path)
    raw = (tmp_path / "phonon/config.json").read_bytes()
    assert report["phonon"]["revision"] == "fixed-revision"
    assert report["phonon"]["sha256"]["config.json"] == hashlib.sha256(raw).hexdigest()
    assert not list(tmp_path.rglob("*.partial"))


def test_hybrid_cli_help_routes_to_runtime_parser(capsys):
    from training_suite.cli import build_parser, cmd_phonon_hybrid

    args = build_parser().parse_args(["phonon-hybrid", "--help"])
    with pytest.raises(SystemExit) as caught:
        cmd_phonon_hybrid(args)
    assert caught.value.code == 0
    assert "download" in capsys.readouterr().out


def test_ollama_managed_speech_accepts_native_runner_request_and_emits_raw_pcm(tmp_path):
    engine = Engine()
    client = create_app(engine, voices_dir=tmp_path, managed_by_ollama=True).test_client()
    result = client.post("/v1/audio/speech", json={"input": "Native speech request.", "voice": "default",
                         "reference_audio": {"data": base64.b64encode(recording()).decode(), "format": "wav"}})
    assert result.status_code == 200
    assert result.content_type == "audio/pcm"
    assert not result.data.startswith(b"RIFF")
    assert len(result.data) == 4800
    assert engine.speakers[-1] is not None
