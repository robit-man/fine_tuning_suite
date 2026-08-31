from __future__ import annotations

import base64
import io
import wave

import pytest

flask = pytest.importorskip("flask")

from training_suite.core.jobs import JobRunner
from training_suite.core.state import StateStore
from training_suite.web import create_app


def test_dashboard_loads(tmp_path) -> None:
    store = StateStore(tmp_path / "suite.sqlite3")
    app = create_app(store=store, runner=JobRunner(store))
    app.testing = True

    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"<h3>Models</h3>" in response.data


def test_omni_audio_contract_and_validation_endpoints(tmp_path) -> None:
    store = StateStore(tmp_path / "suite.sqlite3")
    app = create_app(store=store, runner=JobRunner(store))
    app.testing = True
    client = app.test_client()

    contract = client.get("/api/omni/audio/contract")
    assert contract.status_code == 200
    assert contract.get_json()["input"]["sample_rate_hz"] == 16000

    router = client.get("/api/omni/router/contract")
    assert router.status_code == 200
    assert router.get_json()["request"]["message_audio_field"] == "audios"

    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 160)
    response = client.post(
        "/api/omni/audio/validate",
        json={
            "audio": {
                "mime_type": "audio/wav",
                "encoding": "base64",
                "data": base64.b64encode(out.getvalue()).decode("ascii"),
            }
        },
    )

    assert response.status_code == 200
    assert response.get_json()["audio"]["channels"] == 1

    tagged = client.post(
        "/api/omni/audio/validate",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "understand this",
                    "audios": [
                        {
                            "mime_type": "audio/wav",
                            "encoding": "base64",
                            "data": base64.b64encode(out.getvalue()).decode("ascii"),
                        }
                    ],
                }
            ]
        },
    )
    assert tagged.status_code == 200


def test_omni_cascade_is_disabled_until_stages_are_configured(tmp_path, monkeypatch) -> None:
    for name in (
        "TRAINING_SUITE_OMNI_ASR_URL",
        "TRAINING_SUITE_OMNI_LANGUAGE_MODEL",
        "TRAINING_SUITE_OMNI_TTS_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    store = StateStore(tmp_path / "suite.sqlite3")
    app = create_app(store=store, runner=JobRunner(store))
    app.testing = True

    response = app.test_client().post(
        "/api/omni/cascade",
        json={"prompt": "hello", "audio": "not-needed-before-config-check"},
    )

    assert response.status_code == 503
    assert "not configured" in response.get_json()["error"]
