import numpy as np
import pytest
import torch

from training_suite.phonon_hybrid import create_app
from training_suite.models.phonon_hybrid import PhononHybrid


def model(profiles=None):
    return PhononHybrid(*(torch.nn.Identity() for _ in range(4)), None, None, {},
                        torch.zeros(1, 2, 512), voice_profiles=profiles)


def test_baked_profiles_survive_strict_checkpoint_load_and_are_isolated():
    source = model()
    for index, name in enumerate(["female", "robot", "male"], 1):
        source.add_voice_profile(name, torch.full((1, 256, 1), float(index)))
    saved = {name: value.clone() for name, value in source.state_dict().items()}
    target = model({name: torch.ones(1, 256, 1) for name in ["female", "robot", "male"]})
    target.load_state_dict(saved, strict=True)
    returned = target.voice_embedding("female")
    returned.fill_(99)
    assert target.voice_embedding("female").mean() == 1
    assert target.voice_embedding("robot").mean() == 2
    assert target.voice_embedding("male").mean() == 3
    assert target.voice_names() == ["default", "female", "male", "robot"]
    assert set(saved) >= {"voices.female", "voices.robot", "voices.male"}


@pytest.mark.parametrize("name,value", [
    ("default", torch.ones(1, 256, 1)),
    ("../female", torch.ones(1, 256, 1)),
    ("female", torch.ones(1, 512, 1)),
    ("female", torch.zeros(1, 256, 1)),
    ("female", torch.full((1, 256, 1), float("nan"))),
])
def test_invalid_baked_profiles_are_rejected(name, value):
    with pytest.raises(ValueError):
        model().add_voice_profile(name, value)


def test_named_api_voices_work_without_enrollment_files_and_cannot_be_overwritten(tmp_path):
    class Engine:
        device = "cpu"
        speakers = []

        def voice_names(self):
            return ["default", "female", "robot", "male"]

        def voice_embedding(self, name):
            return np.full((1, 256, 1), {"female": 1, "robot": 2, "male": 3}[name], dtype=np.float32)

        def synthesize(self, text, speaker=None):
            self.speakers.append(float(np.mean(speaker)))
            return np.ones(2400, dtype=np.float32) * 0.1

    engine = Engine()
    client = create_app(engine, voices_dir=tmp_path).test_client()
    assert client.get("/v1/audio/voices").json["voices"] == ["default", "female", "male", "robot"]
    for name in ["female", "robot", "male", "female"]:
        response = client.post("/v1/audio/speech", json={"model": "phonon-hybrid-v1", "input": "A new sentence.",
                                                       "voice": name}, buffered=True)
        assert response.status_code == 200
    assert engine.speakers == [1, 2, 3, 1]
    assert not list(tmp_path.iterdir())
    assert client.post("/v1/audio/voices", data={"name": "female"}).status_code == 409
