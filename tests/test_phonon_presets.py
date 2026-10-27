import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from safetensors.torch import load_file

from training_suite.models.phonon_hybrid import PhononHybrid, verify_bundle
from training_suite.training.phonon_presets import bake


def test_baker_exports_profiles_and_receipts_without_reference_files(tmp_path, monkeypatch):
    class Assets:
        def save_pretrained(self, directory):
            directory.mkdir()
            (directory / "tokenizer_config.json").write_text("{}")

    parts = [torch.nn.Identity() for _ in range(4)]
    for part in parts:
        part.config = SimpleNamespace(to_dict=lambda: {})
        part.generation_config = SimpleNamespace(to_dict=lambda: {})
    model = PhononHybrid(*parts, Assets(), Assets(), {}, torch.zeros(1, 2, 512))
    references = []

    def speaker_embedding(audio, rate):
        references.append((audio, rate))
        return torch.ones(1, 256, 1)

    monkeypatch.setattr(model, "speaker_embedding", speaker_embedding)
    monkeypatch.setattr("training_suite.models.phonon_hybrid_storage.load_ollama_model", lambda *args: model)
    audio = np.zeros(24000, dtype=np.float32)
    audio[8000:16000] = np.sin(np.arange(8000) * 0.1) * 0.1
    reference = tmp_path / "private-reference.wav"
    sf.write(reference, audio, 16000)
    output = tmp_path / "release/model"
    receipt = bake(models_directory=tmp_path, source_model="example/voice", voices={"female": reference}, output=output)
    config = verify_bundle(output)
    assert config["baked_voices"] == ["female"]
    assert config["ollama_runtime"]["extension"] == "phonon-echo.2"
    assert torch.equal(load_file(str(output / "model.safetensors"))["voices.female"], torch.ones(1, 256, 1))
    assert references[0][1] == 16000
    assert len(references[0][0]) < len(audio)
    assert np.max(np.abs(references[0][0])) == pytest.approx(0.9)
    assert not list(output.rglob("*.wav"))
    assert reference.name not in json.dumps(receipt)
    assert receipt["profiles"]["female"]["reference_sha256"]
    assert json.loads((output / "model-card.json").read_text())["source_model"] == "example/voice"
    assert (output.parent / "bake-receipt.json").is_file()
    with pytest.raises(ValueError, match="empty directory"):
        bake(models_directory=tmp_path, source_model="example/voice", voices={"female": reference}, output=output)


def test_suite_cli_parses_baking_options_and_dispatches(tmp_path, monkeypatch):
    from training_suite.cli import build_parser

    captured = {}
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr("training_suite.training.phonon_presets.bake", lambda **kwargs: captured.update(kwargs) or {})
    args = build_parser().parse_args(["phonon-bake-voices", "--voice", "female=ref.wav", "--out", str(tmp_path / "out")])
    args.func(args)
    assert captured["models_directory"] == tmp_path / "models"
    assert captured["voices"]["female"].name == "ref.wav"


@pytest.mark.parametrize("command", ["phonon-bake-voices", "phonon-native-tts"])
def test_suite_preparation_cli_help(command, capsys):
    from training_suite.cli import build_parser

    with pytest.raises(SystemExit) as result:
        args = build_parser().parse_args([command, "--help"])
        args.func(args)
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert ("--out" if command == "phonon-bake-voices" else "prepare") in output


def test_duplicate_profile_names_fail_before_loading_weights():
    from training_suite.cli import build_parser

    args = build_parser().parse_args(["phonon-bake-voices", "--voice", "female=a.wav", "--voice", "female=b.wav", "--out", "out"])
    with pytest.raises(SystemExit, match="duplicate voice"):
        args.func(args)
