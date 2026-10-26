from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file

from training_suite.training.phonon_voice import export_voice, invert_waveform, spectral_loss


def test_voice_export_matches_native_import_layout_and_values(tmp_path):
    # The runtime wants [batch, channels, frames], while the preparation model
    # uses [batch, frames, channels]. Prove values, not just dimensions.
    prompt = torch.arange(3 * 512, dtype=torch.float64).reshape(1, 3, 512)
    path = tmp_path / "voices" / "reference.safetensors"
    report = export_voice(prompt, path)
    tensors = load_file(str(path))
    assert set(tensors) == {"speaker_wavs"}
    actual = tensors["speaker_wavs"]
    assert actual.dtype == torch.float32
    assert actual.shape == (1, 512, 3)
    torch.testing.assert_close(actual.transpose(1, 2), prompt.float())
    assert report["shape"] == [1, 512, 3]


@pytest.mark.parametrize("shape", [(512,), (1, 512), (1, 0, 512), (2, 3, 512), (1, 3, 256)])
def test_voice_export_rejects_unsupported_tensor_shapes(tmp_path, shape):
    with pytest.raises(ValueError, match="shape"):
        export_voice(torch.zeros(shape), tmp_path / "bad.safetensors")
    assert not (tmp_path / "bad.safetensors").exists()


def test_voice_export_rejects_nonfinite_values(tmp_path):
    with pytest.raises(ValueError, match="nonfinite"):
        export_voice(torch.full((1, 3, 512), float("nan")), tmp_path / "bad.safetensors")


def test_spectral_objective_has_finite_gradient_and_zero_for_identical_audio():
    torch.manual_seed(1)
    target = torch.randn(1920)
    assert spectral_loss(target, target).item() == 0
    candidate = torch.randn(1920, requires_grad=True)
    loss = spectral_loss(candidate, target)
    loss.backward()
    assert torch.isfinite(candidate.grad).all()
    assert candidate.grad.abs().max() > 0


def test_decoder_inversion_optimizes_latents_without_changing_frozen_decoder():
    from types import SimpleNamespace

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.8), requires_grad=False)

        def decode(self, latents, **kwargs):
            return SimpleNamespace(audio_values=(latents * self.scale).repeat_interleave(60, dim=-1).flatten())

    decoder = Decoder()
    target_latents = torch.linspace(-0.5, 0.5, 32).reshape(1, 1, 32)
    target = decoder.decode(target_latents).audio_values
    original_scale = decoder.scale.detach().clone()
    fitted, history = invert_waveform(
        decoder, target, torch.zeros(32), torch.ones(32), steps=25, learning_rate=0.08, seed=42
    )
    assert history[-1]["spectral_loss"] < history[0]["spectral_loss"]
    assert fitted.shape == (1, 1, 32)
    assert not fitted.requires_grad
    assert decoder.scale.grad is None
    torch.testing.assert_close(decoder.scale, original_scale)
