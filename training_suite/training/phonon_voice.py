"""Offline Phonon voice fitting; deployment uses stock Ollama, never this module.

This is experimental prompt inversion, not a released zero-shot speaker encoder.
First invert the frozen waveform decoder, then fit only the speaker prompt using
teacher-forced codec targets. A new-sentence voice similarity gate is still needed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


def spectral_loss(predicted, target, *, fft_sizes=(256, 512, 1024)):
    import torch

    predicted = predicted.flatten()
    target = target.flatten()
    length = min(predicted.numel(), target.numel())
    predicted, target = predicted[:length], target[:length]
    losses = []
    for size in fft_sizes:
        window = torch.hann_window(size, device=predicted.device)
        a = torch.stft(predicted, size, size // 4, window=window, return_complex=True).abs()
        b = torch.stft(target, size, size // 4, window=window, return_complex=True).abs()
        convergence = (a - b).norm() / b.norm().clamp_min(1e-6)
        losses.append(convergence + (a.clamp_min(1e-4).log() - b.clamp_min(1e-4).log()).abs().mean())
    return torch.stack(losses).mean()


def invert_waveform(codec, waveform, mean, std, *, steps: int, learning_rate: float, seed: int):
    """Fit continuous decoder inputs without needing the unreleased audio encoder."""
    import torch

    if steps <= 0 or learning_rate <= 0:
        raise ValueError("inversion steps and learning rate must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    frames = math.ceil(waveform.numel() / 1920)
    normalized = torch.nn.Parameter(torch.randn(1, frames, 32, generator=generator) * 0.1)
    optimizer = torch.optim.Adam([normalized], lr=learning_rate)
    history = []
    best_loss, best = float("inf"), None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        decoded = codec.decode(normalized * std + mean, use_cache=False).audio_values
        reconstruction = spectral_loss(decoded, waveform)
        loss = reconstruction + normalized.square().mean() * 0.001
        if not torch.isfinite(loss):
            raise ValueError("nonfinite decoder inversion loss")
        value = float(reconstruction.detach())
        # Save the actual parameters evaluated by this forward, before updating.
        if value < best_loss:
            best_loss, best = value, normalized.detach().clone()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([normalized], 10)
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step == steps - 1:
            history.append({"step": step + 1, "spectral_loss": value})
            print(json.dumps({"stage": "codec-inversion", **history[-1]}), flush=True)
    assert best is not None
    return (best * std + mean).detach(), history


def voice_objective(model, token_ids, voice, target_latents, *, noise):
    """One-step flow reconstruction and EOS losses, differentiable in voice only."""
    import torch
    from torch.nn import functional as F

    config = model.config
    count = target_latents.shape[1]
    ids = torch.tensor([[
        *([config.audio_prompt_token_id] * voice.shape[1]),
        *token_ids,
        config.audio_bos_token_id,
        *([config.audio_latent_token_id] * count),
    ]], dtype=torch.long)
    normalized = (target_latents - model.latent_mean) / model.latent_std.clamp_min(1e-6)
    hidden = model.model(
        input_ids=ids,
        audio_prompt_latents=voice,
        audio_latents=normalized,
        use_cache=False,
    ).last_hidden_state
    audio_positions = (ids == config.audio_bos_token_id) | (ids == config.audio_latent_token_id)
    hidden = hidden[audio_positions]
    times = torch.ones(count, 1, dtype=hidden.dtype)
    predicted = noise + model.flow_head(noise, hidden[:-1], times=[times * 0, times])
    flow = F.mse_loss(predicted, normalized[0])
    eos_labels = torch.zeros(count + 1)
    eos_labels[-1] = 1
    eos = F.binary_cross_entropy_with_logits(model.eos_head(hidden).flatten(), eos_labels)
    return flow + eos * config.eos_loss_weight


def fit_voice_prompt(model, tokenizer, text, initial_voice, target_latents, *, steps,
                     learning_rate, seed, extra_examples=(), prior_weight=0.01):
    import torch

    if steps <= 0 or learning_rate <= 0 or prior_weight < 0:
        raise ValueError("voice steps/rate must be positive and prior weight nonnegative")
    voice = torch.nn.Parameter(initial_voice.clone())
    optimizer = torch.optim.Adam([voice], lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    examples = [(tokenizer.encode(t, add_special_tokens=False), latents)
                for t, latents in [(text, target_latents), *extra_examples]]
    history = []
    # A fixed validation draw makes losses across updates comparable. Training
    # uses fresh draws, matching the stochastic one-step sampler in Ollama.
    validation_noise = [torch.randn(z.shape[1], 32, generator=generator) * math.sqrt(0.3)
                        for _, z in examples]
    best_loss, best = float("inf"), initial_voice.clone()
    for step in range(steps + 1):
        if step % 10 == 0 or step == steps:
            with torch.no_grad():
                value = sum(float(voice_objective(model, ids, voice, z, noise=noise))
                            for (ids, z), noise in zip(examples, validation_noise)) / len(examples)
            if not math.isfinite(value):
                raise ValueError("nonfinite voice validation loss")
            if value < best_loss:
                best_loss, best = value, voice.detach().clone()
            history.append({"step": step, "validation_loss": value})
            print(json.dumps({"stage": "voice-fit", **history[-1]}), flush=True)
        if step == steps:
            break
        optimizer.zero_grad(set_to_none=True)
        ids, latents = examples[step % len(examples)]
        noise = torch.randn(latents.shape[1], 32, generator=generator) * math.sqrt(0.3)
        loss = voice_objective(model, ids, voice, latents, noise=noise)
        loss = loss + (voice - initial_voice).square().mean() * prior_weight
        if not torch.isfinite(loss):
            raise ValueError("nonfinite voice fitting loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([voice], 10)
        optimizer.step()
    return best, history


def export_voice(voice, destination: Path) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file

    if voice.ndim != 3 or voice.shape[0] != 1 or voice.shape[2] != 512 or voice.shape[1] < 1:
        raise ValueError("voice must have shape [1, positive_frames, 512]")
    if not torch.isfinite(voice).all():
        raise ValueError("voice contains nonfinite values")
    native = voice.detach().to(device="cpu", dtype=torch.float32).transpose(1, 2).contiguous()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    save_file({"speaker_wavs": native}, str(partial))
    partial.replace(destination)
    return {"path": str(destination), "tensor": "speaker_wavs", "dtype": "F32", "shape": list(native.shape),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def prepare_voice(*, checkpoint: Path, codec_checkpoint: Path, reference: Path, text: str,
                  output_dir: Path, inversion_steps=100, voice_steps=100, threads=4, seed=42):
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import load_file, save_file
    from scipy.signal import resample_poly
    from transformers import AutoTokenizer, GradiumPhononTTSForConditionalGeneration, PocketTTSAudioTokenizerModel

    started = time.monotonic()
    if not text.strip() or len(text) > 4096:
        raise ValueError("reference text must contain 1–4096 characters")
    if threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    audio, rate = sf.read(reference, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 24000:
        common = math.gcd(rate, 24000)
        audio = resample_poly(audio, 24000 // common, rate // common).astype(np.float32)
    if not np.isfinite(audio).all() or not np.any(audio) or not 0.25 <= len(audio) / 24000 <= 10:
        raise ValueError("reference must be finite nonsilent audio of 0.25–10 seconds")
    waveform = torch.from_numpy(audio.copy())
    # This experiment is explicitly CPU-only; it never initializes CUDA or
    # starts an inference service. GPU training needs the host lease protocol.
    model = GradiumPhononTTSForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.float32, local_files_only=True, attn_implementation="eager"
    ).to("cpu").eval().requires_grad_(False)
    codec = PocketTTSAudioTokenizerModel.from_pretrained(
        codec_checkpoint, dtype=torch.float32, local_files_only=True, attn_implementation="eager"
    ).to("cpu").eval().requires_grad_(False)
    if model.config.hidden_size != 512 or model.config.audio_prompt_dim != 512 or model.config.latent_dim != 32:
        raise ValueError("checkpoint does not match stock Ollama Phonon dimensions")
    if model.config.flow_head_config.flow_type != "lsd":
        raise ValueError("voice fitting requires the stock-compatible LSD flow head")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    initial = load_file(str(checkpoint / "default-voice.safetensors"))["speaker_wavs"].transpose(1, 2).float()
    target, inversion_history = invert_waveform(
        codec, waveform, model.latent_mean, model.latent_std,
        steps=inversion_steps, learning_rate=0.08, seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        reconstruction = codec.decode(target, use_cache=False).audio_values.flatten()[:len(audio)].numpy()
    sf.write(output_dir / "reference-reconstruction.wav", reconstruction, 24000, subtype="PCM_16")
    save_file({"audio_latents": target}, str(output_dir / "reference-latents.safetensors"))
    voice, history = fit_voice_prompt(
        model, tokenizer, text, initial, target,
        steps=voice_steps, learning_rate=0.02, seed=seed,
    )
    artifact = export_voice(voice, output_dir / "voice.safetensors")
    report = {
        "schema": "robit.phonon-offline-voice-fit.v1",
        "status": "experimental-unverified-voice",
        "deployment": {"official_ollama_unmodified": True, "endpoint": "/v1/audio/speech",
                       "selection": "prepared named voice", "per_request_reference_audio": False},
        "checkpoint": str(checkpoint), "codec_checkpoint": str(codec_checkpoint),
        "reference": str(reference), "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "reference_text": text, "reference_seconds": len(audio) / 24000,
        "inversion_history": inversion_history, "voice_fit_history": history, "voice": artifact,
        "base_weights_changed": False, "asr_included": False,
        "native_generation_verified": False, "new_sentence_voice_similarity_verified": False,
        "cuda_initialized": torch.cuda.is_initialized(), "seed": seed,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--codec-checkpoint", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--text", required=True, help="Exact transcript of the reference recording")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--inversion-steps", type=int, default=100)
    parser.add_argument("--voice-steps", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = prepare_voice(checkpoint=args.checkpoint, codec_checkpoint=args.codec_checkpoint,
                           reference=args.reference, text=args.text, output_dir=args.out,
                           inversion_steps=args.inversion_steps, voice_steps=args.voice_steps,
                           threads=args.threads, seed=args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
