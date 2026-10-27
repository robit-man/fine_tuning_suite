"""One composite Phonon + reference voice conversion + Whisper checkpoint.

This executes all neural components locally in one process. It is a new
runtime architecture, not a graph supported by the official Ollama loader.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import threading
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

SCHEMA = "robit.phonon-hybrid.v1"
MODEL_ID = "phonon-hybrid-v1"
SAMPLE_RATE = 24000


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def complete_tokenizer_assets(directory: Path):
    """HF save_pretrained can retain a versioned filename but save tokenizer.json."""
    config_path = directory / "tokenizer_config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text())
    for name in config.get("fast_tokenizer_files", []):
        if Path(name).name != name:
            raise ValueError("tokenizer asset must be a filename")
        target = directory / name
        if not target.exists():
            target.write_bytes((directory / "tokenizer.json").read_bytes())


def verify_bundle(directory: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported hybrid manifest")
    if not {"model.safetensors", "config.json"} <= manifest.get("files", {}).keys():
        raise ValueError("hybrid manifest omits model weights or configuration")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"hybrid file failed integrity verification: {name}")
    config = json.loads((directory / "config.json").read_text())
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported hybrid configuration")
    return config


def resample(audio: np.ndarray, rate: int, target: int) -> np.ndarray:
    if rate == target:
        return np.asarray(audio, dtype=np.float32)
    common = math.gcd(rate, target)
    return resample_poly(audio, target // common, rate // common).astype(np.float32)


def read_audio(data: bytes, *, max_seconds=60, allow_silence=False) -> tuple[np.ndarray, int]:
    try:
        with sf.SoundFile(io.BytesIO(data)) as handle:
            rate = handle.samplerate
            if not 8000 <= rate <= 192000 or not 1 <= handle.channels <= 8:
                raise ValueError("unsupported audio rate or channel count")
            if not 0.25 <= len(handle) / rate <= max_seconds:
                raise ValueError(f"audio must contain 0.25–{max_seconds} seconds")
            audio = handle.read(dtype="float32", always_2d=True).mean(axis=1)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError("invalid audio recording") from exc
    if not np.isfinite(audio).all() or (not allow_silence and np.max(np.abs(audio)) < 1e-5):
        raise ValueError("audio must contain finite, nonsilent samples")
    return audio, rate


def _converter(config):
    from openvoice.models import SynthesizerTrn

    data = config["data"]
    return SynthesizerTrn(0, data["filter_length"] // 2 + 1,
                          n_speakers=data["n_speakers"], **config["model"])


class PhononHybrid(torch.nn.Module):
    def __init__(self, phonon, codec, converter, asr, tokenizer, processor, converter_config, default_voice,
                 voice_profiles=None):
        super().__init__()
        self.phonon = phonon
        self.codec = codec
        self.converter = converter
        self.asr = asr
        self.tokenizer = tokenizer
        self.processor = processor
        self.converter_config = converter_config
        self.register_buffer("default_voice", default_voice)
        self.voices = torch.nn.Module()
        for name, embedding in (voice_profiles or {}).items():
            self.add_voice_profile(name, embedding)
        self.lock = threading.RLock()
        self.eval().requires_grad_(False)

    @property
    def device(self):
        return self.default_voice.device

    def add_voice_profile(self, name, embedding):
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name) or name == "default":
            raise ValueError("invalid baked voice name")
        if name in self.voices._buffers:
            raise ValueError("baked voice already exists")
        value = torch.as_tensor(embedding, dtype=torch.float32).detach().clone()
        if tuple(value.shape) != (1, 256, 1) or not torch.isfinite(value).all() or value.norm() == 0:
            raise ValueError("baked voice requires a finite, nonzero [1,256,1] speaker embedding")
        self.voices.register_buffer(name, value)

    def voice_names(self):
        return ["default", *sorted(self.voices._buffers)]

    def voice_embedding(self, name):
        if name not in self.voices._buffers:
            raise ValueError(f"unknown baked voice: {name}")
        return self.voices._buffers[name].detach().clone()

    @classmethod
    def from_components(cls, *, phonon_dir: Path, codec_dir: Path, converter_dir: Path, asr_dir: Path):
        from safetensors.torch import load_file
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            GradiumPhononTTSForConditionalGeneration,
            PocketTTSAudioTokenizerModel,
            WhisperForConditionalGeneration,
        )

        options = {"local_files_only": True, "dtype": torch.float32, "attn_implementation": "eager"}
        phonon = GradiumPhononTTSForConditionalGeneration.from_pretrained(phonon_dir, **options)
        codec = PocketTTSAudioTokenizerModel.from_pretrained(codec_dir, **options)
        asr = WhisperForConditionalGeneration.from_pretrained(asr_dir, **options)
        converter_config = json.loads((converter_dir / "config.json").read_text())
        converter = _converter(converter_config)
        weights = torch.load(converter_dir / "checkpoint.pth", map_location="cpu", weights_only=True)
        converter.load_state_dict(weights["model"], strict=True)
        voice = load_file(str(phonon_dir / "default-voice.safetensors"))["speaker_wavs"].transpose(1, 2)
        return cls(phonon, codec, converter, asr,
                   AutoTokenizer.from_pretrained(phonon_dir, local_files_only=True),
                   AutoProcessor.from_pretrained(asr_dir, local_files_only=True),
                   converter_config, voice.float())

    def save_pretrained(self, directory: Path):
        from safetensors.torch import save_file

        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "model.safetensors").exists():
            raise FileExistsError(f"refusing to overwrite existing hybrid weights in {directory}")
        config = {"schema": SCHEMA, "model_type": "phonon_hybrid", "model_id": MODEL_ID,
                  "architectures": ["PhononHybrid"], "sample_rate": SAMPLE_RATE,
                  "phonon": self.phonon.config.to_dict(), "codec": self.codec.config.to_dict(),
                  "asr": self.asr.config.to_dict(), "converter": self.converter_config,
                  "phonon_generation": self.phonon.generation_config.to_dict(),
                  "asr_generation": self.asr.generation_config.to_dict(),
                  "default_voice_shape": list(self.default_voice.shape),
                  "baked_voices": sorted(self.voices._buffers),
                  "composition": ["phonon", "mimi_decoder", "openvoice_v2", "whisper"],
                  "ollama_runtime": {"base_version": "0.40.11", "extension": "phonon-echo.2",
                                     "setup": "https://github.com/robit-man/fine_tuning_suite/blob/main/docs/phonon-hybrid.md"},
                  "jointly_trained": False, "official_ollama_loader_supported": False}
        (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        self.tokenizer.save_pretrained(directory / "phonon_tokenizer")
        self.processor.save_pretrained(directory / "asr_processor")
        complete_tokenizer_assets(directory / "phonon_tokenizer")
        complete_tokenizer_assets(directory / "asr_processor")
        license_source = Path(__file__).resolve().parents[1] / "licenses"
        shutil.copytree(license_source, directory / "licenses", dirs_exist_ok=True)
        # Clone tied tensors so safetensors can store a self-contained state dict.
        weights = {key: value.detach().cpu().contiguous().clone() for key, value in self.state_dict().items()}
        partial = directory / "model.safetensors.partial"
        save_file(weights, str(partial), metadata={"schema": SCHEMA})
        partial.rename(directory / "model.safetensors")
        files = {str(path.relative_to(directory)): digest(path)
                 for path in directory.rglob("*") if path.is_file() and path.name != "manifest.json"}
        manifest = {"schema": SCHEMA, "model_id": MODEL_ID, "files": files,
                    "parameters": sum(p.numel() for p in self.parameters())}
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest

    @classmethod
    def from_pretrained(cls, directory: Path, *, device="cpu"):
        config = verify_bundle(directory)
        from safetensors.torch import load_file

        return cls.from_state(directory, config, load_file(str(directory / "model.safetensors")), device=device)

    @classmethod
    def from_state(cls, directory, config, weights, *, device="cpu"):
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            GenerationConfig,
            GradiumPhononTTSConfig,
            GradiumPhononTTSForConditionalGeneration,
            PocketTTSAudioTokenizerConfig,
            PocketTTSAudioTokenizerModel,
            WhisperConfig,
            WhisperForConditionalGeneration,
        )

        phonon = GradiumPhononTTSForConditionalGeneration(GradiumPhononTTSConfig.from_dict(config["phonon"]))
        codec = PocketTTSAudioTokenizerModel(PocketTTSAudioTokenizerConfig.from_dict(config["codec"]))
        asr = WhisperForConditionalGeneration(WhisperConfig.from_dict(config["asr"]))
        phonon.generation_config = GenerationConfig.from_dict(config["phonon_generation"])
        asr.generation_config = GenerationConfig.from_dict(config["asr_generation"])
        model = cls(phonon, codec, _converter(config["converter"]), asr,
                    AutoTokenizer.from_pretrained(directory / "phonon_tokenizer", local_files_only=True),
                    AutoProcessor.from_pretrained(directory / "asr_processor", local_files_only=True),
                    config["converter"], torch.zeros(config["default_voice_shape"]),
                    voice_profiles={name: torch.ones(1, 256, 1) for name in config.get("baked_voices", [])})
        model.load_state_dict(weights, strict=True)
        return model.to(device).eval().requires_grad_(False)

    def _spectrogram(self, audio, rate):
        config = self.converter_config["data"]
        audio = resample(audio, rate, config["sampling_rate"])
        y = torch.from_numpy(audio.copy()).to(self.device).unsqueeze(0)
        padding = (config["filter_length"] - config["hop_length"]) // 2
        y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
        window = torch.hann_window(config["win_length"], device=self.device)
        spectrum = torch.stft(y, config["filter_length"], hop_length=config["hop_length"],
                              win_length=config["win_length"], window=window, center=False,
                              return_complex=True)
        return torch.sqrt(spectrum.abs().square() + 1e-6)

    @torch.inference_mode()
    def speaker_embedding(self, audio, rate):
        with self.lock:
            features = self._spectrogram(audio, rate)
            return self.converter.ref_enc(features.transpose(1, 2)).unsqueeze(-1).cpu()

    @torch.inference_mode()
    def transcribe(self, audio, rate):
        if len(audio) / rate > 30:
            raise ValueError("ASR currently accepts up to 30 seconds per request")
        with self.lock:
            inputs = self.processor(resample(audio, rate, 16000), sampling_rate=16000,
                                    return_tensors="pt", return_attention_mask=True).to(self.device)
            ids = self.asr.generate(**inputs, max_new_tokens=256)
            return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

    @torch.inference_mode()
    def _base_speech(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        prefix = [self.phonon.config.audio_prompt_token_id] * self.default_voice.shape[1]
        inputs = torch.tensor([[*prefix, *ids, self.phonon.config.audio_bos_token_id]], device=self.device)
        budget = min(500, math.ceil((len(ids) / 3 + 2) * 12.5))
        generated = self.phonon.generate(input_ids=inputs, audio_prompt_latents=self.default_voice,
                                         max_new_tokens=budget, do_sample=False,
                                         frames_after_eos=3 if len(text.split()) <= 4 else 1)
        decoded = self.codec.decode(generated.audio_latents, use_cache=False).audio_values.flatten()
        return decoded.cpu().numpy()

    @torch.inference_mode()
    def synthesize(self, text: str, *, speaker=None):
        return np.concatenate(list(self.iter_speech(text, speaker=speaker))).astype(np.float32)

    def iter_speech(self, text: str, *, speaker=None):
        if not isinstance(text, str) or not text.strip() or len(text) > 4096:
            raise ValueError("input must contain 1–4096 characters")
        # Each chunk owns its decoder/cache state. Release the inference lock
        # before yielding to a possibly slow or disconnected HTTP client.
        emitted = False
        for chunk in re.split(r"(?<=[.!?])\s+|\n+", text.strip()):
            words = chunk.split()
            for start in range(0, len(words), 45):
                with self.lock, torch.inference_mode():
                    sentence = " ".join(words[start:start + 45])
                    if sentence[-1] not in ".!?":
                        sentence += "."
                    audio = self._base_speech(sentence)
                    if speaker is not None:
                        features = self._spectrogram(audio, SAMPLE_RATE)
                        source = self.converter.ref_enc(features.transpose(1, 2)).unsqueeze(-1)
                        target = torch.as_tensor(speaker, device=self.device, dtype=source.dtype)
                        lengths = torch.tensor([features.shape[-1]], device=self.device)
                        converted = self.converter.voice_conversion(features, lengths, source, target, tau=0.3)[0]
                        audio = resample(converted[0, 0].cpu().numpy(),
                                         self.converter_config["data"]["sampling_rate"], SAMPLE_RATE)
                    if not len(audio) or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
                        raise RuntimeError("hybrid speech model returned invalid or silent audio")
                if emitted:
                    yield np.zeros(1200, dtype=np.float32)
                emitted = True
                yield audio.astype(np.float32)
