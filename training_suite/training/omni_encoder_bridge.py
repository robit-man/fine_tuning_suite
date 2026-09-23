from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from training_suite.models.omni import (
    QWEN3_OMNI_INSTRUCT,
    assess_trained_encoder_bridge,
)

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pad_sequence
except ModuleNotFoundError:  # Planning remains usable in the lightweight suite venv.
    torch = None
    nn = None
    pad_sequence = None


BRIDGE_SCHEMA = "robit.qwen-omni.encoder-bridge.v1"
IGNORE_INDEX = -100


@dataclass(frozen=True)
class EncoderBridgeConfig:
    source_width: int
    target_width: int
    modality: str = "audio"
    normalization_epsilon: float = 1e-6
    boundary_embeddings: int = 2

    def __post_init__(self) -> None:
        if self.source_width <= 0 or self.target_width <= 0:
            raise ValueError("source_width and target_width must be positive")
        if self.modality != "audio":
            raise ValueError("the first executable bridge supports audio only")
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        if self.boundary_embeddings != 2:
            raise ValueError("audio bridge requires exactly start and end boundaries")

    @property
    def affine_parameters(self) -> int:
        return self.source_width * self.target_width + self.target_width

    @property
    def trainable_parameters(self) -> int:
        source_norm = self.source_width
        target_norm = self.target_width
        boundaries = self.boundary_embeddings * self.target_width
        return self.affine_parameters + source_norm + target_norm + boundaries

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "architecture": "rmsnorm-affine-rmsnorm-with-learned-boundaries",
            "affine_parameters": self.affine_parameters,
            "trainable_parameters": self.trainable_parameters,
            "trainable_bf16_bytes": self.trainable_parameters * 2,
        }


@dataclass(frozen=True)
class Qwen3AFinalProjectorConfig:
    """The bridge shape that can execute in stock llama.cpp's Qwen3A graph."""

    source_width: int
    target_width: int

    def __post_init__(self) -> None:
        if self.source_width <= 0 or self.target_width <= 0:
            raise ValueError("source_width and target_width must be positive")

    @property
    def trainable_parameters(self) -> int:
        return self.source_width * self.target_width + self.target_width

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "architecture": "qwen3a-final-projector",
            "gguf_weight": "mm.a.mlp.2.weight",
            "gguf_bias": "mm.a.mlp.2.bias",
            "trainable_parameters": self.trainable_parameters,
            "trainable_bf16_bytes": self.trainable_parameters * 2,
        }


def build_bridge_manifest(
    *,
    text_config: dict[str, Any],
    omni_config: dict[str, Any],
    target_source: str,
    omni_source: str = QWEN3_OMNI_INSTRUCT,
) -> dict[str, Any]:
    assessment = assess_trained_encoder_bridge(text_config, omni_config)
    audio = assessment["modalities"]["audio"]
    if not assessment["feasible_research_path"]:
        raise ValueError("encoder bridge is blocked: " + "; ".join(assessment["blockers"]))
    if not audio["available"]:
        raise ValueError("Omni donor does not expose an audio encoder")

    config = Qwen3AFinalProjectorConfig(
        source_width=int(audio["encoder_output_width"]),
        target_width=int(audio["language_embedding_width"]),
    )
    return {
        "schema": BRIDGE_SCHEMA,
        "status": "ready-to-initialize",
        "target_language": {
            "source": target_source,
            "frozen": True,
            "text_only_bypass": True,
            "input": "inputs_embeds",
        },
        "omni_encoder": {
            "source": omni_source,
            "component": "thinker.audio_tower",
            "feature_entrypoint": "audio_tower input to proj2 after proj1+GELU",
            "frozen": True,
            "thinker_required_after_export": False,
        },
        "bridge": config.to_dict(),
        "native_vision": {
            "policy": "retain-target-native-path",
            "train_in_stage_one": False,
        },
        "training": {
            "trainable": [
                "bridge.projection",
            ],
            "frozen": ["target language model", "Omni audio encoder"],
            "loss": "target causal-language loss on assistant response tokens only",
            "use_cache": False,
            "boundary_policy": "target prompt text surrounds the injected media span",
            "runtime_export": "replace mm.a.mlp.2 in a combined target-vision/audio GGUF",
        },
        "release_invariants": {
            "text_weights_byte_identical": True,
            "text_only_requests_bypass_bridge": True,
            "audio_never_sets_current_visual_input": True,
            "semantic_text_router_is_rollback": True,
        },
        "assessment": assessment,
    }


def write_bridge_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def audio_output_length(input_frames: int) -> int:
    """Match Qwen3-Omni's three-CNN/deepstack output-length calculation."""
    if input_frames < 0:
        raise ValueError("input_frames must not be negative")
    if input_frames == 0:
        return 0
    remaining = input_frames % 100
    remaining_after_first = (remaining - 1) // 2 + 1
    remaining_output = ((remaining_after_first - 1) // 2 + 1 - 1) // 2 + 1
    return remaining_output + (input_frames // 100) * 13


_AUDIO_BLOCK_TENSOR_MAP = {
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_out.weight": "self_attn.out_proj.weight",
    "attn_out.bias": "self_attn.out_proj.bias",
    "ln1.weight": "self_attn_layer_norm.weight",
    "ln1.bias": "self_attn_layer_norm.bias",
    "ffn_up.weight": "fc1.weight",
    "ffn_up.bias": "fc1.bias",
    "ffn_down.weight": "fc2.weight",
    "ffn_down.bias": "fc2.bias",
    "ln2.weight": "final_layer_norm.weight",
    "ln2.bias": "final_layer_norm.bias",
}


def map_audio_gguf_tensor_name(name: str) -> str | None:
    """Map llama.cpp Qwen3-Omni audio-projector names to Transformers names."""
    direct = {
        "a.post_ln.weight": "ln_post.weight",
        "a.post_ln.bias": "ln_post.bias",
        "a.conv_out.weight": "conv_out.weight",
        "mm.a.mlp.1.weight": "proj1.weight",
        "mm.a.mlp.1.bias": "proj1.bias",
        "mm.a.mlp.2.weight": "proj2.weight",
        "mm.a.mlp.2.bias": "proj2.bias",
    }
    if name in direct:
        return direct[name]
    convolution = re.fullmatch(r"a\.conv2d\.([123])\.(weight|bias)", name)
    if convolution:
        return f"conv2d{convolution.group(1)}.{convolution.group(2)}"
    block = re.fullmatch(r"a\.blk\.(\d+)\.(.+)", name)
    if block and block.group(2) in _AUDIO_BLOCK_TENSOR_MAP:
        return f"layers.{block.group(1)}.{_AUDIO_BLOCK_TENSOR_MAP[block.group(2)]}"
    return None


def _require_torch() -> None:
    if torch is None or nn is None or pad_sequence is None:
        raise RuntimeError(
            "PyTorch is required for bridge execution; planning works without it"
        )


if nn is not None:

    @dataclass
    class BridgeOutput:
        embeddings: Any
        attention_mask: Any


    @dataclass
    class PackedLanguageInputs:
        inputs_embeds: Any
        attention_mask: Any
        labels: Any | None


    class RMSNorm(nn.Module):
        def __init__(self, width: int, epsilon: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))
            self.epsilon = epsilon

        def forward(self, values):
            variance = values.float().pow(2).mean(dim=-1, keepdim=True)
            normalized = values * torch.rsqrt(variance + self.epsilon).to(values.dtype)
            return normalized * self.weight.to(values.dtype)


    class OmniAudioSequenceBridge(nn.Module):
        """Map frozen Omni audio sequences into a frozen language embedding space."""

        def __init__(self, config: EncoderBridgeConfig) -> None:
            super().__init__()
            self.config = config
            self.source_norm = RMSNorm(config.source_width, config.normalization_epsilon)
            self.projection = nn.Linear(config.source_width, config.target_width)
            self.target_norm = RMSNorm(config.target_width, config.normalization_epsilon)
            self.audio_start = nn.Parameter(torch.empty(config.target_width))
            self.audio_end = nn.Parameter(torch.empty(config.target_width))
            nn.init.normal_(self.audio_start, mean=0.0, std=0.02)
            nn.init.normal_(self.audio_end, mean=0.0, std=0.02)

        def initialize_boundaries(self, start_embedding, end_embedding) -> None:
            expected = (self.config.target_width,)
            if tuple(start_embedding.shape) != expected or tuple(end_embedding.shape) != expected:
                raise ValueError(f"boundary embeddings must have shape {expected}")
            with torch.no_grad():
                self.audio_start.copy_(start_embedding)
                self.audio_end.copy_(end_embedding)

        def forward(self, features, feature_attention_mask=None) -> BridgeOutput:
            if features.ndim != 3 or features.shape[-1] != self.config.source_width:
                raise ValueError(
                    "features must have shape [batch, sequence, "
                    f"{self.config.source_width}]"
                )
            batch, sequence, _ = features.shape
            if feature_attention_mask is None:
                feature_attention_mask = torch.ones(
                    (batch, sequence), dtype=torch.bool, device=features.device
                )
            else:
                if tuple(feature_attention_mask.shape) != (batch, sequence):
                    raise ValueError("feature_attention_mask shape does not match features")
                feature_attention_mask = feature_attention_mask.to(
                    device=features.device, dtype=torch.bool
                )

            lengths = feature_attention_mask.sum(dim=1)
            expected_mask = (
                torch.arange(sequence, device=features.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            )
            if not torch.equal(feature_attention_mask, expected_mask):
                raise ValueError("feature_attention_mask must be right-padded and contiguous")

            projected = self.target_norm(self.projection(self.source_norm(features)))
            output = projected.new_zeros((batch, sequence + 2, self.config.target_width))
            output[:, 0] = self.audio_start.to(projected.dtype)
            output[:, 1 : sequence + 1] = projected * feature_attention_mask.unsqueeze(-1)
            batch_indices = torch.arange(batch, device=features.device)
            output[batch_indices, lengths + 1] = self.audio_end.to(projected.dtype)
            output_mask = (
                torch.arange(sequence + 2, device=features.device).unsqueeze(0)
                < (lengths + 2).unsqueeze(1)
            )
            return BridgeOutput(embeddings=output, attention_mask=output_mask)


    class Qwen3AFinalProjectorBridge(nn.Module):
        """Deployable Qwen3A bridge: the exact replacement for ``mm.a.mlp.2``."""

        def __init__(self, config: Qwen3AFinalProjectorConfig) -> None:
            super().__init__()
            self.config = config
            self.projection = nn.Linear(config.source_width, config.target_width)

        def forward(self, features, feature_attention_mask=None) -> BridgeOutput:
            if features.ndim != 3 or features.shape[-1] != self.config.source_width:
                raise ValueError(
                    "features must have shape [batch, sequence, "
                    f"{self.config.source_width}]"
                )
            batch, sequence, _ = features.shape
            if feature_attention_mask is None:
                feature_attention_mask = torch.ones(
                    (batch, sequence), dtype=torch.bool, device=features.device
                )
            elif tuple(feature_attention_mask.shape) != (batch, sequence):
                raise ValueError("feature_attention_mask shape does not match features")
            mask = feature_attention_mask.to(device=features.device, dtype=torch.bool)
            projected = self.projection(features)
            return BridgeOutput(
                embeddings=projected * mask.unsqueeze(-1),
                attention_mask=mask,
            )


    def capture_audio_final_projector_input(
        encoder,
        *,
        input_features,
        feature_lens,
    ):
        """Run the frozen audio tower and capture the tensor entering ``proj2``."""
        captured = []

        def capture(_module, args):
            captured.append(args[0])

        handle = encoder.proj2.register_forward_pre_hook(capture)
        try:
            encoder(
                input_features=input_features,
                feature_lens=feature_lens,
                return_dict=True,
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("audio encoder did not execute its final projector exactly once")
        return captured[0]


    def omni_audio_output_lengths(frame_lengths):
        if frame_lengths.ndim != 1:
            raise ValueError("frame_lengths must be one-dimensional")
        if bool((frame_lengths < 0).any()):
            raise ValueError("frame_lengths must not contain negative values")
        remaining = frame_lengths % 100
        remaining_after_first = torch.where(
            remaining == 0,
            torch.zeros_like(remaining),
            (remaining - 1) // 2 + 1,
        )
        remaining_output = torch.where(
            remaining_after_first == 0,
            torch.zeros_like(remaining_after_first),
            ((remaining_after_first - 1) // 2 + 1 - 1) // 2 + 1,
        )
        return remaining_output + (frame_lengths // 100) * 13


    def pack_audio_feature_batch(input_features, feature_attention_mask):
        """Pack padded `[batch, mel, frames]` features for the Omni audio tower."""
        if input_features.ndim != 3:
            raise ValueError("input_features must have shape [batch, mel, frames]")
        batch, _, frames = input_features.shape
        if tuple(feature_attention_mask.shape) != (batch, frames):
            raise ValueError("feature_attention_mask shape does not match input_features")
        mask = feature_attention_mask.to(device=input_features.device, dtype=torch.bool)
        lengths = mask.sum(dim=1)
        expected = (
            torch.arange(frames, device=input_features.device).unsqueeze(0)
            < lengths.unsqueeze(1)
        )
        if not torch.equal(mask, expected):
            raise ValueError("feature_attention_mask must be right-padded and contiguous")
        packed = torch.cat(
            [input_features[index, :, : int(length)] for index, length in enumerate(lengths)],
            dim=-1,
        )
        return packed, lengths


    def pad_flat_audio_features(flat_features, output_lengths) -> BridgeOutput:
        if flat_features.ndim != 2:
            raise ValueError("flat_features must have shape [total_sequence, width]")
        if output_lengths.ndim != 1:
            raise ValueError("output_lengths must be one-dimensional")
        lengths = [int(value) for value in output_lengths.detach().cpu().tolist()]
        if sum(lengths) != flat_features.shape[0]:
            raise ValueError("audio output lengths do not sum to the flat feature count")
        sequences = torch.split(flat_features, lengths, dim=0)
        padded = pad_sequence(sequences, batch_first=True)
        mask = (
            torch.arange(padded.shape[1], device=flat_features.device).unsqueeze(0)
            < output_lengths.to(flat_features.device).unsqueeze(1)
        )
        return BridgeOutput(embeddings=padded, attention_mask=mask)


    def load_omni_audio_encoder_from_gguf(
        projector_gguf: Path,
        *,
        omni_source: str = QWEN3_OMNI_INSTRUCT,
        dtype=None,
        device: str = "cpu",
    ):
        """Materialize only the frozen Omni audio tower from an existing projector.

        The projector is memory-mapped and tensors are dequantized one at a time.
        No Omni Thinker weights are loaded or copied.
        """
        from gguf import GGUFReader, dequantize
        from transformers import AutoConfig
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
        )

        source = Path(projector_gguf).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if dtype is None:
            dtype = torch.bfloat16

        config = AutoConfig.from_pretrained(omni_source)
        audio_config = config.thinker_config.audio_config
        encoder = Qwen3OmniMoeAudioEncoder._from_config(
            audio_config,
            dtype=dtype,
        ).to(device=device)
        parameters = dict(encoder.named_parameters())
        loaded: set[str] = set()
        reader = GGUFReader(str(source), "r")

        with torch.no_grad():
            for tensor in reader.tensors:
                target_name = map_audio_gguf_tensor_name(tensor.name)
                if target_name is None:
                    continue
                if target_name not in parameters:
                    raise ValueError(
                        f"GGUF tensor {tensor.name!r} maps to unknown parameter {target_name!r}"
                    )
                if target_name in loaded:
                    raise ValueError(f"duplicate mapped audio parameter: {target_name}")
                values = dequantize(tensor.data, tensor.tensor_type)
                source_tensor = torch.from_numpy(values.copy())
                target = parameters[target_name]
                if source_tensor.shape != target.shape and source_tensor.numel() == target.numel():
                    source_tensor = source_tensor.reshape(target.shape)
                if tuple(source_tensor.shape) != tuple(target.shape):
                    raise ValueError(
                        f"shape mismatch for {tensor.name}: GGUF {tuple(source_tensor.shape)} "
                        f"!= Transformers {tuple(target.shape)}"
                    )
                target.copy_(source_tensor.to(device=target.device, dtype=target.dtype))
                loaded.add(target_name)

        missing = sorted(set(parameters) - loaded)
        if missing:
            raise ValueError(
                "projector does not contain the complete Omni audio tower: "
                + ", ".join(missing[:8])
            )
        encoder.requires_grad_(False)
        encoder.eval()
        report = {
            "schema": BRIDGE_SCHEMA,
            "source": str(source),
            "source_size_bytes": source.stat().st_size,
            "omni_source": omni_source,
            "mapped_parameter_tensors": len(loaded),
            "dtype": str(dtype),
            "device": str(device),
            "thinker_loaded": False,
        }
        return encoder, report


    def save_bridge_checkpoint(
        output_dir: Path,
        bridge,
        *,
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save only the compact bridge; never duplicate either frozen model."""
        from safetensors.torch import save_file

        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        weights_path = destination / "bridge.safetensors"
        config_path = destination / "bridge_config.json"
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in bridge.state_dict().items()
        }
        save_file(state, str(weights_path), metadata={"schema": BRIDGE_SCHEMA})
        payload = {
            "schema": BRIDGE_SCHEMA,
            "bridge": bridge.config.to_dict(),
            "frozen_model_weights_included": False,
            "manifest": manifest,
        }
        config_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "weights": str(weights_path),
            "weights_size_bytes": weights_path.stat().st_size,
            "config": str(config_path),
            "state_tensors": len(state),
            "frozen_model_weights_included": False,
        }


    def load_bridge_checkpoint(
        checkpoint_dir: Path,
        *,
        device: str = "cpu",
        dtype=None,
    ) -> OmniAudioSequenceBridge:
        from safetensors.torch import load_file

        source = Path(checkpoint_dir).expanduser().resolve()
        payload = json.loads((source / "bridge_config.json").read_text(encoding="utf-8"))
        if payload.get("schema") != BRIDGE_SCHEMA:
            raise ValueError("unsupported encoder bridge checkpoint schema")
        raw_config = payload["bridge"]
        architecture = raw_config.get("architecture")
        if architecture == "qwen3a-final-projector":
            config = Qwen3AFinalProjectorConfig(
                source_width=int(raw_config["source_width"]),
                target_width=int(raw_config["target_width"]),
            )
            bridge = Qwen3AFinalProjectorBridge(config).to(device=device)
        else:
            config = EncoderBridgeConfig(
                source_width=int(raw_config["source_width"]),
                target_width=int(raw_config["target_width"]),
                modality=str(raw_config["modality"]),
                normalization_epsilon=float(raw_config["normalization_epsilon"]),
                boundary_embeddings=int(raw_config["boundary_embeddings"]),
            )
            bridge = OmniAudioSequenceBridge(config).to(device=device)
        state = load_file(str(source / "bridge.safetensors"), device=device)
        bridge.load_state_dict(state, strict=True)
        if dtype is not None:
            bridge.to(dtype=dtype)
        return bridge


    def pack_language_sections(
        *,
        prefix_embeddings,
        prefix_attention_mask,
        media: BridgeOutput,
        suffix_embeddings,
        suffix_attention_mask,
        suffix_labels=None,
    ) -> PackedLanguageInputs:
        sections = []
        labels = []
        lengths = []
        batch = prefix_embeddings.shape[0]
        if not (
            media.embeddings.shape[0] == batch
            and suffix_embeddings.shape[0] == batch
        ):
            raise ValueError("all embedding sections must have the same batch size")

        for index in range(batch):
            prefix = prefix_embeddings[index][prefix_attention_mask[index].bool()]
            audio = media.embeddings[index][media.attention_mask[index].bool()]
            suffix = suffix_embeddings[index][suffix_attention_mask[index].bool()]
            combined = torch.cat((prefix, audio, suffix), dim=0)
            sections.append(combined)
            lengths.append(combined.shape[0])
            if suffix_labels is not None:
                supervised = suffix_labels[index][suffix_attention_mask[index].bool()]
                labels.append(
                    torch.cat(
                        (
                            supervised.new_full(
                                (prefix.shape[0] + audio.shape[0],), IGNORE_INDEX
                            ),
                            supervised,
                        )
                    )
                )

        packed = pad_sequence(sections, batch_first=True)
        packed_mask = (
            torch.arange(packed.shape[1], device=packed.device).unsqueeze(0)
            < torch.tensor(lengths, device=packed.device).unsqueeze(1)
        )
        packed_labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if labels
            else None
        )
        return PackedLanguageInputs(
            inputs_embeds=packed,
            attention_mask=packed_mask,
            labels=packed_labels,
        )


    class FrozenOmniAudioLanguageAdapter(nn.Module):
        """Stage-one training graph with only the bridge left trainable."""

        def __init__(self, omni_audio_encoder, language_model, bridge) -> None:
            super().__init__()
            self.omni_audio_encoder = omni_audio_encoder
            self.language_model = language_model
            self.bridge = bridge
            for module in (self.omni_audio_encoder, self.language_model):
                module.requires_grad_(False)
                module.eval()

        def train(self, mode: bool = True):
            super().train(mode)
            self.omni_audio_encoder.eval()
            self.language_model.eval()
            self.bridge.train(mode)
            return self

        def _encode_audio(self, input_features, feature_attention_mask, frame_lengths):
            encoder_parameter = next(self.omni_audio_encoder.parameters())
            input_features = input_features.to(
                device=encoder_parameter.device,
                dtype=encoder_parameter.dtype,
            )
            feature_attention_mask = feature_attention_mask.to(encoder_parameter.device)
            frame_lengths = frame_lengths.to(encoder_parameter.device)
            if isinstance(self.bridge, Qwen3AFinalProjectorBridge):
                packed_features, packed_lengths = pack_audio_feature_batch(
                    input_features,
                    feature_attention_mask,
                )
                return capture_audio_final_projector_input(
                    self.omni_audio_encoder,
                    input_features=packed_features,
                    feature_lens=packed_lengths,
                )
            if hasattr(self.omni_audio_encoder, "get_audio_features"):
                return self.omni_audio_encoder.get_audio_features(
                    input_features=input_features,
                    feature_attention_mask=feature_attention_mask,
                    return_dict=True,
                ).last_hidden_state
            packed_features, packed_lengths = pack_audio_feature_batch(
                input_features,
                feature_attention_mask,
            )
            return self.omni_audio_encoder(
                input_features=packed_features,
                feature_lens=packed_lengths,
                return_dict=True,
            ).last_hidden_state

        def forward(
            self,
            *,
            input_features,
            feature_attention_mask,
            prefix_input_ids,
            prefix_attention_mask,
            suffix_input_ids,
            suffix_attention_mask,
            suffix_labels=None,
        ):
            frame_lengths = feature_attention_mask.sum(dim=1)
            with torch.no_grad():
                encoded = self._encode_audio(
                    input_features,
                    feature_attention_mask,
                    frame_lengths,
                )
                output_lengths = omni_audio_output_lengths(frame_lengths)
                padded_audio = pad_flat_audio_features(encoded, output_lengths)
                embeddings = self.language_model.get_input_embeddings()
                embedding_parameter = next(embeddings.parameters())
                prefix_embeddings = embeddings(
                    prefix_input_ids.to(embedding_parameter.device)
                )
                suffix_embeddings = embeddings(
                    suffix_input_ids.to(embedding_parameter.device)
                )

            bridge_parameter = next(self.bridge.parameters())
            bridge_features = padded_audio.embeddings.to(
                device=bridge_parameter.device,
                dtype=bridge_parameter.dtype,
            )
            bridge_mask = padded_audio.attention_mask.to(bridge_parameter.device)

            bridged = self.bridge(
                bridge_features,
                bridge_mask,
            )
            bridged = BridgeOutput(
                embeddings=bridged.embeddings.to(
                    device=prefix_embeddings.device,
                    dtype=prefix_embeddings.dtype,
                ),
                attention_mask=bridged.attention_mask.to(prefix_embeddings.device),
            )
            packed = pack_language_sections(
                prefix_embeddings=prefix_embeddings,
                prefix_attention_mask=prefix_attention_mask.to(prefix_embeddings.device),
                media=bridged,
                suffix_embeddings=suffix_embeddings,
                suffix_attention_mask=suffix_attention_mask.to(prefix_embeddings.device),
                suffix_labels=(
                    suffix_labels.to(prefix_embeddings.device)
                    if suffix_labels is not None
                    else None
                ),
            )
            return self.language_model(
                inputs_embeds=packed.inputs_embeds,
                attention_mask=packed.attention_mask,
                labels=packed.labels,
                use_cache=False,
            )


else:

    def _torch_only(*_args, **_kwargs):  # pragma: no cover - import guard.
        _require_torch()


    class OmniAudioSequenceBridge:  # pragma: no cover - exercised only without torch.
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


    class FrozenOmniAudioLanguageAdapter:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


    class Qwen3AFinalProjectorBridge:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            _require_torch()


    load_bridge_checkpoint = _torch_only
    load_omni_audio_encoder_from_gguf = _torch_only
    save_bridge_checkpoint = _torch_only
