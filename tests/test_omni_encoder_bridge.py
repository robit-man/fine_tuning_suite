from __future__ import annotations

from types import SimpleNamespace

import pytest
from test_omni import OMNI_CONFIG, ORNITH_CONFIG, QWEN38_CONFIG

from training_suite.training.omni_encoder_bridge import (
    BRIDGE_SCHEMA,
    EncoderBridgeConfig,
    Qwen3AFinalProjectorConfig,
    audio_output_length,
    build_bridge_manifest,
    map_audio_gguf_tensor_name,
)


def test_qwen38_audio_bridge_manifest_preserves_the_language_trunk() -> None:
    manifest = build_bridge_manifest(
        text_config=QWEN38_CONFIG,
        omni_config=OMNI_CONFIG,
        target_source="manitcor/Qwen3.8-27B-Obliterated-E03",
    )

    assert manifest["schema"] == BRIDGE_SCHEMA
    assert manifest["target_language"]["frozen"] is True
    assert manifest["target_language"]["text_only_bypass"] is True
    assert "proj2" in manifest["omni_encoder"]["feature_entrypoint"]
    assert manifest["omni_encoder"]["thinker_required_after_export"] is False
    assert manifest["bridge"]["source_width"] == 1280
    assert manifest["bridge"]["target_width"] == 5120
    assert manifest["bridge"]["architecture"] == "qwen3a-final-projector"
    assert manifest["bridge"]["trainable_parameters"] == 6_558_720
    assert manifest["native_vision"]["policy"] == "retain-target-native-path"


def test_ornith_bridge_is_the_smaller_fallback() -> None:
    manifest = build_bridge_manifest(
        text_config=ORNITH_CONFIG,
        omni_config=OMNI_CONFIG,
        target_source="deepreinforce-ai/Ornith-1.5-9B",
    )

    assert manifest["bridge"]["target_width"] == 4096
    assert manifest["bridge"]["trainable_parameters"] == 5_246_976


@pytest.mark.parametrize(
    ("input_frames", "output_tokens"),
    ((0, 0), (1, 1), (8, 1), (9, 2), (99, 13), (100, 13), (101, 14), (200, 26)),
)
def test_audio_output_length_matches_qwen3_omni_deepstack(
    input_frames: int,
    output_tokens: int,
) -> None:
    assert audio_output_length(input_frames) == output_tokens


def test_bridge_rejects_non_audio_stage_one() -> None:
    with pytest.raises(ValueError, match="audio only"):
        EncoderBridgeConfig(source_width=2048, target_width=5120, modality="video")


def test_deployable_bridge_matches_qwen3a_final_projector_shape() -> None:
    config = Qwen3AFinalProjectorConfig(source_width=1280, target_width=5120)

    assert config.trainable_parameters == 6_558_720
    assert config.to_dict()["gguf_weight"] == "mm.a.mlp.2.weight"


def test_deployable_bridge_preserves_sequence_length() -> None:
    torch = pytest.importorskip("torch")
    from training_suite.training.omni_encoder_bridge import Qwen3AFinalProjectorBridge

    bridge = Qwen3AFinalProjectorBridge(
        Qwen3AFinalProjectorConfig(source_width=4, target_width=6)
    )
    output = bridge(
        torch.randn(2, 3, 4),
        torch.tensor([[1, 1, 1], [1, 0, 0]]),
    )

    assert output.embeddings.shape == (2, 3, 6)
    assert output.attention_mask.tolist() == [[True, True, True], [True, False, False]]
    assert torch.count_nonzero(output.embeddings[1, 1:]) == 0


@pytest.mark.parametrize(
    ("gguf_name", "transformers_name"),
    (
        ("a.conv2d.1.weight", "conv2d1.weight"),
        ("a.blk.7.attn_q.weight", "layers.7.self_attn.q_proj.weight"),
        ("a.blk.31.ffn_down.bias", "layers.31.fc2.bias"),
        ("a.blk.0.ln2.weight", "layers.0.final_layer_norm.weight"),
        ("a.post_ln.bias", "ln_post.bias"),
        ("mm.a.mlp.1.weight", "proj1.weight"),
        ("mm.a.mlp.2.bias", "proj2.bias"),
        ("a.position_embd.weight", None),
        ("v.blk.0.attn_q.weight", None),
    ),
)
def test_audio_projector_tensor_mapping(
    gguf_name: str,
    transformers_name: str | None,
) -> None:
    assert map_audio_gguf_tensor_name(gguf_name) == transformers_name


def test_torch_bridge_places_end_boundary_after_each_real_sequence() -> None:
    torch = pytest.importorskip("torch")
    from training_suite.training.omni_encoder_bridge import OmniAudioSequenceBridge

    config = EncoderBridgeConfig(source_width=4, target_width=6)
    bridge = OmniAudioSequenceBridge(config)
    features = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]])

    output = bridge(features, mask)

    assert output.embeddings.shape == (2, 5, 6)
    assert output.attention_mask.tolist() == [
        [True, True, True, True, True],
        [True, True, True, False, False],
    ]
    assert torch.equal(output.embeddings[1, 2], bridge.audio_end)
    assert torch.count_nonzero(output.embeddings[1, 3:]) == 0
    assert sum(parameter.numel() for parameter in bridge.parameters()) == config.trainable_parameters


def test_flat_audio_features_are_rebatched_with_the_reference_lengths() -> None:
    torch = pytest.importorskip("torch")
    from training_suite.training.omni_encoder_bridge import (
        omni_audio_output_lengths,
        pad_flat_audio_features,
    )

    frame_lengths = torch.tensor([9, 100])
    output_lengths = omni_audio_output_lengths(frame_lengths)
    flat = torch.arange(15 * 4, dtype=torch.float32).reshape(15, 4)

    padded = pad_flat_audio_features(flat, output_lengths)

    assert output_lengths.tolist() == [2, 13]
    assert padded.embeddings.shape == (2, 13, 4)
    assert padded.attention_mask.sum(dim=1).tolist() == [2, 13]


def test_padded_audio_features_are_packed_for_the_omni_encoder() -> None:
    torch = pytest.importorskip("torch")
    from training_suite.training.omni_encoder_bridge import pack_audio_feature_batch

    features = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
    mask = torch.tensor(
        [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]],
        dtype=torch.long,
    )

    packed, lengths = pack_audio_feature_batch(features, mask)

    assert packed.shape == (3, 7)
    assert lengths.tolist() == [5, 2]
    assert torch.equal(packed[:, :5], features[0])
    assert torch.equal(packed[:, 5:], features[1, :, :2])


def test_checkpoint_contains_only_the_compact_bridge(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from training_suite.training.omni_encoder_bridge import (
        OmniAudioSequenceBridge,
        load_bridge_checkpoint,
        save_bridge_checkpoint,
    )

    config = EncoderBridgeConfig(source_width=4, target_width=6)
    bridge = OmniAudioSequenceBridge(config)
    report = save_bridge_checkpoint(tmp_path, bridge)
    restored = load_bridge_checkpoint(tmp_path)

    assert report["frozen_model_weights_included"] is False
    assert report["state_tensors"] == len(bridge.state_dict())
    assert restored.config == config
    for name, value in bridge.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])


def test_stage_one_wrapper_freezes_both_large_models_and_trains_bridge_only() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.nn import functional

    from training_suite.training.omni_encoder_bridge import (
        FrozenOmniAudioLanguageAdapter,
        OmniAudioSequenceBridge,
    )

    class FakeOmni(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder_scale = nn.Parameter(torch.ones(1))

        def get_audio_features(self, **_kwargs):
            values = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            return SimpleNamespace(last_hidden_state=values * self.encoder_scale)

    class FakeLanguage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.Embedding(32, 6)
            self.head = nn.Linear(6, 32, bias=False)

        def get_input_embeddings(self):
            return self.embeddings

        def forward(self, *, inputs_embeds, attention_mask, labels, use_cache):
            assert use_cache is False
            assert attention_mask.dtype == torch.bool
            logits = self.head(inputs_embeds)
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
            return SimpleNamespace(loss=loss, logits=logits)

    bridge = OmniAudioSequenceBridge(EncoderBridgeConfig(source_width=4, target_width=6))
    adapter = FrozenOmniAudioLanguageAdapter(FakeOmni(), FakeLanguage(), bridge)
    result = adapter(
        input_features=torch.zeros(2, 4, 9),
        feature_attention_mask=torch.tensor(
            [[1] * 9, [1] + [0] * 8],
            dtype=torch.long,
        ),
        prefix_input_ids=torch.tensor([[1, 2], [3, 0]]),
        prefix_attention_mask=torch.tensor([[1, 1], [1, 0]]),
        suffix_input_ids=torch.tensor([[4, 5], [6, 7]]),
        suffix_attention_mask=torch.tensor([[1, 1], [1, 1]]),
        suffix_labels=torch.tensor([[4, 5], [6, 7]]),
    )
    result.loss.backward()

    assert all(parameter.grad is None for parameter in adapter.omni_audio_encoder.parameters())
    assert all(parameter.grad is None for parameter in adapter.language_model.parameters())
    assert all(parameter.grad is not None for parameter in adapter.bridge.parameters())
