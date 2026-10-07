"""Differentiable frozen AFMoE decoder for training Kolibri vision projectors.

Loads our existing GGUF tensors, skips only release-declared identity padding,
and preserves the stock sigmoid-probability routing, MuP input scaling, dual
norms, and RoPE/SWA conventions. CPU parity must pass before GPU training.
Only F32/F16/BF16 active weights are accepted; quantized training is not silently
substituted. The published Q4 variant will use the same trained vision bridge.
"""
from __future__ import annotations

from pathlib import Path
import math
import sys
import warnings
import importlib.util

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / "training_suite/vendor/kolibri-port/llama.cpp/gguf-py"))
import gguf


def rms(value, weight, epsilon):
    normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + epsilon)
    return (normalized * weight.float()).to(value.dtype)


def neox_rope(value, positions, base):
    half = value.shape[-1] // 2
    angle = positions.float()[:, None] / (base ** (torch.arange(half, device=value.device).float() / half))
    cosine, sine = angle.cos()[:, None], angle.sin()[:, None]
    left, right = value[..., :half].float(), value[..., half:].float()
    return torch.cat([left * cosine - right * sine, right * cosine + left * sine], dim=-1).to(value.dtype)


class FrozenDenseLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight):
        if weight.requires_grad:
            raise ValueError("Dense BF16 helper requires frozen base weights")
        ctx.save_for_backward(weight)
        return torch.mm(value.to(weight.dtype), weight.T, out_dtype=torch.float32)

    @staticmethod
    def backward(ctx, gradient):
        (weight,) = ctx.saved_tensors
        # torch's F32-output BF16 GEMM has no registered derivative in this
        # installed build. Supply the frozen-weight input derivative explicitly.
        value_gradient = torch.mm(gradient.to(weight.dtype), weight, out_dtype=torch.float32)
        return value_gradient, None


def linear_f32(value, weight):
    """Match stock BF16-weight GEMM with F32 output/activation storage."""
    if weight.device.type == "cuda" and weight.dtype in (torch.bfloat16, torch.float16):
        return FrozenDenseLinear.apply(value, weight)
    # CPU offload is exact F32; transient conversion is limited to one active
    # dense/expert matrix at a time, never a second full model allocation.
    return F.linear(value.float(), weight.float())


def batched_f32(value, weight):
    if weight.device.type == "cuda" and weight.dtype in (torch.bfloat16, torch.float16):
        return torch.bmm(value.to(weight.dtype), weight.transpose(1, 2), out_dtype=torch.float32)
    return torch.bmm(value.float(), weight.float().transpose(1, 2))


class FrozenAFMoE(nn.Module):
    def __init__(self, path, devices=("cpu",), checkpoint_layers=True, grouped_experts=False,
                 cpu_embedding=False, cpu_head=False, cpu_layers=0, load_progress=None):
        super().__init__()
        self.reader = gguf.GGUFReader(path)
        field = lambda key, default=None: self.reader.fields[key].contents() if key in self.reader.fields else default
        if field("general.architecture") != "afmoe":
            raise ValueError("The released Kolibri GGUF must use AFMoE")
        self.hidden = field("afmoe.embedding_length")
        self.heads = field("afmoe.attention.head_count")
        self.kv_heads = field("afmoe.attention.head_count_kv")
        self.head_dim = field("afmoe.attention.key_length")
        self.topk = field("afmoe.expert_used_count")
        self.experts = field("afmoe.expert_count")
        self.epsilon = field("afmoe.attention.layer_norm_rms_epsilon")
        self.rope_base = field("afmoe.rope.freq_base")
        self.sliding = field("afmoe.attention.sliding_window")
        count = field("afmoe.block_count")
        self.swa = field("afmoe.attention.sliding_window_pattern", [(i + 1) % 4 != 0 for i in range(count)])
        self.norm_routes = field("afmoe.expert_weights_norm", False)
        self.route_scale = field("afmoe.expert_weights_scale", 1.0)
        self.checkpoint_layers = checkpoint_layers
        self.grouped_experts = grouped_experts
        self.capture = None
        spec = importlib.util.spec_from_file_location("ragged_experts", Path(__file__).with_name("ragged-experts.py"))
        self.ragged = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.ragged)
        mapping = field("kolibri.original_layer_mapping", list(range(count)))
        if len(mapping) != count or len(self.swa) != count:
            raise ValueError("Layer mapping or SWA pattern is incomplete")
        active = [i for i, source in enumerate(mapping) if source != -1]
        self.devices = [torch.device(d) for d in devices]
        self.layer_device = {layer: self.devices[min(len(self.devices) - 1, n * len(self.devices) // len(active))]
                             for n, layer in enumerate(active)}
        for layer in active[:cpu_layers]:
            self.layer_device[layer] = torch.device("cpu")
        tensors = {tensor.name: tensor for tensor in self.reader.tensors}

        def load(name, device):
            tensor = tensors[name]
            shape = tuple(reversed(tensor.shape.tolist()))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
                    value = torch.from_numpy(tensor.data.view(np.uint16).reshape(shape)).view(torch.bfloat16)
                elif tensor.tensor_type == gguf.GGMLQuantizationType.F16:
                    value = torch.from_numpy(tensor.data.view(np.float16).reshape(shape))
                elif tensor.tensor_type == gguf.GGMLQuantizationType.F32:
                    value = torch.from_numpy(tensor.data.view(np.float32).reshape(shape))
                else:
                    raise ValueError(f"Unsupported active training tensor {name}: {tensor.tensor_type}")
            return value.detach().to(device)

        self.embedding = load("token_embd.weight", "cpu" if cpu_embedding else self.devices[0])
        self.head_device = torch.device("cpu") if cpu_head else self.devices[-1]
        self.output_norm = load("output_norm.weight", self.head_device)
        self.output_head = load("output.weight", self.head_device)
        self.layers = []
        for layer in active:
            prefix = f"blk.{layer}."
            weights = {}
            for name in tensors:
                if name.startswith(prefix) and name != prefix + "attn_gate.weight":
                    weights[name.removeprefix(prefix)] = load(name, self.layer_device[layer])
            gate = tensors[prefix + "attn_gate.weight"]
            # Release conversion proves the attention gates are exactly zero;
            # never accept an arbitrary nonzero/quantized gate as a constant.
            if gate.tensor_type == gguf.GGMLQuantizationType.F32:
                if np.any(gate.data):
                    raise ValueError("Nonzero attention gates require a differentiable gate projection")
            elif field("kolibri.source_repository") != "Aleph-Alpha/Kolibri-1-BF16":
                raise ValueError("Only the verified Kolibri zero-gate port is accepted")
            self.layers.append((layer, weights))
            if load_progress:
                load_progress(layer, str(self.layer_device[layer]))

    def embeddings(self, tokens):
        return self.embedding[tokens.to(self.embedding.device)].to(self.devices[0])

    def _experts(self, value, weights, selected, probabilities):
        if value.is_cuda and weights["ffn_gate_exps.weight"].dtype == torch.bfloat16:
            routing = self.ragged.routing_tiles(selected, self.experts)
            order = routing[0]
            tokens = order // self.topk
            sorted_value = value[tokens]
            gate = self.ragged.linear(sorted_value, weights["ffn_gate_exps.weight"], routing)
            up = self.ragged.linear(sorted_value, weights["ffn_up_exps.weight"], routing)
            hidden = F.silu(gate) * up
            projected = self.ragged.linear(hidden, weights["ffn_down_exps.weight"], routing)
            weighted = projected * probabilities.reshape(-1)[order, None]
            return value.new_zeros(value.shape).index_add(0, tokens, weighted)
        if self.grouped_experts:
            # Pack each expert's selected tokens, without copying expert matrices
            # once per token. This retains gradients through inputs and scores.
            flat = selected.reshape(-1)
            order = flat.argsort(stable=True)
            chosen = flat[order]
            counts = torch.bincount(flat, minlength=self.experts)
            max_count = int(counts.max().item())
            starts = torch.cumsum(counts, 0) - counts
            offsets = torch.arange(flat.numel(), device=value.device) - torch.repeat_interleave(starts, counts)
            packed = value.new_zeros((self.experts, max_count, self.hidden))
            token_ids = order // self.topk
            packed = packed.index_put((chosen, offsets), value[token_ids])
            gate = batched_f32(packed, weights["ffn_gate_exps.weight"])
            up = batched_f32(packed, weights["ffn_up_exps.weight"])
            hidden = F.silu(gate) * up
            expert_output = batched_f32(hidden, weights["ffn_down_exps.weight"])
            weighted = expert_output[chosen, offsets] * probabilities.reshape(-1)[order, None].to(value.dtype)
            result = value.new_zeros(value.shape)
            return result.index_add(0, token_ids, weighted)
        output = torch.zeros_like(value)
        for expert in range(self.experts):
            token_ids, slots = torch.where(selected == expert)
            if token_ids.numel() == 0:
                continue
            current = value[token_ids]
            hidden = F.silu(linear_f32(current, weights["ffn_gate_exps.weight"][expert]))
            hidden = hidden * linear_f32(current, weights["ffn_up_exps.weight"][expert])
            result = linear_f32(hidden, weights["ffn_down_exps.weight"][expert])
            output = output.index_add(0, token_ids, result * probabilities[token_ids, slots, None].to(value.dtype))
        return output

    def _layer(self, value, layer, weights):
        # Stock stores residuals, norm results, RoPE, and expert nonlinearities
        # in F32 even when the weights themselves are BF16.
        target = self.layer_device[layer]
        if value.is_cuda and value.device != target:
            # Give cross-device handoff a defined completion point; training
            # correctness takes precedence over overlapping this single copy.
            torch.cuda.synchronize(value.device)
        value = value.to(target, dtype=torch.float32)
        normalized = rms(value, weights["attn_norm.weight"], self.epsilon)
        if self.capture:
            self.capture("attn_norm", layer, normalized)
        query = linear_f32(normalized, weights["attn_q.weight"]).reshape(-1, self.heads, self.head_dim)
        key = linear_f32(normalized, weights["attn_k.weight"]).reshape(-1, self.kv_heads, self.head_dim)
        val = linear_f32(normalized, weights["attn_v.weight"]).reshape(-1, self.kv_heads, self.head_dim)
        query = rms(query, weights["attn_q_norm.weight"], self.epsilon)
        key = rms(key, weights["attn_k_norm.weight"], self.epsilon)
        positions = torch.arange(value.shape[0], device=value.device)
        # Stock AFMoE uses a default four-layer NoPE interval. Added identity
        # blocks ensure the port's real full-attention layers land on that cycle.
        if (layer + 1) % 4 != 0:
            query = neox_rope(query, positions, self.rope_base)
            key = neox_rope(key, positions, self.rope_base)
        key = key.repeat_interleave(self.heads // self.kv_heads, dim=1)
        val = val.repeat_interleave(self.heads // self.kv_heads, dim=1)
        allowed = positions[:, None] >= positions[None, :]
        if self.swa[layer]:
            allowed = allowed & ((positions[:, None] - positions[None, :]) < self.sliding)
        attention = F.scaled_dot_product_attention(query.transpose(0, 1)[None], key.transpose(0, 1)[None],
            val.transpose(0, 1)[None], attn_mask=allowed, dropout_p=0)[0].transpose(0, 1).reshape(value.shape[0], -1)
        attention = linear_f32(attention * .5, weights["attn_output.weight"])
        residual = value + rms(attention, weights["post_attention_norm.weight"], self.epsilon)
        normalized = rms(residual, weights["ffn_norm.weight"], self.epsilon)
        scores = F.linear(normalized.float(), weights["ffn_gate_inp.weight"].float())
        unbiased = scores.sigmoid()
        selected = torch.topk(unbiased + weights["exp_probs_b.bias"], self.topk, dim=-1).indices
        probabilities = unbiased.gather(-1, selected)
        if self.norm_routes:
            probabilities = probabilities / probabilities.sum(-1, keepdim=True)
        probabilities = probabilities * self.route_scale
        routed = self._experts(normalized, weights, selected, probabilities)
        shared = F.silu(linear_f32(normalized, weights["ffn_gate_shexp.weight"]))
        shared = shared * linear_f32(normalized, weights["ffn_up_shexp.weight"])
        shared = linear_f32(shared, weights["ffn_down_shexp.weight"])
        result = residual + rms(routed + shared, weights["post_ffw_norm.weight"], self.epsilon)
        if self.capture:
            self.capture("l_out", layer, result)
        return result

    def forward(self, input_embeddings, output_positions=None):
        # Projector outputs and token embeddings both receive this stock scaling.
        value = input_embeddings * math.sqrt(self.hidden)
        for layer, weights in self.layers:
            if self.checkpoint_layers and torch.is_grad_enabled():
                value = checkpoint(lambda x, layer=layer, weights=weights: self._layer(x, layer, weights),
                                   value, use_reentrant=False)
            else:
                value = self._layer(value, layer, weights)
        if output_positions is not None:
            value = value[output_positions]
        value = rms(value.to(self.head_device), self.output_norm, self.epsilon)
        # Keep the head in F32, matching the published port. Chunk answer
        # positions at the training-loss layer instead of materializing all logits.
        return F.linear(value.float(), self.output_head.float())
