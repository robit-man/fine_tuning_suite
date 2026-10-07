"""Frozen BF16 expert GEMMs without per-expert maximum-token padding.

Sorted routed tokens are contiguous per expert. Tile indices cover only that
expert's actual rows. F32 outputs/gradients use BF16 tensor-core products with
F32 accumulation, matching the deployed weight/activation precision contract.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gemm(INPUT, WEIGHT, OUTPUT, STARTS, COUNTS, TILE_EXPERTS, TILE_ROWS,
         INPUT_WIDTH: tl.constexpr, OUTPUT_WIDTH: tl.constexpr,
         WE: tl.constexpr, WO: tl.constexpr, WI: tl.constexpr,
         TRANSPOSE: tl.constexpr, BM: tl.constexpr=16, BN: tl.constexpr=64, BK: tl.constexpr=64):
    tile = tl.program_id(0)
    expert = tl.load(TILE_EXPERTS + tile)
    tile_row = tl.load(TILE_ROWS + tile)
    start = tl.load(STARTS + expert)
    count = tl.load(COUNTS + expert)
    local_rows = tile_row * BM + tl.arange(0, BM)
    rows = start + local_rows
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    compensation = tl.full((BM, BN), 0, tl.float32)
    for offset in range(tl.cdiv(INPUT_WIDTH, BK)):
        k = offset * BK + inner
        a = tl.load(INPUT + rows[:, None] * INPUT_WIDTH + k[None, :],
            mask=(local_rows[:, None] < count) & (k[None, :] < INPUT_WIDTH), other=0).to(tl.bfloat16)
        if TRANSPOSE:
            pointer = WEIGHT + expert * WE + k[:, None] * WO + columns[None, :] * WI
        else:
            pointer = WEIGHT + expert * WE + columns[None, :] * WO + k[:, None] * WI
        b = tl.load(pointer, mask=(k[:, None] < INPUT_WIDTH) & (columns[None, :] < OUTPUT_WIDTH), other=0)
        # Keep each tensor-core subtotal independent, then compensate the
        # F32 reduction. This avoids precision loss from long MMA accumulators
        # at Kolibri's actual 2560-wide expert dimensions.
        subtotal = tl.dot(a, b, input_precision="ieee")
        corrected = subtotal - compensation
        updated = acc + corrected
        compensation = (updated - acc) - corrected
        acc = updated
    tl.store(OUTPUT + rows[:, None] * OUTPUT_WIDTH + columns[None, :], acc,
        mask=(local_rows[:, None] < count) & (columns[None, :] < OUTPUT_WIDTH))


def launch(value, weight, starts, counts, tile_experts, tile_rows, transpose):
    width = weight.shape[2] if transpose else weight.shape[1]
    output = torch.empty((value.shape[0], width), device=value.device, dtype=torch.float32)
    # Triton selects the current CUDA device/stream, unlike torch operators
    # which dispatch from their input tensors. Explicitly scope both passes.
    with torch.cuda.device(value.device):
        gemm[(tile_experts.numel(), triton.cdiv(width, 64))](value, weight, output,
            starts, counts, tile_experts, tile_rows, value.shape[1], width, *weight.stride(), transpose, enable_fp_fusion=False)
    return output


class FrozenGroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, weight, starts, counts, tile_experts, tile_rows):
        if weight.dtype != torch.bfloat16 or weight.requires_grad:
            raise ValueError("Ragged GEMM requires frozen BF16 weights")
        ctx.save_for_backward(weight, starts, counts, tile_experts, tile_rows)
        return launch(value.contiguous(), weight, starts, counts, tile_experts, tile_rows, False)

    @staticmethod
    def backward(ctx, gradient):
        weight, starts, counts, tile_experts, tile_rows = ctx.saved_tensors
        # Straight-through BF16 input rounding is the standard mixed-precision
        # training surrogate. Keep accumulated input gradients in F32.
        result = launch(gradient.contiguous(), weight, starts, counts, tile_experts, tile_rows, True)
        return result, None, None, None, None, None


def routing_tiles(selected, experts):
    flat = selected.reshape(-1)
    order = flat.argsort(stable=True)
    counts = torch.bincount(flat, minlength=experts).to(torch.int32)
    starts = (counts.cumsum(0) - counts).to(torch.int32)
    tile_counts = torch.div(counts + 15, 16, rounding_mode="floor")
    tile_experts = torch.repeat_interleave(torch.arange(experts, device=flat.device, dtype=torch.int32), tile_counts)
    tile_starts = tile_counts.cumsum(0) - tile_counts
    tile_rows = (torch.arange(tile_experts.numel(), device=flat.device) - torch.repeat_interleave(tile_starts, tile_counts)).to(torch.int32)
    return order, starts, counts, tile_experts, tile_rows


def linear(value, weight, routing):
    _, starts, counts, tile_experts, tile_rows = routing
    return FrozenGroupedLinear.apply(value, weight, starts, counts, tile_experts, tile_rows)
