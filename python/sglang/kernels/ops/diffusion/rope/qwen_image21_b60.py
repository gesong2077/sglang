# SPDX-License-Identifier: Apache-2.0
"""BF16 Qwen-Image 2.1 epilogues for Intel Arc Pro B60 and Data Center GPU Flex 170."""

from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache
def _is_b60(device):
    return "Arc(TM) Pro B60" in torch.xpu.get_device_name(device)


@lru_cache
def _is_flex(device):
    return torch.xpu.get_device_properties(device).device_id == 0x56C0


def supports_b60(x):
    return (
        x.device.type == "xpu"
        and x.dtype is torch.bfloat16
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and _is_b60(x.device)
    )


def can_fuse_qk(x, weight, rope):
    return (
        x.device.type == "xpu"
        and x.dtype is torch.bfloat16
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and (
            _is_b60(x.device) or (_is_flex(x.device) and x.shape == (1, 4096, 16, 128))
        )
        and x.ndim == 4
        and x.shape[0] == 1
        and x.shape[1] >= 128
        and x.shape[-1] == 128
        and x.is_contiguous()
        and weight.shape == (128,)
        and weight.dtype == x.dtype
        and weight.device == x.device
        and weight.is_contiguous()
        and rope.dtype == torch.complex64
        and rope.shape == (x.shape[1], 64)
        and rope.device == x.device
        and rope.is_contiguous()
    )


@triton.jit
def _b60_inverse_rms(x, BLOCK: tl.constexpr, EPS: tl.constexpr):
    # ATen XPU ReduceKernel<MeanOps, vt=4> assigns a 128-wide row to
    # 32 lanes. Each lane adds four strided squares in order. The subgroup
    # then combines adjacent pairs with increasing shuffle distances.
    squares = tl.permute(tl.reshape(x * x, (BLOCK, 4, 32)), (0, 2, 1))
    even, odd = tl.split(tl.reshape(squares, (BLOCK, 32, 2, 2)))
    a, c = tl.split(even)
    b, d = tl.split(odd)
    total = ((a + b) + c) + d
    a, b = tl.split(tl.reshape(total, (BLOCK, 16, 2)))
    a, b = tl.split(tl.reshape(a + b, (BLOCK, 8, 2)))
    a, b = tl.split(tl.reshape(a + b, (BLOCK, 4, 2)))
    a, b = tl.split(tl.reshape(a + b, (BLOCK, 2, 2)))
    a, b = tl.split(tl.reshape(a + b, (BLOCK, 2)))
    return tl.rsqrt((a + b) * (1.0 / 128) + EPS)


@triton.jit
def _flex_inverse_rms(
    X, row, ROWS: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr
):
    # ATen's 128-wide contiguous mean on Flex uses eight lanes and four
    # independent accumulators per lane. Load those strides into registers
    # directly; transposing the entire 128-wide row to match ATen is expensive.
    offset = row[:, None] * 128 + tl.arange(0, 8)[None, :]
    total = tl.full((BLOCK, 8), 0, tl.float32)
    for register in tl.static_range(4):
        partial = tl.full((BLOCK, 8), 0, tl.float32)
        for step in tl.static_range(4):
            value = tl.load(
                X + offset + (4 * step + register) * 8, row[:, None] < ROWS, 0
            ).to(tl.float32)
            partial += value * value
        total += partial
    a, b = tl.split(tl.reshape(total, (BLOCK, 4, 2)))
    a, b = tl.split(tl.reshape(a + b, (BLOCK, 2, 2)))
    a, b = tl.split(tl.reshape(a + b, (BLOCK, 2)))
    return tl.rsqrt((a + b) * (1.0 / 128) + EPS)


@triton.jit
def _qk_kernel(
    X,
    W,
    R,
    Y,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    FLEX: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = tl.arange(0, 128)
    offset = row[:, None] * 128 + col[None, :]
    x = tl.load(X + offset, row[:, None] < ROWS, 0).to(tl.float32)
    if FLEX:
        inv = _flex_inverse_rms(X, row, ROWS, BLOCK, EPS)
    else:
        inv = _b60_inverse_rms(x, BLOCK, EPS)
    weight = tl.load(W + col).to(tl.float32)
    x = (x * inv[:, None]).to(tl.bfloat16).to(tl.float32)
    x = (x * weight[None, :]).to(tl.bfloat16).to(tl.float32)
    real, imag = tl.split(tl.reshape(x, (BLOCK, 64, 2)))
    rotation = tl.load(
        R + (row // HEADS)[:, None] * 128 + col[None, :], row[:, None] < ROWS, 0
    )
    cos, sin = tl.split(tl.reshape(rotation, (BLOCK, 64, 2)))
    re = tl.fma(real, cos, -imag * sin)
    im = tl.fma(real, sin, imag * cos)
    tl.store(Y + offset, tl.reshape(tl.join(re, im), (BLOCK, 128)), row[:, None] < ROWS)


def qk_norm_rope(x, weight, rope, eps):
    out = torch.empty_like(x)
    rows = x.numel() // 128
    flex = _is_flex(x.device)
    block = 32 if flex else 16
    with torch.xpu.device(x.device):
        _qk_kernel[(triton.cdiv(rows, block),)](
            x,
            weight,
            torch.view_as_real(rope),
            out,
            rows,
            x.shape[2],
            eps,
            block,
            flex,
            num_warps=8 if flex else 16,
            enable_fp_fusion=False,
        )
    return out


@lru_cache
def _supports_pointwise(device):
    return _is_b60(device) or _is_flex(device)


def can_fuse_pair(x, other):
    return (
        x.device.type == "xpu"
        and x.dtype is torch.bfloat16
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
        and _supports_pointwise(x.device)
        and x.ndim == 3
        and x.shape[1] >= 128
        and x.shape == other.shape
        and other.dtype == x.dtype
        and other.device == x.device
        and x.is_contiguous()
        and other.is_contiguous()
    )


@triton.jit
def _pointwise_kernel(
    X,
    U,
    G,
    Y,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    RESIDUAL: tl.constexpr,
    ROW_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0)
    update = tl.load(U + i, i < N, 0).to(tl.float32)
    if RESIDUAL:
        gate = tl.load(G + i % WIDTH).to(tl.float32)
        out = x.to(tl.float32) + (update * gate).to(tl.bfloat16).to(tl.float32)
    else:
        bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
        out = tl.load(G + bits).to(tl.float32) * update
    tl.store(Y + i + (i // WIDTH) * ROW_PAD, out, i < N)


@lru_cache
def silu_table(device):
    # BF16 has 65,536 bit patterns. Obtain the exact ATen XPU SiLU for
    # each pattern once, including subnormals and saturated negative inputs.
    values = torch.arange(65536, dtype=torch.int32, device=device)
    return torch.nn.functional.silu(values.to(torch.int16).view(torch.bfloat16))


def fused_pointwise(x, update, gate=None, *, output_row_padding=0):
    width = x.shape[-1]
    if output_row_padding:
        out = x.new_empty((*x.shape[:-1], width + output_row_padding))[..., :width]
    else:
        out = torch.empty_like(x)
    with torch.xpu.device(x.device):
        _pointwise_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            update,
            gate if gate is not None else silu_table(x.device),
            out,
            x.numel(),
            x.shape[-1],
            gate is not None,
            output_row_padding,
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
