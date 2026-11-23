# SPDX-License-Identifier: Apache-2.0
"""XPU inference kernels for dense BF16 MOSS DiT blocks."""

from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.numerics import round_bf16_to_fp32


@triton.jit
def _bf16(x):
    return tl.where(x == x, round_bf16_to_fp32(x), x)


@lru_cache(maxsize=8)
def _situ_tables(device):
    values = (
        torch.arange(65536, device=device, dtype=torch.int32)
        .short()
        .view(torch.bfloat16)
    )
    hidden = 25.0 * torch.tanh(values / 25.0)
    gate = (4.0 * torch.tanh(values / 4.0)) * torch.sigmoid(values)
    # Make the immutable tables visible to users on other streams.
    torch.xpu.current_stream(device).synchronize()
    return hidden, gate


@triton.jit
def _situ(OUT, H, G, HT, GT, N, HS, GS, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = pos // WIDTH, pos % WIDTH
    h = tl.load(H + row * HS + col, pos < N, 0).to(tl.uint16, bitcast=True)
    g = tl.load(G + row * GS + col, pos < N, 0).to(tl.uint16, bitcast=True)
    hf = tl.load(HT + h.to(tl.int32)).to(tl.float32)
    gf = tl.load(GT + g.to(tl.int32)).to(tl.float32)
    tl.store(OUT + pos, hf * gf, pos < N)


def situ_if_supported(hidden, gate):
    if not (
        hidden.dtype == gate.dtype == torch.bfloat16
        and hidden.device == gate.device
        and hidden.ndim == 3
        and hidden.shape == gate.shape
        and hidden.numel() >= 65536
        and all(
            t.stride(2) == 1
            and (t.size(0) == 1 or t.stride(0) == t.size(1) * t.stride(1))
            for t in (hidden, gate)
        )
    ):
        return None
    out = torch.empty_like(hidden, memory_format=torch.contiguous_format)
    with torch.xpu.device(hidden.device):
        if torch.compiler.is_compiling() or torch.xpu.is_current_stream_capturing():
            return None
        _situ[(triton.cdiv(hidden.numel(), 1024),)](
            out,
            hidden,
            gate,
            *_situ_tables(hidden.device),
            hidden.numel(),
            hidden.stride(1),
            gate.stride(1),
            hidden.size(2),
            1024,
            num_warps=8,
        )
    return out


@triton.jit
def _norm_finish(
    X,
    INV,
    W,
    A,
    B,
    OUT,
    N,
    SEQ,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    SA,
    SB,
    ROPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pos = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, col = pos // D, pos % D
    mask = pos < N
    x = tl.load(X + pos, mask, 0).to(tl.float32)
    inv = tl.load(INV + row, mask, 0)
    w = tl.load(W + col).to(tl.float32)
    value = _bf16(_bf16(libdevice.mul_rn(x, inv)) * w)
    if ROPE:
        other = col ^ (D // 2)
        mate = tl.load(X + row * D + other, mask, 0).to(tl.float32)
        mate_w = tl.load(W + other).to(tl.float32)
        mate = _bf16(_bf16(libdevice.mul_rn(mate, inv)) * mate_w)
        mate = tl.where(col < D // 2, -mate, mate)
        off = (row // HEADS) * D + col
        sin = tl.load(A + off, mask, 0).to(tl.float32)
        cos = tl.load(B + off, mask, 0).to(tl.float32)
        result = _bf16(value * cos) + _bf16(mate * sin)
    else:
        batch = row // SEQ
        scale = tl.load(A + batch * SA + col, mask, 0).to(tl.float32)
        shift = tl.load(B + batch * SB + col, mask, 0).to(tl.float32)
        result = _bf16(value * _bf16(1.0 + scale)) + shift
    tl.store(OUT + pos, _bf16(result), mask)


def norm_finish_if_supported(x, weight, a, b, heads, *, rope):
    if x.ndim != 3 or x.size(-1) % heads:
        return None
    batch, seq, width = x.shape
    dim = width // heads
    param_shape = (batch, seq, dim) if rope else (batch, dim)
    if not (
        x.numel() >= 65536
        and (not rope or dim == 128)
        and x.is_contiguous()
        and weight.is_contiguous()
        and weight.shape == (dim,)
        and a.shape == b.shape == param_shape
        and all(
            t.device == x.device and t.dtype == torch.bfloat16
            for t in (x, weight, a, b)
        )
        and a.stride(-1) == b.stride(-1) == 1
        and (not rope or (a.is_contiguous() and b.is_contiguous()))
    ):
        return None
    # Keep aten's reduction order and rsqrt. Even a one-ULP change before
    # rounding to BF16 can affect the trajectory over 50 denoising steps.
    rows = x.unflatten(-1, (heads, dim)) if rope else x
    inv = torch.rsqrt(rows.float().square().mean(-1, keepdim=True) + 1e-6)
    out = torch.empty_like(x)
    with torch.xpu.device(x.device):
        _norm_finish[(triton.cdiv(x.numel(), 512),)](
            x,
            inv,
            weight,
            a,
            b,
            out,
            x.numel(),
            seq,
            heads,
            dim,
            a.stride(0),
            b.stride(0),
            rope,
            512,
            num_warps=8,
            enable_fp_fusion=False,
        )
    return out
