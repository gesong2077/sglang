# SPDX-License-Identifier: Apache-2.0
"""Rowwise activation packing for MOSS DiT's serialized ConvRot W8A8 weights."""
from functools import lru_cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .w8a8_int8_kernels import W8A8Activation, dynamic_w8a8_linear


@triton.jit
def _pack_rows(X, Q, S, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    x = tl.load(X + row * WIDTH + col, valid, other=0.0)
    # PyTorch's scalar division by 127 multiplies by its FP32 reciprocal.
    # Its tensor/tensor division rounds before round-to-nearest-even.
    # For nonnegative IEEE floats the unsigned bits sort in magnitude order,
    # with NaNs last. Preserve torch.amax's NaN propagation across the row.
    bits = tl.abs(x).to(tl.uint32, bitcast=True)
    peak = tl.max(bits, axis=0).to(tl.float32, bitcast=True)
    scale = tl.where(peak < 1e-10, 1e-10, peak) * (1.0 / 127)
    rounded = libdevice.rint(tl.div_rn(x, scale))
    tl.store(Q + row * WIDTH + col, rounded.to(tl.int8), valid)
    tl.store(S + row, scale)


def quantize_rotated(rotated, output_dtype):
    width = rotated.shape[-1]
    rows = rotated.numel() // width
    values = torch.empty((rows, width), dtype=torch.int8, device=rotated.device)
    scales = torch.empty(rows, dtype=torch.float32, device=rotated.device)
    _pack_rows[(rows,)](
        rotated, values, scales, width, triton.next_power_of_2(width), num_warps=8
    )
    return W8A8Activation(values, scales, rotated.shape, output_dtype)


@lru_cache(maxsize=1)
def _native_matmul():
    try:
        from .onednn_w8a8 import scaled_int8_mm
    except OSError:
        return None
    return scaled_int8_mm


def project_packed(x, weight, weight_scale):
    native = _native_matmul()
    if native is not None and x.values.shape[0] >= 128:
        out = native(x.values, weight.T, x.scales, weight_scale, optional=True)
        if out is not None:
            return out.view(*x.shape[:-1], weight.shape[0])
    return dynamic_w8a8_linear(x, weight.T, weight_scale)


@triton.jit
def _regular_h4(x, BLOCK: tl.constexpr, STRIDE: tl.constexpr):
    # In a regular H4 each output has one minus sign, at input i XOR 3.
    tiles = tl.reshape(x, (BLOCK // (4 * STRIDE), 2, 2, STRIDE))
    even, odd = tl.split(tl.permute(tiles, (0, 3, 1, 2)))
    a, c = tl.split(even)
    b, d = tl.split(odd)
    ab, cd = a + b, c + d
    left = tl.join(ab + (c - d), cd + (a - b))
    right = tl.join(ab + (d - c), cd + (b - a))
    return tl.reshape(tl.permute(tl.join(left, right), (0, 2, 3, 1)), (BLOCK,))


@triton.jit
def _rotate_pack_rows(
    X,
    Q,
    S,
    WIDTH: tl.constexpr,
    GROUP: tl.constexpr,
    BLOCK: tl.constexpr,
    VALUE=None,
    SILU_TABLE=None,
    SWIGLU: tl.constexpr = False,
    MODULATION=None,
    MODULATE: tl.constexpr = False,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offsets = row.to(tl.int64) * WIDTH + col
    activation = tl.load(X + offsets, col < WIDTH, 0)
    if SWIGLU:
        # The lookup table reproduces ATen's BF16 SiLU for every input bit pattern.
        # Preserve the following BF16 multiply rounding before the FP32 rotation.
        bits = activation.to(tl.uint16, bitcast=True).to(tl.int32)
        gate = tl.load(SILU_TABLE + bits).to(tl.float32)
        value = tl.load(VALUE + offsets, col < WIDTH, 0).to(tl.float32)
        activation = (gate * value).to(tl.bfloat16)
    if MODULATE:
        # Match the two BF16 roundings in norm(x) * (1 + scale).
        scale = tl.load(MODULATION + col, col < WIDTH, 0).to(tl.float32)
        factor = (1.0 + scale).to(tl.bfloat16)
        activation = (activation.to(tl.float32) * factor.to(tl.float32)).to(tl.bfloat16)
    x = activation.to(tl.float32)
    x = _regular_h4(x, BLOCK, 1)
    x = _regular_h4(x, BLOCK, 4)
    if GROUP >= 64:
        x = _regular_h4(x, BLOCK, 16)
    if GROUP >= 256:
        x = _regular_h4(x, BLOCK, 64)
    if GROUP >= 1024:
        x = _regular_h4(x, BLOCK, 256)
    x = x * (1.0 / tl.sqrt(float(GROUP)))
    bits = tl.abs(x).to(tl.uint32, bitcast=True)
    peak = tl.max(bits, axis=0).to(tl.float32, bitcast=True)
    scale = tl.where(peak < 1e-10, 1e-10, peak) * (1.0 / 127)
    tl.store(Q + offsets, libdevice.rint(tl.div_rn(x, scale)).to(tl.int8), col < WIDTH)
    tl.store(S + row, scale)


def rotate_and_pack(
    x, group, *, swiglu_value=None, silu_table=None, modulation=None
):
    width = x.size(-1)
    rows = x.numel() // width
    values = torch.empty((rows, width), dtype=torch.int8, device=x.device)
    scales = torch.empty(rows, dtype=torch.float32, device=x.device)
    warps = 16 if width == 8192 else 8
    if group == 256 and torch.xpu.get_device_properties(x.device).device_id == 0x56C0:
        warps = {2048: 4, 6144: 16}.get(width, warps)
    with torch.xpu.device(x.device):
        _rotate_pack_rows[(rows,)](
            x,
            values,
            scales,
            width,
            group,
            triton.next_power_of_2(width),
            VALUE=swiglu_value,
            SILU_TABLE=silu_table,
            SWIGLU=swiglu_value is not None,
            MODULATION=modulation,
            MODULATE=modulation is not None,
            num_warps=warps,
        )
    return W8A8Activation(values, scales, x.shape, x.dtype)
