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
