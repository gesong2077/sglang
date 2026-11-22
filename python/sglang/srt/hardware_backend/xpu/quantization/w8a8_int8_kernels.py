"""Dynamic symmetric INT8 W8A8 on XPU: per-token quantization and integer GEMM."""

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_tune_quantize = os.environ.get("SGLANG_XPU_W8A8_TUNE_QUANTIZE") == "1"
_use_onednn = os.environ.get("SGLANG_XPU_W8A8_ONEDNN") == "1"
_use_dp4a = os.environ.get("SGLANG_XPU_W8A8_DP4A") == "1"
_use_dp4a_scaled = os.environ.get("SGLANG_XPU_W8A8_DP4A_SCALED") == "1"
_use_silu_quant = os.environ.get("SGLANG_XPU_MUSE_SILU_QUANT") == "1"
_use_silu_lut = os.environ.get("SGLANG_XPU_MUSE_SILU_LUT") == "1"
if _use_silu_lut:
    from .muse_silu_lut import quantize as silu_lut_quantize
if _use_dp4a:
    from .dp4a_gemv import int8_gemv
if _use_dp4a_scaled:
    from .dp4a_scaled import down_gemv
if _use_onednn:
    from .onednn_w8a8 import scaled_int8_mm


@dataclass(frozen=True)
class W8A8Activation:
    """Internal prequantized activation passed through a parallel linear layer."""

    values: torch.Tensor
    scales: torch.Tensor
    shape: tuple
    dtype: torch.dtype = torch.bfloat16


@triton.jit
def _quantize_rows(X, Q, S, K: tl.constexpr, SX: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * SX + col, col < K, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x)), 1.0e-10) / 127.0
    q = libdevice.rint(x / scale).to(tl.int8)
    tl.store(Q + row * K + col, q, col < K)
    tl.store(S + row, scale)


@triton.jit
def _scale_output(
    C,
    XS,
    WS,
    BIAS,
    Y,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < TOTAL
    value = tl.load(C + idx, valid, other=0).to(tl.float32)
    value *= tl.load(XS + idx // N, valid, other=0)
    value *= tl.load(WS + idx % N, valid, other=0)
    if HAS_BIAS:
        value += tl.load(BIAS + idx % N, valid, other=0).to(tl.float32)
    tl.store(Y + idx, value, valid)


@triton.jit
def _scale_silu(
    C,
    XS,
    WS,
    Y,
    H: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
    PRESCALED: tl.constexpr = False,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < TOTAL
    row, col = i // H, i % H
    gate = tl.load(C + row * (H * 2) + col, valid, 0).to(tl.float32)
    up = tl.load(C + row * (H * 2) + col + H, valid, 0).to(tl.float32)
    if not PRESCALED:
        xs = tl.load(XS + row, valid, 0)
        gate = gate * xs * tl.load(WS + col, valid, 0)
        up = up * xs * tl.load(WS + col + H, valid, 0)
    # Preserve the materialized BF16 linear output before activation.
    gate = gate.to(Y.dtype.element_ty).to(tl.float32)
    up = up.to(Y.dtype.element_ty).to(tl.float32)
    value = gate / (1.0 + libdevice.exp(-gate)) * up
    tl.store(Y + i, value, valid)


@triton.jit
def _gate_scale_quant(
    C,
    XS,
    WS,
    A,
    Q,
    S,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    PRESCALED: tl.constexpr = False,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < H
    gate = tl.load(C + row * H + col, valid, 0).to(tl.float32)
    if not PRESCALED:
        gate = gate * tl.load(XS + row) * tl.load(WS + col, valid, 0)
    gate = gate.to(tl.bfloat16).to(tl.float32)
    sigmoid = tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    value = tl.load(A + row * H + col, valid, 0).to(tl.float32) * sigmoid
    # Keep both the gate and gated activation's existing BF16 rounding.
    value = value.to(tl.bfloat16).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value)), 1.0e-10) / 127.0
    q = libdevice.rint(value / scale).to(tl.int8)
    tl.store(Q + row * H + col, q, valid)
    tl.store(S + row, scale)


@triton.jit
def _silu_quant(C,XS,WS,Q,S,H:tl.constexpr,PRESCALED:tl.constexpr,B:tl.constexpr):
 row=tl.program_id(0);col=tl.arange(0,B);mask=col<H
 gate=tl.load(C+row*2*H+col,mask,0).to(tl.float32)
 up=tl.load(C+row*2*H+col+H,mask,0).to(tl.float32)
 if not PRESCALED:
  xs=tl.load(XS+row)
  gate=gate*xs*tl.load(WS+col,mask,0)
  up=up*xs*tl.load(WS+col+H,mask,0)
 gate=gate.to(tl.bfloat16).to(tl.float32)
 up=up.to(tl.bfloat16).to(tl.float32)
 value=(gate/(1.0+libdevice.exp(-gate))*up).to(tl.bfloat16).to(tl.float32)
 value=tl.where(mask,value,0)
 scale=tl.maximum(tl.max(tl.abs(value)),1.e-10)/127.
 quant=libdevice.rint(value/scale).to(tl.int8)
 tl.store(Q+row*H+col,quant,mask);tl.store(S+row,scale)


def dynamic_w8a8_linear(
    x, weight, weight_scale, bias=None, *, silu_mul=False, output_gate=None,
    pack_output_gate=False,
):
    """weight is the channel-scaled INT8 checkpoint transposed to [K, N]."""
    shape = x.shape
    dtype = x.dtype
    if isinstance(x, W8A8Activation):
        quantized, scales = x.values, x.scales
        m, k = quantized.shape
        device = quantized.device
    else:
        x = x.reshape(-1, shape[-1])
        if x.stride(1) != 1:
            x = x.contiguous()
        m, k = x.shape
        device = x.device
        quantized = torch.empty((m, k), dtype=torch.int8, device=device)
        scales = torch.empty((m,), dtype=torch.float32, device=device)
        tune = _tune_quantize and (k == 9984 or (m == 1 and k == 6656))
        options = {"num_warps": 16, "grf_mode": "128"} if tune else {"num_warps": 8}
        _quantize_rows[(m,)](
            x, quantized, scales, k, x.stride(0), triton.next_power_of_2(k), **options
        )
    n = weight.shape[1]
    if weight_scale.numel() != n:
        raise NotImplementedError(
            "XPU W8A8 currently requires per-channel weight scales"
        )
    if silu_mul and (bias is not None or n % 2 or dtype != torch.bfloat16):
        raise ValueError("Fused W8A8 SiLU requires unbiased, even-width BF16 output")
    if (
        _use_dp4a and _use_dp4a_scaled and (m, k, n) == (1, 9984, 6656)
        and dtype == torch.bfloat16 and bias is None and not silu_mul
        and output_gate is None
    ):
        result = down_gemv(quantized, weight, scales, weight_scale)
        if result is not None:
            return result.reshape(*shape[:-1], n)
    prescaled = _use_onednn and m >= 128 and dtype == torch.bfloat16 and bias is None
    accum = (
        scaled_int8_mm(quantized, weight, scales, weight_scale)
        if prescaled
        else (
            int8_gemv(quantized, weight)
            if _use_dp4a and m == 1
            else torch._int_mm(quantized, weight)
        )
    )
    if output_gate is not None:
        if (
            silu_mul
            or bias is not None
            or dtype != torch.bfloat16
            or output_gate.dtype != torch.bfloat16
            or not output_gate.is_contiguous()
            or output_gate.numel() != m * n
        ):
            raise ValueError(
                "Fused output gate requires matching contiguous BF16 activations"
            )
        if pack_output_gate:
            from sglang.srt.hardware_backend.xpu.kernels.muse_o_pack import quantize
            return quantize(accum, scales, weight_scale, output_gate, prescaled=prescaled)
        q = torch.empty((m, n), dtype=torch.int8, device=device)
        s = torch.empty((m,), dtype=torch.float32, device=device)
        _gate_scale_quant[(m,)](
            accum,
            scales,
            weight_scale,
            output_gate,
            q,
            s,
            n,
            triton.next_power_of_2(n),
            PRESCALED=prescaled,
            num_warps=16,
            grf_mode="128",
        )
        return W8A8Activation(q, s, (*shape[:-1], n), dtype)
    if silu_mul and _use_silu_quant and m >= 1024 and n == 19968:
        if _use_silu_lut and prescaled:
            q, s = silu_lut_quantize(accum)
            return W8A8Activation(q, s, (*shape[:-1], n // 2), dtype)
        q = torch.empty((m, n // 2), dtype=torch.int8, device=device)
        s = torch.empty(m, dtype=torch.float32, device=device)
        _silu_quant[(m,)](
            accum, scales, weight_scale, q, s, n // 2, prescaled,
            triton.next_power_of_2(n // 2), num_warps=32, grf_mode="128",
        )
        return W8A8Activation(q, s, (*shape[:-1], n // 2), dtype)
    if silu_mul:
        result = torch.empty((m, n // 2), dtype=dtype, device=device)
        _scale_silu[(triton.cdiv(m * (n // 2), 512),)](
            accum,
            scales,
            weight_scale,
            result,
            n // 2,
            m * (n // 2),
            512,
            PRESCALED=prescaled,
            num_warps=4,
            grf_mode="128",
        )
        return result.reshape(*shape[:-1], n // 2)
    if prescaled:
        return accum.reshape(*shape[:-1], n)
    result = torch.empty((m, n), dtype=dtype, device=device)
    _scale_output[(triton.cdiv(m * n, 1024),)](
        accum,
        scales,
        weight_scale,
        bias,
        result,
        n,
        m * n,
        bias is not None,
        1024,
        num_warps=4,
    )
    return result.reshape(*shape[:-1], n)
