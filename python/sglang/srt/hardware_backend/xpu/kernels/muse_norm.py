"""Opt-in Muse post-norm/residual/pre-norm/quantization fusion.

The native implementation preserves every existing BF16 materialization and
the pinned sgl-kernel-xpu reduction order. Inputs are never mutated. Build the
companion library explicitly and supply SGLANG_XPU_MUSE_NORM_LIB.
"""

import ctypes as C
import os

import torch

from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import W8A8Activation

_lib = C.CDLL(os.environ["SGLANG_XPU_MUSE_NORM_LIB"])
_lib.mn_run.argtypes = [C.c_void_p] * 9 + [C.c_int] * 3 + [C.c_float] * 2
_lib.mn_run.restype = C.c_int
_lib.mn_error.restype = C.c_char_p


def fused_post_norm(x, residual, norm, pre_norm=None, *, quantize=False):
    m, k = x.shape
    weights = [norm.weight] + ([pre_norm.weight] if pre_norm is not None else [])
    if (
        k != 6656
        or m < 1
        or residual.shape != x.shape
        or any(t.dtype != torch.bfloat16 or not t.is_contiguous()
               or t.device != x.device for t in [x, residual, *weights])
        or any(t.numel() != k for t in weights)
        or (quantize and pre_norm is None)
    ):
        raise ValueError("Fused Muse norms require contiguous BF16 width-6656 tensors")
    mode = 1 if pre_norm is None else 3 if quantize else 2
    y = torch.empty_like(x)
    z = torch.empty_like(x) if mode == 2 else None
    q = torch.empty_like(x, dtype=torch.int8) if mode == 3 else None
    s = torch.empty(m, device=x.device, dtype=torch.float32) if mode == 3 else None
    w2 = pre_norm.weight if pre_norm is not None else None
    if _lib.mn_run(
        torch.xpu.current_stream(x.device).sycl_queue,
        *[t.data_ptr() if t is not None else 0 for t in [x, residual, norm.weight, w2, y, z, q, s]],
        m, k, mode, norm.variance_epsilon,
        pre_norm.variance_epsilon if pre_norm is not None else norm.variance_epsilon,
    ):
        raise RuntimeError(_lib.mn_error().decode())
    if mode == 1:
        return (y,)
    if mode == 2:
        return y, z
    return y, W8A8Activation(q, s, (m, k))
