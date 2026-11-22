"""Opt-in exact INT8 GEMV for single-token XPU decode.

The companion SYCL library submits to PyTorch's current queue, including graph
capture. It consumes the existing [K,N] transposed checkpoint without repacking.
"""
import ctypes as C
import os

import torch

_lib = C.CDLL(os.environ["SGLANG_XPU_W8A8_DP4A_LIB"])
_lib.dg_run.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4
_lib.dg_run.restype = C.c_int
_lib.dg_error.restype = C.c_char_p

_prefetch = None
if os.environ.get("SGLANG_XPU_W8A8_DP4A_PREFETCH") == "1":
    _prefetch = C.CDLL(os.environ["SGLANG_XPU_W8A8_DP4A_PREFETCH_LIB"])
    _prefetch.pf_run.argtypes = [C.c_void_p] * 4 + [C.c_int] * 5
    _prefetch.pf_run.restype = C.c_int
    _prefetch.pf_error.restype = C.c_char_p


def int8_gemv(q, weight):
    m, k = q.shape
    n = weight.shape[1]
    if (
        m != 1
        or k == 0
        or k % 256
        or n == 0
        or q.dtype != torch.int8
        or weight.dtype != torch.int8
        or not q.is_contiguous()
        or weight.shape[0] != k
        or weight.stride() != (1, k)
        or weight.device != q.device
    ):
        return torch._int_mm(q, weight)
    output = torch.empty((1, n), dtype=torch.int32, device=q.device)
    local_size = 32 if k == 2048 else 16
    # Only the two Muse projection shapes with stable gains on both B60 cards.
    # One weight block ahead, L1 bypassed; integer accumulation is unchanged.
    if _prefetch is not None and (k, n) in ((9984, 6656), (2048, 6656)):
        if _prefetch.pf_run(
            torch.xpu.current_stream(q.device).sycl_queue,
            q.data_ptr(), weight.data_ptr(), output.data_ptr(),
            k, n, 1, local_size, 0,
        ):
            raise RuntimeError(_prefetch.pf_error().decode())
        return output
    if _lib.dg_run(
        torch.xpu.current_stream(q.device).sycl_queue,
        q.data_ptr(),
        weight.data_ptr(),
        output.data_ptr(),
        k,
        n,
        256,
        local_size,
    ):
        raise RuntimeError(_lib.dg_error().decode())
    return output
