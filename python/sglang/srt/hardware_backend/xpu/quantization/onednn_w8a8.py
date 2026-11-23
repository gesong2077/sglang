"""Opt-in oneDNN INT8 GEMM with per-token and per-channel scale epilogues.

The companion host bridge is built against oneDNN SYCL. It can be installed
alongside this module or selected with SGLANG_XPU_W8A8_ONEDNN_LIB.
"""

import ctypes as C
import os
from functools import lru_cache
from pathlib import Path

import torch

_sequence_padding = os.environ.get("SGLANG_XPU_MUSE_SEQUENCE_PARALLEL") == "1"
_sequence_alignment = int(os.environ.get("SGLANG_XPU_MUSE_SEQUENCE_ALIGNMENT", "2"))
_lib = C.CDLL(
    os.environ.get(
        "SGLANG_XPU_W8A8_ONEDNN_LIB", str(Path(__file__).with_name("libonednn_w8a8.so"))
    )
)
_lib.ow_error.restype = C.c_char_p
_lib.ow_create.argtypes = [C.c_void_p] + [C.c_int] * 5
_lib.ow_create.restype = C.c_void_p
_lib.ow_scratch.argtypes = [C.c_void_p]
_lib.ow_scratch.restype = C.c_size_t
_lib.ow_run.argtypes = [C.c_void_p] * 8
_lib.ow_run.restype = C.c_int
_lib.ow_destroy.argtypes = [C.c_void_p]


class _Plan:
    def __init__(self, device, m, k, n, transposed):
        queue = torch.xpu.current_stream(device).sycl_queue
        # Binary post-ops preserve (float(accum) * row_scale) * column_scale.
        self._destroy = _lib.ow_destroy
        self.ptr = _lib.ow_create(queue, m, k, n, transposed, 2)
        if not self.ptr:
            raise RuntimeError(_lib.ow_error().decode())
        self.scratch_bytes = max(1, _lib.ow_scratch(self.ptr))

    def __del__(self):
        if getattr(self, "ptr", None):
            self._destroy(self.ptr)


@lru_cache(maxsize=128)
def _plan(device, m, k, n, transposed):
    return _Plan(device, m, k, n, transposed)


@lru_cache(maxsize=128)
def _optional_plan(device, m, k, n, transposed):
    try:
        return _plan(device, m, k, n, transposed)
    except RuntimeError:
        return None


def scaled_int8_mm(q, weight, row_scale, column_scale, *, optional=False):
    m, k = q.shape
    n = weight.shape[1]
    if (
        q.dtype != torch.int8
        or weight.dtype != torch.int8
        or row_scale.dtype != torch.float32
        or column_scale.dtype != torch.float32
        or not q.is_contiguous()
        or not row_scale.is_contiguous()
        or not column_scale.is_contiguous()
        or weight.shape[0] != k
        or row_scale.numel() != m
        or column_scale.numel() != n
        or weight.stride() not in ((1, k), (n, 1))
        or any(t.device != q.device for t in (weight, row_scale, column_scale))
    ):
        raise ValueError("Unsupported oneDNN W8A8 tensor layout or dtype")
    get_plan = _optional_plan if optional else _plan
    plan = get_plan(q.device.index, m, k, n, int(weight.stride(0) == 1))
    if plan is None:
        return None
    # Muse's TP2 reduce-scatter needs an even number of rows. Keep the logical
    # GEMM shape unchanged but reserve a tail so the communication path can
    # initialize just its padding, avoiding a full activation copy.
    rows = m
    if _sequence_padding and n == 6656 and m >= 512:
        if _sequence_alignment < 2 or _sequence_alignment % 2:
            raise ValueError("Muse sequence alignment must be a positive even number")
        rows = ((m + _sequence_alignment - 1) // _sequence_alignment) * _sequence_alignment
    storage = torch.empty((rows, n), dtype=torch.bfloat16, device=q.device)
    output = storage[:m] if rows != m else storage
    # Each call owns its outputs and scratch: no cross-layer or cross-stream alias.
    scratch = torch.empty(plan.scratch_bytes, dtype=torch.uint8, device=q.device)
    tensors = (q, weight, row_scale, column_scale, output, scratch)
    if _lib.ow_run(
        plan.ptr,
        torch.xpu.current_stream(q.device).sycl_queue,
        *[t.data_ptr() for t in tensors],
    ):
        raise RuntimeError(_lib.ow_error().decode())
    return output
