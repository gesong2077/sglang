"""Opt-in prefetched INT8 down GEMV with the original BF16 scale rounding."""
import ctypes as C
import os

import torch

_lib = C.CDLL(os.environ['SGLANG_XPU_W8A8_DP4A_SCALED_LIB'])
_lib.pfs_run.argtypes = [C.c_void_p] * 6 + [C.c_int] * 5
_lib.pfs_run.restype = C.c_int
_lib.pfs_error.restype = C.c_char_p


def down_gemv(q, weight, scales, weight_scale):
    if (
        q.shape != (1, 9984)
        or weight.shape != (9984, 6656)
        or q.dtype != torch.int8
        or weight.dtype != torch.int8
        or not q.is_contiguous()
        or weight.stride() != (1, 9984)
        or scales.numel() != 1
        or weight_scale.numel() != 6656
        or any(t.dtype != torch.float32 or not t.is_contiguous()
               for t in (scales, weight_scale))
        or q.device.type != 'xpu'
        or any(t.device != q.device for t in (weight, scales, weight_scale))
    ):
        return None
    result = torch.empty((1, 6656), device=q.device, dtype=torch.bfloat16)
    if _lib.pfs_run(
        torch.xpu.current_stream(q.device).sycl_queue,
        q.data_ptr(), weight.data_ptr(), result.data_ptr(),
        scales.data_ptr(), weight_scale.data_ptr(), 9984, 6656, 1, 16, 0,
    ):
        raise RuntimeError(_lib.pfs_error().decode())
    return result
