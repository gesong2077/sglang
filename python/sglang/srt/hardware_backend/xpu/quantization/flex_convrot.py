# SPDX-License-Identifier: Apache-2.0
"""Optional ACM-G10 ESIMD rotation for the three Qwen TP2 token buffers."""
import ctypes
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache
def _library():
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name("libflex_convrot.so")))
        entry = lib.flex_convrot_pack
    except (OSError, AttributeError):
        return None
    entry.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int]
    entry.restype = ctypes.c_int
    return entry


def pack_if_supported(x, values, scales, width, modulation=None, table=None):
    """Return the on-device row repair mask, or None for the original dispatch."""
    if (
        x.device.type != "xpu"
        or torch.is_grad_enabled()
        or torch.compiler.is_compiling()
    ):
        return None
    if torch.xpu.get_device_properties(x.device).device_id != 0x56C0:
        return None
    paired = width == 6144
    if width not in (2048, 4096, 6144):
        return None
    if x.shape != (1, 2048, 24576 if paired else width):
        return None
    if values.shape != ((2, 2048, width) if paired else (2048, width)):
        return None
    if scales.shape != ((2, 2048) if paired else (2048,)):
        return None
    if (modulation is not None) != (width == 4096) or (table is not None) != paired:
        return None
    auxiliary = table if paired else modulation
    tensors = [x, values, scales]
    if auxiliary is not None:
        if (
            auxiliary.numel() != (65536 if paired else width)
            or auxiliary.dtype != x.dtype
        ):
            return None
        tensors.append(auxiliary)
    if (
        x.dtype != torch.bfloat16
        or values.dtype != torch.int8
        or scales.dtype != torch.float32
        or any(
            t.device != x.device or not t.is_contiguous() or t.data_ptr() % 16
            for t in tensors
        )
    ):
        return None
    run = _library()
    if run is None:
        return None
    try:
        repair = torch.empty(scales.shape, device=x.device, dtype=torch.uint8)
    except torch.OutOfMemoryError:
        return None
    stream = torch.xpu.current_stream(x.device)
    if run(
        stream.sycl_queue,
        x.data_ptr(),
        auxiliary.data_ptr() if auxiliary is not None else None,
        values.data_ptr(),
        scales.data_ptr(),
        repair.data_ptr(),
        width,
    ):
        return None
    return repair
