# SPDX-License-Identifier: Apache-2.0
"""oneDNN's prepacked BF16 gate/up matrix for the Flex 170 Qwen image shard."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from sglang.kernels.ops.diffusion.sites.bitexact_gate import BitExactFusionGate
from sglang.multimodal_gen.runtime.managers.forward_context import (
    DENOISING_LOOP_STATE,
    FLEX_MLP_LAYOUTS_USED,
    get_forward_context_or_none,
)


_GATE = BitExactFusionGate("Flex 170 prepacked BF16 gate/up", per_signature=True)


@lru_cache
def _library():
    lib = ctypes.CDLL(str(Path(__file__).with_name("libflex_bf16_mlp.so")))
    ptr = ctypes.c_void_p
    lib.flex_mlp_error.restype = ctypes.c_char_p
    lib.flex_mlp_create.argtypes = [ptr]
    lib.flex_mlp_create.restype = ptr
    lib.flex_projection_create.argtypes = [
        ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
    ]
    lib.flex_projection_create.restype = ptr
    lib.flex_mlp_unpack.argtypes = [ptr] * 4
    lib.flex_mlp_destroy.argtypes = [ptr]
    lib.flex_mlp_destroy.restype = None
    for query in (lib.flex_mlp_packed_bytes, lib.flex_mlp_scratch_bytes):
        query.argtypes = [ptr]
        query.restype = ctypes.c_size_t
    lib.flex_mlp_pack.argtypes = [ptr] * 4
    lib.flex_mlp_run.argtypes = [ptr] * 6
    return lib


class _Plan:
    def __init__(self, device, k, n):
        self.lib = _library()
        self.handle = self.lib.flex_projection_create(
            torch.xpu.current_stream(device).sycl_queue, k, n, k == 2048
        )
        if not self.handle:
            raise RuntimeError(self.lib.flex_mlp_error().decode())
        self.packed_bytes = self.lib.flex_mlp_packed_bytes(self.handle)
        self.scratch_bytes = max(1, self.lib.flex_mlp_scratch_bytes(self.handle))

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.flex_mlp_destroy(self.handle)


def _plan(device, k=4096, n=12288):
    return _geometry_plan(device, k, n)


@lru_cache
def _geometry_plan(device, k, n):
    return _Plan(device, k, n)


@torch.library.custom_op(
    "sglang::flex_bf16_gate_up", mutates_args=(), device_types="xpu"
)
def flex_bf16_gate_up(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    return _project(x, packed, 12288)


@torch.library.custom_op(
    "sglang::flex_bf16_projection", mutates_args=(), device_types="xpu"
)
def flex_bf16_projection(x: torch.Tensor, packed: torch.Tensor, n: int) -> torch.Tensor:
    return _project(x, packed, n)


@flex_bf16_projection.register_fake
def _fake_projection(x, packed, n):
    return x.new_empty(1, 4096, n)


def _project(x, packed, n):
    k = x.shape[-1]
    if (k, n) not in ((4096, 12288), (4096, 6144), (2048, 4096)):
        raise ValueError("unsupported Flex projection geometry")
    plan = _plan(x.device, k, n)
    if (
        x.shape != (1, 4096, k)
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or packed.device != x.device
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
        or packed.numel() != plan.packed_bytes
    ):
        raise ValueError("invalid Flex projection activation or packed weight")
    stream = torch.xpu.current_stream(x.device)
    output = x.new_empty(1, 4096, n)
    scratch = packed.new_empty(plan.scratch_bytes)
    status = plan.lib.flex_mlp_run(
        plan.handle,
        stream.sycl_queue,
        x.data_ptr(),
        packed.data_ptr(),
        output.data_ptr(),
        scratch.data_ptr(),
    )
    if status:
        raise RuntimeError(plan.lib.flex_mlp_error().decode())
    packed.record_stream(stream)
    x.record_stream(stream)
    return output


@flex_bf16_gate_up.register_fake
def _fake(x, packed):
    return x.new_empty(1, 4096, 12288)


def _cached_weight(source, parameters, name, size, materialize):
    ctx = get_forward_context_or_none()
    if (
        ctx is None
        or not ctx.is_denoising
        or ctx.forward_batch is None
        or any(w.is_inference() for w in parameters)
    ):
        return None
    # The request owns the derived weights for this denoising loop. Loaders and
    # hooks always see the canonical Parameters, including at other resolutions.
    state = ctx.forward_batch.extra.setdefault(DENOISING_LOOP_STATE, {})
    cache = state.setdefault(name, {})
    key = id(parameters[0])
    stamp = (source.data_ptr(), tuple((id(w), w._version) for w in parameters))
    entry = cache.get(key)
    device = source.device
    stream = torch.xpu.current_stream(device)
    if entry is None or entry[0] != stamp:
        free, _ = torch.xpu.mem_get_info(device)
        reusable = torch.xpu.memory_reserved(device) - torch.xpu.memory_allocated(
            device
        )
        # Allow room for a complete unquantized image block and the TP2 bridge.
        if free + reusable < size() + (1536 << 20):
            return None
        try:
            packed = materialize(stream)
        except torch.OutOfMemoryError:
            # Fragmented allocator blocks may not satisfy an otherwise fitting allocation.
            return None
        source.record_stream(stream)
        ready = torch.xpu.Event()
        ready.record(stream)
        entry = cache[key] = (stamp, packed, stream, ready)
        ctx.forward_batch.extra[FLEX_MLP_LAYOUTS_USED] = True
    _, packed, producer, ready = entry
    if producer != stream:
        stream.wait_event(ready)
        packed.record_stream(stream)
    return packed


def pack_weight(wide, k, n):
    canonical_shape = (n, k) if k == 2048 else (k, n)
    if (
        wide.shape != canonical_shape
        or wide.dtype != torch.bfloat16
        or not wide.is_contiguous()
    ):
        raise ValueError("invalid canonical Flex projection weight")
    plan = _plan(wide.device, k, n)
    packed = wide.new_empty(plan.packed_bytes, dtype=torch.uint8)
    stream = torch.xpu.current_stream(wide.device)
    if plan.lib.flex_mlp_pack(
        plan.handle, stream.sycl_queue, wide.data_ptr(), packed.data_ptr()
    ):
        raise RuntimeError(plan.lib.flex_mlp_error().decode())
    wide.record_stream(stream)
    return packed


def unpack_weight(packed, k, n):
    """Restore an opaque oneDNN projection as canonical contiguous BF16 [N, K]."""
    plan = _plan(packed.device, k, n)
    if packed.numel() != plan.packed_bytes or packed.dtype != torch.uint8:
        raise ValueError("invalid packed Flex projection weight")
    output = packed.new_empty((n, k), dtype=torch.bfloat16)
    stream = torch.xpu.current_stream(packed.device)
    if plan.lib.flex_mlp_unpack(
        plan.handle, stream.sycl_queue, packed.data_ptr(), output.data_ptr()
    ):
        raise RuntimeError(plan.lib.flex_mlp_error().decode())
    packed.record_stream(stream)
    return output


def unpack_gate_up(packed):
    return unpack_weight(packed, 4096, 12288)


def _packed_weight(x, wide, gate, up):
    return _cached_weight(
        wide,
        (gate, up),
        "flex_bf16_gate_up",
        lambda: _plan(x.device).packed_bytes,
        lambda stream: pack_weight(wide, 4096, 12288),
    )


def down_weight(weight):
    """Cache the transposed layout only for a measured 4096-token image."""
    if (
        weight.device.type != "xpu"
        or weight.dtype != torch.bfloat16
        or weight.shape != (4096, 6144)
        or not weight.is_contiguous()
    ):
        return None
    return _cached_weight(
        weight,
        (weight,),
        "flex_bf16_down",
        lambda: weight.numel() * weight.element_size(),
        lambda stream: weight.T.contiguous().T,
    )


def packed_gate_up(x, wide, gate, up, logger):
    if not _GATE.disabled:
        try:
            packed = _packed_weight(x, wide, gate, up)
        except (OSError, RuntimeError) as exc:
            _GATE.on_exception(exc, logger=logger)
        else:
            if packed is not None:
                return _GATE.run(
                    lambda: flex_bf16_gate_up(x, packed),
                    lambda: x @ wide,
                    sig=x.device,
                    logger=logger,
                )
    return x @ wide
