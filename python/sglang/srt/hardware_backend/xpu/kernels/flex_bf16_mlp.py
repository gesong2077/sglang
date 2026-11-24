# SPDX-License-Identifier: Apache-2.0
"""oneDNN's prepacked BF16 gate/up matrix for the Flex 170 Qwen image shard."""

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from sglang.kernels.ops.diffusion.sites.bitexact_gate import BitExactFusionGate
from sglang.multimodal_gen.runtime.managers.forward_context import (
    DENOISING_LOOP_STATE,
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
    lib.flex_mlp_destroy.argtypes = [ptr]
    lib.flex_mlp_destroy.restype = None
    for query in (lib.flex_mlp_packed_bytes, lib.flex_mlp_scratch_bytes):
        query.argtypes = [ptr]
        query.restype = ctypes.c_size_t
    lib.flex_mlp_pack.argtypes = [ptr] * 4
    lib.flex_mlp_run.argtypes = [ptr] * 6
    return lib


class _Plan:
    def __init__(self, device):
        self.lib = _library()
        self.handle = self.lib.flex_mlp_create(
            torch.xpu.current_stream(device).sycl_queue
        )
        if not self.handle:
            raise RuntimeError(self.lib.flex_mlp_error().decode())
        self.packed_bytes = self.lib.flex_mlp_packed_bytes(self.handle)
        self.scratch_bytes = max(1, self.lib.flex_mlp_scratch_bytes(self.handle))

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.flex_mlp_destroy(self.handle)


@lru_cache
def _plan(device):
    return _Plan(device)


@torch.library.custom_op(
    "sglang::flex_bf16_gate_up", mutates_args=(), device_types="xpu"
)
def flex_bf16_gate_up(x: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    plan = _plan(x.device)
    if (
        x.shape != (1, 4096, 4096)
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or packed.device != x.device
        or packed.dtype != torch.uint8
        or not packed.is_contiguous()
        or packed.numel() != plan.packed_bytes
    ):
        raise ValueError("invalid Flex gate/up activation or packed weight")
    stream = torch.xpu.current_stream(x.device)
    output = x.new_empty(1, 4096, 12288)
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


def _packed_weight(x, wide, gate, up):
    ctx = get_forward_context_or_none()
    if (
        ctx is None
        or not ctx.is_denoising
        or ctx.forward_batch is None
        or gate.is_inference()
        or up.is_inference()
    ):
        return None
    # A request owns the derived weights until its denoising stage finishes.
    # This leaves the original Parameters visible to hooks/loaders and frees
    # the cache before VAE decode, including after interrupted requests.
    state = ctx.forward_batch.extra.setdefault(DENOISING_LOOP_STATE, {})
    cache = state.setdefault("flex_bf16_gate_up", {})
    key = id(gate)
    stamp = (wide.data_ptr(), id(up), gate._version, up._version)
    entry = cache.get(key)
    stream = torch.xpu.current_stream(x.device)
    if entry is None or entry[0] != stamp:
        plan = _plan(x.device)
        free, _ = torch.xpu.mem_get_info(x.device)
        reusable = torch.xpu.memory_reserved(x.device) - torch.xpu.memory_allocated(
            x.device
        )
        # Allow room for a complete unquantized image block and the TP2 bridge.
        if free + reusable < plan.packed_bytes + (1536 << 20):
            return None
        packed = torch.empty(plan.packed_bytes, device=x.device, dtype=torch.uint8)
        if plan.lib.flex_mlp_pack(
            plan.handle, stream.sycl_queue, wide.data_ptr(), packed.data_ptr()
        ):
            raise RuntimeError(plan.lib.flex_mlp_error().decode())
        wide.record_stream(stream)
        ready = torch.xpu.Event()
        ready.record(stream)
        entry = cache[key] = (stamp, packed, stream, ready)
    _, packed, producer, ready = entry
    if producer != stream:
        stream.wait_event(ready)
    return packed


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
