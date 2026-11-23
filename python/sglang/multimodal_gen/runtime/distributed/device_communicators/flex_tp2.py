# SPDX-License-Identifier: Apache-2.0
"""Intra-node, two-rank BF16 TP sums for the Flex 170 PCIe host-staging path."""

import ctypes
import logging
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.multimodal_gen import envs

logger = logging.getLogger(__name__)

# Larger and small messages use the process group.
_MAX_BYTES = 32 * 1024**2


@triton.jit
def _add_truncated(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(X + i, i < N, 0).to(tl.float32)
    b = tl.load(Y + i, i < N, 0).to(tl.float32)
    # oneCCL first adds in round-to-nearest FP32, then discards its low
    # 16 bits. A contracted add-to-BF16.rtz differs for small opposite signs.
    total = libdevice.add_rn(a, b)
    high = (total.to(tl.uint32, bitcast=True) >> 16).to(tl.uint16)
    tl.store(X + i, high.to(tl.bfloat16, bitcast=True), i < N)


def _shared_region(coordinator, num_bytes):
    """Return a shared, preallocated byte mapping only when both ranks mapped its inode."""
    group = coordinator.cpu_group
    location = [None]
    path = None
    if coordinator.rank_in_group == 0:
        try:
            with tempfile.NamedTemporaryFile(dir="/dev/shm", delete=False) as file:
                path = file.name
                os.posix_fallocate(file.fileno(), 0, num_bytes)
                stat = os.fstat(file.fileno())
                location[0] = (path, stat.st_dev, stat.st_ino)
        except OSError:
            pass
    try:
        dist.broadcast_object_list(location, src=coordinator.ranks[0], group=group)
        if location[0] is None:
            return None
        name, dev, ino = location[0]
        staging = None
        try:
            stat = os.stat(name)
            if (stat.st_dev, stat.st_ino) == (dev, ino):
                staging = torch.from_file(
                    name, shared=True, size=num_bytes, dtype=torch.uint8, device="cpu"
                )
        except (OSError, RuntimeError):
            pass
        mapped = [None, None]
        dist.all_gather_object(mapped, staging is not None, group=group)
        return staging if all(mapped) else None
    finally:
        if path is not None:
            os.unlink(path)


class _RegisteredSum:
    def __init__(self, staging, device, rank):
        from sglang.srt.hardware_backend import xpu

        default = Path(xpu.__file__).parent / "kernels" / "libflex_tp2_sum.so"
        self.library = ctypes.CDLL(os.getenv("SGLANG_XPU_FLEX_TP2_LIB", default))
        pointer = ctypes.c_void_p
        self.library.flex_tp2_create.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_size_t,
        ]
        self.library.flex_tp2_create.restype = pointer
        self.library.flex_tp2_error.restype = ctypes.c_char_p
        self.library.flex_tp2_sum.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.library.flex_tp2_destroy.argtypes = [pointer]
        self.library.flex_tp2_destroy.restype = None
        self.staging = staging
        self.scratch = torch.empty(_MAX_BYTES, dtype=torch.uint8, device=device)
        self.handle = self.library.flex_tp2_create(
            torch.xpu.current_stream(device).sycl_queue,
            staging.data_ptr(),
            self.scratch.data_ptr(),
            rank,
            _MAX_BYTES,
        )
        if not self.handle:
            raise RuntimeError(self.library.flex_tp2_error().decode())

    def all_reduce(self, x, truncate):
        stream = torch.xpu.current_stream(x.device)
        if self.library.flex_tp2_sum(
            self.handle, stream.sycl_queue, x.data_ptr(), x.numel(), truncate
        ):
            # A collective has started. Retrying locally could double-add or
            # leave the other rank waiting for the wrong generation.
            raise RuntimeError(self.library.flex_tp2_error().decode())
        x.record_stream(stream)
        return x

    def __del__(self):
        if getattr(self, "handle", None):
            self.library.flex_tp2_destroy(self.handle)


class FlexTP2Sum:
    def __init__(self, staging, coordinator, registered=None):
        self.registered = registered
        self.staging = staging
        self.rank = coordinator.rank_in_group
        self.group = coordinator.cpu_group
        self.device_group = coordinator.device_group
        self.device_index = torch.xpu.current_stream(coordinator.device).device_index
        self.round_toward_zero = {}

    @classmethod
    def try_create(cls, coordinator):
        group = coordinator.cpu_group
        rank = coordinator.rank_in_group
        props = torch.xpu.get_device_properties(coordinator.device)
        available = envs.SGLANG_DIFFUSION_FLEX_TP2_SHM and props.device_id == 0x56C0
        permissions = [None, None]
        dist.all_gather_object(permissions, available, group=group)
        if not all(permissions):
            return None

        # Four payload slots (two generations) plus an unregistered control page.
        staging = _shared_region(coordinator, 4 * _MAX_BYTES + 4096)
        if staging is not None:
            registered = None
            error = None
            try:
                registered = _RegisteredSum(staging, coordinator.device, rank)
            except (AttributeError, OSError, RuntimeError) as exc:
                error = str(exc)
            errors = [None, None]
            dist.all_gather_object(errors, error, group=group)
            if all(e is None for e in errors):
                if rank == 0:
                    logger.info("Flex TP2 uses registered host DMA")
                return cls(None, coordinator, registered)
            del registered
            if rank == 0:
                logger.info("Flex TP2 uses blocking host copies: %s", errors)
        else:
            # Smaller tmpfs quotas can still support the blocking implementation.
            staging = _shared_region(coordinator, 2 * _MAX_BYTES)
            if staging is None:
                return None
        rows = staging[: 2 * _MAX_BYTES].view(torch.bfloat16).view(2, -1)
        return cls(rows, coordinator)

    def supports(self, x):
        return (
            x.device.type == "xpu"
            and x.device.index == self.device_index
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and 2**20 <= x.nbytes <= _MAX_BYTES
        )

    def all_reduce(self, x):
        # oneCCL can either truncate or round BF16 depending on its selected
        # transport. Probe the actual size once so switching does not change it.
        if x.numel() not in self.round_toward_zero:
            probe = torch.full_like(x, 1.0 if self.rank == 0 else 0.005859375)
            dist.all_reduce(probe, group=self.device_group)
            self.round_toward_zero[x.numel()] = probe.view(-1)[0].item() == 1.0
        if self.registered is not None:
            return self.registered.all_reduce(x, self.round_toward_zero[x.numel()])
        rows = self.staging[:, : x.numel()]
        rows[self.rank].view_as(x).copy_(x)
        dist.barrier(group=self.group)
        peer = rows[1 - self.rank].view_as(x).to(x.device, non_blocking=False)
        # Both device copies must finish before the next collective reuses the slots.
        dist.barrier(group=self.group)
        if self.round_toward_zero[x.numel()]:
            with torch.xpu.device(x.device):
                _add_truncated[(triton.cdiv(x.numel(), 1024),)](
                    x, peer, x.numel(), 1024
                )
            return x
        return x.add_(peer)
