# SPDX-License-Identifier: Apache-2.0
"""Intra-node, two-rank BF16 TP sums for the Flex 170 PCIe host-staging path."""

import os
import tempfile

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.multimodal_gen import envs

# Two rank-local slots. Larger and small messages use the process group.
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


class FlexTP2Sum:
    def __init__(self, staging, rank, group, device_group):
        self.staging = staging
        self.rank = rank
        self.group = group
        self.device_group = device_group
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

        # The inode check also covers containers with a separate /dev/shm.
        # Reserve space before mapping to handle tmpfs quotas without SIGBUS.
        location = [None]
        path = None
        if rank == 0:
            try:
                with tempfile.NamedTemporaryFile(dir="/dev/shm", delete=False) as file:
                    path = file.name
                    os.posix_fallocate(file.fileno(), 0, 2 * _MAX_BYTES)
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
                        name,
                        shared=True,
                        size=_MAX_BYTES,
                        dtype=torch.bfloat16,
                        device="cpu",
                    ).view(2, -1)
            except (OSError, RuntimeError):
                pass
            dist.all_gather_object(permissions, staging is not None, group=group)
            return (
                cls(staging, rank, group, coordinator.device_group)
                if all(permissions)
                else None
            )
        finally:
            if path is not None:
                os.unlink(path)

    def supports(self, x):
        return (
            x.device.type == "xpu"
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
