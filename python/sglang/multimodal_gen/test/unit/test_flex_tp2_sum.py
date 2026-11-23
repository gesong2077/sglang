# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
def test_bf16_truncation_after_fp32_rounding():
    import triton
    from sglang.multimodal_gen.runtime.distributed.device_communicators.flex_tp2 import (
        _add_truncated,
    )

    # Cross exponent boundaries, including sums smaller than half an FP32 ULP.
    values = torch.arange(0x3F00, 0x4100, dtype=torch.int16).view(torch.bfloat16)
    smaller = torch.cat([values[:128].float() * 2**k for k in range(-32, 4)])
    a = values[:, None].expand(-1, smaller.numel()).contiguous()
    for sign in (1, -1):
        b = (smaller * sign).bfloat16().expand_as(a).contiguous()
        total = a.float() + b.float()
        expected = (total.view(torch.int32) >> 16).to(torch.int16).view(torch.bfloat16)
        out = a.to("xpu")
        _add_truncated[(triton.cdiv(a.numel(), 1024),)](
            out, b.to("xpu"), a.numel(), 1024
        )
        assert torch.equal(out.cpu(), expected)


@pytest.fixture(scope="module")
def flex_tp2_group():
    """Keep one Gloo/XCCL rendezvous for both transports in a torchrun process."""
    import os

    import torch.distributed as dist

    if os.environ.get("WORLD_SIZE") != "2" or torch.xpu.device_count() < 2:
        pytest.skip("requires two torchrun processes on Flex 170 cards")
    from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
        GroupCoordinator,
    )

    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("xpu", rank)
    if torch.xpu.get_device_properties(device).device_id != 0x56C0:
        pytest.skip("requires Flex 170")
    torch.xpu.set_device(device)
    torch.manual_seed(231 + rank)
    created = not dist.is_initialized()
    if created:
        dist.init_process_group("xccl")
    group = GroupCoordinator(
        [[0, 1]], rank, "xccl", use_device_communicator=False, group_name="tp_group"
    )
    try:
        yield group, device
    finally:
        group.destroy()
        if created:
            dist.destroy_process_group()


@pytest.mark.parametrize("peer", [False, True])
def test_registered_tp2_waits_for_delayed_producers(peer, flex_tp2_group):
    """Run under torchrun --standalone --nproc-per-node=2 -m pytest -k delayed_producers."""
    import torch.distributed as dist

    group, device = flex_tp2_group
    if group.flex_tp2 is None or group.flex_tp2.registered is None:
        pytest.skip("requires the registered DMA bridge on both ranks")
    x = torch.empty((4096, 4096), device=device, dtype=torch.bfloat16)
    a, b = torch.randn_like(x), torch.randn_like(x)
    scratch = torch.empty_like(x)
    reference = a.clone()
    dist.all_reduce(reference, group=group.device_group)
    # Initialize the rounding probe and the first generation. The faulty
    # wait-list barrier only appeared on later generations.
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    def reduce(value, use_peer=peer):
        with set_forward_context(0, None, is_denoising=use_peer):
            return group.all_reduce(value)

    reduce(a.clone())
    if peer and not group.flex_tp2.registered.peer_ready:
        pytest.skip(
            "driver or machine cannot establish and verify the peer inboxes"
        )
    streams = [torch.xpu.current_stream(device), torch.xpu.Stream(device)]
    for iteration, stream in enumerate(streams * 3):
        with torch.xpu.stream(stream):
            x.fill_(float("nan"))
            torch.xpu.synchronize(device)
            # Delay the current stream's actual producer. The copy of a
            # must complete before the bridge reads x on this rank.
            for _ in range(4):
                torch.mm(a, b, out=scratch)
            x.copy_(a)
            # Share the same generations across peer and host fallback paths.
            out = reduce(x, use_peer=peer and iteration % 3 != 1)
            assert torch.equal(out, reference)


def _peer_setup_worker(rank, rendezvous):
    """Exercise the real Unix descriptor exchange and asymmetric failures on CPU."""
    import ctypes
    import os
    from types import SimpleNamespace
    import torch.distributed as dist
    from sglang.multimodal_gen.runtime.distributed.device_communicators.flex_tp2 import (
        _RegisteredSum,
    )

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        for failure in ("export", "open", None):
            transport = _RegisteredSum.__new__(_RegisteredSum)
            transport.handle = 1
            transport.peer_ready = None
            transport.peer_fd = None
            fd = os.memfd_create("peer-test", os.MFD_CLOEXEC)
            os.write(fd, bytes([rank]))

            def export(_, output):
                ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = fd
                return int(failure == "export" and rank == 1)

            def open_remote(_, remote):
                assert os.pread(remote, 1, 0) == bytes([1 - rank])
                return int(failure == "open" and rank == 1)

            def no_op(*args):
                pass

            transport.library = SimpleNamespace(
                flex_tp2_peer_export=export,
                flex_tp2_peer_open=open_remote,
                flex_tp2_peer_sum=no_op,
                flex_tp2_sum=SimpleNamespace(argtypes=[]),
                flex_tp2_peer_close=no_op,
                flex_tp2_error=lambda: b"injected failure",
                flex_tp2_destroy=no_op,
            )
            ref = transport.connect_peer(rank, 0, dist.group.WORLD, torch.ones(1))
            assert (ref is not None) == (failure is None)
            if ref is not None:
                assert os.get_inheritable(transport.peer_fd) is False
                transport.close_peer()
            assert transport.peer_fd is None
            os.close(fd)
    finally:
        dist.destroy_process_group()


def test_flex_peer_dma_buf_setup_and_symmetric_fallback(tmp_path):
    from torch.multiprocessing import spawn

    spawn(_peer_setup_worker, args=(f"file://{tmp_path}/gloo",), nprocs=2)
