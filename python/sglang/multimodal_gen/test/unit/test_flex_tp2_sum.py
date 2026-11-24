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


def test_peer_sum_preserves_bf16_bits(flex_tp2_group):
    """Compare the 32 MiB peer vector kernel against the original scalar kernel."""
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )

    group, device = flex_tp2_group
    if group.flex_tp2 is None or group.flex_tp2.registered is None:
        pytest.skip("requires the registered DMA bridge on both ranks")
    count = 16 * 1024 * 1024
    with set_forward_context(0, None, is_denoising=True):
        group.all_reduce(torch.ones(count, dtype=torch.bfloat16, device=device))
    registered = group.flex_tp2.registered
    if not registered.peer_ready:
        pytest.skip("peer DMA-BUF could not be established")

    # Cross every BF16 representation with zeros, subnormals, infinities,
    # different NaN payloads, and normal inputs around BF16 rounding ties.
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    partners = [0, 1, 0x7F, 0x80, 0x3F80, 0x3B80, 0x7F7F, 0x7F80, 0x7F81, 0x7FFF]
    partners += [word | 0x8000 for word in partners]
    partners += list(range(118)) + [word | 0x8000 for word in range(118)]
    if group.rank_in_group == 0:
        words = patterns.repeat(len(partners))
    else:
        words = (
            torch.tensor(partners, dtype=torch.int32)
            .to(torch.int16)
            .repeat_interleave(65536)
        )
    values = words.to(device).view(torch.bfloat16)
    for offset in (0, 1):
        storage = torch.empty(count + offset, dtype=torch.bfloat16, device=device)
        for truncate in (False, True):
            expected = values.clone()
            actual = storage[offset:]
            actual.copy_(values)
            registered.all_reduce(expected, truncate, peer=False)
            registered.all_reduce(actual, truncate, peer=True)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_flex_gated_peer_sum(flex_tp2_group):
    """Compare both epilogues bitwise, including stream switches and intervening sums."""
    from sglang.kernels.ops.diffusion.rope.qwen_image21_b60 import fused_pointwise
    from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context

    group, device = flex_tp2_group
    reducer = group.flex_tp2
    if reducer is None or reducer.registered is None:
        pytest.skip("requires the registered DMA bridge on both ranks")
    shape = (1, 4096, 4096)
    a = torch.randn(shape, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(a)
    gate = torch.randn(1, 1, 4096, dtype=a.dtype, device=device).tanh()
    with set_forward_context(0, None, is_denoising=True):
        group.all_reduce(a.clone())
        if not reducer.gated_residual_ready():
            pytest.skip("requires the gated peer epilogue on both ranks")

        x = torch.empty_like(a)
        scratch = torch.empty_like(a[0])
        initial = torch.xpu.current_stream(device)
        streams = [initial, torch.xpu.Stream(device), initial]
        torch.xpu.synchronize(device)
        register = reducer.registered
        for stream in streams:
            with torch.xpu.stream(stream):
                # Mix registered-host and direct peer generations. The native
                # copy must follow the GEMM and the writer on the caller stream.
                for truncate in (True, False):
                    reference = a.clone()
                    register.all_reduce(reference, truncate, peer=False)
                    expected = fused_pointwise(residual, reference, gate)
                    x.fill_(float("nan"))
                    torch.mm(a[0], a[0], out=scratch)
                    x.copy_(a)
                    register.all_reduce(
                        x, truncate, peer=True, gated_residual=(residual, gate)
                    )
                    assert torch.equal(x.view(torch.int16), expected.view(torch.int16))

        # Cross the payload, gate and residual NaN payloads, infinities, signed
        # zeros and subnormal boundary. Both BF16 operations must round separately.
        codes = torch.tensor(
            [0, 1, 0x7F, 0x80, 0x3F80, 0x7F7F, 0x7F80, 0x7F81, 0x7FC1, 0x7FFF],
            dtype=torch.int32, device=device,
        )
        codes = torch.cat([codes, codes | 0x8000]).to(torch.int16)
        n = codes.numel()
        rows = torch.arange(4096, device=device)
        cols = torch.arange(4096, device=device)
        local_code = rows // n if group.rank_in_group == 0 else rows
        a.view(torch.int16).copy_(codes[local_code % n].view(1, -1, 1))
        residual.view(torch.int16).copy_(codes[(cols // n) % n].view(1, 1, -1))
        gate.view(torch.int16).copy_(codes[cols % n].view(1, 1, -1))
        for truncate in (True, False):
            expected = a.clone()
            register.all_reduce(expected, truncate, peer=True)
            expected = fused_pointwise(residual, expected, gate)
            actual = a.clone()
            register.all_reduce(
                actual, truncate, peer=True, gated_residual=(residual, gate)
            )
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

        offset = torch.empty(a.numel() + 1, device=device, dtype=a.dtype)[1:].view(shape)
        assert reducer.try_gated_residual(offset, residual, gate) is None


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
