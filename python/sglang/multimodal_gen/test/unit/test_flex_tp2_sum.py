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


def test_registered_tp2_waits_for_delayed_producers():
    """Run under torchrun --standalone --nproc-per-node=2 -m pytest -k delayed_producers."""
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
        if group.flex_tp2 is None or group.flex_tp2.registered is None:
            pytest.skip("requires the registered DMA bridge on both ranks")
        x = torch.empty((4096, 4096), device=device, dtype=torch.bfloat16)
        a, b = torch.randn_like(x), torch.randn_like(x)
        scratch = torch.empty_like(x)
        reference = a.clone()
        dist.all_reduce(reference, group=group.device_group)
        # Initialize the rounding probe and the first generation. The faulty
        # wait-list barrier only appeared on later generations.
        group.all_reduce(a.clone())
        streams = [torch.xpu.current_stream(device), torch.xpu.Stream(device)]
        for stream in streams * 2:
            with torch.xpu.stream(stream):
                x.fill_(float("nan"))
                torch.xpu.synchronize(device)
                # Delay the current stream's actual producer. The copy of a
                # must complete before the bridge reads x on this rank.
                for _ in range(4):
                    torch.mm(a, b, out=scratch)
                x.copy_(a)
                out = group.all_reduce(x)
                assert torch.equal(out, reference)
    finally:
        group.destroy()
        if created:
            dist.destroy_process_group()
