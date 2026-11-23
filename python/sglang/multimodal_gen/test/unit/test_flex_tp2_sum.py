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
        _add_truncated[(triton.cdiv(a.numel(), 1024),)](out, b.to("xpu"), a.numel(), 1024)
        assert torch.equal(out.cpu(), expected)
