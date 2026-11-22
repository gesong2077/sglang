# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from sglang.kernels.ops.diffusion.rope import qwen_image21_b60 as b60


def b60_devices():
    return [
        torch.device(f"xpu:{i}")
        for i in range(torch.xpu.device_count())
        if "Arc(TM) Pro B60" in torch.xpu.get_device_name(i)
    ]


@pytest.mark.parametrize("device", b60_devices())
@pytest.mark.parametrize("row_padding", [0, 32])
@torch.inference_mode()
def test_bf16_silu_mul_all_inputs(device, row_padding):
    bits = torch.arange(65536, dtype=torch.int32, device=device)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 256, 256)
    value = torch.ones_like(x)
    expected = torch.nn.functional.silu(x) * value
    actual = b60.fused_pointwise(x, value, output_row_padding=row_padding)
    assert actual.stride(-2) == x.size(-1) + row_padding
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("device", b60_devices())
@pytest.mark.parametrize("tokens,heads", [(129, 15), (4096, 16)])
@torch.inference_mode()
def test_qk_rounding_and_gated_residual(device, tokens, heads):
    gen = torch.Generator(device=device).manual_seed(1983)
    x = torch.randn(
        1, tokens, heads, 128, device=device, dtype=torch.bfloat16, generator=gen
    )
    weight = torch.randn(128, device=device, dtype=x.dtype, generator=gen)
    angles = torch.randn(tokens, 64, device=device, generator=gen)
    rope = torch.polar(torch.ones_like(angles), angles)
    xf = x.float()
    inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
    norm = ((xf * inv).to(x.dtype) * weight).float()
    z = torch.view_as_complex(norm.unflatten(-1, (-1, 2)))
    expected = torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)
    assert b60.can_fuse_qk(x, weight, rope)
    assert torch.equal(b60.qk_norm_rope(x, weight, rope, 1e-6), expected)

    x = x.flatten(-2)
    update = torch.randn(x.shape, device=device, dtype=x.dtype, generator=gen)
    gate = torch.randn(1, 1, x.size(-1), device=device, dtype=x.dtype, generator=gen)
    assert b60.can_fuse_pair(x, update)
    assert torch.equal(b60.fused_pointwise(x, update, gate), x + update * gate)
