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


def flex_devices():
    return [
        torch.device(f"xpu:{i}")
        for i in range(torch.xpu.device_count())
        if torch.xpu.get_device_properties(i).device_id == 0x56C0
    ]


@pytest.mark.parametrize("device", b60_devices() + flex_devices())
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

    # The grouped TP2 GEMM stores gate and up side-by-side in each row.
    gate, up = torch.cat((x, value), dim=-1).chunk(2, dim=-1)
    interleaved = b60.fused_pointwise(
        gate, up, input_row_padding=gate.shape[-1], output_row_padding=row_padding
    )
    torch.testing.assert_close(interleaved, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "device,tokens,heads",
    [(dev, 129, 15) for dev in b60_devices()]
    + [(dev, 4096, 16) for dev in b60_devices() + flex_devices()],
)
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


@pytest.mark.parametrize("device", flex_devices())
@torch.inference_mode()
def test_flex_swiglu_convrot_pack(device):
    from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import (
        rotate_and_pack,
    )

    generator = torch.Generator(device=device).manual_seed(369)
    shape = (1, 4096, 6144)
    gate, value = [
        torch.randn(shape, dtype=torch.bfloat16, generator=generator, device=device)
        for _ in range(2)
    ]
    expected = rotate_and_pack(torch.nn.functional.silu(gate) * value, 256)
    actual = rotate_and_pack(
        gate, 256, swiglu_value=value, silu_table=b60.silu_table(device)
    )
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.scales, expected.scales)


@pytest.mark.parametrize("device", flex_devices())
@torch.no_grad()
def test_flex_group_uses_live_weights_and_preserves_child_forward(device):
    from types import SimpleNamespace
    from unittest.mock import patch

    from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear
    from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
        _FlexTP2ColumnMethod,
    )

    group = SimpleNamespace(world_size=2, rank_in_group=0)
    with torch.device(device):
        layers = [
            ColumnParallelLinear(
                4096, 12288, bias=False, params_dtype=torch.bfloat16, tp_group=group
            ).eval()
            for _ in range(2)
        ]
        x = torch.empty((1, 4096, 4096), dtype=torch.bfloat16)
    _FlexTP2ColumnMethod.prepare(layers)
    shared = _FlexTP2ColumnMethod.for_image(x, layers)
    assert shared is not None
    layers[1].weight[0, 0].fill_(9)
    assert shared[0, 6144].item() == 9
    assert _FlexTP2ColumnMethod.for_image(x[:, :32], layers) is None

    with layers[0].register_forward_pre_hook(lambda *args: None):
        assert _FlexTP2ColumnMethod.for_image(x, layers) is None
    with torch.nn.modules.module.register_module_forward_hook(lambda *args: None):
        assert _FlexTP2ColumnMethod.for_image(x, layers) is None
    layers[0].train()
    assert _FlexTP2ColumnMethod.for_image(x, layers) is None
    layers[0].eval()
    with torch.enable_grad():
        assert _FlexTP2ColumnMethod.for_image(x, layers) is None
    with patch.object(torch.compiler, "is_compiling", return_value=True):
        assert _FlexTP2ColumnMethod.for_image(x, layers) is None

    # Replacing a parameter can break sharing after loading an adapter.
    # Re-read the current storage rather than serving stale packed weights.
    layers[1].weight = torch.nn.Parameter(layers[1].weight.detach().clone())
    assert _FlexTP2ColumnMethod.for_image(x, layers) is None
