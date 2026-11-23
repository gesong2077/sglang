# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.quantization.moss_w8a8 import (
    ConvRotMethod,
    SerializedConvRotConfig,
    _PackedLinear,
    prepare_moss_qkv_input,
)

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")


@pytest.mark.parametrize("rows", [1, 131])
def test_shared_activation_matches_independent_eager_projections(rows, monkeypatch):
    from sglang.srt.hardware_backend.xpu.quantization import moss_convrot

    torch.manual_seed(31)
    width, outputs = 768, 512
    groups = {f"layers.0.proj.{name}": 256 for name in "qkv"}
    config = SerializedConvRotConfig(groups)
    projections = nn.ModuleDict()
    for name in "qkv":
        layer = _PackedLinear(
            width,
            outputs,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=config,
            prefix=f"layers.0.proj.{name}",
        ).to("xpu")
        layer.weight.data.random_(-127, 128)
        layer.weight_scale.data.uniform_(1e-5, 0.02)
        projections[name] = layer
    x = torch.randn((rows, 2 * width), device="xpu", dtype=torch.bfloat16)[:, ::2]
    x[0].zero_()
    packed = prepare_moss_qkv_input(projections, x)
    assert packed.values.shape == x.shape
    eager = ConvRotMethod(256)
    for layer in projections.values():
        expected = eager.apply(layer, x)
        assert torch.equal(layer(packed), expected)
        with monkeypatch.context() as m:
            m.setattr(moss_convrot, "_native_matmul", lambda: None)
            assert torch.equal(layer(x), expected)
    handle = projections["k"].register_forward_pre_hook(lambda mod, args: None)
    try:
        assert prepare_moss_qkv_input(projections, x) is x
    finally:
        handle.remove()


def test_exact_pack_preserves_halfway_rounding_and_zero_rows():
    from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import quantize_rotated

    rotated = torch.tensor(
        [[127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], [0] * 7, [1] * 6 + [float("nan")]],
        device="xpu",
        dtype=torch.float32,
    )
    ref_scale = rotated.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
    expected = (rotated / ref_scale).round().clamp(-127, 127).to(torch.int8)
    packed = quantize_rotated(rotated, torch.bfloat16)
    assert torch.equal(packed.values, expected)
    torch.testing.assert_close(
        packed.scales, ref_scale.flatten(), rtol=0, atol=0, equal_nan=True
    )
