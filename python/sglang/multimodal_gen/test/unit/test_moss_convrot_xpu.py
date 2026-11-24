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
    from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import (
        quantize_rotated,
    )

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


@pytest.mark.parametrize("group", [16, 64, 256, 1024])
def test_butterfly_matches_regular_hadamard(group):
    from sglang.multimodal_gen.runtime.layers.quantization.moss_w8a8 import rotate
    from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import (
        quantize_rotated,
        rotate_and_pack,
    )

    torch.manual_seed(654)
    x = torch.randn(36, 3072, device="xpu", dtype=torch.bfloat16)
    x[0].zero_()
    expected = quantize_rotated(rotate(x.float(), group), x.dtype)
    actual = rotate_and_pack(x, group)
    torch.testing.assert_close(actual.scales, expected.scales, rtol=2e-6, atol=1e-10)
    assert (actual.values.short() - expected.values.short()).abs().max().item() <= 1


def test_xpu_norm_rope_matches_bf16_rounding():
    from sglang.kernels.ops.diffusion.norm.moss_xpu import norm_finish_if_supported

    torch.manual_seed(777)
    x = torch.randn(1, 2372, 3072, device="xpu", dtype=torch.bfloat16)
    weight = torch.randn_like(x[0, 0, :128])
    sin, cos = (torch.randn_like(x[..., :128]) for _ in range(2))
    heads = x.unflatten(-1, (24, 128))
    inv = torch.rsqrt(heads.float().square().mean(-1, keepdim=True) + 1e-6)
    normalized = (heads.float() * inv).bfloat16() * weight
    left, right = normalized.chunk(2, -1)
    expected = normalized * cos.unsqueeze(2) + torch.cat(
        (-right, left), -1
    ) * sin.unsqueeze(2)
    actual = norm_finish_if_supported(x, weight, sin, cos, 24, rope=True)
    assert torch.equal(actual, expected.flatten(2))


def test_modulated_convrot_pack_preserves_bf16_rounding():
    from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import (
        rotate_and_pack,
    )

    gen = torch.Generator(device="xpu").manual_seed(27)
    x = torch.randn(64, 4096, dtype=torch.bfloat16, device="xpu", generator=gen)
    scale = torch.randn(1, 4096, dtype=x.dtype, device=x.device, generator=gen)
    x[0].zero_()
    scale[0, ::2] = -1
    fused = rotate_and_pack(x, 256, modulation=scale)
    reference = rotate_and_pack(x * (1 + scale), 256)
    assert torch.equal(fused.values, reference.values)
    assert torch.equal(
        fused.scales.view(torch.int32), reference.scales.view(torch.int32)
    )


def test_flex_batched_gate_up_and_row_projection_match_both_tp_shards():
    from sglang.kernels.ops.diffusion.rope.qwen_image21_b60 import silu_table
    from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import (
        pack_flex_gate_up_pair,
        project_packed,
        rotate_and_pack,
    )
    from sglang.srt.hardware_backend.xpu.quantization.onednn_w8a8 import (
        flex_token_row_pair,
        supports_flex_token_pairs,
    )

    device = torch.device("xpu", torch.xpu.current_device())
    if torch.xpu.get_device_properties(
        device
    ).device_id != 0x56C0 or not supports_flex_token_pairs(device):
        pytest.skip("requires Flex batched oneDNN projections")
    generator = torch.Generator(device=device).manual_seed(6931)
    wide = torch.randn(
        (1, 2048, 24576), dtype=torch.bfloat16, device=device, generator=generator
    )
    wide[:, 0, :] = 0
    table = silu_table(device)
    codes, scales = pack_flex_gate_up_pair(wide, table)
    weights = torch.randint(
        -127, 128, (2, 4096, 6144), dtype=torch.int8, device=device, generator=generator
    )
    channels = torch.rand(
        (2, 4096), dtype=torch.float32, device=device, generator=generator
    )
    combined = flex_token_row_pair(codes, weights, scales, channels)
    parts = wide.chunk(4, dim=-1)
    for rank in range(2):
        tensor = rotate_and_pack(
            parts[2 * rank].contiguous(),
            256,
            swiglu_value=parts[2 * rank + 1].contiguous(),
            silu_table=table,
        )
        assert torch.equal(tensor.values, codes[rank])
        assert torch.equal(
            tensor.scales.view(torch.int32), scales[rank].view(torch.int32)
        )
        expected = project_packed(tensor, weights[rank], channels[rank])
        assert torch.equal(combined[rank], expected[0])


@pytest.mark.parametrize("width", [2048, 4096, 6144])
@torch.inference_mode()
def test_flex_mixed_rotation_repairs_exceptional_rows_bitwise(width):
    from sglang.kernels.ops.diffusion.rope.qwen_image21_b60 import silu_table
    from sglang.srt.hardware_backend.xpu.quantization.flex_convrot import (
        _library,
        pack_if_supported,
    )
    from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import (
        _rotate_pack_rows,
        pack_flex_token_rows,
    )

    dev = torch.device("xpu", torch.xpu.current_device())
    if torch.xpu.get_device_properties(dev).device_id != 0x56C0 or _library() is None:
        pytest.skip("requires the optional Flex 170 native rotation library")
    paired = width == 6144
    generator = torch.Generator(device=dev).manual_seed(7211)
    x = torch.randn(
        (1, 2048, 24576 if paired else width),
        device=dev,
        dtype=torch.bfloat16,
        generator=generator,
    )
    # Keep the named control row ordinary. Plant both signs and a single BF16
    # subnormal by bit pattern, so the fixture is independent of host FP mode.
    bits = [0x7FC1, 0x7F80, -128, 1, 0x5F80, 0x1F00, 0x7F7F]
    edge = torch.tensor(bits, device=dev, dtype=torch.int16).view(torch.bfloat16)
    x[0, : len(bits), 11] = edge
    x[0, len(bits)].zero_()
    impulse = len(bits) + 1
    x[0, impulse].zero_()
    x[0, impulse, 0] = 1
    if paired:
        x[0, impulse, width] = 1
        x[0, impulse, 2 * width] = 1
        x[0, impulse, 3 * width] = 1
    if paired:
        x[0, : len(bits), 2 * width + 17] = edge.flip([0])
    modulation = (
        torch.zeros((1, 1, width), device=dev, dtype=x.dtype) if width == 4096 else None
    )
    table = silu_table(dev) if paired else None
    shape = (2, 2048, width) if paired else (2048, width)
    codes = torch.empty(shape, device=dev, dtype=torch.int8)
    scales = torch.empty(shape[:-1], device=dev, dtype=torch.float32)
    mask = pack_if_supported(x, codes, scales, width, modulation, table)
    assert mask is not None
    assert mask.reshape(-1, 2048)[:, : len(bits)].all().item()
    assert not mask.reshape(-1, 2048)[:, len(bits)].any().item()
    ref_codes, ref_scales = torch.empty_like(codes), torch.empty_like(scales)
    grid = (2048, 2) if paired else (2048,)
    _rotate_pack_rows[grid](
        x,
        ref_codes,
        ref_scales,
        width,
        256,
        8192 if paired else width,
        SILU_TABLE=table,
        SWIGLU=paired,
        PAIRED_GATE_UP=paired,
        MODULATION=modulation,
        MODULATE=modulation is not None,
        num_warps=16 if paired else (4 if width == 2048 else 8),
    )
    pack_flex_token_rows(
        x, codes, scales, width, modulation=modulation, silu_table=table
    )
    assert torch.equal(codes, ref_codes)
    assert torch.equal(scales.view(torch.int32), ref_scales.view(torch.int32))
    # Independent CPU oracle for a sparse physical row. The regular H4
    # has the negative entry at input index (output index XOR 3).
    host = torch.zeros(width)
    host[0] = (
        torch.nn.functional.silu(torch.tensor(1, dtype=torch.bfloat16)).float()
        if paired
        else 1
    )
    for stride in (1, 4, 16, 64):
        a, b, c, d = host.view(-1, 4, stride).unbind(1)
        host = torch.stack(
            (
                (a + b) + (c - d),
                (a + b) + (d - c),
                (c + d) + (a - b),
                (c + d) + (b - a),
            ),
            1,
        ).reshape(-1)
    host *= 0.0625
    scale = host.abs().max() * (1 / 127)
    expected = (host / scale).round().to(torch.int8)
    for vc, sc in zip(codes.reshape(-1, 2048, width), scales.reshape(-1, 2048)):
        assert torch.equal(vc[impulse].cpu(), expected)
        assert torch.equal(sc[impulse].cpu().view(torch.int32), scale.view(torch.int32))
