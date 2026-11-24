# SPDX-License-Identifier: Apache-2.0
"""Weight lifecycle coverage, independent of access to two Flex devices."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.dits import qwen_image21_flex_bf16 as layouts


def layer():
    linear = nn.Linear(3, 2, bias=False)
    linear.weight.requires_grad_(False)
    linear.weight.weight_loader = object()
    linear.quant_method = layouts.HostLinear()
    return linear


def test_canonical_copy_keeps_checkpoint_and_loader():
    linear = layer()
    old = linear.weight
    with torch.inference_mode():
        host = layouts._host_parameter(old, old)
    assert not host.is_inference()
    assert host.weight_loader is old.weight_loader
    assert torch.equal(host, old) and host.data_ptr() != old.data_ptr()
    linear.weight = host
    assert torch.equal(linear.state_dict()["weight"], old)
    x = torch.rand(4, 3)
    assert torch.equal(
        linear.quant_method.apply(linear, x), nn.functional.linear(x, old)
    )


def test_cache_signature_covers_mutation_replacement_and_epoch(monkeypatch):
    linear = layer()
    stamp = layouts._stamp((linear,))
    linear.weight.add_(1)
    assert stamp != layouts._stamp((linear,))
    stamp = layouts._stamp((linear,))
    linear.weight = nn.Parameter(linear.weight.clone(), requires_grad=False)
    assert stamp != layouts._stamp((linear,))
    stamp = layouts._stamp((linear,))
    monkeypatch.setattr(layouts, "conditioning_weights_epoch", lambda: 43212)
    assert stamp != layouts._stamp((linear,))
    with torch.inference_mode():
        linear.weight = nn.Parameter(torch.ones(2, 3), requires_grad=False)
    assert layouts._stamp((linear,)) is None


def test_cross_stream_wait_and_stale_weight_bypass(monkeypatch):
    linear = layer()
    packed = torch.zeros(4, dtype=torch.uint8)
    producer, reader, ready = Mock(), Mock(), object()
    cache = layouts.PackedWeight(packed, layouts._stamp((linear,)), producer, ready)
    monkeypatch.setattr(torch.xpu, "current_stream", lambda _: reader)
    with patch.object(torch.Tensor, "record_stream", autospec=True) as use:
        assert cache.get((linear,), packed.device) is packed
        reader.wait_event.assert_called_once_with(ready)
        assert use.call_count == 1
        linear.weight.add_(1)
        assert cache.get((linear,), packed.device) is None


@pytest.mark.parametrize("target", [torch.device("cpu"), torch.device("meta")])
def test_retire_before_offload_uses_canonical_current_values(target):
    owner, linear = nn.Module(), layer()
    old = linear.weight
    linear.quant_method = layouts.PackedOutput()
    owner._flex_bf16_pack = SimpleNamespace(data=torch.empty(4))
    layouts.retire(owner, (linear,), fn=lambda t: t.to(target))
    assert linear.weight is old
    assert type(linear.quant_method) is layouts.UnquantizedLinearMethod
    assert not hasattr(owner, "_flex_bf16_pack")


def test_unsupported_hook_and_device_use_regular_module():
    linear = layer()
    with linear.register_forward_hook(lambda *args: None):
        assert not layouts._unobserved(linear)
    linear.eval()
    assert layouts._unobserved(linear)
    assert layouts._active(linear, torch.zeros(1)) is None


def test_short_prefix_uses_restored_weights_and_obeys_hooks(monkeypatch):
    class Column(nn.Module):
        def __init__(self):
            super().__init__()
            self.quant_method = layouts.HostLinear()
            self.eval()

    attn = nn.Module()
    attn.to_q, attn.to_k, attn.to_v = Column(), Column(), Column()
    attn.heads, attn.head_dim = 16, 128
    attn.eval()
    monkeypatch.setattr(layouts, "ColumnParallelLinear", Column)
    monkeypatch.setattr(layouts, "_active", lambda *_: {})
    handle = object()
    monkeypatch.setattr(layouts, "_available", lambda *_: handle)
    # Meta tensors exercise the real linear and reshape geometry without a GPU.
    restore = Mock(return_value=torch.empty(6144, 4096, device="meta"))
    monkeypatch.setattr(layouts.native, "unpack_weight", restore)
    x = torch.empty(1, 32, 4096, device="meta")
    projections = layouts.qkv(attn, x, None)
    assert all(v.shape == (1, 32, 16, 128) for v in projections)
    restore.assert_called_once_with(handle, 4096, 6144)
    with attn.to_q.register_forward_hook(lambda *_: None):
        assert layouts.qkv(attn, x, None) is None
    monkeypatch.setattr(layouts, "_available", lambda *_: None)
    assert layouts.qkv(attn, x, None) is None
    assert restore.call_count == 1
