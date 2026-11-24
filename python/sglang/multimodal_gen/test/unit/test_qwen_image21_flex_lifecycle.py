# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
    QwenImage21Transformer2DModel,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image21_flex_tokens import (
    TokenShards,
)


@pytest.fixture
def shared(monkeypatch):
    model = QwenImage21Transformer2DModel.__new__(QwenImage21Transformer2DModel)
    nn.Module.__init__(model)
    block = nn.Module()
    block._layer_id = 0
    block.attn = nn.Module()
    block.img_mlp = nn.Module()
    block.attn.pack_qkv_weights = Mock()
    block.img_mlp.pack_gate_up_weights = Mock()
    block.img_mlp.pad_b60_down_weight = Mock()
    model.transformer_blocks = nn.ModuleList([block])
    weight = torch.arange(64, dtype=torch.int8).view(2, 4, 8)
    scale = torch.arange(8, dtype=torch.float32).view(2, 4)
    block.weight = nn.Parameter(weight[0], requires_grad=False)
    block.scale = nn.Parameter(scale[0], requires_grad=False)
    state = TokenShards.__new__(TokenShards)
    state.slabs = {0: ((weight, scale),)}
    state.weights = {}
    state.stamp = ("original",)
    state.ready = object()
    state.stream = Mock(device=torch.device("cpu"))
    state.reader_streams = {Mock()}
    model._flex_token_shards = state
    consumer = Mock()
    monkeypatch.setattr(torch.xpu, "current_stream", lambda _: consumer)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    return model, block, state, consumer


@pytest.mark.parametrize(
    "target", [torch.device("cpu"), torch.float64, torch.device("meta")]
)
def test_apply_retires_only_storage_still_shared(shared, target):
    model, block, state, consumer = shared
    readers = state.reader_streams | {state.stream}
    model.to(target)
    for reader in readers:
        consumer.wait_stream.assert_any_call(reader)
    consumer.wait_event.assert_any_call(state.ready)
    assert not hasattr(model, "_flex_token_shards") and not state.slabs
    assert state.stamp is None
    assert block.weight.dtype == torch.int8
    if target != torch.device("meta"):
        assert torch.equal(block.weight, torch.arange(32, dtype=torch.int8).view(4, 8))
        assert block.weight.untyped_storage().nbytes() == block.weight.nbytes
        assert block.scale.untyped_storage().nbytes() == block.scale.nbytes
    else:
        assert block.weight.is_meta and block.scale.is_meta


def test_post_load_observes_stream_and_preserves_changed_values(shared):
    model, block, state, consumer = shared

    def loaded():
        assert state.stamp is None
        consumer.wait_event.assert_called_with(state.ready)
        block.weight.add_(5)
        block.scale = nn.Parameter(torch.ones(4), requires_grad=False)

    block.attn.pack_qkv_weights.side_effect = loaded
    model.post_load_weights()
    assert not hasattr(model, "_flex_token_shards") and not state.slabs
    assert torch.equal(block.weight, torch.arange(32, dtype=torch.int8).view(4, 8) + 5)
    assert torch.equal(block.scale, torch.ones(4))
    assert block.weight.untyped_storage().nbytes() == block.weight.nbytes


def test_cleanup_failure_keeps_invalid_state_for_retry(shared, monkeypatch):
    model, block, state, _ = shared
    with monkeypatch.context() as local:

        def fail(*args):
            raise torch.OutOfMemoryError("injected")

        local.setattr(torch.Tensor, "clone", fail)
        model.to("cpu")
    assert state.stamp is None and model._flex_token_shards is state and state.slabs
    model.to("cpu")
    assert not hasattr(model, "_flex_token_shards") and not state.slabs
    assert block.weight.untyped_storage().nbytes() == block.weight.nbytes


def test_failed_apply_retires_partial_migration(shared):
    model, block, state, _ = shared

    def apply(t):
        if t.dtype == torch.float32:
            raise ValueError("injected partial transform")
        return t.clone()

    with pytest.raises(ValueError, match="injected partial transform"):
        model._apply(apply)
    assert not hasattr(model, "_flex_token_shards") and not state.slabs
    assert block.weight.untyped_storage().nbytes() == block.weight.nbytes
    assert block.scale.untyped_storage().nbytes() == block.scale.nbytes
