# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image ConvRot on two Flex 170s, with one half of the image per rank."""

import torch
import torch.distributed as dist
from torch import nn

from sglang.kernels.ops.diffusion.rope import qwen_image21_b60 as b60
from sglang.multimodal_gen.runtime.cache.conditioning import conditioning_weights_epoch
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_tp_group
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.moss_w8a8 import (
    _DiTConvRotMethod,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    DENOISING_LOOP_STATE,
    FLEX_MLP_LAYOUTS_USED,
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import (
    _rotate_pack_rows,
    pack_flex_gate_up_pair,
    project_packed,
    rotate_and_pack,
)
from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import (
    W8A8Activation,
)

logger = init_logger(__name__)


def _eligible(blocks, device, group):
    from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
        QwenImage21Attention,
        QwenImage21FeedForward,
        QwenImage21TransformerBlock,
    )

    if (
        nn.modules.module._global_forward_hooks
        or nn.modules.module._global_forward_pre_hooks
    ):
        return False
    for block in blocks:
        if type(block) is not QwenImage21TransformerBlock:
            return False
        attn, mlp = block.attn, block.img_mlp
        if (
            type(attn) is not QwenImage21Attention
            or type(mlp) is not QwenImage21FeedForward
        ):
            return False
        if any(
            module.training
            or module._forward_hooks
            or module._forward_pre_hooks
            or module._compiled_call_impl is not None
            or "forward" in vars(module)
            for module in block.modules()
        ):
            return False
        layers = (
            (attn.to_q, ColumnParallelLinear, (2048, 4096)),
            (attn.to_k, ColumnParallelLinear, (2048, 4096)),
            (attn.to_v, ColumnParallelLinear, (2048, 4096)),
            (attn.to_out[0], RowParallelLinear, (4096, 2048)),
            (mlp.gate_layer, ColumnParallelLinear, (6144, 4096)),
            (mlp.proj, ColumnParallelLinear, (6144, 4096)),
            (mlp.out, RowParallelLinear, (4096, 6144)),
        )
        for layer, cls, shape in layers:
            if (
                type(layer) is not cls
                or type(layer.quant_method) is not _DiTConvRotMethod
            ):
                return False
            w = getattr(layer, "weight", None)
            s = getattr(layer, "weight_scale", None)
            if w is None or s is None:
                return False
            if not (
                layer.tp_group is group
                and layer.tp_size == 2
                and layer.bias is None
                and layer.quant_method.fast_rotation
                and layer.quant_method.group == 256
                and w.shape == shape
                and w.dtype == torch.int8
                and w.device == device
                and w.is_contiguous()
                and s.numel() == shape[0]
                and s.dtype == torch.float32
                and s.device == device
                and s.is_contiguous()
                and (
                    not layer.gather_output
                    if cls is ColumnParallelLinear
                    else layer.input_is_parallel and layer.reduce_results
                )
            ):
                return False
    return True


def _replicated_layers(block):
    return (
        block.attn.to_out[0],
        block.img_mlp.gate_layer,
        block.img_mlp.proj,
        block.img_mlp.out,
    )


def _weight_stamp(blocks):
    # Some loaders construct inference tensors without version counters. Their
    # peer copies remain valid only within one frozen denoising request.
    stamp = [conditioning_weights_epoch()]
    for block in blocks:
        for layer in _replicated_layers(block):
            for t in (layer.weight, layer.weight_scale):
                if t.is_inference():
                    return None
                stamp.append((id(t), t.data_ptr(), t._version, t.stride()))
    return tuple(stamp)


def _retire(model):
    state = getattr(model, "_flex_token_shards", None)
    if state is not None:
        state.stamp = None
        if not state.compact(model.transformer_blocks):
            return False
        model.__dict__.pop("_flex_token_shards", None)
    return True


def _all_ready(local, group):
    vote = torch.tensor(int(local), dtype=torch.int32, device="cpu")
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
    return bool(vote.item())


def _is_oom(error):
    text = str(error).lower().replace("_", " ")
    return (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in text
        or "out of device memory" in text
    )


def prepare_tokens(model, x, layouts, prefix_pad, sp):
    if (
        sp != 1
        or len(layouts) != 1
        or prefix_pad is not None
        or x.shape != (1, 4096, 4096)
        or x.device.type != "xpu"
        or x.dtype != torch.bfloat16
        or model.num_layers != 32
        or not b60.is_flex_device(x.device)
        or torch.compiler.is_compiling()
        or torch.is_grad_enabled()
    ):
        _retire(model)
        return None
    ctx = get_forward_context_or_none()
    if ctx is None or not ctx.is_denoising or ctx.forward_batch is None:
        return None
    request = ctx.forward_batch
    loop = request.extra.setdefault(DENOISING_LOOP_STATE, {})
    key = ("flex_convrot_token_shards", id(model))
    if key in loop:
        state = loop[key]
        if state is not None:
            state.on_stream(torch.xpu.current_stream(x.device))
        return state
    group = get_tp_group()
    bridge = getattr(group, "flex_tp2", None)
    if bridge is None:
        return None
    eligible = _eligible(model.transformer_blocks, x.device, group)
    stamp = _weight_stamp(model.transformer_blocks) if eligible else None
    previous = getattr(model, "_flex_token_shards", None)
    reuse = (
        previous is not None
        and stamp is not None
        and previous.stamp == stamp
        and previous.ready is not None
    )
    available = (
        torch.xpu.mem_get_info(x.device)[0]
        + torch.xpu.memory_reserved(x.device)
        - torch.xpu.memory_allocated(x.device)
    )
    needed = (2 << 30) if reuse else (5 << 30)
    eligible = eligible and available >= needed
    vote = torch.tensor(
        [int(eligible) * (2 if reuse else 1), int(stamp is not None)],
        dtype=torch.int32,
        device="cpu",
    )
    dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group.cpu_group)
    mode, persistent = vote.tolist()
    state = None
    if mode != 2:
        compacted = _retire(model)
        previous = None
        if mode and not _all_ready(compacted, group.cpu_group):
            loop[key] = None
            return None
    if mode and bridge.prepare_token_ops():
        if mode == 2:
            state = previous
            state.on_stream(torch.xpu.current_stream(x.device))
        else:
            state = TokenShards(
                bridge,
                model.num_layers,
                _weight_stamp(model.transformer_blocks) if persistent else None,
            )
            if not state.copy_replicas(
                model.transformer_blocks, group.cpu_group, x.device
            ):
                loop[key] = None
                return None
            if persistent:
                model._flex_token_shards = state
        request.extra[FLEX_MLP_LAYOUTS_USED] = True
        logger.info(
            "Flex TP2 rank %d owns 2048 ConvRot image tokens (peer weights %s)",
            bridge.rank,
            "reused" if mode == 2 else "copied",
        )
    loop[key] = state
    return state


def _half_view(raw, rows, width):
    split = rows * width
    return W8A8Activation(
        raw[:split].view(torch.int8).view(rows, width),
        raw[split:].view(torch.float32),
        (1, rows, width),
    )


def _pack_half(x):
    rows, width = x.shape[-2:]
    raw = torch.empty(rows * (width + 4), device=x.device, dtype=torch.uint8)
    packed = _half_view(raw, rows, width)
    _rotate_pack_rows[(rows,)](
        x, packed.values, packed.scales, width, 256, width, num_warps=4
    )
    return raw, packed


class TokenShards:
    def __init__(self, bridge, layers, stamp):
        self.bridge = bridge
        self.rank = bridge.rank
        self.last_layer = layers - 1
        self.weights = {}
        self.slabs = {}
        self.stamp = stamp
        self.ready = None
        self.stream = None
        self.truncate = bridge.round_toward_zero[4096 * 4096]

    def ordered(self, own, other):
        return (own, other) if self.rank == 0 else (other, own)

    def full_input(self, normalized, modulation):
        rows, width = normalized.shape[-2:]
        scale_bytes = rows * 4
        half_bytes = rows * width + scale_bytes
        raw = torch.empty(2 * half_bytes, device=normalized.device, dtype=torch.uint8)
        # Scales bracket both payloads. One peer DMA writes both codes and
        # scales directly to their final locations; the codes are contiguous.
        values = raw[scale_bytes:-scale_bytes].view(torch.int8).view(2 * rows, width)
        scales = (
            raw[:scale_bytes].view(torch.float32),
            raw[-scale_bytes:].view(torch.float32),
        )
        parts = raw.view(2, half_bytes)
        _rotate_pack_rows[(rows,)](
            normalized,
            values.narrow(0, self.rank * rows, rows),
            scales[self.rank],
            width,
            256,
            width,
            MODULATION=modulation,
            MODULATE=True,
            num_warps=8,
        )
        self.bridge.peer_exchange(parts[self.rank], out=parts[1 - self.rank])
        return W8A8Activation(values, torch.cat(scales), (1, 2 * rows, width))

    @staticmethod
    def slab_views(block, slabs, rank):
        out, gate, up, down = _replicated_layers(block)
        for layer, pair, slot in (
            (out, slabs[0], rank),
            (gate, slabs[1], 2 * rank),
            (up, slabs[1], 2 * rank + 1),
            (down, slabs[2], rank),
        ):
            for original, storage in zip((layer.weight, layer.weight_scale), pair):
                local = storage[slot]
                if local.numel() == original.numel():
                    local = local.view_as(original)
                yield original, local

    @torch.inference_mode(False)
    @torch.no_grad()
    def compact(self, blocks):
        # Local Parameters are views of the same allocations as peer weights.
        # Retire a block at a time so dropping the cache really gives memory back.
        for index, block in enumerate(blocks):
            key = getattr(block, "_layer_id", index)
            slabs = self.slabs.get(key)
            if slabs is None:
                continue
            storage_keys = {
                (t.device, t.untyped_storage().data_ptr())
                for pair in slabs
                for t in pair
            }
            # A loader may already have replaced or removed an entire layer.
            # Only copy current Parameters which still own part of the old slab.
            for param in block.parameters():
                if (
                    param.device,
                    param.untyped_storage().data_ptr(),
                ) not in storage_keys:
                    continue
                stream = torch.xpu.current_stream(param.device)
                if stream != self.stream:
                    stream.wait_event(self.ready)
                param.record_stream(stream)
                try:
                    param.data = param.clone()
                except torch.OutOfMemoryError:
                    return False
            del self.slabs[key]
        return True

    @torch.inference_mode(False)
    @torch.no_grad()
    def allocate_slabs(self, blocks):
        for block in blocks:
            out, gate, _, down = _replicated_layers(block)
            self.slabs[block._layer_id] = tuple(
                (
                    layer.weight.new_empty((count, *layer.weight.shape)),
                    layer.weight_scale.new_empty((count, layer.weight_scale.numel())),
                )
                for layer, count in ((out, 2), (gate, 4), (down, 2))
            )

    def copy_replicas(self, blocks, cpu_group, device):
        native = False
        if self.stamp is not None:
            try:
                from sglang.srt.hardware_backend.xpu.quantization.onednn_w8a8 import (
                    supports_flex_token_pairs,
                )

                native = supports_flex_token_pairs(device)
            except OSError:
                pass
        if _all_ready(native, cpu_group):
            try:
                self.allocate_slabs(blocks)
                allocated = True
            except torch.OutOfMemoryError:
                allocated = False
            if _all_ready(allocated, cpu_group):
                self.stream = torch.xpu.current_stream(device)
                for block in blocks:
                    slabs = self.slabs[block._layer_id]
                    for (source, local), (_, peer) in zip(
                        self.slab_views(block, slabs, self.rank),
                        self.slab_views(block, slabs, 1 - self.rank),
                    ):
                        source.record_stream(self.stream)
                        local.copy_(source)
                        self.bridge.peer_exchange(local, out=peer)
                self.ready = self.stream.record_event()
                for block in blocks:
                    for param, local in self.slab_views(
                        block, self.slabs[block._layer_id], self.rank
                    ):
                        param.data = local
                self.stamp = _weight_stamp(blocks)
                logger.info(
                    "Flex TP2 rank %d installed canonical interleaved INT8 weights for %d blocks",
                    self.rank,
                    len(self.slabs),
                )
                return True
            self.slabs.clear()
        return self.copy_separate_replicas(blocks, cpu_group, device)

    def copy_separate_replicas(self, blocks, cpu_group, device):
        # Allocate both ranks' destinations before the first weight collective.
        # An allocation failure on either card makes both ranks take the normal
        # tensor-parallel path without posting an unmatched peer exchange.
        try:
            for block in blocks:
                replicas = []
                for layer in _replicated_layers(block):
                    local = (layer.weight, layer.weight_scale)
                    peer = tuple(torch.empty_like(t) for t in local)
                    replicas.append(self.ordered(local, peer))
                self.weights[block._layer_id] = replicas
            reserved = True
        except torch.OutOfMemoryError:
            reserved = False
        vote = torch.tensor(int(reserved), dtype=torch.int32, device="cpu")
        dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=cpu_group)
        if not vote.item():
            self.weights.clear()
            return False
        for replicas in self.weights.values():
            for shards in replicas:
                for src, dst in zip(shards[self.rank], shards[1 - self.rank]):
                    self.bridge.peer_exchange(src, out=dst)
        self.stream = torch.xpu.current_stream(device)
        self.ready = self.stream.record_event()
        return True

    def on_stream(self, stream):
        if stream != self.stream:
            stream.wait_event(self.ready)
            for slabs in self.slabs.values():
                for weight, scale in slabs:
                    weight.record_stream(stream)
                    scale.record_stream(stream)
            for block in self.weights.values():
                for shards in block:
                    for weights in shards:
                        for tensor in weights:
                            tensor.record_stream(stream)

    def attention_output(self, attention, x, gate, weights):
        local, peer = attention.flatten(2).chunk(2, dim=1)
        local, peer = self.ordered(local, peer)
        send, _ = _pack_half(peer)
        _, owned = _pack_half(local)
        other = _half_view(self.bridge.peer_exchange(send), 2048, 2048)
        inputs = self.ordered(owned, other)
        first, second = (project_packed(inp, *w) for inp, w in zip(inputs, weights))
        return self.bridge.local_update(first, second, x, gate, self.truncate)

    @staticmethod
    def row_pair(values, weights, scales, channel_scales):
        from sglang.srt.hardware_backend.xpu.quantization.onednn_w8a8 import (
            flex_token_row_pair,
        )

        return flex_token_row_pair(values, weights, scales, channel_scales).unbind(0)

    def slab_attention_output(self, attention, x, gate, slab):
        local, peer = self.ordered(*attention.flatten(2).chunk(2, dim=1))
        rows, width = local.shape[-2:]
        sbytes = rows * 4
        half = rows * width + sbytes

        def views(raw, rank):
            if rank == 0:
                return raw[sbytes:].view(torch.int8).view(rows, width), raw[
                    :sbytes
                ].view(torch.float32)
            return raw[:-sbytes].view(torch.int8).view(rows, width), raw[-sbytes:].view(
                torch.float32
            )

        def pack(inp, codes, scales):
            with torch.xpu.device(inp.device):
                _rotate_pack_rows[(rows,)](
                    inp, codes, scales, width, 256, width, num_warps=4
                )

        send = torch.empty(half, device=x.device, dtype=torch.uint8)
        pack(peer, *views(send, self.rank))
        try:
            raw = torch.empty(2 * half, device=x.device, dtype=torch.uint8)
        except torch.OutOfMemoryError:
            raw = None
        if raw is not None:
            parts = raw.view(2, half)
            codes = raw[sbytes:-sbytes].view(torch.int8).view(2, rows, width)
            scales = (
                raw[:sbytes].view(torch.float32),
                raw[-sbytes:].view(torch.float32),
            )
            pack(local, codes[self.rank], scales[self.rank])
            self.bridge.peer_exchange(send, out=parts[1 - self.rank])
            inputs = tuple(zip(codes.unbind(0), scales))
            try:
                a, b = self.row_pair(codes, slab[0], torch.stack(scales), slab[1])
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
            else:
                return self.bridge.local_update(
                    a.view_as(x), b.view_as(x), x, gate, self.truncate
                )
        else:
            owned = torch.empty_like(send)
            own = views(owned, self.rank)
            pack(local, *own)
            received = self.bridge.peer_exchange(send)
            inputs = self.ordered(own, views(received, 1 - self.rank))
        # Both layouts used exactly one identical wire exchange. A single rank
        # can retry the local GEMMs without posting a second collective.
        a, b = (
            project_packed(W8A8Activation(v, s, (1, rows, width)), w, ws)
            for (v, s), w, ws in zip(inputs, *slab)
        )
        return self.bridge.local_update(
            a.view_as(x), b.view_as(x), x, gate, self.truncate
        )

    def slab_mlp(self, packed, columns, down):
        w, s = columns
        wide = project_packed(packed, w.flatten(0, 1), s.flatten())
        vals, scales = pack_flex_gate_up_pair(wide, b60.silu_table(w.device))
        return self.row_pair(vals, down[0], scales, down[1])

    def block_forward(
        self, block, images, mod, prefix_state, prefix_mod, layout, rope, cache
    ):
        from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
            apply_gated_residual,
            apply_modulation,
        )

        x = images.chunk(2, dim=1)[self.rank] if images.shape[1] == 4096 else images
        packed = self.full_input(block.img_norm1(x), mod[0])
        q, k, v = block.attn.project_qkv(packed)
        prefix = (
            apply_modulation(
                prefix_state["hidden_states"], block.img_norm1, prefix_mod[0]
            )
            if not cache
            else None
        )
        attention, prefix_out = block.attn.attend_sample(
            q, k, v, rope, prefix, layout, cache
        )
        slabs = self.slabs.get(block._layer_id)
        if slabs is None:
            out, gates, ups, downs = self.weights[block._layer_id]
            x = self.attention_output(attention, x, mod[1], out)
        else:
            out, columns, down = slabs
            x = self.slab_attention_output(attention, x, mod[1], out)
        packed = rotate_and_pack(block.img_norm2(x), 256, modulation=mod[2])
        partials = None
        if slabs is not None:
            try:
                partials = [
                    part.view_as(x) for part in self.slab_mlp(packed, columns, down)
                ]
            except RuntimeError as exc:
                if not _is_oom(exc):
                    raise
                # No collective runs inside the wide MLP. Retry from its input.
                weights, scales = columns
                gates = zip(weights[::2], scales[::2])
                ups = zip(weights[1::2], scales[1::2])
                downs = zip(*down)
        if partials is None:
            partials = []
            for gate, up, down in zip(gates, ups, downs):
                g = project_packed(packed, *gate)
                u = project_packed(packed, *up)
                hidden = rotate_and_pack(
                    g, 256, swiglu_value=u, silu_table=b60.silu_table(x.device)
                )
                partials.append(project_packed(hidden, *down))
        x = self.bridge.local_update(*partials, x, mod[3], self.truncate)
        if prefix_out is not None:
            prefix = apply_gated_residual(
                prefix_state["hidden_states"], prefix_out, prefix_mod[1]
            )
            prefix_state["hidden_states"] = apply_gated_residual(
                prefix,
                block.img_mlp(apply_modulation(prefix, block.img_norm2, prefix_mod[2])),
                prefix_mod[3],
            )
        if block._layer_id == self.last_layer:
            return torch.cat(self.ordered(x, self.bridge.peer_exchange(x)), dim=1)
        return x
