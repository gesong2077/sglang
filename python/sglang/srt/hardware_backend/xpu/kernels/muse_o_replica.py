"""Move TP2 output-projection exchange before INT8 -> BF16 expansion.

Prefill only. oneCCL transports the existing quantized gate output and its exact
FP32 scales. Each rank evaluates both original o_proj shards on half the rows,
rounds each partial to BF16, then adds them in rank order just as TP2 reduction.
Decode continues to use the original tensor-parallel projection and XPU graph.
"""

import logging
import os

import torch
import torch.distributed as dist

from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import (
    W8A8Activation,
    dynamic_w8a8_linear,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)
_exchange = os.environ.get("SGLANG_XPU_MUSE_O_EXCHANGE", "gather")
if _exchange not in ("gather", "alltoall", "auto"):
    raise ValueError("Unsupported Muse output-projection exchange")


def install_peer_o_shards(model, count):
    parallel = get_parallel()
    layers = model.model.layers
    if parallel.tp_size != 2 or parallel.attn_tp_size != 2:
        raise ValueError("Muse output-projection replicas require TP2")
    if not 1 <= count <= len(layers):
        raise ValueError("Invalid output-projection replica count")
    group = parallel.tp_group.device_group
    rank = parallel.tp_group.rank_in_group
    allocated = 0
    for layer in layers[:count]:
        attn = layer.self_attn
        if not attn.fused_xpu_gate_quant:
            raise ValueError("Output-projection replicas need exact gate quantization")
        weight = attn.o_proj.weight
        raw = weight if tuple(weight.shape) == (6656, 2048) else weight.t()
        if raw.dtype != torch.int8 or not raw.is_contiguous() or tuple(raw.shape) != (6656, 2048):
            raise ValueError("Unexpected Muse output-projection weight layout")
        scale = attn.o_proj.weight_scale.flatten().float().contiguous()
        if scale.numel() != 6656 or raw.device.type != "xpu":
            raise ValueError("Output-projection weights must be resident on XPU")
        for name, value in [("weight", raw), ("scale", scale)]:
            peer = torch.empty_like(value)
            for source in (0, 1):
                transport = value if rank == source else peer
                if name == "weight":
                    transport = transport.view(torch.int32)
                dist.broadcast(transport, src=dist.get_global_rank(group, source), group=group)
            attn.register_buffer("_xpu_peer_o_" + name, peer.t() if name == "weight" else peer, persistent=False)
            allocated += peer.numel() * peer.element_size()
        attn._xpu_has_o_replica = True
    logger.info("Muse TP2 o_proj replicas: %d layers, %.3f GiB per GPU", count, allocated / 2**30)


def local_replicated_o_proj(attn, gated, group, rank):
    from .muse_o_pack import PackedGate, projection
    if isinstance(gated, PackedGate):
        return projection(attn, gated, group, rank)
    if not isinstance(gated, W8A8Activation):
        raise TypeError("Output-projection replicas require prequantized gate output")
    m, k = gated.values.shape
    if k != 2048 or rank not in (0, 1) or m < 512:
        raise ValueError("Unsupported output-projection replica shape")
    local_m = (m + 1) // 2
    begin, end = rank * local_m, min((rank + 1) * local_m, m)
    valid = end - begin
    alltoall = _exchange == "alltoall" or (_exchange == "auto" and m >= 2048)
    if alltoall:
        # Equal counts avoid the slower all-to-all-v path for odd token lengths.
        # The single padding row has a zero scale and is never exposed upstream.
        send_q, send_s = gated.values, gated.scales
        if m % 2:
            send_q = torch.empty((2 * local_m, k), dtype=torch.int8, device=gated.values.device)
            send_s = torch.empty(2 * local_m, dtype=torch.float32, device=gated.values.device)
            send_q[:m].copy_(gated.values)
            send_q[m:].zero_()
            send_s[:m].copy_(gated.scales)
            send_s[m:].zero_()
        full_q = torch.empty_like(send_q)
        full_s = torch.empty_like(send_s)
        dist.all_to_all_single(full_q.view(torch.int32), send_q.view(torch.int32), group=group)
        dist.all_to_all_single(full_s, send_s, group=group)
    else:
        full_q = torch.empty((2 * m, k), dtype=torch.int8, device=gated.values.device)
        full_s = torch.empty(2 * m, dtype=torch.float32, device=gated.values.device)
        # Packing is bit-preserving; all-gather does no INT32 arithmetic.
        dist.all_gather_single(full_q.view(torch.int32), gated.values.view(torch.int32), group=group)
        dist.all_gather_single(full_s, gated.scales, group=group)
    own = (attn.o_proj.weight, attn.o_proj.weight_scale)
    peer = (attn._xpu_peer_o_weight, attn._xpu_peer_o_scale)
    parts = (own, peer) if rank == 0 else (peer, own)
    outputs = []
    for source, (weight, scale) in enumerate(parts):
        start, stop = (source * local_m, source * local_m + valid) if alltoall else (source * m + begin, source * m + end)
        x = W8A8Activation(full_q[start:stop], full_s[start:stop], (end - begin, k))
        outputs.append(dynamic_w8a8_linear(x, weight, scale))
    local = outputs[0] + outputs[1]
    if end - begin != local_m:
        padded = torch.empty((local_m, 6656), dtype=local.dtype, device=local.device)
        padded[:end - begin].copy_(local)
        padded[end - begin:].zero_()
        local = padded
    return local
