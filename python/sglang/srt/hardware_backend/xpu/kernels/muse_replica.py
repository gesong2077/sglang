"""Trade resident peer MLP weights for fewer TP2 prefill collectives.

Both original TP partials retain their own per-token activation scales and BF16
rounding. Only their token rows move to sequence parallel execution; the two
partial outputs are summed locally before the existing post-FFN normalization.
"""

import logging

import torch
import torch.distributed as dist

from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import (
    W8A8Activation,
    dynamic_w8a8_linear,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def install_peer_mlp_shards(model, count):
    parallel = get_parallel()
    layers = model.model.layers
    if parallel.tp_size != 2 or parallel.attn_tp_size != 2:
        raise ValueError("Muse MLP replicas require TP2 and attention TP2")
    if count < 1 or count > len(layers):
        raise ValueError("Replica layer count is outside the model")
    tp = parallel.tp_group
    rank = tp.rank_in_group
    group = tp.device_group
    allocated = 0
    for layer in layers[:count]:
        mlp = layer.mlp
        if not mlp.fused_xpu_mlp:
            raise ValueError("MLP replicas require dynamic symmetric INT8 W8A8")
        for name, value, shape in [
            ("gate_up_weight", mlp.gate_up_proj.weight, (19968, 6656)),
            ("gate_up_scale", mlp.gate_up_proj.weight_scale, (19968,)),
            ("down_weight", mlp.down_proj.weight, (6656, 9984)),
            ("down_scale", mlp.down_proj.weight_scale, (6656,)),
        ]:
            if value.device.type != "xpu":
                raise ValueError("MLP replica weights must be resident on XPU")
            is_weight = name.endswith("weight")
            if is_weight:
                if value.dtype != torch.int8:
                    raise ValueError("MLP replica weights must be INT8")
                # Normal load hooks run before quantization transposes weights;
                # accept already-transposed post-load storage as well.
                raw = value if tuple(value.shape) == shape else value.t()
                if tuple(raw.shape) != shape or not raw.is_contiguous():
                    raise ValueError("Unexpected Muse MLP weight layout")
            else:
                raw = value.flatten().float().contiguous()
                if tuple(raw.shape) != shape:
                    raise ValueError("Unexpected Muse MLP channel scales")
            peer = torch.empty_like(raw)
            for source in (0, 1):
                buffer = raw if rank == source else peer
                transport = buffer.view(torch.int32) if is_weight else buffer
                dist.broadcast(
                    transport, src=dist.get_global_rank(group, source), group=group
                )
            storage = peer.t() if is_weight else peer
            mlp.register_buffer("_xpu_peer_" + name, storage, persistent=False)
            allocated += storage.numel() * storage.element_size()
        mlp._xpu_replica_rank = rank
        mlp._xpu_has_replica = True
    logger.info(
        "Muse TP2 MLP replicas: %d layers, %.3f GiB additional weights per GPU",
        count, allocated / 2**30,
    )


def local_replicated_mlp(mlp, x, *, return_partials=False):
    if not isinstance(x, W8A8Activation):
        raise TypeError("Sequence-local replicated MLP expects prequantized rows")
    own = (
        mlp.gate_up_proj.weight, mlp.gate_up_proj.weight_scale,
        mlp.down_proj.weight, mlp.down_proj.weight_scale,
    )
    peer = (
        mlp._xpu_peer_gate_up_weight, mlp._xpu_peer_gate_up_scale,
        mlp._xpu_peer_down_weight, mlp._xpu_peer_down_scale,
    )
    parts = (own, peer) if mlp._xpu_replica_rank == 0 else (peer, own)
    outputs = []
    for gate_weight, gate_scale, down_weight, down_scale in parts:
        z = dynamic_w8a8_linear(x, gate_weight, gate_scale, silu_mul=True)
        outputs.append(dynamic_w8a8_linear(z, down_weight, down_scale))
    if return_partials:
        return outputs[0], outputs[1]
    return outputs[0] + outputs[1]
