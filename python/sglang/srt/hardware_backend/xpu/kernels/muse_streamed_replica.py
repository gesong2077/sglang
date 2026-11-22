"""Experimental TP2 prefill peer weights via bounded Torch H2D double buffers.

Original BF16 partials and rank-order sum are retained. Copies use non-profiling
events: profiling timestamp commands were observed to serialize copy/compute.
No IPC implementation or new collective transport is introduced here.
"""
import logging

import torch
import torch.distributed as dist

from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import (
    W8A8Activation, dynamic_w8a8_linear,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


class StreamedMLP:
    def __init__(self, rank, device, hosts, scales):
        self.rank = rank
        self.device = device
        self.hosts = hosts
        self.scales = scales
        self.stream = torch.xpu.Stream(device=device)
        self.slots = [
            [torch.empty_like(x, device=device) for x in hosts[0]]
            for _ in range(2)
        ]
        self.ready = {}
        self.next_index = 0
        self.active = False

    def _enqueue(self, index, dependency=None):
        with torch.xpu.stream(self.stream):
            if dependency is not None:
                self.stream.wait_event(dependency)
            for target, source in zip(self.slots[index % 2], self.hosts[index]):
                target.copy_(source, non_blocking=True)
            ready = torch.xpu.Event()
            ready.record(self.stream)
        self.ready[index] = ready

    def begin(self):
        self.ready = {}
        self.next_index = 0
        self.active = True
        # Weights are immutable across requests. Previous forward's consumers
        # must finish before this request can overwrite either slot.
        previous = torch.xpu.current_stream(self.device).record_event()
        self._enqueue(0, previous)
        if len(self.hosts) > 1:
            self._enqueue(1)

    def run(self, index, mlp, x):
        if not self.active or index != self.next_index:
            raise RuntimeError('Streamed Muse layers must execute once in order')
        if not isinstance(x, W8A8Activation):
            raise TypeError('Streamed MLP requires existing quantized activation')
        current = torch.xpu.current_stream(self.device)
        current.wait_event(self.ready.pop(index))
        own = (mlp.gate_up_proj.weight, mlp.gate_up_proj.weight_scale,
               mlp.down_proj.weight, mlp.down_proj.weight_scale)
        weights = self.slots[index % 2]
        peer = (weights[0].t(), self.scales[index][0],
                weights[1].t(), self.scales[index][1])
        parts = (own, peer) if self.rank == 0 else (peer, own)
        outputs = []
        for gw, gs, dw, ds in parts:
            z = dynamic_w8a8_linear(x, gw, gs, silu_mul=True)
            outputs.append(dynamic_w8a8_linear(z, dw, ds))
        # The next copy into this slot waits for both weight consumers, while
        # the other slot's layer can compute independently on the current stream.
        consumed = current.record_event()
        if index + 2 < len(self.hosts):
            self._enqueue(index + 2, consumed)
        self.next_index += 1
        return outputs[0] + outputs[1]


def install_streamed_peer_mlp(model, start, count):
    parallel = get_parallel()
    layers = model.model.layers
    if parallel.tp_size != 2 or parallel.attn_tp_size != 2:
        raise ValueError('Streamed Muse MLP requires TP2 and attention TP2')
    if count < 1 or start < 0 or start + count > len(layers):
        raise ValueError('Streamed layer range is outside the model')
    tp = parallel.tp_group
    rank, group = tp.rank_in_group, tp.device_group
    hosts, scales = [], []
    selected = layers[start:start + count]
    for layer in selected:
        mlp = layer.mlp
        if not mlp.fused_xpu_mlp or getattr(mlp, '_xpu_has_replica', False):
            raise ValueError('Streamed layers must be unbiased W8A8 without resident replicas')
        host, scale = [], []
        for projection, shape in [(mlp.gate_up_proj, (19968, 6656)),
                                  (mlp.down_proj, (6656, 9984))]:
            value = projection.weight
            raw = value if tuple(value.shape) == shape else value.t()
            if raw.dtype != torch.int8 or not raw.is_contiguous() or tuple(raw.shape) != shape:
                raise ValueError('Unexpected Muse checkpoint weight layout')
            peer = torch.empty_like(raw)
            peer_scale = torch.empty_like(projection.weight_scale.flatten().float())
            own_scale = projection.weight_scale.flatten().float().contiguous()
            for source in (0, 1):
                buffer = raw if rank == source else peer
                dist.broadcast(buffer.view(torch.int32), src=dist.get_global_rank(group, source), group=group)
                buffer = own_scale if rank == source else peer_scale
                dist.broadcast(buffer, src=dist.get_global_rank(group, source), group=group)
            cpu = torch.empty_like(raw, device='cpu', pin_memory=True)
            cpu.copy_(peer, non_blocking=False)
            host.append(cpu)
            scale.append(peer_scale)
            del peer
        hosts.append(host)
        scales.append(scale)
    device = selected[0].mlp.gate_up_proj.weight.device
    state = StreamedMLP(rank, device, hosts, scales)
    for index, layer in enumerate(selected):
        layer.mlp._xpu_streamed_replica = (state, index)
    model.model._xpu_streamed_mlp = state
    logger.info('Muse streamed MLP: layers [%d,%d), %.3f GiB pinned host per rank; %.3f GiB device slots',
                start, start+count,
                sum(x.numel() for pair in hosts for x in pair)/2**30,
                sum(x.numel() for pair in state.slots for x in pair)/2**30)
    return state
