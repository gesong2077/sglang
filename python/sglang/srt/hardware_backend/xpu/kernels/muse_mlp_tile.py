"""Tile token-independent INT8 MLP work, retaining per-token/shard quantization."""
import torch
from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import (
    W8A8Activation, dynamic_w8a8_linear,
)


def tiled_mlp(x, w1, s1, w2, s2, tile):
    if not isinstance(x, W8A8Activation) or len(x.shape) != 2 or tile < 1:
        raise ValueError('Expected prequantized two-dimensional token rows')
    m, k = x.shape
    output = torch.empty((m, w2.shape[1]), device=x.values.device, dtype=x.dtype)
    for begin in range(0, m, tile):
        end = min(m, begin+tile)
        inp = W8A8Activation(x.values[begin:end], x.scales[begin:end], (end-begin,k), x.dtype)
        z = dynamic_w8a8_linear(inp, w1, s1, silu_mul=True)
        y = dynamic_w8a8_linear(z, w2, s2)
        output[begin:end].copy_(y)
        del z, y
    return output
