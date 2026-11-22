# SPDX-License-Identifier: Apache-2.0
"""Load component-local MOSS projections with explicit serialized W8A8 targets."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import LinearBase, ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.quantization.compressed_int8 import (
    CompressedInt8Config,
    DynamicInt8Method,
)


@lru_cache(maxsize=32)
def _hadamard(group, device, dtype):
    # Comfy's regular H4 and its Kronecker powers are symmetric and orthogonal.
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    matrix = h4
    while matrix.shape[0] < group:
        matrix = torch.kron(matrix, h4)
    return matrix / group**0.5


def rotate(x, group):
    return (x.reshape(-1, group) @ _hadamard(group, x.device, x.dtype)).reshape_as(x)


class ConvRotMethod(DynamicInt8Method):
    def __init__(self, group):
        self.group = group

    def apply(self, layer, x, bias=None):
        rotated = rotate(x.float(), self.group).reshape(-1, x.shape[-1])
        scale = rotated.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 127
        quantized = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
        accum = torch._int_mm(quantized, layer.weight.T)
        result = accum.float() * scale * layer.weight_scale.T
        if bias is not None:
            result = result + bias
        return result.view(*x.shape[:-1], layer.weight.shape[0]).to(x.dtype)


class SerializedConvRotConfig(CompressedInt8Config):
    def __init__(self, groups):
        if not groups or any(v not in (16, 64, 256) for v in groups.values()):
            raise ValueError(
                "Expected explicit Comfy ConvRot layer names and group sizes"
            )
        super().__init__(groups)
        self.groups = groups

    @classmethod
    def get_name(cls):
        return "convrot_int8"

    @classmethod
    def from_config(cls, config):
        return cls(config["layer_groups"])

    def get_quant_method(self, layer, prefix):
        return ConvRotMethod(self.groups[prefix])


class _PackedLinear(ReplicatedLinear):
    def forward(self, x):
        return super().forward(x)[0]


class _PackedEmbedding(nn.Module):
    def __init__(self, rows, columns, group, dtype):
        super().__init__()
        self.group = group
        self.output_dtype = dtype
        self.weight = nn.Parameter(
            torch.empty(rows, columns, dtype=torch.int8), requires_grad=False
        )
        self.weight_scale = nn.Parameter(
            torch.empty(rows, 1, dtype=torch.float32), requires_grad=False
        )

    def forward(self, ids):
        rows = F.embedding(ids, self.weight)
        scale = F.embedding(ids.to(self.weight_scale.device), self.weight_scale)
        scale = scale.to(rows.device)
        return rotate(rows.float() * scale, self.group).to(self.output_dtype)


def install_moss_projections(model, config, *, text_only=False):
    if not isinstance(config, CompressedInt8Config):
        raise NotImplementedError(
            f"Unsupported MOSS quantization: {type(config).__name__}"
        )
    for path in sorted(config.targets):
        if text_only and path.startswith("visual."):
            continue
        original = model.get_submodule(path)
        rows, columns = original.weight.shape
        if isinstance(original, nn.Embedding) and isinstance(
            config, SerializedConvRotConfig
        ):
            replacement = _PackedEmbedding(
                rows, columns, config.groups[path], original.weight.dtype
            )
        elif isinstance(original, (LinearBase, nn.Linear)):
            replacement = _PackedLinear(
                columns, rows, bias=original.bias is not None,
                quant_config=config, prefix=path, params_dtype=original.weight.dtype,
            )
        else:
            raise ValueError(f"Unsupported MOSS quantization target {path}")
        model.set_submodule(path, replacement)
