# SPDX-License-Identifier: Apache-2.0
"""Load component-local projections with explicit serialized W8A8 targets."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
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


class _DiTConvRotMethod(ConvRotMethod):
    def __init__(self, group, fast_rotation=None):
        from sglang.multimodal_gen import envs

        super().__init__(group)
        self.fast_rotation = (
            envs.SGLANG_MOSS_XPU_FAST_ROTATION
            if fast_rotation is None
            else fast_rotation
        )

    def prepare_input(self, x):
        from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import (
            quantize_rotated,
            rotate_and_pack,
        )

        if self.fast_rotation and x.is_contiguous() and x.size(-1) % self.group == 0:
            return rotate_and_pack(x, self.group)
        return quantize_rotated(rotate(x.float(), self.group), x.dtype)

    def apply(self, layer, x, bias=None):
        if isinstance(x, torch.Tensor):
            if not _can_pack_dit_input(x) or bias is not None:
                return super().apply(layer, x, bias)
            x = self.prepare_input(x)
        from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import (
            project_packed,
        )

        return project_packed(x, layer.weight, layer.weight_scale)


def _can_pack_dit_input(x):
    return (
        x.device.type == "xpu"
        and x.dtype == torch.bfloat16
        and x.numel() > 0
        and x.shape[-1] <= 8192
    )


def prepare_convrot_input(layers, x, layer_type=ColumnParallelLinear):
    """Share a packed activation across plain projections within one forward."""
    if isinstance(x, torch.Tensor) and _can_pack_dit_input(x) and all(
        type(layer) is layer_type
        and isinstance(layer.quant_method, _DiTConvRotMethod)
        and layer.quant_method.group == layers[0].quant_method.group
        and layer.bias is None
        and not layer._forward_pre_hooks
        and not layer._forward_hooks
        for layer in layers
    ):
        return layers[0].quant_method.prepare_input(x)
    return x


def can_pack_flex_modulation(layers, device):
    return all(
        type(layer) is ColumnParallelLinear
        and type(layer.quant_method) is _DiTConvRotMethod
        and layer.quant_method.fast_rotation
        and layer.quant_method.group == 256
        and layer.tp_size == 2
        and not layer.gather_output
        and layer.weight.device == device
        and layer.bias is None
        and not layer.training
        and layer._compiled_call_impl is None
        and "forward" not in layer.__dict__
        and not layer._forward_pre_hooks
        and not layer._forward_hooks
        for layer in layers
    )


def can_pack_flex_mlp_output(layer, gate):
    # Only the measured TP2 shape has a tuned, bit-exact fused pack.
    return (
        gate.shape == (1, 4096, 6144)
        and torch.xpu.get_device_properties(gate.device).device_id == 0x56C0
        and type(layer) is RowParallelLinear
        and layer.input_is_parallel
        and layer.tp_size == 2
        and isinstance(layer.quant_method, _DiTConvRotMethod)
        and layer.quant_method.fast_rotation
        and layer.quant_method.group == 256
        and layer.bias is None
        and not layer._forward_pre_hooks
        and not layer._forward_hooks
    )


def prepare_moss_qkv_input(proj, x):
    return prepare_convrot_input((proj["q"], proj["k"], proj["v"]), x, _PackedLinear)


class SerializedConvRotConfig(CompressedInt8Config):
    def __init__(self, groups):
        if not groups or any(
            v not in (16, 64, 256) and not (v == 1024 and k.startswith("layers."))
            for k, v in groups.items()
        ):
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

    def supports_input_partition(self, prefix, partition_size):
        return partition_size % self.groups.get(prefix, 1) == 0

    def get_quant_method(self, layer, prefix):
        if prefix not in self.groups:
            return super().get_quant_method(layer, prefix)
        if prefix.startswith("transformer_blocks."):
            return _DiTConvRotMethod(self.groups[prefix], fast_rotation=True)
        if prefix.startswith("layers."):
            return _DiTConvRotMethod(self.groups[prefix])
        return ConvRotMethod(self.groups[prefix])


class _PackedLinear(ReplicatedLinear):
    def forward(self, x):
        return super().forward(x)[0]


class ConvRotEmbedding(nn.Module):
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


class FlexTP2ConvRotEmbedding(ConvRotEmbedding):
    """Keep each serialized vocabulary row on its owning Flex GPU."""

    def __init__(self, rows, columns, group, dtype, tp_group):
        if tp_group.world_size != 2 or rows % 2:
            raise ValueError(
                "Flex vocabulary sharding requires an even vocabulary and two ranks"
            )
        self.rows_per_rank = rows // 2
        super().__init__(self.rows_per_rank, columns, group, dtype)
        self.tp_group = tp_group
        self.weight.weight_loader = self._load_shard
        self.weight_scale.weight_loader = self._load_shard

    def _load_shard(self, param, weight):
        from sglang.multimodal_gen.runtime.loader.weight_utils import (
            default_weight_loader,
        )

        if weight.shape[0] == 2 * self.rows_per_rank:
            start = self.tp_group.rank_in_group * self.rows_per_rank
            weight = weight.narrow(0, start, self.rows_per_rank)
        default_weight_loader(param, weight)

    def forward(self, ids):
        if ids.numel() == 0:
            return self.weight.new_empty(
                (*ids.shape, self.weight.shape[1]), dtype=self.output_dtype
            )
        size = self.rows_per_rank
        # Preserve embedding's out-of-range check and every bit of the owning row.
        local_ids = torch.where(
            (ids >= 0) & (ids < 2 * size), ids.remainder(size), size
        )
        local = super().forward(local_ids)
        halves = self.tp_group.all_gather(local.unsqueeze(0).contiguous(), dim=0)
        return torch.where((ids >= size).unsqueeze(-1), halves[1], halves[0])


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
        if isinstance(config, SerializedConvRotConfig) and columns % config.groups[path]:
            raise ValueError(
                f"{path} input width {columns} must divide into rotation groups of {config.groups[path]}"
            )
        if isinstance(original, nn.Embedding) and isinstance(
            config, SerializedConvRotConfig
        ):
            replacement = ConvRotEmbedding(
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
