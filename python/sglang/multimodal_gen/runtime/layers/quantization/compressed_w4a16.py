# SPDX-License-Identifier: Apache-2.0
"""Load packed, weight-only compressed-tensors INT4 linears on Intel XPU."""

import torch
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.srt.hardware_backend.xpu.quantization.int4pack_utils import (
    SUPPORTED_GROUP_SIZES,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    XPUCompressedTensorsWNA16,
)


class PackedInt4LinearMethod(CompressedTensorsLinearMethod):
    def create_weights(self, layer, **weight_attrs):
        # Diffusion's ordinary loader shards packed weights and group scales
        # by their declared dimensions and uses the encoder's local TP rank.
        weight_attrs["weight_loader"] = layer.weight_loader
        super().create_weights(layer, **weight_attrs)


class PackedInt4Config(QuantizationConfig):
    checkpoint_uses_native_qkv_layout = True

    def __init__(self, targets, group_size):
        super().__init__()
        self.targets = frozenset(targets)
        self.group_size = group_size

    @classmethod
    def get_name(cls):
        return "compressed-tensors"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        groups = config.get("config_groups", {})
        if config.get("format") != "pack-quantized" or len(groups) != 1:
            raise ValueError("Expected one packed INT4 weight-only group")
        group = next(iter(groups.values()))
        weights = group.get("weights") or {}
        if (
            group.get("input_activations") is not None
            or group.get("output_activations") is not None
            or group.get("format", "pack-quantized") != "pack-quantized"
            or weights.get("type") != "int"
            or weights.get("num_bits") != 4
            or weights.get("strategy") != "group"
            or weights.get("symmetric") is not True
            or weights.get("dynamic", False)
            or weights.get("actorder") is not None
            or weights.get("group_size") not in SUPPORTED_GROUP_SIZES
        ):
            raise ValueError("Unsupported XPU compressed-tensors W4A16 scheme")
        targets = group.get("targets", [])
        ignored = config.get("ignore") or []
        if (
            not targets
            or any("." not in name or name.startswith("re:") for name in targets)
            or any(name.startswith("re:") for name in ignored)
            or set(targets).intersection(ignored)
        ):
            raise ValueError("Expected explicit, component-local packed linear names")
        return cls(targets, weights["group_size"])

    def supports_input_partition(self, prefix, input_size_per_partition):
        return (
            prefix not in self.targets
            or input_size_per_partition % self.group_size == 0
        )

    def get_quant_method(self, layer, prefix):
        if not isinstance(layer, LinearBase):
            return None
        if prefix not in self.targets:
            return UnquantizedLinearMethod()
        layer.scheme = XPUCompressedTensorsWNA16("group", 4, self.group_size)
        return PackedInt4LinearMethod(self)
