# SPDX-License-Identifier: Apache-2.0
"""Serialized per-channel W8A8 diffusion checkpoints with dynamic token scales."""

import torch
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.utils.weight_attrs import set_weight_attrs


class DynamicInt8Method(LinearMethodBase):
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ):
        rows = sum(output_partition_sizes)
        for name, columns, dtype, attrs in (
            (
                "weight",
                input_size_per_partition,
                torch.int8,
                {"input_dim": 1, "output_dim": 0},
            ),
            ("weight_scale", 1, torch.float32, {"output_dim": 0}),
        ):
            value = torch.nn.Parameter(
                torch.empty(rows, columns, dtype=dtype), requires_grad=False
            )
            set_weight_attrs(value, {**attrs, **extra_weight_attrs})
            layer.register_parameter(name, value)

    def apply(self, layer, x, bias=None):
        if x.device.type != "xpu":
            raise NotImplementedError(
                "This diffusion W8A8 implementation requires Intel XPU"
            )
        from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import (
            dynamic_w8a8_linear,
        )

        return dynamic_w8a8_linear(x, layer.weight.T, layer.weight_scale, bias)


class CompressedInt8Config(QuantizationConfig):
    checkpoint_uses_native_qkv_layout = True

    def __init__(self, targets):
        super().__init__()
        self.targets = frozenset(targets)

    @classmethod
    def get_name(cls):
        return "compressed-tensors"

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        groups = config.get("config_groups", {})
        if config.get("format") != "int-quantized" or len(groups) != 1:
            raise ValueError(
                "Expected a serialized int-quantized W8A8 checkpoint with one group"
            )
        group = next(iter(groups.values()))
        if (
            group.get("output_activations") is not None
            or group.get("format", config["format"]) != "int-quantized"
        ):
            raise ValueError("Expected int-quantized weights without quantized outputs")
        for field, strategy, dynamic in (
            ("weights", "channel", False),
            ("input_activations", "token", True),
        ):
            spec = group.get(field) or {}
            if (
                spec.get("type"),
                spec.get("num_bits"),
                spec.get("symmetric"),
                spec.get("strategy"),
                spec.get("dynamic"),
            ) != ("int", 8, True, strategy, dynamic):
                raise ValueError(f"Unsupported W8A8 {field} specification")
        targets = group["targets"]
        if (
            not targets
            or any("." not in name or name.startswith("re:") for name in targets)
            or config.get("ignore")
        ):
            raise ValueError(
                "This W8A8 adapter requires explicit checkpoint layer names"
            )
        return cls(targets)

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return (
                DynamicInt8Method()
                if prefix in self.targets
                else UnquantizedLinearMethod()
            )
        return None
