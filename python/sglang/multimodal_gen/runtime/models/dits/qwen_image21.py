# Copyright 2026 Qwen-Image Team and The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0

import math
from contextlib import contextmanager

import torch
from torch import nn

from sglang.kernels.ops.diffusion import (
    BitExactFusionGate,
    can_use_fused_layernorm_modulate,
    can_use_qknorm_complex_rope_cuda,
    fused_complex_rope,
    fused_layernorm_modulate,
    fused_silu_mul_bitexact,
    qknorm_complex_rope_cuda,
    qknorm_complex_rope_pack_,
    residual_gate_add,
    rmsnorm_preserve_reduction,
    tensors_equal,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_kv_triton import (
    qknorm_complex_rope_kv,
)
from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_triton import (
    qknorm_complex_rope,
)
from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.breakable_cuda_graph.replay_token import (
    get_current_replay_token,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_world_size,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_parallel_rank,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention, USPAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
    apply_unquantized_linear,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import BaseLayerWithLoRA
from sglang.multimodal_gen.runtime.layers.quantization.convrot_int8_comfy_kitchen import (
    ConvRotInt8ComfyKitchenLinearMethod,
    comfy_kitchen_swiglu_mlp,
)
from sglang.multimodal_gen.runtime.layers.quantization.moss_w8a8 import (
    _DiTConvRotMethod,
    can_pack_flex_mlp_output,
    can_pack_flex_modulation,
    prepare_convrot_input,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

_is_cuda = current_platform.is_cuda()
_is_xpu = current_platform.is_xpu()
flex_bf16 = None
if _is_xpu:
    from sglang.kernels.ops.diffusion.rope import qwen_image21_b60 as b60
    from sglang.multimodal_gen.runtime.models.dits import qwen_image21_flex_bf16 as flex_bf16
    from sglang.srt.hardware_backend.xpu.quantization.moss_convrot import (
        rotate_and_pack,
    )

_FLEX_MODULATION_PACK = BitExactFusionGate(
    "Flex 170 BF16 modulation + ConvRot INT8 pack", per_signature=True
)
_FLEX_SWIGLU_PACK = BitExactFusionGate(
    "Flex 170 BF16 SwiGLU + ConvRot INT8 pack", per_signature=True
)
_FLEX_GATE_UP = BitExactFusionGate(
    "Flex 170 BF16 gate/up and SwiGLU", per_signature=True
)
_FLEX_PADDED_DOWN = BitExactFusionGate(
    "Flex 170 padded BF16 MLP down", per_signature=True
)
_FLEX_PACKED_FA = BitExactFusionGate("Flex Qwen packed Flash Attention", per_signature=True)
_FLEX_WIDE_FA = BitExactFusionGate("Flex wide BF16 QKV packed attention", per_signature=True)
_FLEX_QKV = BitExactFusionGate("Flex 170 BF16 batch Q/K/V", per_signature=True)
_XPU_QK = BitExactFusionGate("XPU Q/K RMSNorm + RoPE", per_signature=True)
_B60_RESIDUAL = BitExactFusionGate("B60 BF16 gated residual")
_B60_SILU = BitExactFusionGate("B60 BF16 SiLU-mul")

logger = init_logger(__name__)
_ROPE_FUSION = BitExactFusionGate("Qwen-Image 2.1 complex RoPE")
_SILU_MUL_FUSION = BitExactFusionGate("Qwen-Image 2.1 SiLU-mul")
_QK_ROPE_FUSION = BitExactFusionGate("Qwen-Image 2.1 Q/K RMSNorm + complex RoPE")
_KV_ROPE_FUSION = BitExactFusionGate("Qwen-Image 2.1 K RMSNorm + RoPE + KV packing")
_QK_ROPE_CUDA_FUSION = BitExactFusionGate(
    "Qwen-Image 2.1 CUDA Q/K RMSNorm + complex RoPE"
)
_KV_PACK_CUDA_FUSION = BitExactFusionGate(
    "Qwen-Image 2.1 CUDA Q/K RMSNorm + RoPE + KV packing"
)
_KV_PROJECT_INTO_FUSION = BitExactFusionGate(
    "Qwen-Image 2.1 K/V projection written into the packed KV buffers"
)
_QKV_PACK_FUSION = BitExactFusionGate(
    "Qwen-Image 2.1 packed Q/K/V projection written into the packed QKV buffer"
)
_QK_NORM_FUSION = BitExactFusionGate("Qwen-Image 2.1 Q/K RMSNorm")
_MODULATION_FUSION = BitExactFusionGate("Qwen-Image 2.1 LayerNorm modulation")


def build_layout(image_slots, image_shapes, axes_dims, device):
    """Expand each condition-image slot to its complete latent grid before denoising."""
    indices, image_indices, positions, segments = [], [], [], []
    cursor = position = image_index = 0
    for text_index, is_image in enumerate(image_slots):
        if not is_image:
            indices.append(text_index)
            positions.append((position, position, position))
            position += 1
            continue
        start = len(indices)
        if start > cursor:
            segments.append((cursor, start, False))
        _, height, width = image_shapes[image_index]
        for h in range(-(height - height // 2), height // 2):
            for w in range(-(width - width // 2), width // 2):
                indices.append(text_index)
                image_indices.append(len(indices) - 1)
                positions.append((position, h, w))
        segments.append((start, len(indices), True))
        cursor = len(indices)
        position += max(height, width)
        image_index += 1
    if image_index != len(image_shapes) - 1:
        raise ValueError("condition-image slots do not match image_shapes")
    if len(indices) > cursor:
        segments.append((cursor, len(indices), False))
    prefix_len = len(indices)
    _, height, width = image_shapes[-1]
    for h in range(-(height - height // 2), height // 2):
        for w in range(-(width - width // 2), width // 2):
            positions.append((position, h, w))
    pos = torch.tensor(positions, device=device, dtype=torch.float32)
    angles = torch.cat(
        [
            pos[:, axis : axis + 1]
            * (10000.0 ** (-torch.arange(0, dim, 2, device=device).float() / dim))
            for axis, dim in enumerate(axes_dims)
        ],
        dim=-1,
    )
    rope = torch.polar(torch.ones_like(angles), angles)
    return dict(
        encoder_seq_len=len(image_slots),
        text_indices=torch.tensor(indices, device=device, dtype=torch.long),
        image_indices=torch.tensor(image_indices, device=device, dtype=torch.long),
        prefix_rope=rope[:prefix_len],
        target_rope=rope[prefix_len:],
        segments=tuple(segments),
    )


def apply_rope(x, rope):
    fused = None
    if _is_cuda and x.is_cuda and _ROPE_FUSION.can_attempt_once():
        fused = fused_complex_rope(x, rope)
        if _ROPE_FUSION.verified:
            return fused
    z = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)
    if fused is not None:
        return _ROPE_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


def apply_qk_norm(x, norm):
    fused = None
    if (
        _is_cuda
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and _QK_NORM_FUSION.can_attempt_once()
    ):
        fused = rmsnorm_preserve_reduction(x, norm.weight, norm.variance_epsilon)
        if _QK_NORM_FUSION.verified:
            return fused
    out = norm(x)
    if fused is not None:
        return _QK_NORM_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


def apply_qk_norm_rope(x, norm, rope):
    if _is_xpu and b60.can_fuse_qk(x, norm.weight, rope):
        return _XPU_QK.run(
            lambda: b60.qk_norm_rope(x, norm.weight, rope, norm.variance_epsilon),
            lambda: apply_rope(norm(x), rope),
            sig=(x.device, x.shape, norm.variance_epsilon),
            logger=logger,
        )
    fused = None
    if (
        _is_cuda
        and x.is_cuda
        and not torch.compiler.is_compiling()
        and can_use_qknorm_complex_rope_cuda(x.dtype, x.shape[-1])
        and _QK_ROPE_CUDA_FUSION.can_attempt_once()
    ):
        fused = qknorm_complex_rope_cuda(x, norm.weight, rope, norm.variance_epsilon)
        if _QK_ROPE_CUDA_FUSION.verified:
            return fused
        out = apply_rope(norm(x), rope)
        return _QK_ROPE_CUDA_FUSION.accept_or_fallback(fused, out, logger=logger)
    if (
        _is_cuda
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.shape[-1] == 128
        and _QK_ROPE_FUSION.can_attempt_once()
    ):
        fused = qknorm_complex_rope(x, norm.weight, rope, norm.variance_epsilon)
        if _QK_ROPE_FUSION.verified:
            return fused
    out = apply_rope(norm(x), rope)
    if fused is not None:
        return _QK_ROPE_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


def can_project_into(layer, x):
    """A plain bf16 ColumnParallelLinear whose forward is just F.linear on a resident weight."""
    return (
        type(layer) is ColumnParallelLinear
        and isinstance(layer.quant_method, UnquantizedLinearMethod)
        and layer.bias is None
        and not layer.gather_output
        and not layer._forward_pre_hooks
        and not layer._forward_hooks
        and layer.weight.is_cuda
        and layer.weight.dtype is torch.bfloat16
        and layer.weight.is_contiguous()
        and x.dtype is torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
        and not torch.compiler.is_compiling()
    )


def project_into(layer, x, out):
    """F.linear(x, layer.weight) written into ``out`` (contiguous rows of a larger buffer).

    A contiguous row slice has the same leading dimension as a fresh tensor, so
    cuBLAS receives the same call as the module forward would issue.
    """
    rows = x.shape[0] * x.shape[1]
    torch.mm(x.reshape(rows, x.shape[-1]), layer.weight.t(), out=out.view(rows, -1))
    return out


class PrefixPad:
    """Rows BCG left-padded onto every cached prefix K/V, read on the host once per forward or graph replay."""

    def __init__(self, rows):
        self.rows = rows
        self.token = self.value = None

    def get(self):
        token = get_current_replay_token()
        if self.value is None or token != self.token:
            self.value, self.token = int(self.rows.item()), token
        return self.value


@eager_on_graph
def attend_past_prefix_pad(attn, q, k, v, prefix_pad, out):
    # a graph break: the pad varies per request, and skipping it hands the
    # kernel exactly the eager K/V, so padded replays stay bit-exact. Calls the
    # impl like USPAttention does without SP, since its forward is a break too
    rows = prefix_pad.get()
    result = attn.attn_impl.forward(
        q, k[:, rows:], v[:, rows:], get_forward_context().attn_metadata
    )
    # copy_ would broadcast a mispredicted shape or cast a dtype silently.
    assert out.shape == result.shape and out.dtype == result.dtype, (
        "Qwen-Image 2.1 prefix-pad attention output mispredicted"
    )
    out.copy_(result)


def cat_outputs(outputs, dim=0):
    # torch.cat on a one-element list degrades to a full copy of the tensor.
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=dim)


def apply_gated_residual(x, update, gate):
    if (
        _is_xpu
        and b60.can_fuse_pair(x, update)
        and gate.shape == (1, 1, x.shape[-1])
        and gate.device == x.device
        and gate.dtype == x.dtype
        and gate.is_contiguous()
    ):
        return _B60_RESIDUAL.run(
            lambda: b60.fused_pointwise(x, update, gate),
            lambda: residual_gate_add(x, update, gate),
            logger=logger,
        )
    return residual_gate_add(x, update, gate)


def _flex_row_kwargs(module, residual, gate):
    """Only pass the epilogue through unwrapped inference modules without observers."""
    if (
        not _is_xpu
        or not _B60_RESIDUAL.verified
        or _B60_RESIDUAL.disabled
        or torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or residual.device.type != "xpu"
        or residual.shape != (1, 4096, 4096)
        or nn.modules.module._global_forward_pre_hooks
        or nn.modules.module._global_forward_hooks
    ):
        return {}
    if type(module) is QwenImage21Attention:
        layer = module.to_out[0]
    elif type(module) is QwenImage21FeedForward:
        layer = module.out
    else:
        return {}
    if type(layer) is not RowParallelLinear or not (
        layer.tp_size == 2 and layer.reduce_results and layer.bias is None
    ):
        return {}
    # Each verified method returns a fresh activation. A custom method may
    # return a scratch buffer and overwrite the residual on its next call.
    if type(layer.quant_method) not in (
        UnquantizedLinearMethod, _FlexTP2DownMethod, _DiTConvRotMethod
    ) and not (flex_bf16 is not None and type(layer.quant_method) is flex_bf16.PackedOutput):
        return {}
    if any(
        part.training
        or "forward" in part.__dict__
        or getattr(part, "_compiled_call_impl", None) is not None
        or part._forward_hooks
        or part._forward_pre_hooks
        for part in (module, layer)
    ):
        return {}
    flex = getattr(layer.tp_group, "flex_tp2", None)
    if flex is None or not flex.gated_residual_ready():
        return {}
    return {"gated_residual": (residual, gate, apply_gated_residual)}


def apply_modulation(x, norm, scale):
    fused = None
    if (
        x.is_cuda
        and can_use_fused_layernorm_modulate(x.dtype, x.shape[-1])
        and _MODULATION_FUSION.can_attempt_once()
    ):
        fused = fused_layernorm_modulate(x, scale.squeeze(1), None, norm.eps)
        if _MODULATION_FUSION.verified:
            return fused
    out = norm(x) * (1 + scale)
    if fused is not None:
        return _MODULATION_FUSION.accept_or_fallback(fused, out, logger=logger)
    return out


def modulate_image_input(x, norm, scale, module):
    """Pack the modulated image once for the input projections that share it."""
    if (
        _is_xpu
        and not torch.compiler.is_compiling()
        and not torch.is_grad_enabled()
        and type(module) in (QwenImage21Attention, QwenImage21FeedForward)
        and not module.training
        and x.device.type == "xpu"
        and x.shape == (1, 4096, 4096)
        and x.dtype == torch.bfloat16
        and scale.shape == (1, 1, 4096)
        and scale.is_contiguous()
        and scale.dtype == x.dtype
        and scale.device == x.device
        and b60.is_flex_device(x.device)
        and not _FLEX_MODULATION_PACK.disabled
        and module._compiled_call_impl is None
        and "forward" not in module.__dict__
        and not module._forward_pre_hooks
        and not module._forward_hooks
        and not nn.modules.module._global_forward_pre_hooks
        and not nn.modules.module._global_forward_hooks
    ):
        if type(module) is QwenImage21Attention:
            layers = (module.to_q, module.to_k, module.to_v)
        else:
            layers = (module.gate_layer, module.proj)
        if can_pack_flex_modulation(layers, x.device):
            hidden = norm(x)
            if (
                hidden.shape != x.shape
                or hidden.dtype != x.dtype
                or hidden.device != x.device
                or not hidden.is_contiguous()
            ):
                return hidden * (1 + scale)
            return _FLEX_MODULATION_PACK.run(
                lambda: rotate_and_pack(hidden, 256, modulation=scale),
                lambda: rotate_and_pack(hidden * (1 + scale), 256),
                sig=x.device,
                equal=lambda a, b: torch.equal(a.values, b.values)
                and torch.equal(a.scales, b.scales),
                logger=logger,
            )
    return apply_modulation(x, norm, scale)


class QwenImage21ZeroCenterRMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        scale = self.weight.float() + 1
        value = x.float()
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
            * scale
        ).to(x.dtype)


class QwenImage21TextProjection(nn.Module):
    def __init__(self, context_dim, dim, eps):
        super().__init__()
        self.text_norm = QwenImage21ZeroCenterRMSNorm(context_dim, eps)
        self.in_layer = nn.Linear(context_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(
            nn.functional.gelu(self.in_layer(self.text_norm(x)), approximate="tanh")
        )


class QwenImage21TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = nn.Module()
        self.timestep_embedder.linear_1 = nn.Linear(256, dim, bias=False)
        self.timestep_embedder.linear_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, t, dtype):
        freq = torch.exp(
            -math.log(10000) * torch.arange(128, device=t.device).float() / 128
        )
        angles = t.float()[:, None] * 1000 * freq
        x = torch.cat([angles.cos(), angles.sin()], dim=-1).to(dtype)
        return self.timestep_embedder.linear_2(
            nn.functional.silu(self.timestep_embedder.linear_1(x))
        )


def _stacked_rows(a, b):
    """[a; b] as one view when ``b`` directly follows ``a`` in the same storage, else None."""
    # Layerwise offload leaves a flat placeholder in place of each weight.
    if not (
        a.dim() == 2
        and a.is_cuda
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
        and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        and b.storage_offset() == a.storage_offset() + a.numel()
    ):
        return None
    return a.as_strided((2 * a.shape[0], a.shape[1]), (a.shape[1], 1))


class _FlexTP2ColumnMethod(UnquantizedLinearMethod):
    """Keep one copy of each weight in the measured 4096-token GEMM layout."""

    @classmethod
    @torch.no_grad()
    def prepare(cls, layers):
        for layer in layers:
            if type(layer) is not ColumnParallelLinear or layer.tp_size != 2:
                continue
            if type(layer.quant_method) not in (UnquantizedLinearMethod, cls):
                continue
            w = layer.weight
            if (
                w.device.type == "xpu"
                and w.dtype == torch.bfloat16
                and w.shape in ((2048, 4096), (6144, 4096))
                and w.is_contiguous()
                and torch.xpu.get_device_properties(w.device).device_id == 0x56C0
            ):
                w.data = w.T.contiguous().T
                layer.quant_method = cls()

        # Own exactly one copy of each shard. Q/K/V remain contiguous [K,N]
        # matrices, while gate/up become adjacent slices of a single [K,2N].
        if torch.compiler.is_compiling() or not all(
            type(layer) is ColumnParallelLinear and type(layer.quant_method) is cls
            for layer in layers
        ):
            return
        shards = [layer.weight for layer in layers]
        device = shards[0].device
        if (
            device.type != "xpu"
            or not b60.is_flex_device(device)
            or not all(w.device == device and w.T.is_contiguous() for w in shards)
        ):
            return
        if cls.shared_weight(layers) is not None:
            return
        if len(shards) == 3 and all(w.shape == (2048, 4096) for w in shards):
            combined = torch.stack([w.T for w in shards])
            views = combined.unbind(0)
        elif len(shards) == 2 and all(w.shape == (6144, 4096) for w in shards):
            combined = torch.cat([w.T for w in shards], dim=1)
            views = combined.chunk(2, dim=1)
        else:
            return
        for w, view in zip(shards, views, strict=True):
            w.data = view.T

    @classmethod
    def shared_weight(cls, layers):
        first = layers[0].weight
        if first.ndim != 2 or first.device.type != "xpu":
            return None
        n, k = first.shape
        count = len(layers)
        if count == 3 and n == 2048:
            shape, strides = (3, k, n), (k * n, n, 1)
            row_stride, shard_step = n, k * n
        elif count == 2 and n == 6144:
            shape, strides = (k, 2 * n), (2 * n, 1)
            row_stride, shard_step = 2 * n, n
        else:
            return None
        base = first.untyped_storage().data_ptr()
        offset = first.storage_offset()
        for i, layer in enumerate(layers):
            w = layer.weight
            if not (
                w.shape == (n, k)
                and w.stride() == (1, row_stride)
                and w.dtype == torch.bfloat16
                and w.device == first.device
                and w.untyped_storage().data_ptr() == base
                and w.storage_offset() == offset + i * shard_step
            ):
                return None
        return first.as_strided(shape, strides)

    @classmethod
    def for_image(cls, x, layers):
        if (
            torch.compiler.is_compiling()
            or torch.is_grad_enabled()
            or not torch.is_tensor(x)
            or x.device.type != "xpu"
            or x.dtype != torch.bfloat16
            or x.shape != (1, 4096, 4096)
            or not x.is_contiguous()
            or not b60.is_flex_device(x.device)
            or nn.modules.module._global_forward_pre_hooks
            or nn.modules.module._global_forward_hooks
        ):
            return None
        if not all(
            type(layer) is ColumnParallelLinear
            and type(layer.quant_method) is cls
            and not layer.training
            and layer.tp_size == 2
            and not layer.gather_output
            and layer.bias is None
            and layer.weight.device == x.device
            and "forward" not in layer.__dict__
            and layer._compiled_call_impl is None
            and not layer._forward_pre_hooks
            and not layer._forward_hooks
            for layer in layers
        ):
            return None
        return cls.shared_weight(layers)

    def apply(self, layer, x, bias=None):
        w = layer.weight
        # oneDNN can choose a different accumulation order on short prefixes.
        # Restore the original physical layout for every unmeasured geometry.
        if not (
            x.device.type == "xpu"
            and x.dtype == torch.bfloat16
            and x.shape == (1, 4096, 4096)
            and x.is_contiguous()
            and not torch.compiler.is_compiling()
            and torch.xpu.get_device_properties(x.device).device_id == 0x56C0
        ):
            w = w.contiguous()
        elif not w.T.is_contiguous():
            # A hook or adapter may require an individual image projection.
            w = w.T.contiguous().T
        return apply_unquantized_linear(x, w, bias)


class _FlexTP2DownMethod(UnquantizedLinearMethod):
    """Use the transposed down matrix for the padded 4096-token image only."""

    def apply(self, layer, x, bias=None):
        w = layer.weight
        if w.shape == (4096, 6144) and w.stride() == (1, 4096):
            if w.device.type == "xpu":
                w.record_stream(torch.xpu.current_stream(w.device))
            if x.shape != (1, 4096, 6144) or x.stride(-2) != 6160 or x.stride(-1) != 1:
                w = w.contiguous()
            return apply_unquantized_linear(x, w, bias)
        if (
            x.device.type == "xpu"
            and x.shape == (1, 4096, 6144)
            and x.stride(-2) == 6160
            and x.stride(-1) == 1
            and x.dtype == torch.bfloat16
            and not _FLEX_PADDED_DOWN.disabled
            and not torch.compiler.is_compiling()
            and not torch.is_grad_enabled()
        ):
            from sglang.srt.hardware_backend.xpu.kernels.flex_bf16_mlp import down_weight

            transposed = down_weight(w)
            if transposed is not None:
                return _FLEX_PADDED_DOWN.run(
                    lambda: apply_unquantized_linear(x, transposed, bias),
                    lambda: apply_unquantized_linear(x.contiguous(), w, bias),
                    sig=x.device,
                    logger=logger,
                )
        return apply_unquantized_linear(x, w, bias)


class QwenImage21FeedForward(nn.Module):
    def __init__(self, dim, ratio, quant_config, prefix):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim,
            dim * ratio,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.gate_layer = ColumnParallelLinear(
            dim,
            dim * ratio,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_layer",
        )
        self.out = RowParallelLinear(
            dim * ratio,
            dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out",
        )

    @torch.no_grad()
    def pad_b60_down_weight(self):
        # Avoid the 12 KiB row stride of a contiguous TP2 BF16 down projection.
        # Keep the Parameter and logical shape used by checkpoint/LoRA loaders.
        if not _is_xpu or type(self.out) is not RowParallelLinear:
            return
        if not isinstance(self.out.quant_method, UnquantizedLinearMethod):
            return
        weight = self.out.weight
        if (
            type(self.out.quant_method) is UnquantizedLinearMethod
            and self.out.tp_size == 2
            and weight.device.type == "xpu"
            and weight.dtype == torch.bfloat16
            and weight.shape == (4096, 6144)
            and b60.is_flex_device(weight.device)
            and weight.is_contiguous()
        ):
            self.out.quant_method = _FlexTP2DownMethod()
            return
        if (
            weight.shape != (4096, 6144)
            or not b60.supports_b60(weight)
            or not weight.is_contiguous()
        ):
            return
        rows, cols = weight.shape
        padded = weight.new_empty(rows, cols + 64)[:, :cols]
        padded.copy_(weight)
        weight.data = padded

    def pack_gate_up_weights(self):
        """Share one [2F, C] INT8 weight and [2F, 1] scale storage between gate_layer
        and proj so the comfy_kitchen path projects both with one GEMM.

        As with ``pack_qkv_weights``, parameters keep their names and shapes and
        no reference to the shared storage is kept. Layers that are not plain
        comfy_kitchen INT8 projections are left alone.
        """
        gate, up = self.gate_layer, self.proj
        if _is_xpu:
            _FlexTP2ColumnMethod.prepare((gate, up))
        if self.packed_gate_up() is not None:
            return
        if not (
            all(
                type(layer) is ColumnParallelLinear
                and isinstance(layer.quant_method, ConvRotInt8ComfyKitchenLinearMethod)
                and layer.weight.is_cuda
                and layer.weight.dtype is torch.int8
                and layer.weight.dim() == 2
                for layer in (gate, up)
            )
            and gate.weight.shape == up.weight.shape
            and gate.quant_method.group_size == up.quant_method.group_size
        ):
            return
        rows = gate.weight.shape[0]
        weight = torch.cat([gate.weight.data, up.weight.data], dim=0)
        scale = torch.cat([gate.weight_scale.data, up.weight_scale.data], dim=0)
        gate.weight.data, up.weight.data = weight[:rows], weight[rows:]
        gate.weight_scale.data, up.weight_scale.data = scale[:rows], scale[rows:]

    def packed_gate_up(self):
        """The stacked [gate; up] weight and scale views when the comfy_kitchen
        fused MLP can run, else None."""
        gate, up, out = self.gate_layer, self.proj, self.out
        # LoRA wrappers replace the layers; TP > 1 would need the down all-reduce.
        if not (
            type(gate) is ColumnParallelLinear
            and type(up) is ColumnParallelLinear
            and type(out) is RowParallelLinear
            and out.tp_size == 1
            and isinstance(gate.quant_method, ConvRotInt8ComfyKitchenLinearMethod)
            and isinstance(out.quant_method, ConvRotInt8ComfyKitchenLinearMethod)
            and out.weight.dtype is torch.int8
        ):
            return None
        weight = _stacked_rows(gate.weight, up.weight)
        scale = _stacked_rows(gate.weight_scale, up.weight_scale)
        if weight is None or scale is None:
            return None
        return weight, scale

    def _apply(self, fn, recurse=True):
        # .to() and CPU offload rebuild every parameter in its own storage;
        # re-share them once the weights are back on the GPU.
        resident = hasattr(self, "_flex_bf16_pack")
        if resident:
            flex_bf16.retire(self, (self.gate_layer, self.proj), fn=fn)
        module = super()._apply(fn, recurse)
        if resident:
            self.out.weight.data = self.out.weight.contiguous()
            if type(self.out.quant_method) is _FlexTP2DownMethod:
                self.out.quant_method = UnquantizedLinearMethod()
        self.pack_gate_up_weights()
        self.pad_b60_down_weight()
        return module

    def forward(self, x, *, gated_residual=None):
        row_kwargs = (
            {"gated_residual": gated_residual} if gated_residual is not None else {}
        )
        if _is_xpu and envs.SGLANG_DIFFUSION_FLEX_BF16_PERSISTENT:
            hidden = flex_bf16.mlp_hidden(self, x, _FlexTP2ColumnMethod, _FlexTP2DownMethod)
            if hidden is not None:
                return self.out(hidden, **row_kwargs)[0]
        # Storage-identity checks cannot be traced; a compiled graph keeps the
        # separate projections.
        packed = None if torch.compiler.is_compiling() else self.packed_gate_up()
        if packed is not None:
            return comfy_kitchen_swiglu_mlp(
                x, *packed, self.gate_layer.quant_method.group_size, self.out
            )
        if _is_xpu:
            wide = _FlexTP2ColumnMethod.for_image(x, (self.gate_layer, self.proj))
            if wide is not None and not _FLEX_GATE_UP.disabled:

                def grouped():
                    from sglang.srt.hardware_backend.xpu.kernels.flex_bf16_mlp import (
                        down_weight,
                        packed_gate_up,
                    )

                    can_pack = (
                        type(self.out) is RowParallelLinear
                        and type(self.out.quant_method) is _FlexTP2DownMethod
                        and not self.out.training
                        and "forward" not in self.out.__dict__
                        and self.out._compiled_call_impl is None
                        and not self.out._forward_pre_hooks
                        and not self.out._forward_hooks
                        and not _FLEX_PADDED_DOWN.disabled
                    )
                    if can_pack:
                        projected = packed_gate_up(
                            x, wide, self.gate_layer.weight, self.proj.weight, logger
                        )
                        pad = down_weight(self.out.weight) is not None
                    else:
                        projected = x @ wide
                        pad = False
                    gate, up = projected.chunk(2, dim=-1)
                    return b60.fused_pointwise(
                        gate,
                        up,
                        input_row_padding=gate.shape[-1],
                        output_row_padding=16 if pad else 0,
                    )

                hidden = _FLEX_GATE_UP.run(
                    grouped,
                    lambda: nn.functional.silu(self.gate_layer(x)[0]) * self.proj(x)[0],
                    sig=x.device,
                    logger=logger,
                )
                return self.out(hidden, **row_kwargs)[0]
            x = prepare_convrot_input((self.gate_layer, self.proj), x)
        gate, value = self.gate_layer(x)[0], self.proj(x)[0]
        if _is_xpu and b60.can_fuse_pair(gate, value):
            if can_pack_flex_mlp_output(self.out, gate):
                hidden = _FLEX_SWIGLU_PACK.run(
                    lambda: rotate_and_pack(
                        gate,
                        256,
                        swiglu_value=value,
                        silu_table=b60.silu_table(gate.device),
                    ),
                    lambda: rotate_and_pack(nn.functional.silu(gate) * value, 256),
                    sig=gate.device,
                    equal=lambda a, b: torch.equal(a.values, b.values)
                    and torch.equal(a.scales, b.scales),
                    logger=logger,
                )
                return self.out(hidden, **row_kwargs)[0]
            # Write the activation in the measured BF16 TP2 down-GEMM layout.
            # Both leading dimensions then avoid a 12 KiB row stride.
            pad = (
                32
                if gate.shape == (1, 4096, 6144)
                and type(self.out) is RowParallelLinear
                and self.out.tp_size == 2
                and isinstance(self.out.quant_method, UnquantizedLinearMethod)
                and self.out.weight.stride() == (6208, 1)
                else 0
            )
            hidden = _B60_SILU.run(
                lambda: b60.fused_pointwise(gate, value, output_row_padding=pad),
                lambda: nn.functional.silu(gate) * value,
                logger=logger,
            )
            return self.out(hidden, **row_kwargs)[0]
        fused = None
        if (
            _is_cuda
            and gate.dtype is torch.bfloat16
            and _SILU_MUL_FUSION.can_attempt_once()
        ):
            fused = fused_silu_mul_bitexact(gate, value)
            if _SILU_MUL_FUSION.verified:
                return self.out(fused, **row_kwargs)[0]
        hidden = nn.functional.silu(gate) * value
        if fused is not None:
            hidden = _SILU_MUL_FUSION.accept_or_fallback(fused, hidden, logger=logger)
        return self.out(hidden, **row_kwargs)[0]


class QwenImage21Attention(nn.Module):
    def __init__(self, ac, quant_config, prefix):
        super().__init__()
        dim = ac.hidden_size
        self.heads = ac.num_attention_heads // get_tp_world_size()
        self.head_dim = ac.attention_head_dim
        self.to_q = ColumnParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.to_q"
        )
        self.to_k = ColumnParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.to_k"
        )
        self.to_v = ColumnParallelLinear(
            dim, dim, bias=False, quant_config=quant_config, prefix=f"{prefix}.to_v"
        )
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.to_out.0",
                )
            ]
        )
        self.norm_q = RMSNorm(
            self.head_dim, ac.eps, cast_x_before_out_mul=True, force_native=True
        )
        self.norm_k = RMSNorm(
            self.head_dim, ac.eps, cast_x_before_out_mul=True, force_native=True
        )
        backends = QwenImage21Transformer2DModel._supported_attention_backends
        self.local_attn = LocalAttention(
            self.heads, self.head_dim, supported_attention_backends=backends
        )
        self.target_attn = USPAttention(
            self.heads, self.head_dim, supported_attention_backends=backends
        )

    def pack_qkv_weights(self):
        """Share one [3C, C] storage between to_q/to_k/to_v so one GEMM projects all three.

        The parameters keep their names and shapes (state_dict, LoRA merge and
        weight updates that copy in place all stay valid); only their storage
        moves. No reference to the shared storage is kept: ``packed_qkv_weight``
        re-derives the [3C, C] view from the parameters, so offloading them
        frees the whole buffer. Layers that are not plain bf16 unquantized
        projections are left alone and the packed path stays off.
        """
        layers = (self.to_q, self.to_k, self.to_v)
        if _is_xpu:
            _FlexTP2ColumnMethod.prepare(layers)
        if self.packed_qkv_weight() is not None:
            return
        if not all(
            type(layer) is ColumnParallelLinear
            and isinstance(layer.quant_method, UnquantizedLinearMethod)
            and layer.bias is None
            and not layer.gather_output
            and layer.weight.is_cuda
            and layer.weight.dtype is torch.bfloat16
            and layer.weight.dim() == 2
            and layer.weight.is_contiguous()
            and layer.weight.shape == self.to_q.weight.shape
            for layer in layers
        ):
            return
        packed = torch.cat([layer.weight.data for layer in layers], dim=0)
        rows = self.to_q.weight.shape[0]
        for index, layer in enumerate(layers):
            layer.weight.data = packed[index * rows : (index + 1) * rows]

    def packed_qkv_weight(self):
        """The [3C, C] view over to_q/to_k/to_v when they are consecutive slices of one storage, else None."""
        # LoRA wrappers keep the projection, and its quant method, in base_layer.
        if not all(
            isinstance(
                (
                    layer.base_layer if isinstance(layer, BaseLayerWithLoRA) else layer
                ).quant_method,
                UnquantizedLinearMethod,
            )
            for layer in (self.to_q, self.to_k, self.to_v)
        ):
            return None
        q, k, v = self.to_q.weight, self.to_k.weight, self.to_v.weight
        # Layerwise offload leaves a flat placeholder in place of each weight.
        if q.dim() != 2:
            return None
        rows, cols = q.shape
        if not (
            q.is_cuda
            and q.dtype is torch.bfloat16
            and k.dtype is q.dtype
            and v.dtype is q.dtype
            and k.shape == q.shape
            and v.shape == q.shape
            and q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
        ):
            return None
        base = q.untyped_storage().data_ptr()
        if not (
            k.untyped_storage().data_ptr() == base
            and v.untyped_storage().data_ptr() == base
        ):
            return None
        stride = rows * cols
        if not (
            k.storage_offset() == q.storage_offset() + stride
            and v.storage_offset() == k.storage_offset() + stride
        ):
            return None
        return q.as_strided((3 * rows, cols), (cols, 1))

    def _apply(self, fn, recurse=True):
        # .to() and CPU offload rebuild every parameter in its own storage;
        # re-share them once the weights are back on the GPU so the packed
        # projection survives the round trip (a no-op anywhere else).
        if hasattr(self, "_flex_bf16_pack"):
            flex_bf16.retire(self, (self.to_q, self.to_k, self.to_v), fn=fn)
            flex_bf16.retire(self.to_out[0], (self.to_out[0],), fn=fn)
            if isinstance(self.to_out[0].quant_method, flex_bf16.PackedOutput):
                self.to_out[0].quant_method = UnquantizedLinearMethod()
        module = super()._apply(fn, recurse)
        self.pack_qkv_weights()
        return module

    def project_qkv(self, x):
        if _is_xpu:
            if envs.SGLANG_DIFFUSION_FLEX_BF16_PERSISTENT:
                projections = flex_bf16.qkv(self, x, _FlexTP2ColumnMethod)
                if projections is not None:
                    return projections
            layers = (self.to_q, self.to_k, self.to_v)
            batched = _FlexTP2ColumnMethod.for_image(x, layers)
            if batched is not None and not _FLEX_QKV.disabled:
                # The expanded input has zero batch stride; BMM reads it once
                # per projection and writes three individually contiguous rows.
                projections = _FLEX_QKV.run(
                    lambda: torch.bmm(x.expand(3, -1, -1), batched).split(1),
                    lambda: tuple(layer(x)[0] for layer in layers),
                    sig=x.device,
                    equal=tensors_equal,
                    logger=logger,
                )
                return tuple(
                    z.unflatten(-1, (self.heads, self.head_dim)) for z in projections
                )
            x = prepare_convrot_input(layers, x)
        q = self.to_q(x)[0].unflatten(-1, (self.heads, self.head_dim))
        k = self.to_k(x)[0].unflatten(-1, (self.heads, self.head_dim))
        v = self.to_v(x)[0].unflatten(-1, (self.heads, self.head_dim))
        return q, k, v

    def qkv(self, x, rope):
        q, k, v = self.project_qkv(x)
        return (
            apply_qk_norm_rope(q, self.norm_q, rope),
            apply_qk_norm_rope(k, self.norm_k, rope),
            v,
        )

    def _pack_kv_cuda(self, q, k, v, rope, kp, vp, k_out=None, v_out=None):
        """One CUDA launch: Q normed+roped in place, K normed+roped and V packed behind the prefix.

        With ``k_out``/``v_out`` the projection already wrote the raw K/V into
        rows ``[P:]`` (``k``/``v`` are those views), so K is normalized in place
        and V is not copied. Returns ``(q, k_out, v_out)`` or ``None`` when the
        dtype/head dimension is unsupported. The first call also runs the eager chain and
        keeps its result on a mismatch.
        """
        if not (
            _is_cuda
            and q.is_cuda
            and not torch.compiler.is_compiling()
            and can_use_qknorm_complex_rope_cuda(q.dtype, self.head_dim)
        ):
            return None
        eps = self.norm_k.variance_epsilon
        if self.norm_q.variance_epsilon != eps:
            return None
        prefix = kp.shape[1]
        if k_out is None:
            k_out = k.new_empty(k.shape[0], prefix + k.shape[1], *k.shape[2:])
            v_out = torch.empty_like(k_out)
            k_src, v_src = k, v
        else:
            k_src, v_src = None, None
        reference = None
        if not _KV_PACK_CUDA_FUSION.verified:
            reference = (
                apply_rope(self.norm_q(q), rope),
                torch.cat([kp, apply_rope(self.norm_k(k), rope)], 1),
                torch.cat([vp, v], 1),
            )
        qknorm_complex_rope_pack_(
            q,
            k_out,
            v_out,
            self.norm_q.weight,
            self.norm_k.weight,
            rope,
            kp,
            vp,
            k_src,
            v_src,
            eps,
        )
        if reference is None:
            return q, k_out, v_out
        return _KV_PACK_CUDA_FUSION.accept_or_fallback(
            (q, k_out, v_out), reference, equal=tensors_equal, logger=logger
        )

    def _project_qkv_packed(self, x, layouts, caches):
        """One GEMM writes Q, K and V into a ``[1, P+S, 3C]`` buffer; ``None`` keeps the plain path.

        Q/K/V become column views with token stride ``3C`` (heads stay contiguous);
        the fused norm + RoPE kernel and SDPA both accept that layout. The first
        call compares every slice against the module forwards.
        """
        # Storage-identity checks and the one-shot verification cannot be traced;
        # a compiled graph keeps the plain projections instead of breaking here.
        if not _is_cuda or torch.compiler.is_compiling():
            return None
        layers = (self.to_q, self.to_k, self.to_v)
        packed = self.packed_qkv_weight()
        if not (
            packed is not None
            and x.shape[0] == 1
            and len(layouts) == 1
            and get_sp_world_size() == 1
            and _QKV_PACK_FUSION.can_attempt_once()
            and all(can_project_into(layer, x) for layer in layers)
        ):
            return None
        cache = caches[0]
        prefix = cache["key"].shape[1] if cache else layouts[0]["prefix_rope"].shape[0]
        seq = x.shape[1]
        rows = packed.shape[0] // 3
        # a buffer kept by an image-conditioned prefill still holds the prefix K/V rows
        buffer = cache.get("qkv") if cache else None
        if buffer is None:
            buffer = x.new_empty(1, prefix + seq, 3 * rows)
        torch.mm(x.view(seq, x.shape[-1]), packed.t(), out=buffer[0, prefix:])
        q = buffer[:, prefix:, :rows].view(1, seq, self.heads, self.head_dim)
        k_out = buffer[:, :, rows : 2 * rows].view(
            1, prefix + seq, self.heads, self.head_dim
        )
        v_out = buffer[:, :, 2 * rows :].view(
            1, prefix + seq, self.heads, self.head_dim
        )
        k, v = k_out[:, prefix:], v_out[:, prefix:]
        if not _QKV_PACK_FUSION.verified:
            reference = tuple(
                layer(x)[0].unflatten(-1, (self.heads, self.head_dim))
                for layer in layers
            )
            accepted = _QKV_PACK_FUSION.accept_or_fallback(
                (q, k, v), reference, equal=tensors_equal, logger=logger
            )
            if accepted is reference:
                return None
        return q, k, v, k_out, v_out, buffer

    def _project_kv_into_buffers(self, x, layouts, caches):
        """Project K/V straight into ``[1, P+S, H, D]`` buffers; ``None`` keeps the plain path."""
        if not (
            _is_cuda
            and x.shape[0] == 1
            and len(layouts) == 1
            and get_sp_world_size() == 1
            and _KV_PROJECT_INTO_FUSION.can_attempt_once()
            and can_project_into(self.to_k, x)
            and can_project_into(self.to_v, x)
        ):
            return None
        cache = caches[0]
        prefix = cache["key"].shape[1] if cache else layouts[0]["prefix_rope"].shape[0]
        k_out = x.new_empty(1, prefix + x.shape[1], self.heads, self.head_dim)
        v_out = torch.empty_like(k_out)
        k = project_into(self.to_k, x, k_out[:, prefix:])
        v = project_into(self.to_v, x, v_out[:, prefix:])
        if not _KV_PROJECT_INTO_FUSION.verified:
            reference = (
                self.to_k(x)[0].unflatten(-1, (self.heads, self.head_dim)),
                self.to_v(x)[0].unflatten(-1, (self.heads, self.head_dim)),
            )
            accepted = _KV_PROJECT_INTO_FUSION.accept_or_fallback(
                (k, v), reference, equal=tensors_equal, logger=logger
            )
            if accepted is reference:
                return reference[0], reference[1], None, None
        return k, v, k_out, v_out

    def attend_target(self, q, k, v, prefix_pad):
        if prefix_pad is None:
            return self.target_attn(q, k, v)
        out = q.new_empty((*q.shape[:-1], v.shape[-1]))
        attend_past_prefix_pad(self.target_attn, q, k, v, prefix_pad, out)
        return out

    def attend_sample(
        self,
        q,
        k,
        v,
        rope,
        prefix,
        layout,
        cache,
        k_out=None,
        v_out=None,
        prefix_pad=None,
        qkv=None,
    ):
        # Packed BF16 Q/K/V share one row. Their consumers require contiguous
        # individual projections unless the measured strided kernel can run.
        if _is_xpu and q.device.type == "xpu" and q.shape == (1, 4096, 16, 128) and q.stride(1) == 6144:
            impl = getattr(self.target_attn, "attn_impl", None)
            attention = getattr(impl, "qwen_packed_attention", None)
            if (cache and prefix_pad is None and attention is not None and get_sp_world_size() == 1
                    and self.norm_q.variance_epsilon == self.norm_k.variance_epsilon
                    and b60.can_pack_wide_qkv(q, k, v, cache["key"], cache["value"],
                                               self.norm_q.weight, self.norm_k.weight, rope)):
                def packed():
                    query, kv = b60.pack_wide_qkv(q, self.norm_q.weight, self.norm_k.weight, rope,
                                                  cache["key"], cache["value"], self.norm_q.variance_epsilon)
                    return attention(query[0], kv, impl.softmax_scale).unsqueeze(0)
                def reference():
                    return self.attend_sample(q.contiguous(), k.contiguous(), v.contiguous(), rope,
                                              prefix, layout, cache, k_out, v_out, prefix_pad, qkv)[0]
                return _FLEX_WIDE_FA.run(packed, reference, sig=q.device, logger=logger), None
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        has_prefix_cache = bool(cache)
        if cache and qkv is not None and qkv is cache.get("qkv"):
            # rows [0:P] of the kept buffer are this prefix's K/V: only Q and the target K change
            rows = cache["key"].shape[1]
            empty = k_out.new_empty(1, 0, *k_out.shape[2:])
            q = self._pack_kv_cuda(
                q, k, v, rope, empty, empty, k_out[:, rows:], v_out[:, rows:]
            )[0]
            return self.attend_target(q, k_out, v_out, prefix_pad), None
        if cache:
            kp, vp = cache["key"], cache["value"]
            prefix_output = None
        else:
            qp, kp, vp = self.qkv(prefix, layout["prefix_rope"])
            outputs = []
            # text runs are causal; image blocks see the entire preceding sequence and themselves
            for start, end, is_image in layout["segments"]:
                mask = None
                if not is_image:
                    mask = (
                        torch.arange(end, device=q.device)[None, :]
                        <= torch.arange(start, end, device=q.device)[:, None]
                    )
                    mask = mask[None, None]
                outputs.append(
                    self.local_attn(
                        qp[:, start:end], kp[:, :end], vp[:, :end], attn_mask=mask
                    )
                )
            prefix_output = self.to_out[0](cat_outputs(outputs, dim=1).flatten(2))[0]
            if cache is not None:
                cache.update(key=kp, value=vp)
        if get_sp_world_size() == 1 and _KV_PACK_CUDA_FUSION.can_attempt_once():
            cuda_packed = self._pack_kv_cuda(q, k, v, rope, kp, vp, k_out, v_out)
            if cuda_packed is not None:
                if (
                    prefix_output is not None
                    and cache is not None
                    and qkv is not None
                    and prefix_pad is None
                    and _KV_PACK_CUDA_FUSION.verified
                    and any(is_image for _, _, is_image in layout["segments"])
                ):
                    # an image prefix is as long as the target: copying it every step
                    # costs more than keeping the packed buffer for the request
                    rows = kp.shape[1]
                    cache.update(qkv=qkv, key=k_out[:, :rows], value=v_out[:, :rows])
                return self.attend_target(*cuda_packed, prefix_pad), prefix_output
        q = apply_qk_norm_rope(q, self.norm_q, rope)
        if (
            _is_xpu
            and q.device.type == "xpu"
            and has_prefix_cache
            and prefix_pad is None
            and get_sp_world_size() == 1
            and not _FLEX_PACKED_FA.disabled
        ):
            impl = getattr(self.target_attn, "attn_impl", None)
            attention = getattr(impl, "qwen_packed_attention", None)
            if attention is not None and b60.can_pack_flex_qwen_kv(
                q, k, v, kp, vp, self.norm_k.weight, rope
            ):
                def packed_attention():
                    kv = b60.pack_flex_qwen_kv(
                        k, v, kp, vp, self.norm_k.weight, rope, self.norm_k.variance_epsilon
                    )
                    return attention(q[0], kv, impl.softmax_scale).unsqueeze(0)

                def ordinary_attention():
                    key = apply_qk_norm_rope(k, self.norm_k, rope)
                    return self.target_attn.forward_with_replicated_kv_prefix(q, kp, vp, key, v)

                out = _FLEX_PACKED_FA.run(packed_attention, ordinary_attention, sig=q.device, logger=logger)
                return out, prefix_output
        packed = None
        if (
            get_sp_world_size() == 1
            and _is_cuda
            and k.is_cuda
            and k.dtype in (torch.float16, torch.bfloat16)
            and self.head_dim == 128
            and _KV_ROPE_FUSION.can_attempt_once()
        ):
            packed = qknorm_complex_rope_kv(
                k, self.norm_k.weight, rope, v, kp, vp, self.norm_k.variance_epsilon
            )
            if not _KV_ROPE_FUSION.verified:
                reference = (
                    torch.cat([kp, apply_rope(self.norm_k(k), rope)], 1),
                    torch.cat([vp, v], 1),
                )
                packed = _KV_ROPE_FUSION.accept_or_fallback(
                    packed,
                    reference,
                    equal=tensors_equal,
                    logger=logger,
                )
        if packed is not None:
            out = self.attend_target(q, *packed, prefix_pad)
        else:
            k = apply_qk_norm_rope(k, self.norm_k, rope)
            if prefix_pad is None:
                out = self.target_attn.forward_with_replicated_kv_prefix(
                    q, kp, vp, k, v
                )
            else:
                out = self.attend_target(
                    q, torch.cat([kp, k], 1), torch.cat([vp, v], 1), prefix_pad
                )
        return out, prefix_output

    def forward(
        self, x, ropes, prefixes, layouts, caches, prefix_pad, *, gated_residual=None
    ):
        # batch target projections while retaining each sample's unpadded prefix
        packed = self._project_qkv_packed(x, layouts, caches)
        qkv = None
        if packed is not None:
            q, k, v, k_out, v_out, qkv = packed
        else:
            buffers = self._project_kv_into_buffers(x, layouts, caches)
            if buffers is None:
                q, k, v = self.project_qkv(x)
                k_out = v_out = None
            else:
                q = self.to_q(x)[0].unflatten(-1, (self.heads, self.head_dim))
                k, v, k_out, v_out = buffers
        outputs, prefix_outputs = [], []
        for sample, layout in enumerate(layouts):
            out, prefix_out = self.attend_sample(
                q[sample : sample + 1],
                k[sample : sample + 1],
                v[sample : sample + 1],
                ropes[sample],
                prefixes[sample],
                layout,
                caches[sample],
                k_out=k_out,
                v_out=v_out,
                prefix_pad=prefix_pad,
                qkv=qkv,
            )
            outputs.append(out)
            prefix_outputs.append(prefix_out)
        row_kwargs = (
            {"gated_residual": gated_residual} if gated_residual is not None else {}
        )
        output = self.to_out[0](cat_outputs(outputs).flatten(2), **row_kwargs)[0]
        return output, prefix_outputs


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, ac, quant_config, prefix, layer_id):
        super().__init__()
        self._layer_id = layer_id
        self.img_norm1 = nn.LayerNorm(
            ac.hidden_size, eps=ac.eps, elementwise_affine=False
        )
        self.img_norm2 = nn.LayerNorm(
            ac.hidden_size, eps=ac.eps, elementwise_affine=False
        )
        self.attn = QwenImage21Attention(ac, quant_config, f"{prefix}.attn")
        self.img_mlp = QwenImage21FeedForward(
            ac.hidden_size, ac.mlp_ratio, quant_config, f"{prefix}.img_mlp"
        )
        self.attn._flex_bf16_layer = self.attn.to_out[0]._flex_bf16_layer = layer_id
        self.img_mlp._flex_bf16_layer = layer_id

    def forward(
        self,
        hidden_states,
        modulation,
        prefix_states,
        prefix_modulation,
        layouts,
        ropes,
        caches,
        prefix_pad,
        flex_tokens=None,
    ):
        # Cache-DiT's UnifiedBlocks forwards the same args to every layer.
        # Slice here so prefix KV stays per-layer after that wrap.
        caches = [cache[self._layer_id] for cache in caches]
        if flex_tokens is not None:
            return flex_tokens.block_forward(
                self, hidden_states, modulation, prefix_states[0], prefix_modulation,
                layouts[0], ropes[0], caches[0],
            )
        scale1, gate1, scale2, gate2 = modulation
        prefixes = [
            apply_modulation(
                state["hidden_states"], self.img_norm1, prefix_modulation[0]
            )
            if not cache
            else None
            for state, cache in zip(prefix_states, caches, strict=True)
        ]
        row_kwargs = _flex_row_kwargs(self.attn, hidden_states, gate1)
        attention, prefix_attentions = self.attn(
            modulate_image_input(hidden_states, self.img_norm1, scale1, self.attn),
            ropes,
            prefixes,
            layouts,
            caches,
            prefix_pad,
            **row_kwargs,
        )
        hidden_states = (
            attention if row_kwargs else apply_gated_residual(hidden_states, attention, gate1)
        )
        row_kwargs = _flex_row_kwargs(self.img_mlp, hidden_states, gate2)
        mlp_input = modulate_image_input(
            hidden_states, self.img_norm2, scale2, self.img_mlp
        )
        update = self.img_mlp(mlp_input, **row_kwargs)
        hidden_states = (
            update if row_kwargs else apply_gated_residual(hidden_states, update, gate2)
        )
        for state, attention in zip(prefix_states, prefix_attentions, strict=True):
            if attention is not None:
                _, pg1, ps2, pg2 = prefix_modulation
                prefix = apply_gated_residual(state["hidden_states"], attention, pg1)
                state["hidden_states"] = apply_gated_residual(
                    prefix,
                    self.img_mlp(apply_modulation(prefix, self.img_norm2, ps2)),
                    pg2,
                )
        return hidden_states


class QwenImage21OutputNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)

    def forward(self, x, temb):
        return apply_modulation(
            x, self.norm, self.linear(nn.functional.silu(temb))[:, None]
        )


_FUSED_GATE_UP = ".img_mlp.gate_up"


def _split_fused_gate_up_lora(adapter, *, ffn_dim):
    """Map ComfyUI/ai-toolkit ``img_mlp.gate_up`` LoRAs onto ``gate_layer`` and ``proj``.

    ComfyUI's native checkpoint fuses the SwiGLU inputs row-wise as [gate; up],
    so one shared A feeds both halves of B.
    """
    fused_bases = [
        key[: -len(".lora_A")]
        for key in adapter
        if key.endswith(f"{_FUSED_GATE_UP}.lora_A")
    ]
    if not fused_bases:
        return adapter
    split = dict(adapter)
    for fused in fused_bases:
        lora_a = split.pop(f"{fused}.lora_A")
        lora_b = split.pop(f"{fused}.lora_B", None)
        alpha = split.pop(f"{fused}.alpha", None)
        if lora_b is None:
            raise ValueError(f"Qwen-Image 2.1 LoRA is missing {fused}.lora_B")
        if lora_b.dim() != 2 or lora_b.shape[0] != 2 * ffn_dim:
            raise ValueError(
                f"Qwen-Image 2.1 LoRA {fused}.lora_B must be [{2 * ffn_dim}, rank], "
                f"got {list(lora_b.shape)}"
            )
        mlp = fused[: -len(".gate_up")]
        halves = (("gate_layer", lora_b[:ffn_dim]), ("proj", lora_b[ffn_dim:]))
        for name, rows in halves:
            target = f"{mlp}.{name}"
            if f"{target}.lora_A" in split or f"{target}.lora_B" in split:
                raise ValueError(
                    f"Qwen-Image 2.1 LoRA has both {fused} and {target} weights"
                )
            split[f"{target}.lora_A"] = lora_a
            split[f"{target}.lora_B"] = rows.contiguous()
            if alpha is not None:
                split[f"{target}.alpha"] = alpha
    return split


class QwenImage21Transformer2DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.SAGE_ATTN,
        AttentionBackendEnum.SAGE_ATTN_3,
        AttentionBackendEnum.TORCH_SDPA,
    }
    _fsdp_shard_conditions = [
        lambda name, module: isinstance(module, QwenImage21TransformerBlock)
    ]
    _compile_conditions = _fsdp_shard_conditions
    layer_names = ["transformer_blocks"]
    param_names_mapping = {}
    # Plain nn.Linear outside the blocks: GGUF quantizes them, so dequantize on load.
    gguf_dequantize_prefixes = (
        "img_in.",
        "txt_in.",
        "time_text_embed.",
        "modulation.",
        "norm_out.",
        "proj_out.",
    )

    def __init__(self, config, hf_config, quant_config=None, **kwargs):
        super().__init__(config, hf_config=hf_config, **kwargs)
        ac = self.config
        if ac.patch_size != 1 or not ac.causal_condition or not ac.causal_block:
            raise ValueError(
                "Qwen-Image 2.1 requires patch_size=1, causal_condition=True and causal_block=True"
            )
        self.hidden_size = ac.hidden_size
        self.num_attention_heads = ac.num_attention_heads
        self.num_channels_latents = ac.in_channels
        self.img_in = nn.Linear(ac.in_channels, ac.hidden_size, bias=False)
        self.txt_in = QwenImage21TextProjection(
            ac.context_in_dim, ac.hidden_size, ac.eps
        )
        self.time_text_embed = QwenImage21TimeEmbedding(ac.hidden_size)
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(ac.hidden_size, ac.hidden_size * 4, bias=False)
        )
        self.num_layers = ac.num_layers
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(
                    ac, quant_config, f"transformer_blocks.{i}", i
                )
                for i in range(ac.num_layers)
            ]
        )
        self.norm_out = QwenImage21OutputNorm(ac.hidden_size, ac.eps)
        self.proj_out = nn.Linear(ac.hidden_size, ac.out_channels, bias=False)

    @contextmanager
    def _changing_flex_token_weights(self):
        state = getattr(self, "_flex_token_shards", None)
        if state is None:
            yield
            return
        # _apply can return the original INT8 Parameter (for instance, a dtype
        # cast) or move its storage to another device. Wait before fn sees it.
        state.stamp = None
        state.wait_before_weight_change()
        try:
            yield
        finally:
            from sglang.multimodal_gen.runtime.models.dits.qwen_image21_flex_tokens import (
                _retire,
            )

            # Only still-aliased Parameters need independent local storage.
            # If a clone runs out of memory, _retire keeps the invalid state
            # so the next request can finish reclaiming its peer allocations.
            _retire(self)

    def _apply(self, fn, recurse=True):
        with self._changing_flex_token_weights():
            return super()._apply(fn, recurse)

    def post_load_weights(self):
        with self._changing_flex_token_weights():
            super().post_load_weights()
            for block in self.transformer_blocks:
                if hasattr(block.attn, "_flex_bf16_pack"):
                    attn = block.attn
                    flex_bf16.retire(attn, (attn.to_q, attn.to_k, attn.to_v))
                    flex_bf16.retire(attn.to_out[0], (attn.to_out[0],))
                    if isinstance(attn.to_out[0].quant_method, flex_bf16.PackedOutput):
                        attn.to_out[0].quant_method = UnquantizedLinearMethod()
                if hasattr(block.img_mlp, "_flex_bf16_pack"):
                    mlp = block.img_mlp
                    flex_bf16.retire(mlp, (mlp.gate_layer, mlp.proj))
                    mlp.out.weight.data = mlp.out.weight.contiguous()
                block.attn.pack_qkv_weights()
                block.img_mlp.pack_gate_up_weights()
                block.img_mlp.pad_b60_down_weight()

    def prepare_lora_adapter(self, adapter):
        return _split_fused_gate_up_lora(
            adapter, ffn_dim=self.config.hidden_size * self.config.mlp_ratio
        )

    def prepare_modulation(self, temb):
        # All blocks share these gates. Preserve the native tanh and its dtype,
        # but compute it once per timestep instead of once per block.
        scale1, gate1, scale2, gate2 = self.modulation(temb)[:, None].chunk(4, dim=-1)
        return scale1, gate1.tanh(), scale2, gate2.tanh()

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        layouts,
        condition_latents=None,
        prefix_caches=None,
        prefix_kv=None,
        prefix_pad=None,
        **kwargs,
    ):
        if isinstance(encoder_hidden_states, list):
            encoder_hidden_states = encoder_hidden_states[0]
        if prefix_kv is not None:
            # BCG stacks every layer's padded prefix K/V so a replay copies one input
            prefix_caches = [
                [
                    {"key": key, "value": value}
                    for key, value in prefix_kv.unflatten(0, (-1, 2))
                ]
            ]
        sp = get_sp_world_size()
        target_len = hidden_states.shape[1]
        if target_len % sp:
            raise ValueError(
                f"target token count {target_len} must be divisible by SP degree {sp}"
            )
        local_len = target_len // sp
        rank = get_sp_parallel_rank()
        start, end = rank * local_len, (rank + 1) * local_len
        images = self.img_in(hidden_states[:, start:end])
        bf16_state = None
        if _is_xpu and envs.SGLANG_DIFFUSION_FLEX_BF16_PERSISTENT:
            bf16_state = flex_bf16.begin(self, images, layouts, prefix_pad, sp)
        temb = self.time_text_embed((timestep.to(images.dtype) / 1000), images.dtype)
        modulation = self.prepare_modulation(temb)
        prefix_modulation = None
        if prefix_caches is None or any(not cache[0] for cache in prefix_caches):
            zero_temb = self.time_text_embed(
                timestep.new_zeros(1).to(images.dtype), images.dtype
            )
            prefix_modulation = self.prepare_modulation(zero_temb)
        if prefix_caches is None:
            prefix_caches = [[None] * self.num_layers for _ in layouts]
        prefix_states, ropes = [], []
        for sample, layout in enumerate(layouts):
            prefix = None
            if not prefix_caches[sample][0]:
                prefix = self.txt_in(
                    encoder_hidden_states[
                        sample : sample + 1, : layout["encoder_seq_len"]
                    ]
                ).index_select(1, layout["text_indices"])
                if condition_latents is not None:
                    prefix[:, layout["image_indices"]] = self.img_in(
                        condition_latents[sample : sample + 1]
                    )
            prefix_states.append({"hidden_states": prefix})
            ropes.append(layout["target_rope"][start:end])
        if prefix_pad is not None:
            prefix_pad = PrefixPad(prefix_pad)
        # Same extras for every block so Cache-DiT's UnifiedBlocks wrap is valid.
        # Each block slices prefix_caches by _layer_id. Visit once per layer for
        # layerwise offload.
        token_args = {}
        if _is_xpu and envs.SGLANG_DIFFUSION_FLEX_CONVROT_TOKEN_PARALLEL:
            from sglang.multimodal_gen.runtime.models.dits.qwen_image21_flex_tokens import (
                prepare_tokens,
            )

            state = prepare_tokens(self, images, layouts, prefix_pad, sp)
            if state is not None:
                token_args["flex_tokens"] = state
        for block in self.transformer_blocks:
            images = block(
                images,
                modulation,
                prefix_states,
                prefix_modulation,
                layouts,
                ropes,
                prefix_caches,
                prefix_pad,
                **token_args,
            )
        if bf16_state is not None:
            flex_bf16.finish(bf16_state, images.device)
        output = self.proj_out(self.norm_out(images, temb))
        if sp > 1:
            output = sequence_model_parallel_all_gather(output, dim=1)
        return output


EntryClass = QwenImage21Transformer2DModel
