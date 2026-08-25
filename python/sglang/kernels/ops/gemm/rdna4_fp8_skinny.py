from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "rdna4_fp8_blockwise_skinny",
        cuda_files=["gemm/rdna4_fp8_blockwise_skinny.cuh"],
        cuda_wrappers=[
            ("rdna4_fp8_blockwise_skinny", "rdna4_fp8_blockwise_skinny"),
            ("rdna4_gdn_dual_projection", "rdna4_gdn_dual_projection"),
            ("rdna4_fp8_lm_head_refine", "rdna4_fp8_lm_head_refine"),
        ],
    )


@register_custom_op(
    op_name="rdna4_fp8_blockwise_skinny",
    mutates_args=["output"],
)
def _rdna4_fp8_blockwise_skinny_custom_op(
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> None:
    _jit_module().rdna4_fp8_blockwise_skinny(
        output, activation, weight, activation_scale, weight_scale
    )


def rdna4_fp8_blockwise_skinny(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """RDNA4 batch-one 128x128 block-scaled FP8 GEMV with BF16 output."""
    output = torch.empty(
        (1, weight.shape[0]),
        dtype=torch.bfloat16,
        device=activation.device,
    )
    _rdna4_fp8_blockwise_skinny_custom_op(
        output, activation, weight, activation_scale, weight_scale
    )
    return output


@register_custom_op(
    op_name="rdna4_fp8_lm_head_refine",
    mutates_args=[
        "candidate_indices",
        "candidate_logits",
        "out_values",
        "out_indices",
    ],
)
def _rdna4_fp8_lm_head_refine_custom_op(
    local_logits: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_logits: torch.Tensor,
    out_values: torch.Tensor,
    out_indices: torch.Tensor,
) -> None:
    _jit_module().rdna4_fp8_lm_head_refine(
        local_logits,
        hidden,
        weight,
        candidate_indices,
        candidate_logits,
        out_values,
        out_indices,
    )


def rdna4_fp8_lm_head_refine(
    local_logits: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    partitions: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine partitioned FP8 logits with the original BF16 lm_head."""
    batch = local_logits.shape[0]
    candidate_indices = torch.empty(
        (batch, partitions), dtype=torch.int32, device=local_logits.device
    )
    candidate_logits = torch.empty(
        (batch, partitions), dtype=local_logits.dtype, device=local_logits.device
    )
    out_values = torch.empty(
        (batch,), dtype=local_logits.dtype, device=local_logits.device
    )
    out_indices = torch.empty(
        (batch,), dtype=torch.int64, device=local_logits.device
    )
    _rdna4_fp8_lm_head_refine_custom_op(
        local_logits,
        hidden,
        weight,
        candidate_indices,
        candidate_logits,
        out_values,
        out_indices,
    )
    return out_values, out_indices


@register_custom_op(
    op_name="rdna4_gdn_dual_projection",
    mutates_args=["q_output", "ba_output"],
)
def _rdna4_gdn_dual_projection_custom_op(
    q_output: torch.Tensor,
    ba_output: torch.Tensor,
    hidden: torch.Tensor,
    q_activation: torch.Tensor,
    q_weight: torch.Tensor,
    q_activation_scale: torch.Tensor,
    q_weight_scale: torch.Tensor,
    ba_weight: torch.Tensor,
) -> None:
    _jit_module().rdna4_gdn_dual_projection(
        q_output,
        ba_output,
        hidden,
        q_activation,
        q_weight,
        q_activation_scale,
        q_weight_scale,
        ba_weight,
    )


def rdna4_gdn_dual_projection(
    hidden: torch.Tensor,
    q_activation: torch.Tensor,
    q_weight: torch.Tensor,
    q_activation_scale: torch.Tensor,
    q_weight_scale: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Qwen3.5/3.8-27B TP2/TP4 M=1 GDN qkvz and ba projections."""
    q_output = torch.empty(
        (1, q_weight.shape[0]), dtype=torch.bfloat16, device=hidden.device
    )
    ba_output = torch.empty(
        (1, ba_weight.shape[0]), dtype=torch.bfloat16, device=hidden.device
    )
    _rdna4_gdn_dual_projection_custom_op(
        q_output,
        ba_output,
        hidden,
        q_activation,
        q_weight,
        q_activation_scale,
        q_weight_scale,
        ba_weight,
    )
    return q_output, ba_output
