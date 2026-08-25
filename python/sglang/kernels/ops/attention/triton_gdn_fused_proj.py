from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

# =============================================================================
# Fused kernel — reads INTERLEAVED input format
# Used by Qwen3-Next whose checkpoint stores fused in_proj_qkvz weights
# in per-head-group interleaved layout:
#   [g0_q, g0_k, g0_v, g0_z, g1_q, g1_k, g1_v, g1_z, ...]
# =============================================================================


@triton.jit
def fused_qkvzba_split_reshape_cat_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)
    QKVZ_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V * 2
    BA_DIM_T: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK * 2
    QKV_DIM_T: tl.constexpr = HEAD_QK * 2 + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    q_end: tl.constexpr = HEAD_QK
    blk_q_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(0, q_end)
    )
    k_end: tl.constexpr = q_end + HEAD_QK
    blk_k_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(q_end, k_end)
    )
    v_end: tl.constexpr = k_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_v_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(k_end, v_end)
    )
    z_end: tl.constexpr = v_end + NUM_HEADS_V // NUM_HEADS_QK * HEAD_V
    blk_z_ptr = (
        mixed_qkvz
        + i_bs * NUM_HEADS_QK * QKVZ_DIM_T
        + i_qk * QKVZ_DIM_T
        + tl.arange(v_end, z_end)
    )
    blk_q_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_v_st_ptr = (
        mixed_qkv
        + i_bs * NUM_HEADS_QK * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )
    blk_z_st_ptr = (
        z
        + i_bs * NUM_HEADS_V * HEAD_V
        + i_qk * HEAD_V * NUM_HEADS_V // NUM_HEADS_QK
        + tl.arange(0, HEAD_V * NUM_HEADS_V // NUM_HEADS_QK)
    )
    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))
    b_end: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK
    a_end: tl.constexpr = b_end + NUM_HEADS_V // NUM_HEADS_QK
    for i in tl.static_range(b_end):
        blk_b_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))
    for i in tl.static_range(b_end, a_end):
        blk_a_ptr = mixed_ba + i_bs * NUM_HEADS_QK * BA_DIM_T + i_qk * BA_DIM_T + i
        blk_a_st_ptr = (
            a + i_bs * NUM_HEADS_V + i_qk * NUM_HEADS_V // NUM_HEADS_QK + (i - b_end)
        )
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def fused_qkvzba_split_reshape_cat(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    batch, seq_len = mixed_qkvz.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)
    if _is_hip and batch * seq_len == 0:
        return mixed_qkv, z, b, a
    grid = (batch * seq_len, num_heads_qk)
    fused_qkvzba_split_reshape_cat_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=1,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


# =============================================================================
# Fused kernel — reads CONTIGUOUS input format
# Used by Qwen3.5 whose checkpoint stores in_proj_qkv and in_proj_z separately.
# After MergedColumnParallelLinear loads them, the matmul output is contiguous:
#   mixed_qkvz: [all_q | all_k | all_v | all_z]
#   mixed_ba:   [all_b | all_a]
#
# Output format is identical to the interleaved kernel (same downstream consumer).
# =============================================================================


@triton.jit
def fused_qkvzba_split_reshape_cat_contiguous_kernel(
    mixed_qkv,
    z,
    b,
    a,
    mixed_qkvz,
    mixed_ba,
    NUM_HEADS_QK: tl.constexpr,
    NUM_HEADS_V: tl.constexpr,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
):
    i_bs, i_qk = tl.program_id(0), tl.program_id(1)

    V_PER_GROUP: tl.constexpr = NUM_HEADS_V // NUM_HEADS_QK

    # ── Input dimensions (contiguous layout) ──
    TOTAL_Q: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_K: tl.constexpr = NUM_HEADS_QK * HEAD_QK
    TOTAL_V: tl.constexpr = NUM_HEADS_V * HEAD_V
    TOTAL_QKVZ: tl.constexpr = TOTAL_Q + TOTAL_K + TOTAL_V + TOTAL_V
    TOTAL_BA: tl.constexpr = NUM_HEADS_V * 2

    # ── Output dimensions ──
    QKV_DIM_T: tl.constexpr = TOTAL_Q + TOTAL_K + TOTAL_V

    # ── Read from contiguous input ──
    # q for head group i_qk: in the all_q region, offset i_qk * HEAD_QK
    blk_q_ptr = mixed_qkvz + i_bs * TOTAL_QKVZ + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
    # k for head group i_qk: in the all_k region
    blk_k_ptr = (
        mixed_qkvz
        + i_bs * TOTAL_QKVZ
        + TOTAL_Q
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    # v for head group i_qk: in the all_v region
    blk_v_ptr = (
        mixed_qkvz
        + i_bs * TOTAL_QKVZ
        + TOTAL_Q
        + TOTAL_K
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )
    # z for head group i_qk: in the all_z region
    blk_z_ptr = (
        mixed_qkvz
        + i_bs * TOTAL_QKVZ
        + TOTAL_Q
        + TOTAL_K
        + TOTAL_V
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )

    # ── Write to output (identical layout to the interleaved kernel) ──
    blk_q_st_ptr = mixed_qkv + i_bs * QKV_DIM_T + i_qk * HEAD_QK + tl.arange(0, HEAD_QK)
    blk_k_st_ptr = (
        mixed_qkv
        + i_bs * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK
        + i_qk * HEAD_QK
        + tl.arange(0, HEAD_QK)
    )
    blk_v_st_ptr = (
        mixed_qkv
        + i_bs * QKV_DIM_T
        + NUM_HEADS_QK * HEAD_QK * 2
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )
    blk_z_st_ptr = (
        z
        + i_bs * NUM_HEADS_V * HEAD_V
        + i_qk * V_PER_GROUP * HEAD_V
        + tl.arange(0, V_PER_GROUP * HEAD_V)
    )

    tl.store(blk_q_st_ptr, tl.load(blk_q_ptr))
    tl.store(blk_k_st_ptr, tl.load(blk_k_ptr))
    tl.store(blk_v_st_ptr, tl.load(blk_v_ptr))
    tl.store(blk_z_st_ptr, tl.load(blk_z_ptr))

    # ── b and a from contiguous [all_b | all_a] ──
    for i in tl.static_range(V_PER_GROUP):
        blk_b_ptr = mixed_ba + i_bs * TOTAL_BA + i_qk * V_PER_GROUP + i
        blk_b_st_ptr = b + i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
        tl.store(blk_b_st_ptr, tl.load(blk_b_ptr))

    for i in tl.static_range(V_PER_GROUP):
        blk_a_ptr = mixed_ba + i_bs * TOTAL_BA + NUM_HEADS_V + i_qk * V_PER_GROUP + i
        blk_a_st_ptr = a + i_bs * NUM_HEADS_V + i_qk * V_PER_GROUP + i
        tl.store(blk_a_st_ptr, tl.load(blk_a_ptr))


def fused_qkvzba_split_reshape_cat_contiguous(
    mixed_qkvz,
    mixed_ba,
    num_heads_qk,
    num_heads_v,
    head_qk,
    head_v,
):
    """Fused split/reshape/cat for CONTIGUOUS input format (Qwen3.5).

    Input layout:
        mixed_qkvz: [all_q | all_k | all_v | all_z]
        mixed_ba:   [all_b | all_a]

    Output layout (same as fused_qkvzba_split_reshape_cat):
        mixed_qkv: [all_q | all_k | all_v]  (z stripped)
        z: [num_v_heads, head_v]
        b: [num_v_heads]
        a: [num_v_heads]
    """
    batch, seq_len = mixed_qkvz.shape[0], 1
    qkv_dim_t = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = torch.empty(
        [batch * seq_len, qkv_dim_t],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    z = torch.empty(
        [batch * seq_len, num_heads_v, head_v],
        dtype=mixed_qkvz.dtype,
        device=mixed_qkvz.device,
    )
    b = torch.empty(
        [batch * seq_len, num_heads_v],
        dtype=mixed_ba.dtype,
        device=mixed_ba.device,
    )
    a = torch.empty_like(b)
    if _is_hip and batch * seq_len == 0:
        return mixed_qkv, z, b, a
    grid = (batch * seq_len, num_heads_qk)
    # Each program moves `v_per_group * head_v` elements for both v and z. For
    # the small head-group ratios (<= 512 elements) a single warp is the best
    # fit; wider ratios (e.g. 8 v-heads per k-head) need more lanes so the
    # per-program vector load/store does not serialize. The threshold was tuned
    # on MI355X, so it is confined to the HIP/aiter path; every other backend
    # keeps the original `num_warps=1`.
    num_warps = 1
    if _use_aiter:
        v_elems_per_program = (num_heads_v // num_heads_qk) * head_v
        num_warps = 1 if v_elems_per_program <= 512 else 4
    fused_qkvzba_split_reshape_cat_contiguous_kernel[grid](
        mixed_qkv,
        z,
        b,
        a,
        mixed_qkvz,
        mixed_ba,
        num_heads_qk,
        num_heads_v,
        head_qk,
        head_v,
        num_warps=num_warps,
        num_stages=3,
    )
    return mixed_qkv, z, b, a


@triton.jit
def fused_qkv_split_gdn_prefill_kernel(
    q,
    k,
    v,
    mixed_qkv,
    MIXED_QKV_STRIDE_T: tl.constexpr,
    MIXED_QKV_STRIDE_D: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    HEAD_V: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i_t = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    q_dim: tl.constexpr = NUM_Q_HEADS * HEAD_Q
    k_dim: tl.constexpr = NUM_K_HEADS * HEAD_K
    v_dim: tl.constexpr = NUM_V_HEADS * HEAD_V
    qk_dim: tl.constexpr = q_dim + k_dim
    qkv_dim: tl.constexpr = qk_dim + v_dim

    mask = offsets < qkv_dim
    values = tl.load(
        mixed_qkv + i_t * MIXED_QKV_STRIDE_T + offsets * MIXED_QKV_STRIDE_D,
        mask=mask,
    )

    q_mask = offsets < q_dim
    tl.store(q + i_t * q_dim + offsets, values, mask=q_mask)

    k_offsets = offsets - q_dim
    k_mask = (offsets >= q_dim) & (offsets < qk_dim)
    tl.store(k + i_t * k_dim + k_offsets, values, mask=k_mask)

    v_offsets = offsets - qk_dim
    v_mask = (offsets >= qk_dim) & (offsets < qkv_dim)
    tl.store(v + i_t * v_dim + v_offsets, values, mask=v_mask)


def fused_qkv_split_gdn_prefill(
    mixed_qkv: torch.Tensor,
    num_q_heads: int,
    num_k_heads: int,
    num_v_heads: int,
    head_q: int,
    head_k: int,
    head_v: int,
):
    """Split packed post-conv GDN QKV into contiguous FLA prefill tensors.

    `mixed_qkv` is laid out per token as `[all_q | all_k | all_v]`. The FLA
    chunk kernels consume separate contiguous `[1, T, H, D]` tensors, so this
    fused split replaces three independent `aten::copy_` kernels from the
    generic FLA input guard. `mixed_qkv` may be a strided `[T, qkv_dim]` view.
    """
    seq_len = mixed_qkv.shape[0]
    q = torch.empty(
        (1, seq_len, num_q_heads, head_q),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    k = torch.empty(
        (1, seq_len, num_k_heads, head_k),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    v = torch.empty(
        (1, seq_len, num_v_heads, head_v),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )

    qkv_dim = num_q_heads * head_q + num_k_heads * head_k + num_v_heads * head_v
    if _is_hip and seq_len == 0:
        return q, k, v
    fused_qkv_split_gdn_prefill_kernel[(seq_len,)](
        q,
        k,
        v,
        mixed_qkv,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        num_q_heads,
        num_k_heads,
        num_v_heads,
        head_q,
        head_k,
        head_v,
        BLOCK_SIZE=triton.next_power_of_2(qkv_dim),
        num_warps=8,
        num_stages=3,
    )
    return q, k, v


# Adapted from the local vLLM GDN prefill preparation kernel.  Keeping the
# implementation here avoids importing vLLM at runtime and lets SGLang use the
# fused path independently.
@triton.jit
def fused_gdn_prefill_prep_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    q,
    k,
    v,
    g,
    beta,
    stride_x_t,
    stride_a_t,
    stride_b_t,
    stride_q_t,
    stride_k_t,
    stride_v_t,
    SEQ_LEN,
    NUM_K_HEADS: tl.constexpr,
    NUM_V_HEADS: tl.constexpr,
    HEAD_K: tl.constexpr,
    HEAD_V: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
    L2NORM_EPS: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    """Split Q/K/V, normalize Q/K, and form GDN gates in one launch."""
    token_block = tl.program_id(0)
    head = tl.program_id(1)
    token_offsets = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    token_mask = token_offsets < SEQ_LEN
    qk_dim: tl.constexpr = NUM_K_HEADS * HEAD_K

    if head < NUM_K_HEADS:
        feature_offsets = tl.arange(0, BLOCK_K)
        feature_mask = feature_offsets < HEAD_K
        mask = token_mask[:, None] & feature_mask[None, :]

        q_offsets = (
            token_offsets[:, None] * stride_x_t
            + head * HEAD_K
            + feature_offsets[None, :]
        )
        k_offsets = q_offsets + qk_dim
        q_values = tl.load(mixed_qkv + q_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        k_values = tl.load(mixed_qkv + k_offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        # Use the same division/sqrt form as FLA's l2norm_fwd kernel.  On
        # gfx1201 tl.rsqrt has a measurably different approximation and shifts
        # the recurrent state enough to change greedy output on long runs.
        q_values /= tl.sqrt(
            tl.sum(q_values * q_values, axis=1)[:, None] + L2NORM_EPS
        )
        k_values /= tl.sqrt(
            tl.sum(k_values * k_values, axis=1)[:, None] + L2NORM_EPS
        )

        q_out_offsets = (
            token_offsets[:, None] * stride_q_t
            + head * HEAD_K
            + feature_offsets[None, :]
        )
        k_out_offsets = (
            token_offsets[:, None] * stride_k_t
            + head * HEAD_K
            + feature_offsets[None, :]
        )
        tl.store(q + q_out_offsets, q_values, mask=mask)
        tl.store(k + k_out_offsets, k_values, mask=mask)
    else:
        value_head = head - NUM_K_HEADS
        value_offsets = tl.arange(0, BLOCK_V)
        value_mask = value_offsets < HEAD_V
        mask = token_mask[:, None] & value_mask[None, :]
        packed_v_offsets = (
            token_offsets[:, None] * stride_x_t
            + 2 * qk_dim
            + value_head * HEAD_V
            + value_offsets[None, :]
        )
        v_values = tl.load(mixed_qkv + packed_v_offsets, mask=mask, other=0.0)
        v_out_offsets = (
            token_offsets[:, None] * stride_v_t
            + value_head * HEAD_V
            + value_offsets[None, :]
        )
        tl.store(v + v_out_offsets, v_values, mask=mask)

        gate_offsets = token_offsets * NUM_V_HEADS + value_head
        a_values = tl.load(
            a + token_offsets * stride_a_t + value_head,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        b_values = tl.load(
            b + token_offsets * stride_b_t + value_head,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        a_log = tl.load(A_log + value_head).to(tl.float32)
        bias = tl.load(dt_bias + value_head).to(tl.float32)
        x = a_values + bias
        # Match SGLang's existing fused_gdn_gating arithmetic.
        softplus = tl.where(
            x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x
        )
        tl.store(
            g + gate_offsets,
            -tl.exp(a_log) * softplus,
            mask=token_mask,
        )
        tl.store(
            beta + gate_offsets,
            tl.sigmoid(b_values).to(b.dtype.element_ty),
            mask=token_mask,
        )


def fused_gdn_prefill_prep(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k: int,
    head_v: int,
):
    """Return normalized Q/K plus V/G/Beta in the GDN prefill layouts."""
    seq_len = mixed_qkv.shape[0]
    expected_dim = 2 * num_k_heads * head_k + num_v_heads * head_v
    if mixed_qkv.shape[1] != expected_dim:
        raise ValueError(
            f"packed GDN width {mixed_qkv.shape[1]} does not match {expected_dim}"
        )

    q = torch.empty(
        (1, seq_len, num_k_heads, head_k),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    k = torch.empty_like(q)
    v = torch.empty(
        (1, seq_len, num_v_heads, head_v),
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    g = torch.empty(
        (1, seq_len, num_v_heads), dtype=torch.float32, device=mixed_qkv.device
    )
    beta = torch.empty_like(g)
    if seq_len == 0:
        return q, k, v, g, beta

    block_t = 16
    fused_gdn_prefill_prep_kernel[
        (triton.cdiv(seq_len, block_t), num_k_heads + num_v_heads)
    ](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        q.stride(1),
        k.stride(1),
        v.stride(1),
        seq_len,
        num_k_heads,
        num_v_heads,
        head_k,
        head_v,
        BLOCK_T=block_t,
        BLOCK_K=triton.next_power_of_2(head_k),
        BLOCK_V=triton.next_power_of_2(head_v),
        L2NORM_EPS=1e-6,
        SOFTPLUS_THRESHOLD=20.0,
        num_warps=4,
        num_stages=2,
    )
    return q, k, v, g, beta
