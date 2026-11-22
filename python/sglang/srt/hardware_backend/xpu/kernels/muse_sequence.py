"""TP2 prefill communication reduction without additional quantization.

Only the token-local post-attention/MLP path is sharded. Attention still sees
the original token order and KV metadata. Decode uses the original graph path.
"""

import os

import torch
import torch.distributed as dist
import triton

from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import (
    W8A8Activation,
    _quantize_rows,
)

_release_temps = os.environ.get("SGLANG_XPU_MUSE_RELEASE_TEMPS") == "1"

_exchange_sum = int(os.environ.get("SGLANG_XPU_MUSE_MLP_EXCHANGE_SUM", "0"))
if _exchange_sum:
    from .muse_peer_norm import fused_peer_norm

_use_fused_norm = os.environ.get("SGLANG_XPU_MUSE_FUSED_NORM") == "1"
if _use_fused_norm:
    from .muse_norm import fused_post_norm


def sequence_parallel_post_attention(
    partial, residual, post_attn_norm, pre_mlp_norm, mlp, post_ffn_norm, group, rank,
    *, alignment=2, local_mlp=None, original_rows=None,
):
    """Replace two BF16 all-reduces with RS, INT8 AG, RS, BF16 AG.

    INT8 values and FP32 scales are exactly those of the existing per-token
    quantizer. The int32 view is only a transport packing operation: no integer
    arithmetic is applied to the activation bytes during all-gather.
    """
    peer_part = None
    m, k = partial.shape
    if original_rows is not None:
        m = original_rows
    if partial.dtype != torch.bfloat16 or k % 4 or rank not in (0, 1):
        raise ValueError("Muse sequence communication requires BF16 TP2 rows")
    if alignment < 2 or alignment % 2:
        raise ValueError("TP2 padding alignment must be a positive even number")
    padded = ((m + alignment - 1) // alignment) * alignment
    local_m = padded // 2
    if original_rows is not None:
        if partial.shape != (local_m, k):
            raise ValueError("Sequence-local attention output has wrong shape")
        local = partial
    elif padded != m:
        base = partial._base
        if (
            base is not None
            and base.shape == (padded, k)
            and base.dtype == partial.dtype
            and base.is_contiguous()
            and partial.storage_offset() == 0
            and partial.is_contiguous()
        ):
            full = base
        else:
            full = torch.empty((padded, k), device=partial.device, dtype=partial.dtype)
            full[:m].copy_(partial)
        full[m:].zero_()
    else:
        full = partial
    if original_rows is None:
        local = torch.empty((local_m, k), device=partial.device, dtype=partial.dtype)
        dist.reduce_scatter_single(local, full, group=group)
    begin = rank * local_m
    valid = min(local_m, max(0, m - begin))
    if valid == local_m:
        residual_slice = residual[begin : begin + local_m]
    else:
        residual_slice = torch.empty_like(local)
        residual_slice[:valid].copy_(residual[begin : begin + valid])
        residual_slice[valid:].zero_()
    if _use_fused_norm and k == 6656:
        local_residual, inp = fused_post_norm(
            local, residual_slice, post_attn_norm, pre_mlp_norm, quantize=True
        )
        q, scales = inp.values, inp.scales
    else:
        local_residual = residual_slice + post_attn_norm(local)
        inp = pre_mlp_norm(local_residual)
        q = torch.empty((local_m, k), device=partial.device, dtype=torch.int8)
        scales = torch.empty(local_m, device=partial.device, dtype=torch.float32)
        _quantize_rows[(local_m,)](
            inp, q, scales, k, inp.stride(0), triton.next_power_of_2(k), num_warps=8
        )
    if local_mlp is not None:
        # Replicated original weight shards consume only the local token rows.
        # This removes INT8/scales all-gather and the MLP reduce-scatter.
        result = local_mlp(W8A8Activation(q, scales, (local_m, k)))
        if isinstance(result, tuple):
            # Both original BF16 projections are available in TP rank order.
            local, peer_part = result
        else:
            local = result
        del result
    else:
        full_q = torch.empty((padded, k), device=partial.device, dtype=torch.int8)
        full_scales = torch.empty(padded, device=partial.device, dtype=torch.float32)
        dist.all_gather_single(full_q.view(torch.int32), q.view(torch.int32), group=group)
        dist.all_gather_single(full_scales, scales, group=group)
        if _release_temps and m >= 8192:
            # Collectives join the caller stream. The original local activation
            # is consumed; keep only the gathered activation for the MLP.
            del inp, q, scales
        activation = W8A8Activation(full_q, full_scales, (padded, k))
        down_partial = mlp(activation, skip_all_reduce=True)
        use_exchange = _exchange_sum and _use_fused_norm and k == 6656 and (_exchange_sum == 1 or m >= 8192)
        if use_exchange or (_release_temps and m >= 8192):
            del activation, full_q, full_scales
        if use_exchange:
            # Existing oneCCL transfers the original BF16 partials. The local
            # fusion preserves the original BF16 sum before RMSNorm/residual.
            exchanged = torch.empty_like(down_partial)
            dist.all_to_all_single(
                exchanged.view(torch.int32), down_partial.view(torch.int32), group=group
            )
            local, peer_part = exchanged[:local_m], exchanged[local_m:]
            del down_partial, exchanged
        else:
            dist.reduce_scatter_single(local, down_partial, group=group)
            if _release_temps and m >= 8192:
                del down_partial
    if peer_part is not None:
        local_output = fused_peer_norm(local, peer_part, local_residual, post_ffn_norm)
    elif _use_fused_norm and k == 6656:
        local_output = fused_post_norm(local, local_residual, post_ffn_norm)[0]
    else:
        local_output = local_residual + post_ffn_norm(local)
    output = torch.empty((padded, k), device=partial.device, dtype=partial.dtype)
    dist.all_gather_single(output, local_output, group=group)
    return output[:m]
