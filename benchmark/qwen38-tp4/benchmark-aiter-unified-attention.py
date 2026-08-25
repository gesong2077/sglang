#!/usr/bin/env python3
"""Microbenchmark AITER unified attention with Qwen3.8 TP4 decode shapes."""

import argparse
import json
import os
import statistics

import torch

from aiter.ops.triton.attention.unified_attention import unified_attention


@torch.inference_mode()
def reference_attention(q, k, v, seq_len, scale):
    k = k.reshape(-1, k.shape[2], k.shape[3])[:seq_len].float()
    v = v.reshape(-1, v.shape[2], v.shape[3])[:seq_len].float()
    repeats = q.shape[1] // k.shape[1]
    k = torch.repeat_interleave(k, repeats, dim=1)
    v = torch.repeat_interleave(v, repeats, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float() * scale, k)
    query_pos = torch.arange(q.shape[0], device=q.device) + seq_len - q.shape[0]
    key_pos = torch.arange(seq_len, device=q.device)
    scores.masked_fill_(key_pos[None, None, :] > query_pos[None, :, None], float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probs, v).to(q.dtype)


@torch.inference_mode()
def run_case(
    seq_len,
    page_size,
    warmups,
    repeats,
    iterations,
    config_max_context,
    expand_kv_heads,
    query_tokens,
    q_heads,
    kv_heads,
    head_size,
    device_index,
    skip_reference,
):
    torch.manual_seed(20260824)
    device = torch.device(f"cuda:{device_index}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    num_blocks = (seq_len + page_size - 1) // page_size

    q = torch.randn(query_tokens, q_heads, head_size, device=device, dtype=dtype)
    k = torch.randn(
        num_blocks, page_size, kv_heads, head_size, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    if expand_kv_heads:
        k = k.expand(-1, -1, q_heads, -1)
        v = v.expand(-1, -1, q_heads, -1)
    out = torch.empty_like(q)
    cu_seqlens_q = torch.tensor([0, query_tokens], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([seq_len], device=device, dtype=torch.int32)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32)[None]
    scale = head_size**-0.5

    def invoke():
        return unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=query_tokens,
            seqused_k=seqused_k,
            max_seqlen_k=config_max_context or seq_len,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )

    for _ in range(warmups):
        invoke()
    torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            invoke()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / iterations)

    if skip_reference:
        max_abs_error = None
        mean_abs_error = None
        allclose = None
    else:
        ref = reference_attention(q, k, v, seq_len, scale)
        abs_diff = (out.float() - ref.float()).abs()
        max_abs_error = abs_diff.max().item()
        mean_abs_error = abs_diff.mean().item()
        allclose = torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
    return {
        "decode_split_cu_multiplier": int(
            os.getenv("AITER_GFX1201_DECODE_SPLIT_CU_MULTIPLIER", "1")
        ),
        "decode_3d_warps": int(os.getenv("AITER_GFX1201_DECODE_3D_WARPS", "8")),
        "decode_3d_waves_per_eu": int(
            os.getenv("AITER_GFX1201_DECODE_3D_WAVES_PER_EU", "2")
        ),
        "long_prefill_block_m": int(
            os.getenv("AITER_GFX1201_LONG_PREFILL_BLOCK_M", "64")
        ),
        "long_prefill_warps": int(
            os.getenv("AITER_GFX1201_LONG_PREFILL_WARPS", "4")
        ),
        "long_prefill_waves_per_eu": int(
            os.getenv("AITER_GFX1201_LONG_PREFILL_WAVES_PER_EU", "4")
        ),
        "seq_len": seq_len,
        "page_size": page_size,
        "config_max_context": config_max_context or seq_len,
        "median_us": statistics.median(timings),
        "min_us": min(timings),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "allclose": allclose,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 16384, 65536])
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--query-tokens", type=int, default=1)
    parser.add_argument("--q-heads", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument(
        "--expand-kv-heads",
        action="store_true",
        help="Benchmark the current SGLang workaround that exposes GQA KV as Q heads.",
    )
    parser.add_argument(
        "--config-max-context",
        type=int,
        default=0,
        help="Compile the kernel for this max context while using each actual seq_len.",
    )
    args = parser.parse_args()

    for seq_len in args.contexts:
        print(
            json.dumps(
                run_case(
                    seq_len,
                    args.page_size,
                    args.warmups,
                    args.repeats,
                    args.iterations,
                    args.config_max_context,
                    args.expand_kv_heads,
                    args.query_tokens,
                    args.q_heads,
                    args.kv_heads,
                    args.head_size,
                    args.device,
                    args.skip_reference,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
