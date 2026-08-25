#!/usr/bin/env python3
"""Microbenchmark SGLang Triton decode attention with Qwen3.8 TP4 shapes."""

import argparse
import json
import os
import statistics

import torch

from sglang.kernels.ops.attention.decode_attention import decode_attention_fwd


@torch.inference_mode()
def reference_attention(q, k, v, scale):
    if k.ndim == 4:
        k = k.flatten(0, 1)
        v = v.flatten(0, 1)
    k = torch.repeat_interleave(k.float(), q.shape[1] // k.shape[1], dim=1)
    v = torch.repeat_interleave(v.float(), q.shape[1] // v.shape[1], dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float() * scale, k)
    return torch.einsum("hqk,khd->qhd", torch.softmax(scores, dim=-1), v).to(
        q.dtype
    )


@torch.inference_mode()
def run_case(
    seq_len,
    num_splits,
    warmups,
    repeats,
    iterations,
    q_heads,
    kv_heads,
    head_size,
    device,
    page_size,
    skip_reference,
):
    torch.manual_seed(20260824)
    device = torch.device(f"cuda:{device}")
    torch.cuda.set_device(device)
    dtype = torch.bfloat16
    scale = head_size**-0.5

    q = torch.randn(1, q_heads, head_size, device=device, dtype=dtype)
    if page_size == 1:
        k = torch.randn(seq_len, kv_heads, head_size, device=device, dtype=dtype)
    else:
        if seq_len % page_size:
            raise ValueError("seq_len must be divisible by page_size")
        k = torch.randn(
            seq_len // page_size,
            page_size,
            kv_heads,
            head_size,
            device=device,
            dtype=dtype,
        )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    kv_indptr = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    kv_indices = torch.arange(seq_len, device=device, dtype=torch.int32)
    splits = torch.tensor([num_splits], device=device, dtype=torch.int32)
    logits = torch.empty(
        1, q_heads, num_splits, head_size, device=device, dtype=torch.float32
    )
    lse = torch.empty(1, q_heads, num_splits, device=device, dtype=torch.float32)

    def invoke():
        decode_attention_fwd(
            q,
            k,
            v,
            out,
            kv_indptr,
            kv_indices,
            logits,
            lse,
            splits,
            num_splits,
            scale,
            1.0,
            1.0,
            page_size=page_size,
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
        ref = reference_attention(q, k, v, scale)
        abs_diff = (out.float() - ref.float()).abs()
        max_abs_error = abs_diff.max().item()
        mean_abs_error = abs_diff.mean().item()
        allclose = torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
    return {
        "block_h": 16,
        "block_n": int(os.getenv("SGLANG_GFX1201_GQA_DECODE_BLOCK_N", "16")),
        "num_warps": int(
            os.getenv("SGLANG_GFX1201_GQA_DECODE_NUM_WARPS", "4")
        ),
        "waves_per_eu": int(
            os.getenv("SGLANG_GFX1201_GQA_DECODE_WAVES_PER_EU", "4")
        ),
        "seq_len": seq_len,
        "page_size": page_size,
        "num_splits": num_splits,
        "median_us": statistics.median(timings),
        "min_us": min(timings),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "allclose": allclose,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 16384, 65536])
    parser.add_argument("--splits", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--q-heads", type=int, default=6)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=256)
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=1)
    parser.add_argument("--skip-reference", action="store_true")
    args = parser.parse_args()

    for seq_len in args.contexts:
        for num_splits in args.splits:
            print(
                json.dumps(
                    run_case(
                        seq_len,
                        num_splits,
                        args.warmups,
                        args.repeats,
                        args.iterations,
                        args.q_heads,
                        args.kv_heads,
                        args.head_size,
                        args.device,
                        args.page_size,
                        args.skip_reference,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
