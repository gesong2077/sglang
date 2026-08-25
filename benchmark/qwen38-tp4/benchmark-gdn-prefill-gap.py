#!/usr/bin/env python3
"""Compare SGLang FLA and local AITER GDN prefill on the TP2 Qwen shape."""

from __future__ import annotations

import argparse
import statistics

import torch


def elapsed_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(20260825)

    from aiter.ops.triton.gated_delta_net.gated_delta_rule import (
        chunk_gated_delta_rule_opt_vk,
    )
    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule

    print(
        "tokens sglang_ms aiter_ms speedup output_rel_l2 output_max_abs "
        "state_rel_l2 state_max_abs"
    )
    for tokens in args.tokens:
        q = torch.randn(1, tokens, 8, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(
            1, tokens, 24, 128, device="cuda", dtype=torch.bfloat16
        )
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, tokens, 24, device="cuda", dtype=torch.float32)
        )
        beta = torch.sigmoid(
            torch.randn(1, tokens, 24, device="cuda", dtype=torch.float32)
        )
        cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        state_pool = torch.zeros(
            1, 24, 128, 128, device="cuda", dtype=torch.bfloat16
        )
        state_indices = torch.zeros(1, device="cuda", dtype=torch.int64)

        def run_sglang():
            output, _, state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state_pool,
                initial_state_indices=state_indices,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            return output, state

        def run_aiter():
            return chunk_gated_delta_rule_opt_vk(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                use_chunk_hip=False,
                use_chunk_flydsl=False,
                state_dtype=torch.float32,
                use_exp2=False,
                num_decodes=0,
                num_decode_tokens=0,
            )

        sglang_output, _ = run_sglang()
        sglang_state = state_pool.clone()
        aiter_output, aiter_state = run_aiter()
        output_diff = (sglang_output.float() - aiter_output.float()).abs()
        state_diff = (sglang_state.float() - aiter_state.float()).abs()
        output_rel_l2 = float(
            torch.linalg.vector_norm(output_diff)
            / torch.linalg.vector_norm(sglang_output.float())
        )
        state_rel_l2 = float(
            torch.linalg.vector_norm(state_diff)
            / torch.linalg.vector_norm(sglang_state.float())
        )
        sglang_ms = elapsed_ms(run_sglang, args.warmup, args.repeats)
        aiter_ms = elapsed_ms(run_aiter, args.warmup, args.repeats)
        print(
            f"{tokens:6d} {sglang_ms:9.4f} {aiter_ms:8.4f} "
            f"{sglang_ms / aiter_ms:7.4f} {output_rel_l2:.6g} "
            f"{float(output_diff.max()):.6g} {state_rel_l2:.6g} "
            f"{float(state_diff.max()):.6g}"
        )


if __name__ == "__main__":
    main()
