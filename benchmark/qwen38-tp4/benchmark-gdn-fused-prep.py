#!/usr/bin/env python3
"""A/B the RDNA4 fused GDN prefill preparation and the existing path."""

from __future__ import annotations

import argparse
import statistics

import torch


def median_ms(fn, warmup: int, repeats: int) -> float:
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


def relative_l2(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float]:
    diff = (lhs.float() - rhs.float()).abs()
    return (
        float(
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(lhs.float()).clamp_min(1e-12)
        ),
        float(diff.max()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 4096, 8192])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(20260825)

    from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
    from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_gdn_prefill_prep,
        fused_qkv_split_gdn_prefill,
    )

    print(
        "tokens old_prep_ms fused_prep_ms prep_speedup old_full_ms fused_full_ms "
        "full_speedup q_rel q_max k_rel k_max g_rel g_max beta_rel beta_max "
        "out_rel out_max state_rel state_max"
    )
    for tokens in args.tokens:
        mixed = torch.randn(
            tokens, 5120, device="cuda", dtype=torch.bfloat16
        ) * 0.125
        a = torch.randn(tokens, 24, device="cuda", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        A_log = torch.randn(24, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(24, device="cuda", dtype=torch.float32)
        cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
        state_index = torch.zeros(1, device="cuda", dtype=torch.int64)
        old_state = torch.zeros(
            1, 24, 128, 128, device="cuda", dtype=torch.bfloat16
        )
        fused_state = torch.zeros_like(old_state)

        def old_prep():
            q, k, v = fused_qkv_split_gdn_prefill(
                mixed, 8, 8, 24, 128, 128, 128
            )
            g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
            return q, k, v, g, beta

        def new_prep():
            return fused_gdn_prefill_prep(
                mixed, a, b, A_log, dt_bias, 8, 24, 128, 128
            )

        q0, k0, v0, g0, beta0 = old_prep()
        q1, k1, v1, g1, beta1 = new_prep()

        def old_full():
            old_state.zero_()
            return chunk_gated_delta_rule(
                q0,
                k0,
                v0,
                g0,
                beta0,
                initial_state=old_state,
                initial_state_indices=state_index,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )[0]

        def new_full():
            fused_state.zero_()
            return chunk_gated_delta_rule(
                q1,
                k1,
                v1,
                g1,
                beta1,
                initial_state=fused_state,
                initial_state_indices=state_index,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
            )[0]

        old_output = old_full()
        new_output = new_full()
        q_err = relative_l2(l2norm_fwd(q0), q1)
        k_err = relative_l2(l2norm_fwd(k0), k1)
        g_err = relative_l2(g0, g1)
        beta_err = relative_l2(beta0, beta1)
        output_err = relative_l2(old_output, new_output)
        state_err = relative_l2(old_state, fused_state)
        old_prep_ms = median_ms(old_prep, args.warmup, args.repeats)
        fused_prep_ms = median_ms(new_prep, args.warmup, args.repeats)
        old_full_ms = median_ms(old_full, args.warmup, args.repeats)
        fused_full_ms = median_ms(new_full, args.warmup, args.repeats)
        print(
            f"{tokens:6d} {old_prep_ms:11.4f} {fused_prep_ms:13.4f} "
            f"{old_prep_ms / fused_prep_ms:12.4f} {old_full_ms:11.4f} "
            f"{fused_full_ms:13.4f} {old_full_ms / fused_full_ms:12.4f} "
            f"{q_err[0]:.6g} {q_err[1]:.6g} {k_err[0]:.6g} {k_err[1]:.6g} "
            f"{g_err[0]:.6g} {g_err[1]:.6g} {beta_err[0]:.6g} {beta_err[1]:.6g} "
            f"{output_err[0]:.6g} {output_err[1]:.6g} "
            f"{state_err[0]:.6g} {state_err[1]:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
