#!/usr/bin/env python3
"""Microbenchmark Qwen3.8 TP2 prefill activation/attention-gate FP8 fusions."""

import argparse

import torch


def bench(fn, warmup: int = 10, repeats: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repeats


def compare(lhs, rhs):
    lhs_q, lhs_s = lhs
    rhs_q, rhs_s = rhs
    lhs_deq = lhs_q.float().view(lhs_s.shape[0], lhs_s.shape[1], 128) * lhs_s.unsqueeze(-1)
    rhs_deq = rhs_q.float().view(rhs_s.shape[0], rhs_s.shape[1], 128) * rhs_s.unsqueeze(-1)
    diff = (lhs_deq - rhs_deq).abs()
    rel_l2 = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(lhs_deq).clamp_min(1e-12)
    return float(rel_l2), float(diff.max()), bool(torch.equal(lhs_q, rhs_q)), bool(torch.equal(lhs_s, rhs_s))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 1024, 4608, 8192])
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    torch.cuda.set_device(args.device)

    import aiter
    from aiter.ops.triton.activation import act_mul_and_fp8_group_quant
    from aiter.ops.triton.quant.fused_fp8_quant import (
        fused_rms_gated_fp8_group_quant,
        fused_sigmoid_mul_fp8_group_quant,
    )
    from sgl_kernel import silu_and_mul
    from sglang.kernels.ops.attention.fla.layernorm_gated import rms_norm_gated
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

    quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    print("tokens path fused_us baseline_us speedup rel_l2 max_abs q_exact s_exact")
    for tokens in args.tokens:
        torch.manual_seed(17 + tokens)
        attn = torch.randn(tokens, 3072, device=args.device, dtype=torch.bfloat16) * 0.125
        gate = torch.randn_like(attn) * 0.125
        baseline_gated = fused_sigmoid_mul(attn.clone(), gate, inplace=True)
        baseline = quant(baseline_gated, quant_dtype=aiter.dtypes.fp8, transpose_scale=False)
        fused = fused_sigmoid_mul_fp8_group_quant(
            attn, gate, group_size=128, out_dtype=torch.float8_e4m3fn
        )
        rel_l2, max_abs, q_exact, s_exact = compare(baseline, fused)
        attn_work = attn.clone()

        def baseline_attn():
            gated = fused_sigmoid_mul(attn_work, gate, inplace=True)
            return quant(gated, quant_dtype=aiter.dtypes.fp8, transpose_scale=False)

        fused_us = bench(
            lambda: fused_sigmoid_mul_fp8_group_quant(
                attn, gate, group_size=128, out_dtype=torch.float8_e4m3fn
            ),
            repeats=args.repeats,
        )
        baseline_us = bench(baseline_attn, repeats=args.repeats)
        print(
            f"{tokens:6d} attn {fused_us:9.3f} {baseline_us:11.3f} "
            f"{baseline_us/fused_us:7.3f} {rel_l2:.6g} {max_abs:.6g} {q_exact} {s_exact}"
        )

        gate_up = torch.randn(tokens, 17408, device=args.device, dtype=torch.bfloat16) * 0.125
        act_out = torch.empty(tokens, 8704, device=args.device, dtype=torch.bfloat16)
        silu_and_mul(gate_up, act_out)
        baseline_act = quant(
            act_out, quant_dtype=aiter.dtypes.fp8, transpose_scale=False
        )
        fused_act = act_mul_and_fp8_group_quant(
            gate_up,
            activation="silu",
            group_size=128,
            dtype_quant=torch.float8_e4m3fn,
        )
        rel_l2, max_abs, q_exact, s_exact = compare(baseline_act, fused_act)
        fused_us = bench(
            lambda: act_mul_and_fp8_group_quant(
                gate_up,
                activation="silu",
                group_size=128,
                dtype_quant=torch.float8_e4m3fn,
            ),
            repeats=args.repeats,
        )
        baseline_us = bench(
            lambda: (
                silu_and_mul(gate_up, act_out),
                quant(act_out, quant_dtype=aiter.dtypes.fp8, transpose_scale=False),
            ),
            repeats=args.repeats,
        )
        print(
            f"{tokens:6d} mlp  {fused_us:9.3f} {baseline_us:11.3f} "
            f"{baseline_us/fused_us:7.3f} {rel_l2:.6g} {max_abs:.6g} {q_exact} {s_exact}"
        )

        rows = tokens * 24
        gdn_x = torch.randn(rows, 128, device=args.device, dtype=torch.bfloat16) * 0.125
        gdn_z = torch.randn_like(gdn_x) * 0.125
        gdn_weight = torch.ones(128, device=args.device, dtype=torch.bfloat16)
        gdn_norm = rms_norm_gated(
            x=gdn_x,
            weight=gdn_weight,
            bias=None,
            z=gdn_z,
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            is_rms_norm=True,
            activation="silu",
        )
        baseline_gdn = quant(
            gdn_norm, quant_dtype=aiter.dtypes.fp8, transpose_scale=False
        )
        fused_gdn = fused_rms_gated_fp8_group_quant(
            gdn_x,
            gdn_weight,
            None,
            gdn_z,
            1e-6,
            norm_before_gate=True,
            activation="silu",
            out_dtype=torch.float8_e4m3fn,
            fp8_min_scaling_factor=1e-10 / 448.0,
            group_size=128,
            round_output_to_input_dtype=True,
        )
        rel_l2, max_abs, q_exact, s_exact = compare(baseline_gdn, fused_gdn)

        def baseline_gdn_fn():
            normed = rms_norm_gated(
                x=gdn_x,
                weight=gdn_weight,
                bias=None,
                z=gdn_z,
                eps=1e-6,
                group_size=None,
                norm_before_gate=True,
                is_rms_norm=True,
                activation="silu",
            )
            return quant(normed, quant_dtype=aiter.dtypes.fp8, transpose_scale=False)

        fused_us = bench(
            lambda: fused_rms_gated_fp8_group_quant(
                gdn_x,
                gdn_weight,
                None,
                gdn_z,
                1e-6,
                norm_before_gate=True,
                activation="silu",
                out_dtype=torch.float8_e4m3fn,
                fp8_min_scaling_factor=1e-10 / 448.0,
                group_size=128,
                round_output_to_input_dtype=True,
            ),
            repeats=args.repeats,
        )
        baseline_us = bench(baseline_gdn_fn, repeats=args.repeats)
        print(
            f"{tokens:6d} gdn  {fused_us:9.3f} {baseline_us:11.3f} "
            f"{baseline_us/fused_us:7.3f} {rel_l2:.6g} {max_abs:.6g} {q_exact} {s_exact}"
        )

if __name__ == "__main__":
    main()
