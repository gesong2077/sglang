#!/usr/bin/env python3
"""Compare the FP8 M=1 kernels used by the Qwen3.5 TP2 decode path.

This benchmark imports only the active SGLang/AITER environment.  The shapes
are the local TP2 projection sizes observed for Qwen3.5-27B.
"""

from __future__ import annotations

import argparse
import gc
import statistics

import torch

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale as aiter_triton_gemm,
)
from sglang.kernels.ops.gemm.rdna4_fp8_skinny import (
    rdna4_fp8_blockwise_skinny,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    w8a8_block_fp8_matmul_triton,
)


SHAPES = [
    (8192, 5120),
    (5120, 3072),
    (17408, 5120),
    (5120, 8704),
    (7168, 5120),
]


def time_us(fn, warmup: int, repeats: int, inner: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / inner)
    return statistics.median(samples)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = actual.float() - expected.float()
    denom = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return (torch.linalg.vector_norm(delta) / denom).item()


def run_shape(n: int, k: int, args: argparse.Namespace) -> None:
    torch.manual_seed(7)
    fp8 = torch.float8_e4m3fn
    x = (torch.rand((1, k), device="cuda", dtype=torch.float32) - 0.5).to(fp8)
    weight = (
        torch.rand((n, k), device="cuda", dtype=torch.float32) - 0.5
    ).to(fp8)
    x_scale = (
        torch.rand((1, k // 128), device="cuda", dtype=torch.float32) * 0.02
        + 0.005
    )
    weight_scale = (
        torch.rand((n // 128, k // 128), device="cuda", dtype=torch.float32)
        * 0.02
        + 0.005
    )

    def skinny():
        return rdna4_fp8_blockwise_skinny(x, weight, x_scale, weight_scale)

    def aiter():
        return aiter_triton_gemm(
            x, weight, x_scale, weight_scale, dtype=torch.bfloat16
        )

    def sgl_triton():
        return w8a8_block_fp8_matmul_triton(
            x,
            weight,
            x_scale,
            weight_scale,
            [128, 128],
            output_dtype=torch.bfloat16,
        )

    outputs = {
        "skinny": skinny(),
        "aiter": aiter(),
        "sgl_triton": sgl_triton(),
    }
    torch.cuda.synchronize()

    x_dequant = x.float() * x_scale.repeat_interleave(128, dim=1)
    weight_dequant = weight.float() * weight_scale.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)
    reference = torch.nn.functional.linear(x_dequant, weight_dequant).bfloat16()
    del x_dequant, weight_dequant

    timings = {
        name: time_us(fn, args.warmup, args.repeats, args.inner)
        for name, fn in (
            ("skinny", skinny),
            ("aiter", aiter),
            ("sgl_triton", sgl_triton),
        )
    }
    best = min(timings.values())
    print(f"M=1 N={n} K={k}")
    for name in ("skinny", "aiter", "sgl_triton"):
        output = outputs[name]
        max_abs = (output.float() - reference.float()).abs().max().item()
        print(
            f"  {name:11s} {timings[name]:9.3f} us  "
            f"x{timings[name] / best:5.2f}  rel_l2={relative_l2(output, reference):.6g}  "
            f"max_abs={max_abs:.6g}"
        )
    print(
        "  pairwise skinny/aiter rel_l2="
        f"{relative_l2(outputs['skinny'], outputs['aiter']):.6g}"
    )
    del x, weight, x_scale, weight_scale, reference, outputs
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--inner", type=int, default=50)
    parser.add_argument(
        "--shape",
        action="append",
        help="Optional N,K pair; may be repeated (default: Qwen TP2 shapes)",
    )
    args = parser.parse_args()
    shapes = SHAPES
    if args.shape:
        shapes = [tuple(map(int, value.split(","))) for value in args.shape]
    print(f"device={torch.cuda.get_device_name(0)} shapes={shapes}")
    for n, k in shapes:
        run_shape(n, k, args)


if __name__ == "__main__":
    main()
