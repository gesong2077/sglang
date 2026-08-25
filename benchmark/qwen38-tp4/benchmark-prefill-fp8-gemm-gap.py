#!/usr/bin/env python3
"""Compare the live SGLang/AITER Qwen TP2 block-FP8 prefill GEMM paths.

This benchmark imports only the active SGLang inference environment.  The
candidate tile maps are transcribed from the locally retained, already tuned
gfx1201 source configuration; no vLLM module or executable is imported.
"""

from __future__ import annotations

import argparse
import gc
from typing import Callable

import torch
import triton


SHAPES = (
    ("mlp_gate_up", 17408, 5120),
    ("gdn_qkvz", 7168, 5120),
    ("gdn_ba", 5120, 3072),
    ("mlp_down", 5120, 8704),
    ("attn_o", 8192, 5120),
)


TUNED_CONFIGS = {
    512: {
        (17408, 5120): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=2, kpack=1),
        (7168, 5120): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=32, num_warps=4, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=2),
        (5120, 3072): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=2),
        (5120, 8704): dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=8, num_stages=1, matrix_instr_nonkdim=16, waves_per_eu=0, kpack=2),
        (8192, 5120): dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, GROUP_SIZE_M=32, num_warps=8, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=0),
    },
    # The retained gfx1201 table has no explicit M=1024 row; its nearest-M
    # selector resolves to the M=512 entries below.
    1024: {
        (17408, 5120): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=2, kpack=1),
        (7168, 5120): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=32, num_warps=4, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=2),
        (5120, 3072): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=2),
        (5120, 8704): dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=8, num_stages=1, matrix_instr_nonkdim=16, waves_per_eu=0, kpack=2),
        (8192, 5120): dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, GROUP_SIZE_M=32, num_warps=8, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=0),
    },
    4096: {
        (17408, 5120): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=32, num_warps=4, num_stages=1),
        (7168, 5120): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1),
        (5120, 3072): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1),
        (5120, 8704): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1),
        (8192, 5120): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1),
    },
    4608: {
        (17408, 5120): dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=8, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=0),
        (7168, 5120): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1),
        (5120, 3072): dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=8, num_stages=1, matrix_instr_nonkdim=16, waves_per_eu=0),
        (5120, 8704): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1, matrix_instr_nonkdim=16, waves_per_eu=2, kpack=1),
        (8192, 5120): dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=8, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=0),
    },
    8192: {
        (17408, 5120): dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, GROUP_SIZE_M=32, num_warps=8, num_stages=1, matrix_instr_nonkdim=16, waves_per_eu=0),
        (7168, 5120): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1, waves_per_eu=1),
        (5120, 3072): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=32, num_warps=4, num_stages=2, waves_per_eu=1),
        (5120, 8704): dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=4, num_stages=1, matrix_instr_nonkdim=0, waves_per_eu=2),
        (8192, 5120): dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, GROUP_SIZE_M=8, num_warps=8, num_stages=1, matrix_instr_nonkdim=16, waves_per_eu=0),
    },
}


def bench_us(fn: Callable[[], torch.Tensor], warmup: int, repeats: int) -> float:
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
    return start.elapsed_time(end) * 1000.0 / repeats


def make_config(tokens: int, n: int, k: int) -> dict:
    config = dict(TUNED_CONFIGS[tokens][(n, k)])
    config.update(BLOCK_SIZE_K=128, NUM_KSPLIT=1)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--tokens", type=int, nargs="+", default=[4608, 8192])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(20260825)

    import aiter
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        _w8a8_block_fp8_matmul,
        w8a8_block_fp8_matmul_triton,
    )

    quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    print(
        "tokens shape current_us tuned_us sgl_tuned_us candidate_mi0_us "
        "candidate_mi16_us sgl_file_us tuned/current "
        "current_tflops tuned_tflops aiter_rel_l2 aiter_max_abs "
        "sgl_rel_l2 sgl_max_abs candidate_rel_l2 candidate_max_abs"
    )
    for tokens in args.tokens:
        if tokens not in TUNED_CONFIGS:
            raise ValueError(f"no tuned map for M={tokens}")
        for name, n, k in SHAPES:
            x = torch.randn(tokens, k, device=args.device, dtype=torch.bfloat16) * 0.125
            qx, sx = quant(
                x, quant_dtype=aiter.dtypes.fp8, transpose_scale=False
            )
            del x
            weight_bf16 = (
                torch.randn((n, k), device=args.device, dtype=torch.bfloat16)
                * 0.125
            )
            weight = weight_bf16.to(torch.float8_e4m3fn)
            del weight_bf16
            weight_scale = (
                torch.rand(
                    (n // 128, k // 128),
                    device=args.device,
                    dtype=torch.float32,
                )
                * 0.002
                + 0.0001
            )

            def current() -> torch.Tensor:
                return gemm_a8w8_blockscale(
                    qx, weight, sx, weight_scale, dtype=torch.bfloat16
                )

            tuned_config = make_config(tokens, n, k)

            def tuned() -> torch.Tensor:
                return gemm_a8w8_blockscale(
                    qx,
                    weight,
                    sx,
                    weight_scale,
                    dtype=torch.bfloat16,
                    config=tuned_config.copy(),
                )

            sgl_tuned_config = tuned_config.copy()
            sgl_tuned_config.pop("NUM_KSPLIT")

            def launch_sgl(config: dict) -> torch.Tensor:
                output = torch.empty(
                    (tokens, n), device=args.device, dtype=torch.bfloat16
                )
                grid = (
                    triton.cdiv(tokens, config["BLOCK_SIZE_M"])
                    * triton.cdiv(n, config["BLOCK_SIZE_N"]),
                )
                _w8a8_block_fp8_matmul[grid](
                    qx,
                    weight,
                    output,
                    sx,
                    weight_scale,
                    tokens,
                    n,
                    k,
                    128,
                    128,
                    qx.stride(0),
                    qx.stride(1),
                    weight.stride(1),
                    weight.stride(0),
                    output.stride(0),
                    output.stride(1),
                    sx.stride(0),
                    sx.stride(1),
                    weight_scale.stride(1),
                    weight_scale.stride(0),
                    needs_masking=False,
                    EVEN_M=tokens % config["BLOCK_SIZE_M"] == 0,
                    EVEN_N=n % config["BLOCK_SIZE_N"] == 0,
                    EVEN_K=k % config["BLOCK_SIZE_K"] == 0,
                    CONTIGUOUS_FAST_PATH=True,
                    SCALAR_B_SCALE=(
                        config["BLOCK_SIZE_M"] <= 64
                        and config["BLOCK_SIZE_N"] <= 128
                    ),
                    **config,
                )
                return output

            def sgl_tuned() -> torch.Tensor:
                return launch_sgl(sgl_tuned_config)

            candidate_configs = []
            if (n, k) in ((17408, 5120), (5120, 8704)):
                for matrix_instr_nonkdim in (0, 16):
                    candidate = sgl_tuned_config.copy()
                    candidate.update(
                        BLOCK_SIZE_M=64,
                        BLOCK_SIZE_N=128,
                        BLOCK_SIZE_K=128,
                        num_warps=4,
                        num_stages=2,
                        waves_per_eu=1,
                        matrix_instr_nonkdim=matrix_instr_nonkdim,
                    )
                    candidate_configs.append(candidate)

            def sgl() -> torch.Tensor:
                return w8a8_block_fp8_matmul_triton(
                    qx,
                    weight,
                    sx,
                    weight_scale,
                    [128, 128],
                    output_dtype=torch.bfloat16,
                )

            current_out = current()
            tuned_out = tuned()
            sgl_tuned_out = sgl_tuned()
            diff = (current_out.float() - tuned_out.float()).abs()
            rel_l2 = float(
                torch.linalg.vector_norm(diff)
                / torch.linalg.vector_norm(current_out.float()).clamp_min(1e-12)
            )
            max_abs = float(diff.max())
            sgl_diff = (current_out.float() - sgl_tuned_out.float()).abs()
            sgl_rel_l2 = float(
                torch.linalg.vector_norm(sgl_diff)
                / torch.linalg.vector_norm(current_out.float()).clamp_min(1e-12)
            )
            sgl_max_abs = float(sgl_diff.max())
            candidate_errors = []
            for candidate in candidate_configs:
                candidate_out = launch_sgl(candidate)
                candidate_diff = (current_out.float() - candidate_out.float()).abs()
                candidate_errors.append(
                    (
                        float(
                            torch.linalg.vector_norm(candidate_diff)
                            / torch.linalg.vector_norm(current_out.float()).clamp_min(
                                1e-12
                            )
                        ),
                        float(candidate_diff.max()),
                    )
                )
                del candidate_out, candidate_diff
            current_us = bench_us(current, args.warmup, args.repeats)
            tuned_us = bench_us(tuned, args.warmup, args.repeats)
            sgl_tuned_us = bench_us(sgl_tuned, args.warmup, args.repeats)
            candidate_us = [
                bench_us(
                    lambda candidate=candidate: launch_sgl(candidate),
                    args.warmup,
                    args.repeats,
                )
                for candidate in candidate_configs
            ]
            sgl_us = bench_us(sgl, args.warmup, args.repeats)
            flops = 2.0 * tokens * n * k
            current_tflops = flops / current_us / 1e6
            tuned_tflops = flops / tuned_us / 1e6
            print(
                f"{tokens:6d} {name:11s} {current_us:10.3f} {tuned_us:9.3f} "
                f"{sgl_tuned_us:12.3f} "
                f"{(candidate_us[0] if candidate_us else float('nan')):16.3f} "
                f"{(candidate_us[1] if candidate_us else float('nan')):17.3f} "
                f"{sgl_us:11.3f} "
                f"{current_us/tuned_us:13.4f} "
                f"{current_tflops:14.2f} {tuned_tflops:13.2f} "
                f"{rel_l2:.6g} {max_abs:.6g} "
                f"{sgl_rel_l2:.6g} {sgl_max_abs:.6g} "
                f"{(max((x[0] for x in candidate_errors), default=float('nan'))):.6g} "
                f"{(max((x[1] for x in candidate_errors), default=float('nan'))):.6g}",
                flush=True,
            )
            del qx, sx, weight, weight_scale
            del current_out, tuned_out, sgl_tuned_out, diff, sgl_diff
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
