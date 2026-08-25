#!/usr/bin/env python3
"""Screen the live gfx1201 TP2 FP8 tiles at arbitrary prefill token counts."""

from __future__ import annotations

import argparse
import gc
import statistics

import torch
import triton


SHAPES = (
    ("mlp_gate_up", 17408, 5120),
    ("gdn_qkvz", 7168, 5120),
    ("gdn_ba", 5120, 3072),
    ("mlp_down", 5120, 8704),
    ("attn_o", 8192, 5120),
)


def config_key(config: dict) -> tuple:
    return tuple(sorted(config.items()))


def elapsed_us(fn, warmup: int, repeats: int) -> float:
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
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1152, 2176, 4224]
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(20260825)

    import aiter
    from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
        gemm_a8w8_blockscale,
    )
    from sglang.kernels.ops.quantization.fp8_kernel import (
        _w8a8_block_fp8_matmul,
    )
    from sglang.srt.layers.quantization.fp8_utils import (
        _GFX1201_QWEN_TP2_PREFILL_FP8_CONFIGS,
        _resolve_gfx1201_qwen_tp2_prefill_fp8_config,
    )

    candidate_maps: dict[tuple[int, int], list[tuple[int, dict]]] = {}
    for source_m, shape_map in _GFX1201_QWEN_TP2_PREFILL_FP8_CONFIGS.items():
        for shape, source_config in shape_map.items():
            config = source_config.copy()
            config.pop("NUM_KSPLIT", None)
            entries = candidate_maps.setdefault(shape, [])
            if config_key(config) not in {config_key(item[1]) for item in entries}:
                entries.append((source_m, config))

    quant = aiter.get_hip_quant(aiter.QuantType.per_1x128)
    print(
        "tokens shape current_us selected_us best_us current/selected "
        "selected/best source_m config rel_l2 max_abs",
        flush=True,
    )
    for tokens in args.tokens:
        for name, n, k in SHAPES:
            x = torch.randn(tokens, k, device=args.device, dtype=torch.bfloat16)
            qx, sx = quant(x, quant_dtype=aiter.dtypes.fp8, transpose_scale=False)
            del x
            weight = torch.randn(
                (n, k), device=args.device, dtype=torch.bfloat16
            ).to(torch.float8_e4m3fn)
            weight_scale = torch.rand(
                (n // 128, k // 128), device=args.device, dtype=torch.float32
            )

            def current():
                return gemm_a8w8_blockscale(
                    qx, weight, sx, weight_scale, dtype=torch.bfloat16
                )

            current_out = current()
            current_us = elapsed_us(current, args.warmup, args.repeats)
            results = []
            for source_m, config in candidate_maps[(n, k)]:
                output = torch.empty(
                    (tokens, n), device=args.device, dtype=torch.bfloat16
                )

                def candidate():
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

                candidate_out = candidate().clone()
                diff = (current_out.float() - candidate_out.float()).abs()
                rel_l2 = float(
                    torch.linalg.vector_norm(diff)
                    / torch.linalg.vector_norm(current_out.float()).clamp_min(1e-12)
                )
                max_abs = float(diff.max())
                candidate_us = elapsed_us(candidate, args.warmup, args.repeats)
                results.append(
                    (candidate_us, source_m, config.copy(), rel_l2, max_abs)
                )
                del output, candidate_out, diff

            best_us, source_m, config, rel_l2, max_abs = min(results)
            selected_config = _resolve_gfx1201_qwen_tp2_prefill_fp8_config(
                tokens, n, k
            ).copy()
            selected_config.pop("NUM_KSPLIT", None)
            selected_result = next(
                result
                for result in results
                if config_key(result[2]) == config_key(selected_config)
            )
            selected_us = selected_result[0]
            compact_config = ",".join(
                f"{key}={value}"
                for key, value in config.items()
                if key
                in {
                    "BLOCK_SIZE_M",
                    "BLOCK_SIZE_N",
                    "GROUP_SIZE_M",
                    "num_warps",
                    "num_stages",
                    "matrix_instr_nonkdim",
                    "waves_per_eu",
                    "kpack",
                }
            )
            print(
                f"{tokens:6d} {name:11s} {current_us:10.3f} "
                f"{selected_us:11.3f} {best_us:9.3f} "
                f"{current_us / selected_us:16.4f} "
                f"{selected_us / best_us:13.4f} {source_m:8d} {compact_config} "
                f"{rel_l2:.6g} {max_abs:.6g}",
                flush=True,
            )
            del qx, sx, weight, weight_scale, current_out, results
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
