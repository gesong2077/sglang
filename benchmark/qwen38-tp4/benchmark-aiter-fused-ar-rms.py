#!/usr/bin/env python3
"""TP2 gfx1201 correctness/stress check for AITER fused AR+RMSNorm.

This is intentionally independent of vLLM. It exercises the AITER source used
by the SGLang environment, including changing-input HIP-graph replay.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
from dataclasses import dataclass, asdict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass
class RankResult:
    rank: int
    eager_residual_exact: bool
    eager_norm_rel_l2: float
    graph_max_residual_abs: float
    graph_max_norm_rel_l2: float
    graph_nonfinite_steps: int
    fused_us: float
    nccl_norm_us: float


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reference(
    local_x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    group,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = local_x.clone()
    dist.all_reduce(reduced, group=group)
    residual_out = reduced + residual
    value = residual_out.float()
    inv_rms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    norm_out = (value * inv_rms * gamma.float()).to(local_x.dtype)
    return norm_out, residual_out


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(actual.float() - expected.float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def _worker(
    rank: int,
    world_size: int,
    port: int,
    iterations: int,
    timing_iterations: int,
    output_queue,
) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)

    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
    )
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
    )
    ensure_model_parallel_initialized(world_size, 1)
    tp_group = get_tp_group()
    device_group = tp_group.device_group
    device = torch.device("cuda", rank)
    dtype = torch.bfloat16
    shape = (1, 5120)
    eps = 1e-6

    generator = torch.Generator(device=device)
    generator.manual_seed(20260824 + rank)
    gamma = (1.0 + 0.05 * torch.randn(shape[-1], generator=generator, device=device)).to(dtype)
    x = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    residual = torch.randn(shape, generator=generator, device=device, dtype=dtype)

    # Align ranks and force communicator/JIT initialization before capture.
    dist.all_reduce(torch.zeros(1, device=device), group=device_group)
    eager_out, eager_residual = tensor_model_parallel_fused_allreduce_rmsnorm(
        x, residual, gamma, eps
    )
    torch.cuda.synchronize(device)
    eager_ref, eager_residual_ref = _reference(x, residual, gamma, eps, device_group)
    eager_residual_exact = bool(torch.equal(eager_residual, eager_residual_ref))
    eager_norm_rel_l2 = _rel_l2(eager_out, eager_ref)

    static_x = x.clone()
    static_residual = residual.clone()
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as capture_context, torch.cuda.graph(
        graph, stream=capture_context.stream
    ):
        graph_out, graph_residual = tensor_model_parallel_fused_allreduce_rmsnorm(
            static_x, static_residual, gamma, eps
        )
    torch.cuda.synchronize(device)

    max_residual_abs = 0.0
    max_norm_rel_l2 = 0.0
    nonfinite_steps = 0
    for step in range(iterations):
        # Change every element on every replay. Rank- and step-dependent inputs
        # make a stale peer read immediately visible.
        x_step = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        residual_step = torch.randn(
            shape, generator=generator, device=device, dtype=dtype
        )
        expected, expected_residual = _reference(
            x_step, residual_step, gamma, eps, device_group
        )
        static_x.copy_(x_step)
        static_residual.copy_(residual_step)
        graph.replay()
        torch.cuda.synchronize(device)
        if not torch.isfinite(graph_out).all() or not torch.isfinite(graph_residual).all():
            nonfinite_steps += 1
        max_residual_abs = max(
            max_residual_abs,
            float((graph_residual - expected_residual).abs().max().item()),
        )
        max_norm_rel_l2 = max(max_norm_rel_l2, _rel_l2(graph_out, expected))

    # Time graph replay. A barrier brackets each experiment so neither rank
    # starts a collective after its peer has already left the loop.
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier(group=device_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(timing_iterations):
        graph.replay()
    end.record()
    end.synchronize()
    fused_us = start.elapsed_time(end) * 1000.0 / timing_iterations
    dist.barrier(group=device_group)

    # Comparable eager NCCL + BF16 residual/RMS math. This is a diagnostic
    # upper bound; the production baseline uses its own fused norm/quant kernel.
    def nccl_norm_once():
        reduced = static_x.clone()
        dist.all_reduce(reduced, group=device_group)
        value = (reduced + static_residual).float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * gamma.float()).to(dtype)

    for _ in range(20):
        nccl_norm_once()
    torch.cuda.synchronize(device)
    dist.barrier(group=device_group)
    start.record()
    for _ in range(timing_iterations):
        nccl_norm_once()
    end.record()
    end.synchronize()
    nccl_norm_us = start.elapsed_time(end) * 1000.0 / timing_iterations
    dist.barrier(group=device_group)

    output_queue.put(
        asdict(
            RankResult(
                rank=rank,
                eager_residual_exact=eager_residual_exact,
                eager_norm_rel_l2=eager_norm_rel_l2,
                graph_max_residual_abs=max_residual_abs,
                graph_max_norm_rel_l2=max_norm_rel_l2,
                graph_nonfinite_steps=nonfinite_steps,
                fused_us=fused_us,
                nccl_norm_us=nccl_norm_us,
            )
        )
    )

    destroy_model_parallel()
    destroy_distributed_environment()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--timing-iterations", type=int, default=1000)
    args = parser.parse_args()

    world_size = 2
    port = _free_port()
    context = mp.get_context("spawn")
    output_queue = context.SimpleQueue()
    mp.spawn(
        _worker,
        args=(world_size, port, args.iterations, args.timing_iterations, output_queue),
        nprocs=world_size,
        join=True,
    )
    results = sorted((output_queue.get() for _ in range(world_size)), key=lambda x: x["rank"])
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
