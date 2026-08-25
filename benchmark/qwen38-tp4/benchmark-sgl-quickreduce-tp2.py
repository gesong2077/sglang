#!/usr/bin/env python3
"""Correctness, graph-replay, and latency checks for SGLang TP2 QuickReduce."""

import argparse
import os
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _measure(fn, warmup: int, repeats: int) -> float:
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


def _worker(rank: int, world_size: int, port: int, args) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
    )
    cpu_group = dist.new_group(backend="gloo")

    from sglang.srt.distributed.device_communicators import (
        custom_all_reduce_ops as ops,
    )

    ptr = ops.init_custom_qr(rank, world_size, 96 * 1024 * 1024)
    handle = ops.qr_get_handle(ptr)
    handles = [None] * world_size
    dist.all_gather_object(handles, handle, group=cpu_group)
    ops.qr_open_handles(ptr, handles)

    results = []
    for tokens in args.tokens:
        elements = tokens * args.hidden_size
        generator = torch.Generator(device=f"cuda:{rank}")
        generator.manual_seed(12345)
        base = torch.randn(elements, dtype=torch.bfloat16, device=rank, generator=generator) * 0.125
        inp = base + rank * 0.0625
        qr_out = torch.empty_like(inp)
        rccl_out = inp.clone()

        ops.qr_all_reduce(ptr, inp, qr_out, 0, args.cast_bf16_to_fp16)
        dist.all_reduce(rccl_out)
        torch.cuda.synchronize()
        diff = (qr_out.float() - rccl_out.float()).abs()
        rel_l2 = float(
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(rccl_out.float()).clamp_min(1e-12)
        )
        max_abs = float(diff.max())
        finite = bool(torch.isfinite(qr_out).all())

        qr_us = _measure(
            lambda: ops.qr_all_reduce(
                ptr, inp, qr_out, 0, args.cast_bf16_to_fp16
            ),
            args.warmup,
            args.repeats,
        )
        rccl_us = _measure(
            lambda: dist.all_reduce(rccl_out), args.warmup, args.repeats
        )
        results.append((tokens, elements * 2, qr_us, rccl_us, rel_l2, max_abs, finite))

    # Changing-input eager stress.
    stress_elements = args.stress_tokens * args.hidden_size
    stress_in = torch.empty(stress_elements, dtype=torch.bfloat16, device=rank)
    stress_out = torch.empty_like(stress_in)
    for iteration in range(args.stress_repeats):
        stress_in.fill_(rank + iteration * 0.0009765625)
        ops.qr_all_reduce(ptr, stress_in, stress_out, 0, args.cast_bf16_to_fp16)
    torch.cuda.synchronize()
    expected = stress_in.float() * 2 + (1 if rank == 0 else -1)
    # Above expression reconstructs x(rank=0) + x(rank=1) from either rank's x.
    eager_max_abs = float((stress_out.float() - expected).abs().max())

    # Captured graph must allocate a fresh device epoch on every replay.
    graph_in = torch.empty(stress_elements, dtype=torch.bfloat16, device=rank)
    graph_out = torch.empty_like(graph_in)
    graph_in.fill_(rank)
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.qr_all_reduce(ptr, graph_in, graph_out, 0, args.cast_bf16_to_fp16)
    dist.barrier()
    for iteration in range(args.graph_repeats):
        graph_in.fill_(rank + iteration * 0.0009765625)
        graph.replay()
    torch.cuda.synchronize()
    expected = graph_in.float() * 2 + (1 if rank == 0 else -1)
    graph_max_abs = float((graph_out.float() - expected).abs().max())

    gathered = [None] * world_size
    dist.all_gather_object(
        gathered,
        {
            "rank": rank,
            "sizes": results,
            "eager_max_abs": eager_max_abs,
            "graph_max_abs": graph_max_abs,
        },
        group=cpu_group,
    )
    if rank == 0:
        print(f"cast_bf16_to_fp16={args.cast_bf16_to_fp16}")
        for rank_result in gathered:
            print(
                f"rank={rank_result['rank']} eager_max_abs={rank_result['eager_max_abs']:.6g} "
                f"graph_max_abs={rank_result['graph_max_abs']:.6g}"
            )
        print("tokens bytes qr_us rccl_us speedup rel_l2 max_abs finite")
        for index, row in enumerate(gathered[0]["sizes"]):
            tokens, size_bytes, qr_us, rccl_us, rel_l2, max_abs, finite = row
            peer_qr_us = gathered[1]["sizes"][index][2]
            peer_rccl_us = gathered[1]["sizes"][index][3]
            qr_us = max(qr_us, peer_qr_us)
            rccl_us = max(rccl_us, peer_rccl_us)
            print(
                f"{tokens:6d} {size_bytes:9d} {qr_us:9.3f} {rccl_us:9.3f} "
                f"{rccl_us / qr_us:7.3f} {rel_l2:.6g} {max_abs:.6g} {finite}"
            )

    dist.barrier()
    ops.qr_destroy(ptr)
    dist.destroy_process_group(cpu_group)
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 1024, 4608, 8192])
    parser.add_argument("--stress-tokens", type=int, default=4608)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--stress-repeats", type=int, default=1000)
    parser.add_argument("--graph-repeats", type=int, default=1000)
    parser.add_argument("--cast-bf16-to-fp16", type=int, choices=[0, 1], default=1)
    args = parser.parse_args()
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    mp.spawn(_worker, args=(2, _free_port(), args), nprocs=2, join=True)


if __name__ == "__main__":
    main()
