#!/usr/bin/env python3
"""Compare Qwen3.8 TP2 GDN decode kernels on one ROCm GPU.

This benchmark does not import or execute vLLM.  It compares the local SGLang
and AITER implementations at the exact per-rank Qwen3.8-27B TP2 shapes.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch

from aiter.ops.triton.conv.causal_conv1d_update_single_token import (
    causal_conv1d_update_single_token as aiter_causal_conv1d_update,
    fused_reshape_causal_conv1d_update_single_token,
)
from aiter.ops.triton.gated_delta_net.fused_rearrange_sigmoid_gdr import (
    fused_rearrange_sigmoid_gated_delta_rule as aiter_fused_recurrent,
)
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as sgl_fused_recurrent,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_update as sgl_causal_conv1d_update,
)


def timed_us(fn: Callable[[], object], warmup: int, repeats: int, loops: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(loops):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / loops)
    return statistics.median(samples), samples


def diff(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    a = actual.float()
    e = expected.float()
    abs_diff = (a - e).abs()
    denom = e.abs().clamp_min(1e-5)
    print(
        f"accuracy {name}: max_abs={abs_diff.max().item():.6g} "
        f"mean_abs={abs_diff.mean().item():.6g} "
        f"max_rel={(abs_diff / denom).max().item():.6g} "
        f"allclose={torch.allclose(a, e, rtol=2e-2, atol=2e-2)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--loops", type=int, default=500)
    args = parser.parse_args()

    torch.manual_seed(1234)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    batch = args.batch

    # Qwen3.8-27B TP2: global HK=16, HV=48, K=V=128.
    num_k_heads = 8
    num_v_heads = 24
    head_k_dim = 128
    head_v_dim = 128
    q_dim = num_k_heads * head_k_dim
    k_dim = num_k_heads * head_k_dim
    v_dim = num_v_heads * head_v_dim
    qkv_dim = q_dim + k_dim + v_dim
    z_dim = v_dim
    qkvz_dim = qkv_dim + z_dim
    width = 4

    qkvz = (torch.randn(batch, qkvz_dim, device=device) * 0.1).to(dtype)
    ba = (torch.randn(batch, 2 * num_v_heads, device=device) * 0.1).to(dtype)
    mixed_qkv = qkvz[:, :qkv_dim].contiguous()
    z_ref = qkvz[:, qkv_dim:].contiguous().view(batch, 1, num_v_heads, head_v_dim)
    b_ref = ba[:, :num_v_heads].contiguous()
    a_ref = ba[:, num_v_heads:].contiguous()
    weight = (torch.randn(qkv_dim, width, device=device) * 0.1).to(dtype)
    cache_indices = torch.arange(batch, device=device, dtype=torch.int32)

    # Match SGLang's [slot, dim, width-1] state layout.
    conv_seed = (torch.randn(args.slots, qkv_dim, width - 1, device=device) * 0.1).to(dtype)

    sgl_state = conv_seed.clone()
    aiter_state = conv_seed.clone()
    fused_state = conv_seed.clone()
    sgl_conv = sgl_causal_conv1d_update(
        mixed_qkv.clone(),
        sgl_state,
        weight,
        activation="silu",
        conv_state_indices=cache_indices,
    )
    aiter_input = mixed_qkv.clone()
    aiter_conv = aiter_causal_conv1d_update(
        aiter_input,
        aiter_state,
        weight,
        activation="silu",
        conv_state_indices=cache_indices,
    )
    z_fused = torch.empty_like(z_ref)
    core_scratch = torch.empty_like(z_ref)
    fused_conv, fused_b, fused_a = fused_reshape_causal_conv1d_update_single_token(
        qkvz,
        batch,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        ba,
        z_fused,
        core_scratch,
        fused_state,
        weight,
        activation="silu",
        conv_state_indices=cache_indices,
        qkvz_layout="flat",
    )
    torch.cuda.synchronize()
    print(
        f"device={torch.cuda.get_device_name(device)} batch={batch} "
        f"qkv={qkv_dim} qkvz={qkvz_dim} state_stride={conv_seed.stride()}"
    )
    diff("aiter_plain_conv/output", aiter_conv, sgl_conv)
    diff("aiter_plain_conv/state", aiter_state, sgl_state)
    diff("aiter_fused_conv/output", fused_conv, sgl_conv)
    diff("aiter_fused_conv/state", fused_state, sgl_state)
    diff("aiter_fused_conv/z", z_fused, z_ref)
    diff("aiter_fused_conv/b", fused_b, b_ref)
    diff("aiter_fused_conv/a", fused_a, a_ref)

    # Persistent tensors used by the timing closures. State mutation is part
    # of the real decode contract; all alternatives perform the same update.
    sgl_conv_input = mixed_qkv.clone()
    sgl_conv_state = conv_seed.clone()
    aiter_conv_input = mixed_qkv.clone()
    aiter_conv_state = conv_seed.clone()
    fused_conv_state = conv_seed.clone()
    z_fused_timing = torch.empty_like(z_ref)
    core_fused_timing = torch.empty_like(z_ref)

    conv_fns: dict[str, Callable[[], object]] = {
        "sgl_conv": lambda: sgl_causal_conv1d_update(
            sgl_conv_input,
            sgl_conv_state,
            weight,
            activation="silu",
            conv_state_indices=cache_indices,
        ),
        "aiter_plain_conv": lambda: aiter_causal_conv1d_update(
            aiter_conv_input,
            aiter_conv_state,
            weight,
            activation="silu",
            conv_state_indices=cache_indices,
        ),
        "aiter_fused_reshape_conv": lambda: fused_reshape_causal_conv1d_update_single_token(
            qkvz,
            batch,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            ba,
            z_fused_timing,
            core_fused_timing,
            fused_conv_state,
            weight,
            activation="silu",
            conv_state_indices=cache_indices,
            qkvz_layout="flat",
        ),
    }
    timings: dict[str, float] = {}
    for name, fn in conv_fns.items():
        median, samples = timed_us(fn, args.warmup, args.repeats, args.loops)
        timings[name] = median
        print(f"timing {name}: median={median:.3f} us samples={[round(x, 3) for x in samples]}")

    A_log = (torch.randn(num_v_heads, device=device) * 0.1).float()
    dt_bias = (torch.randn(num_v_heads, device=device) * 0.1).float()
    ssm_seed = (torch.randn(args.slots, num_v_heads, head_v_dim, head_k_dim, device=device) * 0.01).to(dtype)
    sgl_ssm = ssm_seed.clone()
    aiter_ssm = ssm_seed.clone()
    sgl_out = torch.empty(batch, 1, num_v_heads, head_v_dim, dtype=dtype, device=device)
    aiter_out_scratch = torch.empty_like(sgl_out)

    sgl_fused_recurrent(
        mixed_qkv=sgl_conv,
        a=a_ref,
        b=b_ref,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=head_k_dim**-0.5,
        initial_state=sgl_ssm,
        out=sgl_out,
        ssm_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
    )
    aiter_out, _ = aiter_fused_recurrent(
        A_log=A_log,
        a=a_ref,
        b=b_ref,
        dt_bias=dt_bias,
        qkv=sgl_conv,
        key_dim=q_dim,
        value_dim=v_dim,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        scale=head_k_dim**-0.5,
        initial_state=aiter_ssm,
        inplace_final_state=True,
        ssm_state_indices=cache_indices,
        use_qk_l2norm_in_kernel=True,
        core_attn_out=aiter_out_scratch,
    )
    torch.cuda.synchronize()
    diff("aiter_recurrent/output", aiter_out, sgl_out)
    diff("aiter_recurrent/state", aiter_ssm, sgl_ssm)

    sgl_ssm_timing = ssm_seed.clone()
    aiter_ssm_timing = ssm_seed.clone()
    sgl_out_timing = torch.empty_like(sgl_out)
    aiter_out_timing = torch.empty_like(sgl_out)
    recurrent_fns: dict[str, Callable[[], object]] = {
        "sgl_packed_recurrent": lambda: sgl_fused_recurrent(
            mixed_qkv=sgl_conv,
            a=a_ref,
            b=b_ref,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=head_k_dim**-0.5,
            initial_state=sgl_ssm_timing,
            out=sgl_out_timing,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
        ),
        "aiter_fused_recurrent": lambda: aiter_fused_recurrent(
            A_log=A_log,
            a=a_ref,
            b=b_ref,
            dt_bias=dt_bias,
            qkv=sgl_conv,
            key_dim=q_dim,
            value_dim=v_dim,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            scale=head_k_dim**-0.5,
            initial_state=aiter_ssm_timing,
            inplace_final_state=True,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            core_attn_out=aiter_out_timing,
        ),
    }
    for name, fn in recurrent_fns.items():
        median, samples = timed_us(fn, args.warmup, args.repeats, args.loops)
        timings[name] = median
        print(f"timing {name}: median={median:.3f} us samples={[round(x, 3) for x in samples]}")

    print(
        "combined kernel-time estimates: "
        f"current={timings['sgl_conv'] + timings['sgl_packed_recurrent']:.3f} us "
        f"aiter_plain+sgl_rec={timings['aiter_plain_conv'] + timings['sgl_packed_recurrent']:.3f} us "
        f"aiter_fused+aiter_rec={timings['aiter_fused_reshape_conv'] + timings['aiter_fused_recurrent']:.3f} us"
    )


if __name__ == "__main__":
    main()
