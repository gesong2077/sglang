"""Observe dispatched XPU operations by input dtype. Timing runs have no dispatch mode."""

from collections import Counter
from contextlib import contextmanager
import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class Counts(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.shapes = Counter()
        self.other = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func._schema.name.split("::")[-1]
        if (
            not args
            or not isinstance(args[0], torch.Tensor)
            or args[0].device.type != "xpu"
        ):
            return out
        x = args[0]
        if name == "linear":
            w = args[1]
            desc = (
                "gemm",
                str(x.dtype),
                x.numel() // x.shape[-1],
                x.shape[-1],
                w.shape[0],
            )
        elif name == "matmul" and x.ndim >= 2 and args[1].ndim >= 2:
            desc = (
                "gemm",
                str(x.dtype),
                out.numel() // out.shape[-1],
                x.shape[-1],
                out.shape[-1],
            )
        elif name in ("mm", "bmm", "_int_mm", "addmm", "baddbmm"):
            a, b = args[1:3] if "add" in name else args[:2]
            desc = (
                "gemm",
                str(a.dtype),
                a.numel() // a.shape[-1],
                a.shape[-1],
                b.shape[-1],
            )
        elif name in (
            "convolution",
            "_convolution",
            "mkldnn_convolution",
            "conv1d",
            "conv2d",
            "conv3d",
        ):
            w = args[1]
            desc = ("conv", str(x.dtype), out.numel(), math.prod(w.shape[1:]), 1)
        elif "scaled_dot_product" in name and len(args) >= 3:
            q, k, v = args[:3]
            # Dense-equivalent QK and PV, including masked positions in fused SDPA.
            desc = (
                "attention",
                str(q.dtype),
                math.prod(q.shape[:-1]),
                k.shape[-2],
                q.shape[-1] + v.shape[-1],
            )
        else:
            self.other[name] += 1
            return out
        self.shapes[desc] += 1
        return out


@contextmanager
def count_native_matmuls(counter):
    from unittest.mock import patch
    from sglang.srt.hardware_backend.xpu.quantization import moss_convrot

    native = moss_convrot._native_matmul()
    if native is None:
        yield
        return

    def counted(q, weight, *args, **kwargs):
        out = native(q, weight, *args, **kwargs)
        if out is not None:
            rows, width = q.shape
            counter.shapes[("gemm", str(q.dtype), rows, width, weight.shape[1])] += 1
            counter.other["native_onednn_gemm"] += 1
        return out

    with patch.object(moss_convrot, "_native_matmul", return_value=counted):
        yield


class QwenCounts(Counts):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        op = func._schema.name.split("::")[-1]
        if func._schema.name == "sglang::flex_token_row_pair":
            x, w = args[:2]
            out = func(*args, **(kwargs or {}))
            batch, rows, width = x.shape
            self.shapes["gemm", str(x.dtype), rows, width, w.shape[1]] += batch
            self.other[op] += 1
            return out
        if op in ("flex_bf16_gate_up", "flex_bf16_projection"):
            self.other[op] += 1
            x = args[0]
            out = func(*args, **(kwargs or {}))
            self.shapes[
                "gemm",
                str(x.dtype),
                x.numel() // x.shape[-1],
                x.shape[-1],
                out.shape[-1],
            ] += 1
            return out
        if func._schema.name == "sgl_kernel::hpg_flash_qwen_packed":
            q, kv = args[:2]
            panels = kv.shape[1]
            assert kv.shape[0] == 2 and kv.shape[2] * q.shape[-1] == kv.shape[3]
            self.shapes[
                "attention",
                str(q.dtype),
                q.shape[0] * q.shape[1],
                panels * 16,
                2 * q.shape[-1],
            ] += 1
        elif op == "hpg_flash_varlen_prefix":
            q, kp, _, k, v = args[:5]
            self.shapes[
                "attention",
                str(q.dtype),
                q.shape[0] * q.shape[1],
                kp.shape[0] + k.shape[0],
                q.shape[-1] + v.shape[-1],
            ] += 1
        elif op == "hpg_flash_varlen":
            q, k, v = args[:3]
            desc = (
                "attention",
                str(q.dtype),
                q.shape[0] * q.shape[1],
                k.shape[0],
                q.shape[-1] + v.shape[-1],
            )
            self.shapes[desc] += 1
        return super().__torch_dispatch__(func, types, args, kwargs)
