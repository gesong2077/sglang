"""Single-pass attention output gating with the unfused BF16 rounding semantics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_mul(X, G, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + idx, idx < TOTAL, other=0)
    gate = tl.load(G + idx, idx < TOTAL, other=0)
    sigmoid = tl.sigmoid(gate.to(tl.float32)).to(gate.dtype)
    result = x.to(tl.float32) * sigmoid.to(tl.float32)
    tl.store(Y + idx, result, idx < TOTAL)


def fused_gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if x.shape != gate.shape or not x.is_contiguous() or not gate.is_contiguous():
        return torch.sigmoid(gate) * x
    out = torch.empty_like(x)
    _sigmoid_mul[(triton.cdiv(x.numel(), 1024),)](
        x, gate, out, x.numel(), 1024, num_warps=4
    )
    return out
