"""Replay both attention primitives using identical captured Q/K/V and no packing in the timed call."""

from pathlib import Path
import os
import torch
import triton
import triton.language as tl
from sglang.kernels.ops.diffusion.rope.qwen_image21_b60 import _flex_packed_kv_offsets
from sgl_kernel.flash_attn import hpg_flash_qwen_packed


@triton.jit
def unpack(
    src, k, v, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, B: tl.constexpr
):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    token = idx // (H * D)
    head = idx // D % H
    col = idx % D
    kp, vp = _flex_packed_kv_offsets(token, head, col)
    tl.store(k + idx, tl.load(src + kp))
    tl.store(v + idx, tl.load(src + T * H * D + vp))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rank = int(os.environ["FLEX_CAPTURE_RANK"])
        device = torch.device("xpu", int(os.environ.get("FLEX_DEVICE_INDEX", "0")))
        if "FLEX_OA_UUID" in os.environ:
            assert (
                str(torch.xpu.get_device_properties(device).uuid)
                == os.environ["FLEX_OA_UUID"]
            )
        record = torch.load(
            Path(os.environ["FLEX_CAPTURE_DIR"]) / f"layer5-step2-rank{rank}.pt",
            weights_only=True,
            mmap=True,
        )
        self.q = record["q"].to(device)
        self.kv = record["kv"].to(device)
        self.k = self.q.new_empty((4128, 16, 128))
        self.v = torch.empty_like(self.k)
        with torch.xpu.device(device):
            unpack[(self.k.numel() // 256,)](
                self.kv, self.k, self.v, 4128, 16, 128, 256, num_warps=8
            )
        self.dense = [v.transpose(0, 1).unsqueeze(0) for v in (self.q, self.k, self.v)]
        self.invoke = (
            self.flash
            if os.environ.get("FLEX_ATTENTION_BACKEND") == "fa"
            else self.sdpa
        )

    def flash(self):
        return hpg_flash_qwen_packed(self.q, self.kv)

    def sdpa(self):
        y = torch.nn.functional.scaled_dot_product_attention(
            *self.dense, dropout_p=0.0, is_causal=False
        )
        return y.squeeze(0).transpose(0, 1)

    def forward(self):
        return self.invoke()


def get_inputs():
    return []
