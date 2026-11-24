"""Replay one captured BF16 long noncausal MOSS attention call under Xe-Forge."""

import os
from pathlib import Path

import torch
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (
    XPUAttentionImpl,
)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        rank = int(os.environ["FLEX_CAPTURE_RANK"])
        path = Path(os.environ["FLEX_CAPTURE_DIR"]) / f"moss-rank{rank}.pt"
        record = torch.load(path, map_location="cpu", weights_only=True)
        device = torch.device("xpu", int(os.environ.get("FLEX_DEVICE_INDEX", "0")))
        self.inputs = []
        for key, stride in zip("qkv", record["strides"]):
            src = record[key]
            tensor = torch.empty_strided(
                src.shape, stride, device=device, dtype=src.dtype
            )
            self.inputs.append(tensor.copy_(src))
        backend = {"fa": XPUAttentionImpl, "sdpa": SDPAImpl}[
            os.environ["FLEX_ATTENTION_BACKEND"]
        ]
        _, _, heads, width = self.inputs[0].shape
        self.impl = backend(heads, width, False, record["scale"])

    def forward(self):
        return self.impl.forward(*self.inputs, None)


def get_inputs():
    return []
