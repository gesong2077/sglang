"""Save one actual rank-local Q and packed K/V during the separate full-work diagnostic."""

from pathlib import Path
import os
import torch


def install(executor):
    from sglang.multimodal_gen.runtime.layers.attention.backends.xpu_backend import (
        XPUAttentionImpl,
    )
    from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
        QwenImage21TransformerBlock as Block,
    )

    original = Block.forward
    stage = executor._run_stage_with_executor_hooks
    init = XPUAttentionImpl.__init__
    state = {"enabled": False, "wanted": False, "saved": False}

    def track(self, *a, **kw):
        if self._layer_id == 5 and state["enabled"]:
            state["visits"] += 1
            state["wanted"] = state["visits"] == 2
        try:
            return original(self, *a, **kw)
        finally:
            state["wanted"] = False

    def setup(self, *a, **kw):
        init(self, *a, **kw)
        fn = self.qwen_packed_attention
        if fn is None:
            return

        def captured(q, kv, *args, **kwargs):
            if state["wanted"] and not state["saved"]:
                assert q.shape == (4096, 16, 128) and kv.shape == (2, 258, 16, 2048)
                rank = torch.distributed.get_rank()
                torch.save(
                    dict(
                        q=q.detach().cpu().contiguous(),
                        kv=kv.detach().cpu().contiguous(),
                    ),
                    Path(os.environ["QWEN_RESULT"]) / f"layer5-step2-rank{rank}.pt",
                )
                state["saved"] = True
            return fn(q, kv, *args, **kwargs)

        self.qwen_packed_attention = captured

    def apply(self, st, idx, batch, *a, **kw):
        path = getattr(batch, "perf_dump_path", None)
        if (
            st._component_stage_name() != "denoising_stage"
            or not path
            or Path(path).stem != "diagnostic"
        ):
            return stage(self, st, idx, batch, *a, **kw)
        state.update(enabled=True, visits=0)
        try:
            return stage(self, st, idx, batch, *a, **kw)
        finally:
            state["enabled"] = False

    Block.forward = track
    XPUAttentionImpl.__init__ = setup
    executor._run_stage_with_executor_hooks = apply


def install_moss(executor):
    """Capture the first long, noncausal BF16 call in the diagnostic request."""
    from unittest.mock import patch
    from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl

    stage = executor._run_stage_with_executor_hooks
    forward = SDPAImpl.forward

    def observe(self, st, idx, batch, *args, **kwargs):
        path = getattr(batch, "perf_dump_path", None)
        if (
            st._component_stage_name() != "denoising_stage"
            or not path
            or Path(path).stem != "diagnostic"
        ):
            return stage(self, st, idx, batch, *args, **kwargs)
        rank = torch.distributed.get_rank()
        output = Path(path).with_name(f"moss-rank{rank}.pt")
        saved = False

        def capture(impl, q, k, v, *a, **kw):
            nonlocal saved
            if (
                not saved
                and not impl.causal
                and q.shape == (1, (2372, 2314)[rank], 24, 128)
            ):
                torch.save(
                    dict(
                        q=q.detach().cpu(),
                        k=k.detach().cpu(),
                        v=v.detach().cpu(),
                        scale=impl.softmax_scale,
                        strides=[x.stride() for x in (q, k, v)],
                    ),
                    output,
                )
                saved = True
            return forward(impl, q, k, v, *a, **kw)

        with patch.object(SDPAImpl, "forward", capture):
            result = stage(self, st, idx, batch, *args, **kwargs)
        if not saved:
            raise RuntimeError(f"No matching MOSS attention input for rank {rank}")
        return result

    executor._run_stage_with_executor_hooks = observe
