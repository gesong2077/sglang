"""Measure synchronized rank-local DiT wall time and count dispatched work separately."""

import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from counts import Counts, QwenCounts, count_native_matmuls


def install(executor):
    original = executor._run_stage_with_executor_hooks

    def measure(self, stage, index, batch, *args, **kwargs):
        dump = getattr(batch, "perf_dump_path", None)
        if not dump or stage._component_stage_name() != "denoising_stage":
            return original(self, stage, index, batch, *args, **kwargs)
        rank = torch.distributed.get_rank()
        counter = None
        torch.xpu.synchronize()
        start = time.perf_counter()
        with ExitStack() as stack:
            if Path(dump).stem in ("diagnostic", "short"):
                counter = (
                    QwenCounts()
                    if os.environ["FLEX_BENCH_FAMILY"] == "qwen"
                    else Counts()
                )
                stack.enter_context(counter)
                stack.enter_context(count_native_matmuls(counter))
            output = original(self, stage, index, batch, *args, **kwargs)
        torch.xpu.synchronize()
        row = {"rank": rank, "seconds": time.perf_counter() - start}
        if counter is not None:
            row["ops"] = [[*key, n] for key, n in counter.shapes.items()]
        Path(f"{dump}.rank{rank}.json").write_text(json.dumps(row))
        return output

    executor._run_stage_with_executor_hooks = measure
