from pathlib import Path
import argparse, json, os
import torch
from xe_forge.core.executor import KernelBenchExecutor
from xe_forge.core.reference_workload import PreparedCall
from workload import Model

parser = argparse.ArgumentParser(
    description="Compare FA and SDPA on one captured Qwen rank"
)
parser.add_argument("directory", type=Path)
parser.add_argument("rank", type=int, choices=[0, 1])
args = parser.parse_args()
d = args.directory.resolve()
r = args.rank
dev = torch.device("xpu", r)
torch.xpu.set_device(dev)
os.environ.update(
    FLEX_CAPTURE_DIR=str(d), FLEX_CAPTURE_RANK=str(r), FLEX_DEVICE_INDEX=str(r)
)
with torch.inference_mode():
    model = Model()
    a, b = model.flash(), model.sdpa()
    assert (
        a.shape == b.shape == model.q.shape
        and torch.isfinite(a).all()
        and torch.isfinite(b).all()
    )
    delta = a.float() - b.float()
    sel = torch.arange(32) * 128 + 17
    q = model.q.cpu()[sel].transpose(0, 1).double()
    k = model.k.cpu().transpose(0, 1).double()
    v = model.v.cpu().transpose(0, 1).double()
    expected = ((q @ k.transpose(1, 2)) * (128**-0.5)).softmax(-1) @ v
    xe = KernelBenchExecutor(
        device=str(dev), warmup_iters=3, benchmark_iters=14, rounds=6
    )
    f, g = PreparedCall(model.flash, ()), PreparedCall(model.sdpa, ())
    at, bt, _ = xe.time_interleaved(f, (), g, ())
    ct, dt, _ = xe.time_interleaved(g, (), f, ())

    def rel(y):
        sample = y.cpu()[sel].transpose(0, 1).double()
        return float((sample - expected).norm() / expected.norm())

    rows = dict(
        rank=r,
        flash_us=[at, dt],
        sdpa_us=[bt, ct],
        max_absolute=float(delta.abs().max()),
        different=int((a.view(torch.int16) != b.view(torch.int16)).sum()),
        elements=a.numel(),
        relative_l2=float(delta.norm() / b.float().norm()),
        rows=sel.tolist(),
        flash_relative_l2_to_fp64=rel(a),
        sdpa_relative_l2_to_fp64=rel(b),
    )
    (d / f"bench-rank{r}.json").write_text(json.dumps(rows, indent=2) + "\n")
