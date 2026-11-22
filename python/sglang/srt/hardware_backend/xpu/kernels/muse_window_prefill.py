"""Experimental paged Intel FlashAttention for the Muse TP2 rank geometry."""

import json
import logging
from pathlib import Path
import ctypes
from functools import lru_cache
import torch


@lru_cache(maxsize=4)
def _load(library):
    native = ctypes.CDLL(library)
    invoke = native.muse_fa_tile
    invoke.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_float]
    invoke.restype = ctypes.c_int
    logging.getLogger(__name__).info(
        "Loaded Muse sliding-window prefill on %s from %s",
        torch.xpu.current_device(),
        library,
    )
    return invoke


def try_prefill(library, q, k, v, pages, offsets, lengths, scale, window):
    if (
        q.ndim != 3
        or q.shape[1:] != (16, 128)
        or q.shape[0] < 2048
        or k.ndim != 4
        or k.shape[1:] != (64, 1, 128)
        or k.shape != v.shape
        or pages is None
        or pages.ndim != 2
        or pages.shape[0] != 1
        or offsets is None
        or offsets.shape != (2,)
        or lengths is None
        or lengths.shape != (1,)
        or q.device.type != "xpu"
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.dtype != torch.bfloat16
        or window != (2047, 0)
        or any(x.dtype != torch.int32 for x in (pages, offsets, lengths))
        or any(
            x.device != q.device or not x.is_contiguous()
            for x in (q, k, v, pages, offsets, lengths)
        )
    ):
        return None
    if any(t.data_ptr() % 64 for t in (q, k, v)):
        return None
    out = torch.empty_like(q)
    rc = _load(library)(
        *[x.data_ptr() for x in (q, k, v, out, pages, offsets, lengths)],
        q.size(0),
        pages.size(1),
        k.size(0),
        window[0],
        scale,
    )
    if rc:
        raise RuntimeError(f"Intel paged FlashAttention failed with status {rc}")
    return out


def verify_shadow(candidate, reference, layer, lengths, directory):
    exact = torch.equal(candidate, reference)
    record = dict(
        device=candidate.device.index,
        layer=layer,
        rows=candidate.size(0),
        kv_length=int(lengths.item()),
        exact=exact,
    )
    if not exact:
        record["max_absolute"] = (candidate - reference).abs().max().item()
        record["different"] = int((candidate != reference).sum().item())
    with (Path(directory) / f"shadow-{candidate.device.index}.jsonl").open("a") as out:
        print(json.dumps(record), file=out)
    if not exact:
        raise RuntimeError(f"Experimental Intel FA failed real-input shadow: {record}")
