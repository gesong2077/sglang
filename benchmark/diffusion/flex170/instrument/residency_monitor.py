"""Observe the live weights without retaining tensors, synchronizing XPU, or moving components."""

from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import weakref
import torch


NAMES = ("text_encoder", "transformer", "vae")


def _nested(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (dict, tuple, list)):
        for item in obj.values() if isinstance(obj, dict) else obj:
            yield from _nested(item)


def census(modules):
    """Logical named bytes, unique physical backing storages and cross-component ownership."""
    all_rows = {}
    owners = defaultdict(set)
    sizes = {}
    report = {}
    for role, mod in modules.items():
        rows = []
        seen = set()
        details = {}
        logical = Counter()
        amount = Counter()
        embeddings = []
        layouts = {}

        def account(kind, name, t):
            dev = str(t.device)
            logical[f"{kind}|{dev}|{t.dtype}"] += t.nbytes
            amount[f"{kind}|{dev}|{t.dtype}"] += 1
            storage = t.untyped_storage()
            addr = 0 if dev == "meta" else storage.data_ptr()
            key = (dev, addr)
            if addr:
                owners[key].add(role)
                sizes[key] = max(sizes.get(key, 0), storage.nbytes())
                seen.add(key)
            row = [
                kind,
                name,
                dev,
                str(t.dtype),
                list(t.shape),
                t.nbytes,
                hex(addr),
                storage.nbytes(),
            ]
            rows.append(row)
            if "embed_tokens." in name or "word_embeddings." in name:
                embeddings.append(row)

        for kind, iterator in (
            ("parameter", mod.named_parameters(remove_duplicate=False)),
            ("buffer", mod.named_buffers(remove_duplicate=False)),
        ):
            for name, t in iterator:
                account(kind, name, t)
        state = getattr(mod, "_flex_token_shards", None)
        if state is not None:
            for key in ("slabs", "weights"):
                container = getattr(state, key, {})
                layouts[key] = len(container)
                for i, t in enumerate(_nested(container)):
                    account("tp_aux", f"{key}.{i}", t)
        managers = getattr(mod, "layerwise_offload_managers", ()) or ()
        details.update(
            type=type(mod).__name__,
            logical_bytes=dict(logical),
            tensor_counts=dict(amount),
            embeddings=embeddings,
            offload_managers=len(managers),
            tp_layouts=layouts,
        )
        report[role] = details
        all_rows[role] = rows
    for role, details in report.items():
        total = Counter()
        exclusive = Counter()
        for key, names in owners.items():
            if role in names:
                total[key[0]] += sizes[key]
                if len(names) == 1:
                    exclusive[key[0]] += sizes[key]
        details.update(
            unique_storage_bytes=dict(total), exclusive_storage_bytes=dict(exclusive)
        )
    return report, all_rows


def install(executor):
    root = Path(os.environ["QWEN_RESULT"])
    register = executor.begin_component_residency_request
    run_stage = executor._run_stage_with_executor_hooks
    refs = {}
    anchors = {}
    known = set()

    def discover(st):
        for name in NAMES:
            obj = getattr(st, name, None)
            if isinstance(obj, torch.nn.Module):
                old = refs.get(name)
                if old is None or old() is not obj:
                    refs[name] = weakref.ref(obj)
                    anchors.pop(name, None)

    @wraps(register)
    def begin(self, stages, *args, **kwargs):
        for st in stages:
            discover(st)
        return register(self, stages, *args, **kwargs)

    executor.begin_component_residency_request = begin

    @wraps(run_stage)
    def scoped(self, stage, stage_idx, req, *args, **kwargs):
        name = stage._component_stage_name()
        if name not in ("conditioning_stage", "denoising_stage", "decoding_stage"):
            return run_stage(self, stage, stage_idx, req, *args, **kwargs)
        discover(stage)
        rank = torch.distributed.get_rank()
        dev = torch.xpu.current_device()
        dump = getattr(req, "perf_dump_path", None)
        case = Path(dump).stem if dump else "priming"
        full = case in {"priming", "short"}
        doc = []
        calls = Counter()
        path = root / f"residency.rank{rank}.jsonl"

        def snap(where, step=None, do_full=False):
            t0 = time.perf_counter_ns()
            free, total = torch.xpu.mem_get_info(dev)
            allocated = torch.xpu.memory_allocated(dev)
            reserved = torch.xpu.memory_reserved(dev)
            entry = dict(
                case=case,
                rank=rank,
                stage=name,
                point=where,
                step=step,
                time=time.time(),
                xpu=dict(
                    index=dev,
                    allocated=allocated,
                    reserved=reserved,
                    driver_free=free,
                    driver_total=total,
                    allocator_peak_allocated=torch.xpu.max_memory_allocated(dev),
                    allocator_peak_reserved=torch.xpu.max_memory_reserved(dev),
                ),
            )
            objs = {k: v() for k, v in refs.items() if v() is not None}
            if do_full:
                totals, manifest = census(objs)
                packed = json.dumps(manifest, separators=(",", ":")).encode()
                sig = hashlib.sha256(packed).hexdigest()
                if sig not in known:
                    (root / f"resident-weights.rank{rank}.{sig[:16]}.json").write_bytes(
                        packed
                    )
                    known.add(sig)
                entry.update(
                    components=totals,
                    manifest=f"resident-weights.rank{rank}.{sig[:16]}.json",
                )
                for role, rows in manifest.items():
                    params = [r for r in rows if r[0] == "parameter"]
                    chosen = (
                        params[:1]
                        + params[-1:]
                        + [r for r in params if "embed_tokens." in r[1]]
                    )
                    anchors[role] = tuple(dict.fromkeys(r[1] for r in chosen))
            else:
                probes = {}
                for role, obj in objs.items():
                    items = {}
                    for key in anchors.get(role, ()):
                        try:
                            weight = obj.get_parameter(key)
                        except (AttributeError, KeyError):
                            continue
                        items[key] = [str(weight.device), weight.nbytes]
                    probes[role] = items
                entry["anchors"] = probes
            entry["monitor_cpu_ms"] = (time.perf_counter_ns() - t0) * 1e-6
            # Flush individually on the first request so an allocator failure
            # still leaves the immediately preceding residency and free VRAM.
            if full:
                with path.open("a") as f:
                    f.write(json.dumps(entry) + "\n")
            else:
                doc.append(entry)

        def hook(st, obj, method, label, each_step=False):
            f = getattr(type(obj), method)

            @wraps(f)
            def traced(module, *a, **kw):
                if module is not obj:
                    return f(module, *a, **kw)
                i = calls[label]
                calls[label] += 1
                deep = full and (not each_step or i < 2)
                snap(label + ".before", i, deep)
                try:
                    return f(module, *a, **kw)
                finally:
                    snap(label + ".after", i, deep)

            st.enter_context(patch.object(type(obj), method, traced))

        snap("stage.before", do_full=full)
        with ExitStack() as st:
            if (
                name == "conditioning_stage"
                and getattr(stage, "text_encoder", None) is not None
            ):
                hook(st, stage.text_encoder, "forward", "encoder.forward")
            if (
                name == "denoising_stage"
                and getattr(stage, "transformer", None) is not None
            ):
                hook(st, stage.transformer, "forward", "dit.forward", True)
            if name == "decoding_stage":
                hook(st, stage, "decode", "vae.decode")
            try:
                return run_stage(self, stage, stage_idx, req, *args, **kwargs)
            finally:
                snap("stage.after", do_full=full)
                doc.append(
                    dict(
                        case=case,
                        rank=rank,
                        stage=name,
                        point="calls",
                        calls=dict(calls),
                    )
                )
                with path.open("a") as f:
                    for r in doc:
                        f.write(json.dumps(r) + "\n")

    executor._run_stage_with_executor_hooks = scoped
