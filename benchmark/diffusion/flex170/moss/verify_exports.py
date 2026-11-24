import argparse
import json
from pathlib import Path
import torch
from safetensors import safe_open
from export_moss import BASE, destination, source_file, text_targets

torch.set_num_threads(6)


def load_comp(comp, root):
    d = root / comp
    return d, json.loads(next(d.glob("*index.json")).read_text())["weight_map"]


def read(d, index, k):
    with safe_open(d / index[k], framework="pt") as s:
        return s.get_tensor(k)


def invert(rows, g):
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    )
    h = h4
    while h.shape[0] < g:
        h = torch.kron(h, h4)
    return ((rows.reshape(-1, g) @ h) / (g**0.5)).reshape_as(rows)


parser = argparse.ArgumentParser(
    description="Verify MOSS exports against their source tensors"
)
parser.add_argument(
    "--modes",
    nargs="+",
    choices=["both", "comfy", "rtn", "gptq"],
    default=["both", "comfy", "rtn", "gptq"],
)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
results = {}
for mode in args.modes:
    root = destination(mode)
    if not root.exists():
        raise FileNotFoundError(root)
    result = {}
    for comp in ("text_encoder", "transformer", "vae"):
        d, idx = load_comp(comp, root) if comp != "vae" else (root / comp, None)
        if idx is None:
            assert next(d.glob("*.safetensors")).samefile(
                next((BASE / comp).glob("*.safetensors"))
            )
            continue
        base, orig = load_comp(comp, BASE)
        cfg = json.loads((d / "config.json").read_text()).get("quantization_config", {})
        if not cfg:
            assert all(
                (d / shard).samefile(base / shard) for shard in set(orig.values())
            )
            continue
        groups = cfg.get("layer_groups", {})
        targets = sorted(groups or text_targets())
        assert set(idx) == set(orig) | {k + ".weight_scale" for k in targets}
        err = {}
        for k in targets:
            key = k + ".weight"
            w = read(d, idx, key)
            scale = read(d, idx, k + ".weight_scale")
            assert w.dtype == torch.int8 and scale.dtype == torch.float32
            assert (
                scale.shape == (w.shape[0], 1)
                and scale.isfinite().all()
                and (scale != 0).all()
            )
            rows = torch.linspace(0, w.shape[0] - 1, 5).long()
            fp = read(base, orig, key)[rows].float()
            dq = w[rows].float() * scale[rows]
            if groups:
                dq = invert(dq, groups[k])
            err[k] = float(
                torch.nn.functional.cosine_similarity(dq, fp, dim=-1, eps=1e-12).mean()
            )
        if groups:
            with safe_open(source_file(comp), framework="pt") as packed:
                packed_keys = set(packed.keys())
                for key in orig:
                    src_key = ("model." if comp == "text_encoder" else "") + key
                    segment = None
                    if src_key not in packed_keys:
                        stem, projection, suffix = src_key.rsplit(".", 2)
                        segment = ("q", "k", "v").index(projection)
                        src_key = stem + ".qkv." + suffix
                    pairs = [(key, src_key)]
                    if key.removesuffix(".weight") in groups:
                        pairs.append(
                            (
                                key.removesuffix(".weight") + ".weight_scale",
                                src_key.removesuffix(".weight") + ".weight_scale",
                            )
                        )
                    for dest_key, source_key in pairs:
                        expected = packed.get_tensor(source_key)
                        if segment is not None:
                            expected = expected.chunk(3)[segment]
                        actual = read(d, idx, dest_key)
                        assert actual.dtype == expected.dtype and torch.equal(
                            actual, expected
                        ), dest_key
        if mode in ("rtn", "gptq") and comp == "text_encoder":
            for key in [
                k
                for k in orig
                if k.startswith("visual.") or k == "language_model.embed_tokens.weight"
            ]:
                current = read(d, idx, key)
                reference = read(base, orig, key)
                assert (
                    current.dtype == reference.dtype == torch.bfloat16
                    and torch.equal(current, reference)
                )
        result[comp] = {
            "count": len(targets),
            "min_sampled_cosine": min(err.values()),
            "lowest": sorted(err.items(), key=lambda a: a[1])[:6],
        }
        print(mode, comp, result[comp], flush=True)
    results[mode] = result
if args.output:
    args.output.write_text(json.dumps(results, indent=2) + "\n")
