"""Create MOSS variants, preserving source components and Comfy's packed values."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODELS = Path(os.environ.get("FLEX_MODELS_DIR", Path.home() / "models"))
CALIBRATION = Path(os.environ.get("FLEX_CALIBRATION_DIR", "calibration")).resolve()
BASE = MODELS / "MOSS-Image-SFT"
COMFY = MODELS / "MOSS-Image"
NAMES = {
    "both": "DiT-text_encoder-int8-convrot-w8a8",
    "comfy": "text_encoder-int8-convrot-w8a8",
    "rtn": "text_encoder-int8-autoround-w8a8",
    "gptq": "text_encoder-int8-gptq-w8a8",
}


def destination(key):
    return MODELS / (BASE.name + "-" + NAMES[key])


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def source_file(component):
    directory = {"transformer": "diffusion_models", "text_encoder": "text_encoders"}[
        component
    ]
    (path,) = (COMFY / directory).glob("*.safetensors")
    return path


def text_targets():
    with safe_open(source_file("text_encoder"), framework="pt") as packed:
        return sorted(
            k.removeprefix("model.").removesuffix(".weight")
            for k in packed.keys()
            if k.startswith("model.language_model.layers.")
            and k.endswith(".weight")
            and packed.get_slice(k).get_dtype() == "I8"
        )


def ct_config(targets):
    base = dict(type="int", num_bits=8, symmetric=True)
    group = dict(
        targets=sorted(targets),
        format="int-quantized",
        weights=dict(base, strategy="channel", dynamic=False),
        input_activations=dict(base, strategy="token", dynamic=True),
    )
    return dict(
        quant_method="compressed-tensors",
        format="int-quantized",
        quantization_status="compressed",
        config_groups={"group_0": group},
        ignore=[],
    )


def link(src, dst):
    return os.link(Path(src).resolve(), dst)


def components(stage, exclude):
    stage.mkdir()
    for src in BASE.iterdir():
        if src.name in exclude or src.name.startswith("."):
            continue
        dst = stage / src.name
        if src.is_dir():
            shutil.copytree(src, dst, copy_function=link)
        else:
            link(src, dst)


def export_component(stage, component, mode, gptq=None):
    source = BASE / component
    out = stage / component
    out.mkdir()
    (indexfile,) = source.glob("*.safetensors.index.json")
    original = json.loads(indexfile.read_text())["weight_map"]
    targets = set(text_targets()) if mode in ("rtn", "gptq") else set()
    groups, written, total = {}, {}, 0
    for shard in sorted(set(original.values())):
        tensors = {}
        with ExitStack() as ctx:
            raw = ctx.enter_context(safe_open(source / shard, framework="pt"))
            if mode == "comfy":
                packed = ctx.enter_context(
                    safe_open(source_file(component), framework="pt")
                )
            for name in raw.keys():
                if original[name] != shard:
                    raise ValueError(name)
                prefix = name.removesuffix(".weight")
                if mode == "comfy":
                    ck = ("model." if component == "text_encoder" else "") + name
                    split = None
                    if ck not in packed.keys():
                        stem, proj, suffix = ck.rsplit(".", 2)
                        if (
                            component != "transformer"
                            or proj not in ("q", "k", "v")
                            or suffix != "weight"
                        ):
                            raise KeyError(ck)
                        ck = stem + ".qkv.weight"
                        split = ("q", "k", "v").index(proj)
                    w = packed.get_tensor(ck)
                    if w.dtype == torch.int8:
                        root = ck.removesuffix(".weight")
                        mark = json.loads(
                            bytes(packed.get_tensor(root + ".comfy_quant").tolist())
                        )
                        assert mark["convrot"] and mark["format"] == "int8_tensorwise"
                        groups[prefix] = mark["convrot_groupsize"]
                        scale = packed.get_tensor(root + ".weight_scale")
                        assert scale.dtype == torch.float32 and scale.shape == (
                            w.shape[0],
                            1,
                        )
                        if split is not None:
                            w, scale = w.chunk(3)[split], scale.chunk(3)[split]
                        tensors[prefix + ".weight_scale"] = scale
                elif mode == "rtn" and prefix in targets and name.endswith(".weight"):
                    from auto_round.data_type.int import quant_tensor_rtn_sym

                    w = raw.get_tensor(name)
                    # The v0.16.1 RTN implementation returns dequantized weights,
                    # per-channel scales and an optional zero point.
                    chunks, scales = [], []
                    for rows in w.split(512):
                        dq, scale, zp = quant_tensor_rtn_sym(
                            rows.float(),
                            bits=8,
                            group_size=-1,
                            scale_dtype=torch.float32,
                        )
                        assert torch.isfinite(scale).all() and (scale != 0).all()
                        chunks.append(
                            (dq / scale.reshape(-1, 1))
                            .round()
                            .clamp(-128, 127)
                            .to(torch.int8)
                        )
                        scales.append(scale.reshape(-1, 1))
                    w = torch.cat(chunks)
                    tensors[prefix + ".weight_scale"] = torch.cat(scales)
                elif mode == "gptq" and name in gptq:
                    part = ctx.enter_context(safe_open(gptq[name], framework="pt"))
                    w = part.get_tensor(name)
                    if prefix in targets:
                        tensors[prefix + ".weight_scale"] = part.get_tensor(
                            prefix + ".weight_scale"
                        )
                        assert w.dtype == torch.int8
                else:
                    if mode == "gptq" and prefix in targets:
                        raise KeyError(name)
                    w = raw.get_tensor(name)
                assert list(w.shape) == raw.get_slice(name).get_shape(), name
                tensors[name] = w
            save_file(tensors, out / shard, metadata={"format": "pt"})
            written.update(dict.fromkeys(tensors, shard))
            total += sum(v.numel() * v.element_size() for v in tensors.values())
        print(component, shard, flush=True)
        del tensors
    qcfg = (
        dict(quant_method="convrot_int8", layer_groups=groups)
        if mode == "comfy"
        else ct_config(targets)
    )
    assert len(groups if mode == "comfy" else targets) == (
        252 if component == "transformer" else 309 if mode == "comfy" else 200
    )
    config = json.loads((source / "config.json").read_text())
    config["quantization_config"] = qcfg
    json_write(out / "config.json", config)
    json_write(
        out / indexfile.name,
        dict(metadata={"total_size": total}, weight_map=dict(sorted(written.items()))),
    )
    for file in source.glob("*.py"):
        link(file, out / file.name)


def export(mode, gptq_dir=None):
    dst = destination(mode)
    if dst.exists():
        raise FileExistsError(dst)
    with tempfile.TemporaryDirectory(dir=MODELS, prefix=".moss-export-") as tmp:
        stage = Path(tmp) / "model"
        components(
            stage,
            {"text_encoder", "transformer"} if mode == "both" else {"text_encoder"},
        )
        if mode == "both":
            if (destination("comfy") / "text_encoder").exists():
                shutil.copytree(
                    destination("comfy") / "text_encoder",
                    stage / "text_encoder",
                    copy_function=link,
                )
            else:
                export_component(stage, "text_encoder", "comfy")
            export_component(stage, "transformer", "comfy")
        else:
            tables = {}
            if mode == "gptq":
                for f in sorted(gptq_dir.glob("*.safetensors")):
                    with safe_open(f, framework="pt") as s:
                        for k in s.keys():
                            if k in tables:
                                raise ValueError(k)
                            tables[k] = f
            export_component(stage, "text_encoder", mode, tables)
        stage.rename(dst)
    print(dst, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modes", choices=NAMES, nargs="+")
    parser.add_argument("--gptq-dir", type=Path, default=CALIBRATION / "gptq")
    args = parser.parse_args()
    torch.set_num_threads(8)
    for mode in args.modes:
        export(mode, args.gptq_dir)
