"""Repackage original Comfy-Org tensor ranges into the SGLang diffusers layout."""

import argparse
import hashlib, json, os, shutil, struct, tempfile
from pathlib import Path
from collections import Counter


def link(src, dst):
    return os.link(Path(src).resolve(), dst)


def header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def schema(component):
    orig = {}
    for p in component.glob("*.safetensors"):
        with p.open("rb") as f:
            orig.update(header(f)[0])
    return orig


def package(file, src, dst, component):
    dst.mkdir()
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and "index.json" not in f.name:
            shutil.copy2(f, dst / f.name)
    expected = schema(src)
    markers = {}
    cmeta = {}
    out = {}
    ranges = []
    offset = 0
    with file.open("rb") as f:
        tensors, base = header(f)
        for key, t in tensors.items():
            name = key
            if component == "text_encoder" and key.startswith(
                ("model.embed_tokens.", "model.layers.", "model.norm.")
            ):
                name = "model.language_model." + key[6:]
            a, b = t["data_offsets"]
            shape = t["shape"]
            names = (
                [
                    name.replace(".img_mlp.gate_up.", ".img_mlp." + p + ".")
                    for p in ("gate_layer", "proj")
                ]
                if ".img_mlp.gate_up." in name
                else [name]
            )
            if key.endswith((".comfy_quant", ".comfy_attention.config")):
                f.seek(base + a)
                value = json.loads(f.read(b - a))
                if key.endswith(".config"):
                    cmeta[key] = value
                else:
                    assert (
                        value["format"] == "int8_tensorwise"
                        and value["convrot"] is True
                    )
                    for n in names:
                        markers[n.removesuffix(".comfy_quant")] = value[
                            "convrot_groupsize"
                        ]
                continue
            size = (b - a) // len(names)
            assert size * len(names) == b - a
            for i, n in enumerate(names):
                sh = [shape[0] // 2, *shape[1:]] if len(names) > 1 else shape
                assert n not in out
                out[n] = dict(
                    dtype=t["dtype"], shape=sh, data_offsets=[offset, offset + size]
                )
                ranges.append((base + a + size * i, size))
                offset += size
        assert {k for k in out if not k.endswith(".weight_scale")} == set(expected)
        assert all(v["shape"] == out[k]["shape"] for k, v in expected.items())
        assert set(markers) == {
            k.removesuffix(".weight") for k, v in out.items() if v["dtype"] == "I8"
        }
        for k in markers:
            assert (
                out[k + ".weight_scale"]["shape"] == [out[k + ".weight"]["shape"][0], 1]
                and out[k + ".weight_scale"]["dtype"] == "F32"
            )
        assert len(markers) == (224 if component == "transformer" else 254)
        cfg = json.loads((dst / "config.json").read_text())
        cfg["quantization_config"] = dict(
            quant_method="convrot_int8", layer_groups=markers
        )
        (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        if cmeta:
            (dst / "comfy_attention_config.json").write_text(
                json.dumps(cmeta, indent=2) + "\n"
            )
        target = dst / (
            "diffusion_pytorch_model.safetensors"
            if component == "transformer"
            else "model.safetensors"
        )
        data = json.dumps(
            {"__metadata__": {"format": "pt"}, **out}, separators=(",", ":")
        ).encode()
        data += b" " * (-len(data) % 8)
        digest = hashlib.sha256()
        with target.open("wb") as g:
            g.write(struct.pack("<Q", len(data)) + data)
            for start, size in ranges:
                f.seek(start)
                while size:
                    chunk = f.read(min(size, 8 << 20))
                    assert chunk
                    g.write(chunk)
                    digest.update(chunk)
                    size -= len(chunk)
        with target.open("rb") as g:
            h, b = header(g)
            assert h == out
            assert hashlib.file_digest(g, "sha256").digest() == digest.digest()
        return dict(
            file=file.name,
            tensors=len(out),
            int8_layers=len(markers),
            groups=dict(Counter(markers.values())),
            payload_sha256=digest.hexdigest(),
            attention_markers=len(cmeta),
        )


def export(root, variants):
    comfy = root / "ComfyUI-Qwen-Image-2.1"
    records = {}
    for turbo in variants:
        model = root / ("Qwen-Image-2.1" + ("-Turbo" if turbo else ""))
        target = model.with_name(model.name + "-int8-convrot-w8a8")
        if target.exists():
            raise FileExistsError(target)
        with tempfile.TemporaryDirectory(dir=root, prefix=".qwen21-export-") as temp:
            stage = Path(temp) / "model"
            stage.mkdir()
            for n in (
                "model_index.json",
                "scheduler",
                "processor",
                "vae",
                "configuration.json",
                "LICENSE",
            ):
                f = model / n
                if f.is_dir():
                    shutil.copytree(f, stage / n, copy_function=link)
                else:
                    link(f, stage / n)
            rec = {}
            for component in ("transformer", "text_encoder"):
                f = (
                    comfy / "text_encoders/qwen3vl_8b_int8_convrot.safetensors"
                    if component == "text_encoder"
                    else comfy
                    / "diffusion_models"
                    / (
                        "qwen_image_2.1_"
                        + ("turbo_" if turbo else "")
                        + "int8_convrot.safetensors"
                    )
                )
                if (
                    turbo
                    and component == "text_encoder"
                    and "Qwen-Image-2.1" in records
                ):
                    first = root / "Qwen-Image-2.1-int8-convrot-w8a8" / component
                    assert (
                        schema(model / component).keys()
                        == schema(root / "Qwen-Image-2.1" / component).keys()
                    )
                    shutil.copytree(
                        first,
                        stage / component,
                        copy_function=shutil.copy2,
                        ignore=shutil.ignore_patterns("*.safetensors"),
                    )
                    link(
                        first / "model.safetensors",
                        stage / component / "model.safetensors",
                    )
                    new = json.loads((model / component / "config.json").read_text())
                    new["quantization_config"] = json.loads(
                        (first / "config.json").read_text()
                    )["quantization_config"]
                    (stage / component / "config.json").write_text(
                        json.dumps(new, indent=2) + "\n"
                    )
                    shutil.copy2(
                        model / component / "generation_config.json",
                        stage / component / "generation_config.json",
                    )
                    rec[component] = records["Qwen-Image-2.1"][component]
                else:
                    rec[component] = package(
                        f, model / component, stage / component, component
                    )
                print(model.name, component, rec[component], flush=True)
            for w in (model / "vae").glob("*.safetensors"):
                assert w.samefile(stage / "vae" / w.name)
            stage.rename(target)
            records[model.name] = rec
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ.get("FLEX_MODELS_DIR", Path.home() / "models")),
    )
    parser.add_argument("--variant", choices=("base", "turbo", "both"), default="both")
    parser.add_argument(
        "--manifest", type=Path, help="Save payload checksums and quantization metadata"
    )
    args = parser.parse_args()
    variants = {"base": (False,), "turbo": (True,), "both": (False, True)}[args.variant]
    records = export(args.models_dir, variants)
    report = json.dumps(records, indent=2)
    if args.manifest:
        args.manifest.write_text(report + "\n")
    print(report)
