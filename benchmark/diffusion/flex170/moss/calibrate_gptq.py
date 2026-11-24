"""Layerwise calibration of the MOSS text-only path with llm-compressor 0.15.0."""

from unittest.mock import patch
import argparse
import json
import os
import sys
from pathlib import Path

# MoviePy needs Pillow <12, while llm-compressor 0.15 needs Pillow >=12.
# Both wheels live in the requested venv, with the latter scoped to calibration.
sys.path.insert(0, str(Path(sys.prefix) / "opt" / "llmcompressor"))

import torch
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.utils.helpers import calculate_qparams
from llmcompressor.modifiers.gptq.gptq_quantize import quantize_weight
from llmcompressor.modifiers.gptq.helpers import accumulate_hessian, factorize_hessian
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)
from export_moss import BASE, CALIBRATION, text_targets

root = CALIBRATION
src = BASE / "text_encoder"


def original(key):
    with safe_open(src / index[key], framework="pt") as f:
        return f.get_tensor(key)


def paths(stage, idx):
    return root / stage / f"{idx:02d}.safetensors"


def save_atomic(path, tensors):
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, tmp)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def share_groups(idx):
    # A hook on the first projection observes exactly the inputs of its siblings.
    if idx % 4 == 3:
        attn = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
        output = "self_attn.o_proj"
    else:
        attn = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]
        output = "linear_attn.out_proj"
    return [attn, [output], ["mlp.gate_proj", "mlp.up_proj"], ["mlp.down_proj"]]


def stats(idx, dev):
    cfg = AutoConfig.from_pretrained(src, local_files_only=True).text_config
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        block = Qwen3_5DecoderLayer(cfg, idx)
    prefix = f"language_model.layers.{idx}."
    block.load_state_dict(
        {k[len(prefix) :]: original(k) for k in index if k.startswith(prefix)},
        assign=True,
    )
    block = block.eval().to(dev)
    # Qwen3.5 uses the exact decoder-layer forward and three-axis text RoPE.
    rope = Qwen3_5TextRotaryEmbedding(cfg, device=dev)
    ids = torch.arange(2048, device=dev)[None, None, :].expand(3, 1, -1)
    position = rope(torch.empty((), device=dev, dtype=torch.bfloat16), ids)
    pre = None
    if idx == 0:
        pre = original("language_model.embed_tokens.weight")
        samples = json.loads((root / "packed.json").read_text())["samples"]
        get_input = lambda n: torch.nn.functional.embedding(
            torch.tensor(samples[n]["input_ids"]), pre
        )
    else:
        previous = load_file(paths("hidden", idx - 1))["hidden"]
        get_input = lambda n: previous[n]
    # 4 GiB on the host. Only a single sequence and one layer live on the XPU.
    hidden = torch.empty(256, 2048, 4096, dtype=torch.bfloat16)
    collected = {}
    handles = []
    need_stats = not paths("stats", idx).exists()
    for group in share_groups(idx) if need_stats else []:
        name = group[0]
        module = block.get_submodule(name)
        k = module.weight.shape[1]
        h = torch.zeros((k, k), device=dev, dtype=torch.float32)
        count = torch.zeros((), device=dev, dtype=torch.float32)
        lo = torch.full((k,), float("inf"), device=dev)
        hi = -lo

        def hook(mod, args, h=h, count=count, lo=lo, hi=hi):
            x = args[0]
            accumulate_hessian(x, mod, h, count)
            x = x.flatten(0, -2)
            torch.minimum(lo, x.amin(0), out=lo)
            torch.maximum(hi, x.amax(0), out=hi)

        handles.append(module.register_forward_pre_hook(hook))
        collected[name] = (h, count, lo, hi)
    for n in range(256):
        x = get_input(n).unsqueeze(0).to(dev)
        hidden[n] = (
            block(x, position_embeddings=position, position_ids=ids[0]).squeeze(0).cpu()
        )
        if (n + 1) % 32 == 0 or n == 0:
            print("calibrated", idx, n + 1, flush=True)
    if not torch.isfinite(hidden.reshape(-1, 4096)[::128]).all():
        raise ValueError(f"non-finite hidden states at layer {idx}")
    output = {}
    for name, (h, count, lo, hi) in collected.items():
        assert count.item() == 256
        output[name + ".H"] = (h / count).cpu()
        output[name + ".range"] = (hi - lo).cpu()
    save_atomic(paths("hidden", idx), {"hidden": hidden})
    if need_stats:
        save_atomic(paths("stats", idx), output)
    if idx >= 2:
        paths("hidden", idx - 2).unlink(missing_ok=True)


def quantize(idx, dev):
    moments = load_file(paths("stats", idx))
    prefix = f"language_model.layers.{idx}."
    smooth = SmoothQuantModifier(smoothing_strength=0.5)
    qargs = QuantizationArgs(
        num_bits=8, type="int", strategy="channel", symmetric=True, dynamic=False
    )
    result = {}
    for j, group in enumerate(share_groups(idx)):
        h = moments[group[0] + ".H"]
        balance = list(group)
        if j == 0 and idx % 4 != 3:
            # The BF16 gate projections also consume the same norm output.
            balance += ["linear_attn.in_proj_a", "linear_attn.in_proj_b"]
        weights = {n: original(prefix + n + ".weight").to(dev) for n in balance}
        if j in (0, 2):
            modules = []
            for w in weights.values():
                mod = torch.nn.Module()
                mod.weight = torch.nn.Parameter(w, requires_grad=False)
                modules.append(mod)
            scales = (
                smooth._calculate_smoothing_scales(
                    modules, moments[group[0] + ".range"].to(dev)
                )
                .float()
                .clamp_min(1e-5)
            )
            norm = prefix + (
                "input_layernorm.weight"
                if j == 0
                else "post_attention_layernorm.weight"
            )
            # HF Qwen3.5 RMSNorm multiplies by 1 + weight.
            gamma = original(norm).float().to(dev)
            result[norm] = ((1 + gamma) / scales - 1).bfloat16().cpu()
            cpu_scales = scales.cpu()
            h = h / cpu_scales[:, None] / cpu_scales[None, :]
            weights = {n: (w.float() * scales).bfloat16() for n, w in weights.items()}
        # oneMKL on this Flex driver falls back to host Cholesky. Factorize on
        # the host once per shared input and reuse the unmodified factor.
        fallback_flag = torch.zeros(1, dtype=torch.bool)
        dead = h.diagonal() == 0
        inverse = factorize_hessian(
            torch.zeros(1, 1, h.shape[0]), h.clone()[None], 0.01, fallback_flag
        )
        if fallback_flag.item() or not inverse.isfinite().all():
            raise ValueError(f"bad Hessian {idx} {group}")

        def factored(w, hess, damp, status):
            w.masked_fill_(dead.to(w.device)[None, None, :], 0)
            return inverse.to(w.device)

        for n, w in weights.items():
            key = prefix + n
            if key not in targets:
                result[key + ".weight"] = w.cpu()
                continue
            wf = w.float()
            scale, zp = calculate_qparams(
                wf.amin(-1, keepdim=True), wf.amax(-1, keepdim=True), qargs
            )

            def run_gptq(device):
                with patch(
                    "llmcompressor.modifiers.gptq.gptq_quantize.factorize_hessian",
                    factored,
                ):
                    return quantize_weight(
                        w.to(device, torch.float32, copy=True).unsqueeze(0),
                        h.unsqueeze(0).to(device),
                        scale.to(device).unsqueeze(0),
                        zp.to(device).unsqueeze(0),
                        None,
                        qargs,
                        blocksize=128,
                        percdamp=0.01,
                    )

            dq, loss, fallback = run_gptq(dev)
            if fallback.any() or not loss.isfinite().all() or not dq.isfinite().all():
                print(
                    "Retrying GPTQ with CPU linear algebra:",
                    key,
                    "fallback",
                    fallback.tolist(),
                    "finite loss",
                    loss.isfinite().sum().item(),
                    "/",
                    loss.numel(),
                    flush=True,
                )
                dq, loss, fallback = run_gptq("cpu")
                scale = scale.cpu()
            if fallback.any() or not loss.isfinite().all() or not dq.isfinite().all():
                raise RuntimeError(f"GPTQ factorization failed for {key}")
            rel = (dq[0] - w.to(dq.device).float()).norm() / w.float().norm().to(
                dq.device
            )
            if not (0 <= rel.item() < 0.15):
                raise ValueError(f"unexpected GPTQ relative weight error {key}: {rel}")
            result[key + ".weight"] = (
                (dq[0] / scale).round().clamp(-128, 127).to(torch.int8).cpu()
            )
            result[key + ".weight_scale"] = scale.cpu()
            print("GPTQ", key, flush=True)
    save_atomic(paths("gptq", idx), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["stats", "gptq"])
    parser.add_argument("--device", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=32)
    args = parser.parse_args()
    index = json.loads(next(src.glob("*index.json")).read_text())["weight_map"]
    targets = set(text_targets())
    torch.set_num_threads(8)
    with torch.inference_mode():
        for idx in range(args.start, args.stop):
            if args.stage == "gptq" and paths("gptq", idx).exists():
                continue
            if (
                args.stage == "stats"
                and paths("stats", idx).exists()
                and paths("hidden", idx).exists()
            ):
                continue
            if args.stage == "gptq" and not paths("stats", idx).exists():
                raise FileNotFoundError(paths("stats", idx))
            (stats if args.stage == "stats" else quantize)(idx, args.device)
