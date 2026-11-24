# SPDX-License-Identifier: Apache-2.0
"""Resident oneDNN BF16 projections for a 4096-token image on two Flex 170s."""
from dataclasses import dataclass
from weakref import ref

import torch
from torch import nn
from torch.nn import functional as F

from sglang.kernels.ops.diffusion.rope import qwen_image21_b60 as b60
from sglang.multimodal_gen.runtime.cache.conditioning import conditioning_weights_epoch
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    DENOISING_LOOP_STATE,
    FLEX_MLP_LAYOUTS_USED,
    get_forward_context_or_none,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.hardware_backend.xpu.kernels import flex_bf16_mlp as native

logger = init_logger(__name__)
_IMAGE = (1, 4096, 4096)
_MAX_LAYERS = 24
_KEY = "qwen_flex_bf16_persistent"


def _headroom(device):
    floor = 1536 << 20
    free, _ = torch.xpu.mem_get_info(device)
    if (
        free < floor
        and torch.xpu.memory_reserved(device) - torch.xpu.memory_allocated(device)
        >= 256 << 20
    ):
        torch.xpu.empty_cache()
        free, _ = torch.xpu.mem_get_info(device)
    return free >= floor


def begin(model, x, layouts, prefix_pad, sp):
    """Set request-local geometry once; module-owned packs survive request cleanup."""
    if (
        torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or x.shape != _IMAGE
        or x.device.type != "xpu"
        or x.dtype != torch.bfloat16
        or sp != 1
        or len(layouts) != 1
        or prefix_pad is not None
        or model.num_layers != 32
    ):
        return None
    ctx = get_forward_context_or_none()
    if ctx is None or not ctx.is_denoising or ctx.forward_batch is None:
        return None
    loop = ctx.forward_batch.extra.setdefault(DENOISING_LOOP_STATE, {})
    if _KEY not in loop:
        first = model.transformer_blocks[0]
        layer = first.attn.to_q
        group = getattr(layer, "tp_group", None)
        valid = (
            not model.training
            and b60.is_flex_device(x.device)
            and layer.tp_size == 2
            and layer.weight.dtype == torch.bfloat16
            and getattr(group, "flex_tp2", None) is not None
            and getattr(first.attn.target_attn.attn_impl, "qwen_packed_attention", None)
            is not None
            and not is_layerwise_offloaded_module(model)
        )
        loop[_KEY] = {"veto": set(), "changed": False} if valid else None
        if valid:
            _headroom(x.device)
            if hasattr(first.attn, "_flex_bf16_pack") or hasattr(
                first.img_mlp, "_flex_bf16_pack"
            ):
                ctx.forward_batch.extra[FLEX_MLP_LAYOUTS_USED] = True
    return loop[_KEY]


def finish(request_state, device):
    if request_state and request_state["changed"]:
        request_state["changed"] = False
        _headroom(device)


def _active(owner, x):
    if (
        not isinstance(x, torch.Tensor)
        or getattr(owner, "_flex_bf16_layer", _MAX_LAYERS) >= _MAX_LAYERS
        or torch.compiler.is_compiling()
        or torch.is_grad_enabled()
        or x.device.type != "xpu"
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
    ):
        return None
    ctx = get_forward_context_or_none()
    if ctx is None or not ctx.is_denoising or ctx.forward_batch is None:
        return None
    loop = ctx.forward_batch.extra.get(DENOISING_LOOP_STATE, {})
    return loop.get(_KEY)


def _unobserved(*modules):
    return not (
        nn.modules.module._global_forward_pre_hooks
        or nn.modules.module._global_forward_hooks
        or any(
            m.training
            or m._forward_pre_hooks
            or m._forward_hooks
            or m._compiled_call_impl is not None
            or "forward" in vars(m)
            for m in modules
        )
    )


def _can_build(owner, state, x):
    if id(owner) in state["veto"]:
        return False
    if not _headroom(x.device):
        state["veto"].add(id(owner))
        return False
    return True


def _stamp(layers):
    return _weight_stamp([m.weight for m in layers])


def _weight_stamp(weights):
    if any(w.is_inference() for w in weights):
        return None
    return (
        conditioning_weights_epoch(),
        tuple(
            (id(w), w._version, w.data_ptr(), w.shape, w.stride(), w.dtype, w.device)
            for w in weights
        ),
    )


def _host_parameter(old, data):
    # CPU canonical values remain visible to state_dict and to weight/adaptor loaders.
    with torch.inference_mode(False), torch.no_grad():
        weight = nn.Parameter(
            data.detach().to("cpu", copy=True), requires_grad=old.requires_grad
        )
    weight.__dict__.update(old.__dict__)
    return weight


@dataclass
class PackedWeight:
    data: torch.Tensor
    signature: tuple
    stream: object
    ready: object
    layers: tuple = ()

    def get(self, layers, device):
        if (
            self.data.device != device
            or self.signature is None
            or self.signature != _stamp(layers)
        ):
            return None
        stream = torch.xpu.current_stream(device)
        if stream != self.stream:
            stream.wait_event(self.ready)
            self.data.record_stream(stream)
        return self.data


class HostLinear(UnquantizedLinearMethod):
    """Individual calls, hooks, other image sizes and changed weights use canonical values."""

    def apply(self, layer, x, bias=None):
        w = layer.weight.to(x.device)
        if x.shape == _IMAGE and w.ndim == 2 and w.shape[1] == 4096:
            w = w.T.contiguous().T
        else:
            w = w.contiguous()
        return F.linear(x, w, bias)


def _install(owner, layers, packed, params, state):
    stream = torch.xpu.current_stream(packed.device)
    pack = PackedWeight(
        packed,
        _weight_stamp(params),
        stream,
        stream.record_event(),
        tuple(ref(m) for m in layers),
    )
    for layer, param in zip(layers, params, strict=True):
        layer.weight = param
        if not isinstance(layer.quant_method, PackedOutput):
            layer.quant_method = HostLinear()
    owner._flex_bf16_pack = pack
    state["changed"] = True
    get_forward_context_or_none().forward_batch.extra[FLEX_MLP_LAYOUTS_USED] = True


def _available(owner, layers, device):
    pack = getattr(owner, "_flex_bf16_pack", None)
    return None if pack is None else pack.get(layers, device)


def _decline(owner, state, exc=None):
    state["veto"].add(id(owner))
    if exc is not None:
        logger.warning("Skipping resident Flex BF16 projection: %s", exc)


def qkv(attn, x, column_type):
    layers = (attn.to_q, attn.to_k, attn.to_v)
    state = _active(attn, x)
    if state is None or not _unobserved(attn, *layers):
        return None
    if not all(type(layer) is ColumnParallelLinear for layer in layers):
        return None
    packed = _available(attn, layers, x.device)
    if x.shape != _IMAGE:
        if (
            packed is not None
            and x.ndim == 3
            and x.shape[0] == 1
            and 0 < x.shape[1] <= 128
            and x.shape[2] == 4096
            and all(type(layer.quant_method) is HostLinear for layer in layers)
        ):
            # HostLinear uses contiguous [N,K] weights at short M. Restore the
            # same BF16 words on-device and keep the three original reductions.
            matrix = native.unpack_weight(packed, 4096, 6144)
            return tuple(
                F.linear(x, part).unflatten(-1, (attn.heads, attn.head_dim))
                for part in matrix.chunk(3)
            )
        return None
    if packed is not None and all(
        type(layer.quant_method) is HostLinear for layer in layers
    ):
        raw = native.flex_bf16_projection(x, packed, 6144)
    else:
        if hasattr(attn, "_flex_bf16_pack") or not _can_build(attn, state, x):
            return None
        batched = column_type.for_image(x, layers)
        row = attn.to_out[0]
        if (
            batched is None
            or type(row) is not RowParallelLinear
            or type(row.quant_method) is not UnquantizedLinearMethod
        ):
            return None
        try:
            ref = torch.bmm(x.expand(3, -1, -1), batched)
            packed = native.pack_weight(torch.cat(batched.unbind(0), dim=1), 4096, 6144)
            raw = native.flex_bf16_projection(x, packed, 6144)
            if not all(
                torch.equal(a[0].view(torch.int16), b.view(torch.int16))
                for a, b in zip(raw.chunk(3, -1), ref.unbind(0), strict=True)
            ):
                _decline(attn, state, "QKV did not match the original BMM")
                return None
            params = [
                _host_parameter(layer.weight, layer.weight.contiguous())
                for layer in layers
            ]
        except (OSError, RuntimeError) as exc:
            _decline(attn, state, exc)
            return None
        _install(attn, layers, packed, params, state)
        row.quant_method = PackedOutput()
    return tuple(z.unflatten(-1, (attn.heads, attn.head_dim)) for z in raw.chunk(3, -1))


class PackedOutput(HostLinear):
    def apply(self, layer, x, bias=None):
        state = _active(layer, x)
        layers = (layer,)
        if (
            state is not None
            and x.ndim == 3
            and x.shape[0] == 1
            and 0 < x.shape[1] <= 128
            and x.shape[2] == 2048
            and bias is None
            and _unobserved(layer)
        ):
            packed = _available(layer, layers, x.device)
            if packed is not None:
                return F.linear(x, native.unpack_weight(packed, 2048, 4096))
        if (
            state is not None
            and x.shape == (1, 4096, 2048)
            and bias is None
            and _unobserved(layer)
        ):
            packed = _available(layer, layers, x.device)
            if packed is not None:
                return native.flex_bf16_projection(x, packed, 4096)
            if (
                not hasattr(layer, "_flex_bf16_pack")
                and layer.weight.device == x.device
                and layer.weight.shape == (4096, 2048)
                and layer.weight.is_contiguous()
                and _can_build(layer, state, x)
            ):
                ref = F.linear(x, layer.weight)
                try:
                    packed = native.pack_weight(layer.weight, 2048, 4096)
                    y = native.flex_bf16_projection(x, packed, 4096)
                    if not torch.equal(y.view(torch.int16), ref.view(torch.int16)):
                        _decline(
                            layer,
                            state,
                            "output projection did not match the original linear",
                        )
                        return ref
                    p = _host_parameter(layer.weight, layer.weight)
                except (OSError, RuntimeError) as exc:
                    _decline(layer, state, exc)
                    return ref
                _install(layer, layers, packed, (p,), state)
                return y
        return super().apply(layer, x, bias)


def mlp_hidden(mlp, x, column_type, down_type):
    state = _active(mlp, x)
    layers = (mlp.gate_layer, mlp.proj)
    if (
        state is None
        or not _unobserved(mlp, mlp.out, *layers)
        or type(mlp.out) is not RowParallelLinear
        or type(mlp.out.quant_method) is not down_type
        or not all(type(layer) is ColumnParallelLinear for layer in layers)
    ):
        return None
    packed = _available(mlp, layers, x.device)
    if packed is None:
        if (
            x.shape != _IMAGE
            or hasattr(mlp, "_flex_bf16_pack")
            or not _can_build(mlp, state, x)
        ):
            return None
        wide = column_type.for_image(x, layers)
        down = mlp.out.weight
        if wide is None or down.shape != (4096, 6144) or not down.is_contiguous():
            return None
        try:
            packed = native.pack_weight(wide, 4096, 12288)
            result = native._GATE.run(
                lambda: native.flex_bf16_gate_up(x, packed),
                lambda: x @ wide,
                sig=x.device,
                logger=logger,
            )
            if not native._GATE.is_verified(x.device) or native._GATE.disabled:
                _decline(mlp, state)
                return None
            with torch.inference_mode(False), torch.no_grad():
                host = wide.detach().to("cpu")
                params = []
                for layer, view in zip(layers, host.chunk(2, dim=1), strict=True):
                    param = nn.Parameter(
                        view.T, requires_grad=layer.weight.requires_grad
                    )
                    param.__dict__.update(layer.weight.__dict__)
                    params.append(param)
            transposed = down.T.contiguous().T
        except (OSError, RuntimeError) as exc:
            _decline(mlp, state, exc)
            return None
        _install(mlp, layers, packed, params, state)
        down.data = transposed
    elif not all(type(layer.quant_method) is HostLinear for layer in layers):
        return None
    elif x.shape == _IMAGE:
        result = native.flex_bf16_gate_up(x, packed)
    else:
        # oneDNN's short prefix dispatch uses the checkpoint [N,K] layout.
        gate, up = native.unpack_gate_up(packed).chunk(2, dim=0)
        return F.silu(F.linear(x, gate)) * F.linear(x, up)
    gate, up = result.chunk(2, dim=-1)
    return b60.fused_pointwise(gate, up, input_row_padding=6144, output_row_padding=16)


def retire(owner, layers, *, fn=None):
    """Reconstitute canonical device placement before model movement or a reload."""
    pack = getattr(owner, "_flex_bf16_pack", None)
    if pack is None:
        return
    if getattr(pack, "layers", ()):
        layers = [target for saved in pack.layers if (target := saved()) is not None]
    if not layers:
        del owner._flex_bf16_pack
        return
    device = pack.data.device
    # .cpu() / .to_empty(meta) can release the derived XPU storage directly.
    # dtype-only conversions must retain the original logical device.
    source_device = device
    if fn is not None:
        sentinel = torch.empty(0, device=device, dtype=layers[0].weight.dtype)
        target = fn(sentinel).device
        if target.type in ("cpu", "meta"):
            source_device = torch.device("cpu")
    params = []
    for layer in layers:
        w = layer.weight
        if w.device.type == "cpu" and source_device.type != "cpu":
            with torch.inference_mode(False), torch.no_grad():
                p = nn.Parameter(
                    w.detach().to(source_device), requires_grad=w.requires_grad
                )
                p.__dict__.update(w.__dict__)
        else:
            p = w
        params.append(p)
    for layer, p in zip(layers, params, strict=True):
        layer.weight = p
        if type(layer.quant_method) in (HostLinear, PackedOutput):
            layer.quant_method = UnquantizedLinearMethod()
    del owner._flex_bf16_pack
