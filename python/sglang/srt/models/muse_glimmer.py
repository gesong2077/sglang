# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import os
from array import array
from typing import Iterable, List, Optional, Tuple

import torch

_use_xpu_qk_norm_rope = os.environ.get("SGLANG_XPU_MUSE_QK_NORM_ROPE") == "1"
from torch import nn
from transformers.activations import ACT2FN
from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_position_ids,
    get_vision_window_index,
)

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.vision import (
    VisionAttention,
    VisionAttentionMetadata,
    prepare_vision_attention_metadata,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    general_mm_embed_routine,
    pad_mm_tokens,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.utils import WeightsMapper, apply_qk_norm, permute_inv
from sglang.srt.runtime_context import get_forward, get_parallel
from sglang.srt.utils import add_prefix, is_cpu, is_cuda, is_xpu

_is_cuda = is_cuda()
_is_cpu = is_cpu()
_xpu_replica_layers = int(os.environ.get("SGLANG_XPU_MUSE_REPLICA_LAYERS", "0"))
_xpu_streamed_layers = int(os.environ.get("SGLANG_XPU_MUSE_STREAMED_MLP_LAYERS", "0"))
_fuse_replica_sum = (
    os.environ.get("SGLANG_XPU_MUSE_REPLICA_SUM_NORM") == "1"
    and os.environ.get("SGLANG_XPU_MUSE_FUSED_NORM") == "1"
    and bool(int(os.environ.get("SGLANG_XPU_MUSE_MLP_EXCHANGE_SUM", "0")))
)
_host_memory_audit = os.environ.get("SGLANG_XPU_MUSE_HOST_MEMORY_AUDIT")
_xpu_o_replica_layers = int(os.environ.get("SGLANG_XPU_MUSE_O_REPLICA_LAYERS", "0"))
_use_xpu_packed_o = is_xpu() and os.environ.get("SGLANG_XPU_MUSE_PACKED_O_EXCHANGE") == "1"
if _xpu_o_replica_layers:
    if not is_xpu() or os.environ.get("SGLANG_XPU_MUSE_SEQUENCE_PARALLEL") != "1" or os.environ.get("SGLANG_XPU_MUSE_SEQUENCE_ALIGNMENT", "2") != "2":
        raise ValueError("o_proj replicas require XPU sequence parallel with alignment 2")
    from sglang.srt.hardware_backend.xpu.kernels.muse_o_replica import (
        install_peer_o_shards, local_replicated_o_proj,
    )
_xpu_mlp_tile = int(os.environ.get("SGLANG_XPU_MUSE_MLP_TILE", "0"))
if _xpu_mlp_tile < 0:
    raise ValueError("Muse MLP tile must be non-negative")
if _xpu_mlp_tile:
    from sglang.srt.hardware_backend.xpu.kernels.muse_mlp_tile import tiled_mlp
if _xpu_replica_layers < 0:
    raise ValueError("Muse replica layer count must be non-negative")
if _xpu_replica_layers:
    if not is_xpu():
        raise ValueError("Muse MLP replicas are only supported on XPU")
    from sglang.srt.hardware_backend.xpu.kernels.muse_replica import (
        install_peer_mlp_shards, local_replicated_mlp,
    )

_use_fused_xpu_norm = is_xpu() and os.environ.get("SGLANG_XPU_MUSE_FUSED_NORM") == "1"
_xpu_input_norm_mode = int(os.environ.get("SGLANG_XPU_MUSE_INPUT_NORM_QUANT", "0"))
_use_xpu_input_norm_quant = is_xpu() and _xpu_input_norm_mode in (1, 2)
if _use_xpu_input_norm_quant:
    from sglang.srt.hardware_backend.xpu.kernels.muse_input_norm import input_norm_quant
if _use_fused_xpu_norm:
    from sglang.srt.hardware_backend.xpu.kernels.muse_norm import fused_post_norm

_use_fused_xpu_gate = is_xpu() and os.environ.get("SGLANG_XPU_MUSE_FUSED_GATE") == "1"
if _use_fused_xpu_gate:
    from sglang.srt.hardware_backend.xpu.kernels.muse_gate import fused_gate

_use_fused_xpu_mlp = is_xpu() and os.environ.get("SGLANG_XPU_MUSE_FUSED_MLP") == "1"
_use_fused_xpu_gate_quant = (
    is_xpu() and os.environ.get("SGLANG_XPU_MUSE_FUSED_GATE_QUANT") == "1"
)
_use_xpu_sequence = (
    is_xpu() and os.environ.get("SGLANG_XPU_MUSE_SEQUENCE_PARALLEL") == "1"
)
_use_xpu_merged_qkv = (
    is_xpu() and os.environ.get("SGLANG_XPU_MUSE_MERGED_QKV") == "1"
)
_xpu_sequence_alignment = int(
    os.environ.get("SGLANG_XPU_MUSE_SEQUENCE_ALIGNMENT", "2")
)
if _use_xpu_sequence:
    from sglang.srt.hardware_backend.xpu.kernels.muse_sequence import (
        sequence_parallel_post_attention,
    )
if _use_fused_xpu_mlp or _use_fused_xpu_gate_quant:
    from sglang.srt.hardware_backend.xpu.quantization.w8a8_int8_kernels import (
        W8A8Activation,
        dynamic_w8a8_linear,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )

if _is_cuda:
    from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
elif _is_cpu:
    _fused_sigmoid_mul_cpu = torch.ops.sgl_kernel.fused_sigmoid_mul_cpu

    def fused_sigmoid_mul(x, gate, inplace=True):
        if not inplace:
            x = x.clone()
        _fused_sigmoid_mul_cpu(x, gate)
        return x


logger = logging.getLogger(__name__)

# The four per-layer norms whose checkpoint weight is an offset from 1.0.
_OFFSET_NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attn_norm.weight",
    "post_attention_layernorm.weight",
    "post_ffn_norm.weight",
)

_VISION_NAME_FRAGMENTS = (
    "vision_encoder",
    "vision_tower",
    "vision_adapter",
    "vision_projection",
    "perception_emb_norm",
)

# Shared by _vendor_weight_name and hf_to_sglang_mapper;
# a rule missing from one silently breaks the other.
_VENDOR_TO_SGLANG = WeightsMapper(
    orig_to_new_prefix={
        "model.language_model.": "model.",
        # The vision modules hang off the entry class, not off ``model``.
        "model.vision_": "vision_",
    },
    # Only the first matching substring is applied; keep these non-overlapping.
    orig_to_new_substr={
        "post_attention_layernorm": "post_attn_norm",
        "pre_feedforward_layernorm": "post_attention_layernorm",
        "post_feedforward_layernorm": "post_ffn_norm",
        "self_attn.gate_proj": "self_attn.output_gate_proj",
    },
)


def _vendor_weight_name(name: str) -> str:
    return _VENDOR_TO_SGLANG.apply_list([name])[0]


def get_attention_sliding_window_size(config) -> int:
    return config.sliding_window - 1


class MuseGlimmerMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Muse Glimmer expects hidden_act=silu, got {config.hidden_act}"
            )
        self.act_fn = SiluAndMul()
        scheme = getattr(self.gate_up_proj, "scheme", None)
        self.fused_xpu_mlp = (
            _use_fused_xpu_mlp
            and isinstance(scheme, CompressedTensorsW8A8Int8)
            and not scheme.is_static_input_scheme
            and scheme.input_symmetric
            and not self.gate_up_proj.gather_output
            and self.gate_up_proj.bias is None
        )

    def forward(self, x: torch.Tensor, *, skip_all_reduce=False) -> torch.Tensor:
        if (
            self.fused_xpu_mlp and skip_all_reduce and _xpu_mlp_tile
            and isinstance(x, W8A8Activation) and x.shape[0] > _xpu_mlp_tile
        ):
            return tiled_mlp(
                x, self.gate_up_proj.weight, self.gate_up_proj.weight_scale,
                self.down_proj.weight, self.down_proj.weight_scale, _xpu_mlp_tile,
            )
        if self.fused_xpu_mlp and x.dtype == torch.bfloat16:
            x = dynamic_w8a8_linear(
                x,
                self.gate_up_proj.weight,
                self.gate_up_proj.weight_scale,
                silu_mul=True,
            )
        else:
            gate_up, _ = self.gate_up_proj(x)
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, skip_all_reduce=skip_all_reduce)
        return x


class MuseGlimmerAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_parallel().tp_size
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling = config.qk_scale_factor / self.head_dim

        self.use_rope = config.no_rope_layers[layer_id] == 1
        self.is_sliding = config.layer_types[layer_id] == "sliding_attention"
        self.use_qk_norm = config.use_qk_norm
        self.use_output_gate = config.use_attn_output_gate

        # Mixed quantization normally needs separate modules. When all three
        # projections are explicitly excluded, they can share one BF16 GEMM.
        merge_unquantized_qkv = False
        if _use_xpu_merged_qkv and quant_config is not None:
            from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
                CompressedTensorsConfig,
            )
            from sglang.srt.layers.quantization.compressed_tensors.utils import (
                should_ignore_layer,
            )

            merge_unquantized_qkv = isinstance(
                quant_config, CompressedTensorsConfig
            ) and all(
                should_ignore_layer(add_prefix(name, prefix), quant_config.ignore)
                for name in ("q_proj", "k_proj", "v_proj")
            )
        self.unfused_qkv = quant_config is not None
        self.decode_merged_qkv = merge_unquantized_qkv
        if merge_unquantized_qkv:
            self.unfused_qkv = False
        if self.unfused_qkv:
            if self.total_num_kv_heads < tp_size:
                raise ValueError(
                    "Muse Glimmer unfused q/k/v needs num_key_value_heads >= tp_size "
                    f"(got {self.total_num_kv_heads} kv heads, tp_size={tp_size}); "
                    "the KV-head replication QKVParallelLinear does is not "
                    "reproduced here."
                )
            self.q_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
            )
            self.k_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("k_proj", prefix),
            )
            self.v_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_kv_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("v_proj", prefix),
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                config.hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=None if merge_unquantized_qkv else quant_config,
                prefix=add_prefix("qkv_proj", prefix),
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        if self.use_output_gate:
            self.output_gate_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("output_gate_proj", prefix),
            )

        self.fused_xpu_gate_quant = (
            _use_fused_xpu_gate_quant
            and self.use_output_gate
            and self.o_proj.input_is_parallel
            and not self.output_gate_proj.gather_output
            and all(
                isinstance(getattr(layer, "scheme", None), CompressedTensorsW8A8Int8)
                and not layer.scheme.is_static_input_scheme
                and layer.scheme.input_symmetric
                and layer.bias is None
                for layer in (self.output_gate_proj, self.o_proj)
            )
        )

        self.qk_norm = (
            RMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
                has_weight=False,
            )
            if self.use_qk_norm
            else None
        )

        # Reference RoPE pairs adjacent elements (GPT-J), not NeoX.
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=config.rope_is_neox_style,
            )
            if self.use_rope
            else None
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=(
                get_attention_sliding_window_size(config) if self.is_sliding else -1
            ),
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        skip_all_reduce=False,
        gate_activation=None,
    ) -> torch.Tensor:
        if self.unfused_qkv:
            q, _ = self.q_proj(hidden_states)
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)
        elif self.decode_merged_qkv and not forward_batch.forward_mode.is_decode():
            # Keep prefill's original three GEMMs and rounding. Decode uses
            # one GEMV; views share the loaded BF16 checkpoint storage.
            qw, kw, vw = self.qkv_proj.weight.split(
                [self.q_size, self.kv_size, self.kv_size], dim=0
            )
            q = torch.nn.functional.linear(hidden_states, qw)
            k = torch.nn.functional.linear(hidden_states, kw)
            v = torch.nn.functional.linear(hidden_states, vw)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        fused_qk = (
            _use_xpu_qk_norm_rope
            and forward_batch.forward_mode.is_decode()
            and self.qk_norm is not None
            and q.device.type == "xpu"
            and q.shape == (1, 2048)
            and k.shape == (1, 128)
            and q.is_contiguous()
            and k.is_contiguous()
            and q.dtype == k.dtype == torch.bfloat16
            and (
                self.rotary_emb is None
                or self.rotary_emb.rotary_dim == 128
            )
        )
        if fused_qk:
            from sglang.srt.hardware_backend.xpu.kernels.muse_qk_norm_rope import forward
            q, k = forward(
                q, k, self.qk_norm.weight, self.qk_norm.variance_epsilon,
                positions if self.rotary_emb is not None else None,
                self.rotary_emb.cos_sin_cache if self.rotary_emb is not None else None,
                is_neox=self.rotary_emb.is_neox_style if self.rotary_emb is not None else False,
            )
        else:
            if self.qk_norm is not None:
                q, k = apply_qk_norm(
                    q,
                    k,
                    q_norm=self.qk_norm,
                    k_norm=self.qk_norm,
                    head_dim=self.head_dim,
                )
            if self.rotary_emb is not None:
                q, k = self.rotary_emb(positions, q, k)

        attn_out = self.attn(q, k, v, forward_batch)

        if self.use_output_gate:
            if (
                self.fused_xpu_gate_quant
                and hidden_states.dtype == attn_out.dtype == torch.bfloat16
                and attn_out.is_contiguous()
                and not get_forward().sp_active
            ):
                gated = dynamic_w8a8_linear(
                    gate_activation if gate_activation is not None else hidden_states,
                    self.output_gate_proj.weight,
                    self.output_gate_proj.weight_scale,
                    output_gate=attn_out,
                    pack_output_gate=(
                        _use_xpu_packed_o and skip_all_reduce
                        and getattr(self, "_xpu_has_o_replica", False)
                    ),
                )
                if skip_all_reduce and getattr(self, "_xpu_has_o_replica", False):
                    tp = get_parallel().tp_group
                    return local_replicated_o_proj(self, gated, tp.device_group, tp.rank_in_group)
                # The original RowParallelLinear still owns TP communication.
                out, _ = self.o_proj(gated, skip_all_reduce=skip_all_reduce)
                return out
            gate, _ = self.output_gate_proj(hidden_states)
            if _is_cuda or _is_cpu:
                attn_out = fused_sigmoid_mul(attn_out, gate, inplace=True)
            elif _use_fused_xpu_gate:
                attn_out = fused_gate(attn_out, gate)
            else:
                attn_out = torch.sigmoid(gate) * attn_out

        out, _ = self.o_proj(attn_out, skip_all_reduce=skip_all_reduce)
        return out


class MuseGlimmerDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = MuseGlimmerAttention(
            config,
            layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.post_attn_norm = RMSNorm(config.hidden_size, eps=config.post_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = MuseGlimmerMLP(
            config, quant_config=quant_config, prefix=add_prefix("mlp", prefix)
        )
        self.post_ffn_norm = RMSNorm(config.hidden_size, eps=config.post_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        gate_activation = None
        if (
            _use_xpu_input_norm_quant and self.self_attn.fused_xpu_gate_quant
            and (_xpu_input_norm_mode == 1 or forward_batch.forward_mode.is_decode())
        ):
            hidden_states, gate_activation = input_norm_quant(hidden_states, self.input_layernorm)
        else:
            hidden_states = self.input_layernorm(hidden_states)
        if (
            _use_xpu_sequence
            and self.mlp.fused_xpu_mlp
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[0] >= 512
            and forward_batch.forward_mode.is_extend()
            and get_parallel().tp_size == 2
            and get_parallel().attn_tp_size == 2
            and not get_forward().sp_active
        ):
            hidden_states = self.self_attn(
                positions, hidden_states, forward_batch, skip_all_reduce=True,
                gate_activation=gate_activation,
            )
            tp = get_parallel().tp_group
            local_mlp = None
            if getattr(self.mlp, "_xpu_has_replica", False):
                local_mlp = lambda x: local_replicated_mlp(
                    self.mlp, x, return_partials=_fuse_replica_sum and x.shape[-1] == 6656
                )
            elif hasattr(self.mlp, "_xpu_streamed_replica"):
                stream_state, stream_index = self.mlp._xpu_streamed_replica
                if stream_state.active:
                    local_mlp = lambda x: stream_state.run(stream_index, self.mlp, x)
            return sequence_parallel_post_attention(
                hidden_states,
                residual,
                self.post_attn_norm,
                self.post_attention_layernorm,
                self.mlp,
                self.post_ffn_norm,
                tp.device_group,
                tp.rank_in_group,
                alignment=_xpu_sequence_alignment,
                original_rows=positions.numel() if getattr(self.self_attn, "_xpu_has_o_replica", False) else None,
                local_mlp=local_mlp,
            )
        hidden_states = self.self_attn(positions, hidden_states, forward_batch,
                                       gate_activation=gate_activation)
        if (_use_fused_xpu_norm and hidden_states.dtype == torch.bfloat16
                and hidden_states.shape[-1] == 6656):
            residual, hidden_states = fused_post_norm(
                hidden_states, residual, self.post_attn_norm,
                self.post_attention_layernorm, quantize=self.mlp.fused_xpu_mlp,
            )
            hidden_states = self.mlp(hidden_states)
            return fused_post_norm(hidden_states, residual, self.post_ffn_norm)[0]
        hidden_states = residual + self.post_attn_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.post_ffn_norm(hidden_states)
        return hidden_states


class MuseGlimmerModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.embed_norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps, has_weight=False)
            if config.normalize_tok_embeddings
            else None
        )
        self.layers = nn.ModuleList(
            [
                MuseGlimmerDecoderLayer(
                    config,
                    i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # Reference MuseGlimmerFinalRMSNorm: weight is the scale, not an offset.
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layers_to_capture: List[int] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if self.embed_norm is not None:
            hidden_states = self.embed_norm(hidden_states)

        stream_state = getattr(self, "_xpu_streamed_mlp", None)
        if stream_state is not None:
            stream_state.active = False
            if (
                _use_xpu_sequence and hidden_states.dtype == torch.bfloat16
                and hidden_states.shape[0] >= 8192
                and forward_batch.forward_mode.is_extend()
                and not get_forward().sp_active
            ):
                stream_state.begin()

        aux_hidden_states = []
        for i, layer in enumerate(self.layers):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states)
            hidden_states = layer(positions, hidden_states, forward_batch)

        hidden_states = self.norm(hidden_states)
        if (
            _host_memory_audit and hidden_states.shape[0] == 1
            and forward_batch.forward_mode.is_extend()
        ):
            from sglang.srt.hardware_backend.xpu.kernels.muse_host_memory_audit import snapshot
            snapshot(self, _host_memory_audit)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


def muse_glimmer_bilinear_pos_embed_lookup(
    grid_thw: torch.Tensor, num_grid_per_side: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bilinear resample of the position-embedding grid; returns (indices, weights) to gather-and-sum."""
    side = num_grid_per_side
    device = grid_thw.device
    index_parts: List[List[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: List[List[torch.Tensor]] = [[] for _ in range(4)]

    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        h_grid = (torch.arange(h, device=device).float() + 0.5) * (side / h) - 0.5
        w_grid = (torch.arange(w, device=device).float() + 0.5) * (side / w) - 0.5

        h_floor = torch.floor(h_grid).long()
        w_floor = torch.floor(w_grid).long()
        h_ceil = h_floor + 1
        w_ceil = w_floor + 1
        h_frac = h_grid - h_floor.float()
        w_frac = w_grid - w_floor.float()

        h_floor_valid = (h_floor >= 0) & (h_floor <= side - 1)
        h_ceil_valid = (h_ceil >= 0) & (h_ceil <= side - 1)
        w_floor_valid = (w_floor >= 0) & (w_floor <= side - 1)
        w_ceil_valid = (w_ceil >= 0) & (w_ceil <= side - 1)
        h_floor = h_floor.clamp(0, side - 1)
        h_ceil = h_ceil.clamp(0, side - 1)
        w_floor = w_floor.clamp(0, side - 1)
        w_ceil = w_ceil.clamp(0, side - 1)

        h_floor_offset = h_floor * side
        h_ceil_offset = h_ceil * side
        corner_indices = [
            (h_floor_offset[:, None] + w_floor[None, :]).flatten(),
            (h_floor_offset[:, None] + w_ceil[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_floor[None, :]).flatten(),
            (h_ceil_offset[:, None] + w_ceil[None, :]).flatten(),
        ]
        corner_weights = [
            (
                (1 - h_frac)[:, None]
                * (1 - w_frac)[None, :]
                * (h_floor_valid[:, None] & w_floor_valid[None, :])
            ).flatten(),
            (
                (1 - h_frac)[:, None]
                * w_frac[None, :]
                * (h_floor_valid[:, None] & w_ceil_valid[None, :])
            ).flatten(),
            (
                h_frac[:, None]
                * (1 - w_frac)[None, :]
                * (h_ceil_valid[:, None] & w_floor_valid[None, :])
            ).flatten(),
            (
                h_frac[:, None]
                * w_frac[None, :]
                * (h_ceil_valid[:, None] & w_ceil_valid[None, :])
            ).flatten(),
        ]
        for corner in range(4):
            index_parts[corner].append(corner_indices[corner].repeat(t))
            weight_parts[corner].append(corner_weights[corner].repeat(t))

    indices = torch.stack([torch.cat(part) for part in index_parts])
    weights = torch.stack([torch.cat(part) for part in weight_parts])
    return indices, weights


class MuseGlimmerVisionPatchEmbedder(nn.Module):
    """Linear patch projection plus a bilinearly resampled learned position grid."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_grid_per_side = config.pos_emb_height
        self.patch_embedding = ReplicatedLinear(
            config.patch_temporal * 3 * config.patch_size**2,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("patch_embedding", prefix),
        )
        self.position_embedding_table = nn.Embedding(
            config.pos_emb_height * config.pos_emb_width, config.hidden_size
        )

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        embeddings, _ = self.patch_embedding(pixel_values)
        indices, weights = muse_glimmer_bilinear_pos_embed_lookup(
            grid_thw, self.num_grid_per_side
        )
        indices = indices.to(embeddings.device)
        weights = weights.to(device=embeddings.device, dtype=torch.float32)
        pos_embeds = (
            self.position_embedding_table(indices).float() * weights[:, :, None]
        ).sum(0)
        return embeddings + pos_embeds.to(embeddings.dtype)


class MuseGlimmerVisionRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        spatial_dim = head_dim // 2
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (total_patches, 2) laid out as (width, height).
        freqs = position_ids.float()[:, :, None] * self.inv_freq[None, None, :]
        freq = torch.cat([freqs[:, 0], freqs[:, 1]], dim=-1)
        return freq.cos(), freq.sin()


class MuseGlimmerVisionMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )
        self.act = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return x


class MuseGlimmerVisionEncoderLayer(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            projection_size=config.num_attention_heads * config.head_dim,
            use_qkv_parallel=True,
            qkv_bias=True,
            proj_bias=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        self.mlp = MuseGlimmerVisionMLP(
            config, quant_config=quant_config, prefix=add_prefix("mlp", prefix)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        forward_metadata: VisionAttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            forward_metadata=forward_metadata,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MuseGlimmerVisionModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.merge_size = config.merge_size
        self.patch_size = config.patch_size
        # Square position grid; the window is the full extent of that grid.
        self.window_size = config.pos_emb_height * config.patch_size
        self.attention_types = config.attention_types

        self.patch_embedder = MuseGlimmerVisionPatchEmbedder(
            config,
            quant_config=quant_config,
            prefix=add_prefix("patch_embedder", prefix),
        )
        self.rotary_emb = MuseGlimmerVisionRotaryEmbedding(
            config.head_dim, config.rope_theta
        )
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            [
                MuseGlimmerVisionEncoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    @property
    def dtype(self) -> torch.dtype:
        return self.ln_pre.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.ln_pre.weight.device

    def pixel_shuffle(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Fold each ``merge_size x merge_size`` patch block into one wider token."""
        factor = self.merge_size
        dim = hidden_states.shape[-1]
        outputs = []
        offset = 0

        for t, h, w in grid_thw.tolist():
            t, h, w = int(t), int(h), int(w)
            num_tokens = t * h * w
            chunk = hidden_states[offset : offset + num_tokens]
            offset += num_tokens

            num_out_per_frame = (h // factor) * (w // factor)
            block_perm = torch.arange(h * w, device=hidden_states.device)
            block_perm = (
                block_perm.view(h // factor, factor, w // factor, factor)
                .permute(0, 2, 1, 3)
                .reshape(-1)
            )
            if t > 1:
                frame_offsets = (
                    torch.arange(t, device=hidden_states.device) * h * w
                ).view(t, 1)
                block_perm = (block_perm.unsqueeze(0) + frame_offsets).reshape(-1)

            merged = chunk[block_perm].view(t * num_out_per_frame, factor * factor, dim)
            merged = (
                merged.permute(0, 2, 1)
                .contiguous()
                .view(t * num_out_per_frame, dim * factor * factor)
            )
            outputs.append(merged)

        return torch.cat(outputs, dim=0)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embedder(pixel_values, grid_thw)
        hidden_states = self.ln_pre(hidden_states)

        cu_seqlens = get_vision_cu_seqlens(grid_thw).to(self.device)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw,
            spatial_merge_size=1,
            window_size=self.window_size,
            patch_size=self.patch_size,
        )
        window_index = window_index.to(self.device)
        cu_window_seqlens = cu_window_seqlens.to(self.device)

        # The reference offsets patch coordinates by 1 and orders them (w, h).
        position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1)
        position_ids = (position_ids.flip(-1) + 1).to(self.device)

        hidden_states = hidden_states[window_index]
        cos, sin = self.rotary_emb(position_ids[window_index])
        position_embeddings = (
            cos.to(self.dtype),
            sin.to(self.dtype),
        )

        metadata_by_type = {
            "full_attention": prepare_vision_attention_metadata(
                cu_seqlens, device=self.device
            ),
            "window_attention": prepare_vision_attention_metadata(
                cu_window_seqlens, device=self.device
            ),
        }
        cu_seqlens_by_type = {
            "full_attention": cu_seqlens,
            "window_attention": cu_window_seqlens,
        }

        hidden_states = hidden_states.unsqueeze(0)
        for i, layer in enumerate(self.layers):
            attention_type = self.attention_types[i]
            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens_by_type[attention_type],
                position_embeddings=position_embeddings,
                forward_metadata=metadata_by_type[attention_type],
            )
        hidden_states = hidden_states.squeeze(0)

        hidden_states = hidden_states[permute_inv(window_index)]
        hidden_states = self.ln_post(hidden_states)
        return self.pixel_shuffle(hidden_states, grid_thw)


class MuseGlimmerVisionAdapter(nn.Module):
    """Two-layer projector between the shuffled ViT output and the decoder width."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        sharded_size = config.padded_projector_hidden_size
        self.fc1 = ColumnParallelLinear(
            config.out_hidden_size,
            sharded_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = RowParallelLinear(
            sharded_size,
            config.projector_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fc2", prefix),
        )
        self.act = ACT2FN[config.projector_hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act(x)
        x, _ = self.fc2(x)
        return self.act(x)


class MuseGlimmerForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    checkpoint_uses_vendor_names = False

    builds_vision_tower = False

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.checkpoint_norms_are_absolute = (
            quant_config is not None and quant_config.get_name() == "gguf"
        )

        if config.output_soft_cap_temp is not None:
            config.final_logit_softcapping = config.output_soft_cap_temp

        self.model = MuseGlimmerModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            # Pad vocab to 128-align for the MXFP8 kernel.
            padding_size=128,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(
            config, logit_scale=config.output_multiplier
        )
        self.logits_processor.xpu_compact_greedy = (
            os.environ.get("SGLANG_XPU_MUSE_COMPACT_GREEDY") == "1"
            and is_xpu() and get_parallel().tp_size == 2
            and config.vocab_size == 202048
            and self.logits_processor.final_logit_softcapping == 20.0
        )
        self.capture_aux_hidden_states = False

    def get_attention_sliding_window_size(self):
        return get_attention_sliding_window_size(self.config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def set_dflash_layers_to_capture(self, layer_ids: List[int]):
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        num_layers = len(self.model.layers)
        bad = [i for i in layer_ids if not 0 <= i < num_layers]
        if bad:
            raise ValueError(
                f"DFLASH target layer ids {bad} are out of range for a "
                f"{num_layers}-layer Muse Glimmer target."
            )
        self.capture_aux_hidden_states = True
        self.model.layers_to_capture = list(layer_ids)

    def post_load_weights(self):
        if _xpu_replica_layers:
            install_peer_mlp_shards(self, _xpu_replica_layers)
        if _xpu_o_replica_layers:
            install_peer_o_shards(self, _xpu_o_replica_layers)
        if _xpu_streamed_layers:
            from sglang.srt.hardware_backend.xpu.kernels.muse_streamed_replica import install_streamed_peer_mlp
            install_streamed_peer_mlp(self, _xpu_replica_layers, _xpu_streamed_layers)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded = 0

        for name, loaded_weight in weights:
            # SGLang derives RoPE itself; the checkpoint ships a cached freqs buffer.
            if "rotary_emb.freqs" in name or "rotary_emb.inv_freq" in name:
                continue
            if not self.builds_vision_tower and any(
                fragment in name for fragment in _VISION_NAME_FRAGMENTS
            ):
                continue

            if self.checkpoint_uses_vendor_names:
                name = _vendor_weight_name(name)

            # Fold the reference's +1; model.norm is exempt.
            if not self.checkpoint_norms_are_absolute and name.endswith(
                _OFFSET_NORM_SUFFIXES
            ):
                loaded_weight = loaded_weight + 1.0

            # ModelOpt holds KV scales on self_attn, not self_attn.attn.
            if name.endswith((".k_scale", ".v_scale")):
                remapped = maybe_remap_kv_scale_name(name, params_dict)
                if remapped is None:
                    continue
                name = remapped

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "output_gate_proj" in name:
                    continue
                mapped = name.replace(weight_name, param_name)
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded += 1
                break
            else:
                if name not in params_dict:
                    logger.warning(
                        "Muse Glimmer: unexpected checkpoint weight %s", name
                    )
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded += 1

        logger.info("Muse Glimmer: loaded %d weight tensors", loaded)
        self.post_load_weights()


class MuseGlimmerForConditionalGeneration(MuseGlimmerForCausalLM):
    """Vendor multimodal HF export: the MuseGlimmerForCausalLM decoder plus the image tower."""

    # Only this class reads vendor-named checkpoints, so only it needs the mapper.
    hf_to_sglang_mapper = _VENDOR_TO_SGLANG
    checkpoint_uses_vendor_names = True
    builds_vision_tower = True

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.language_model_only = bool(config.language_model_only)
        if self.language_model_only:
            self.builds_vision_tower = False
            self.vision_tower = None
            self.vision_adapter = None
            self.vision_projection = None
            self.perception_emb_norm = None
            return
        if config.vision_config is None:
            raise ValueError(
                "MuseGlimmerForConditionalGeneration needs a vision_config; a checkpoint "
                "with no vision tower should declare MuseGlimmerForCausalLM instead."
            )
        self.vision_tower = MuseGlimmerVisionModel(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_tower", prefix),
        )
        self.vision_adapter = MuseGlimmerVisionAdapter(
            config,
            quant_config=quant_config,
            prefix=add_prefix("vision_adapter", prefix),
        )
        self.vision_projection = ReplicatedLinear(
            config.projector_hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("vision_projection", prefix),
        )
        self.perception_emb_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )

    def pad_input_ids(self, input_ids: array, mm_inputs: MultimodalInputs) -> array:
        return pad_mm_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        image_grid_thw = torch.cat([item.image_grid_thw for item in items], dim=0).cpu()
        assert pixel_values.dim() == 2, pixel_values.dim()
        assert image_grid_thw.dim() == 2, image_grid_thw.dim()

        features = self.vision_tower(pixel_values, image_grid_thw)
        features = self.vision_adapter(features)
        features, _ = self.vision_projection(features)
        return self.perception_emb_norm(features)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        if self.language_model_only:
            # Embed here: prefill BCG captures input_embeds.
            if input_embeds is None:
                input_embeds = self.get_input_embeddings()(input_ids)
                if forward_batch.input_embeds is not None:
                    forward_batch.input_embeds.copy_(input_embeds)
                    input_embeds = forward_batch.input_embeds
            hidden_states = self.model(None, positions, forward_batch, input_embeds)
        else:
            hidden_states = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                multimodal_model=self,
                positions=positions,
            )

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )


EntryClass = [MuseGlimmerForCausalLM, MuseGlimmerForConditionalGeneration]
