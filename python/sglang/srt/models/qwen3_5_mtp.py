# Copyright 2023-2024 SGLang Team
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

"""Inference-only Qwen3_5 MTP model."""

import copy
import logging
from contextlib import ExitStack
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.kernels.ops.gemm.rdna4_fp8_skinny import (
    rdna4_fp8_blockwise_skinny,
    rdna4_fp8_lm_head_refine,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
)
from sglang.srt.distributed import get_pp_group, get_tp_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.runtime_context import (
    get_model,
    get_parallel,
    get_spec,
)
from sglang.srt.utils import add_prefix, get_bool_env_var, is_gfx1201_supported, is_npu

logger = logging.getLogger(__name__)


def _mtp_quant_config(quant_config):
    """The quantization the MTP module itself is built with.

    The MTP module often ships unquantized even though the target checkpoint is
    quantized; the loader's fusion gate has to see the same normalization the
    constructor applies, or it would answer for the target's quantization.
    """
    # Serialized Qwen3.5 ModelOpt checkpoints keep embedded MTP weights in
    # BF16. Disable quantization for those checkpoints; non-serialized
    # modelopt_fp4 still converts MoE expert weights on load.
    if quant_config and (
        quant_config.get_name() == "modelopt_mixed"
        or (
            quant_config.get_name() == "modelopt_fp4"
            and quant_config.is_checkpoint_nvfp4_serialized
        )
    ):
        return None
    if is_npu() and get_spec().speculative_draft_model_quantization is None:
        return None
    # Quark-quantized Qwen3.5 MXFP4 checkpoints ship the MTP module in bf16;
    # every `mtp.*` layer appears under the quantization exclude list. Detect
    # that and skip quantization here so linear/MoE weight loaders allocate
    # bf16 shapes (see sgl-project/sglang#23113).
    if quant_config and quant_config.get_name() == "quark":
        exclude_layers = getattr(quant_config, "exclude_layers", [])
        if any(
            isinstance(layer, str) and layer.startswith("mtp.")
            for layer in exclude_layers
        ):
            return None
    return quant_config


class Qwen3_5ForCausalLMMTP(nn.Module):

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return Qwen3_5ForCausalLM.shared_experts_fusion_disable_reason(
            getattr(hf_config, "text_config", hf_config),
            _mtp_quant_config(quant_config),
        )

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.is_multimodal = hasattr(config, "text_config")
        if self.is_multimodal:
            config = config.text_config

        # Deep-copy so MTP mutations below don't leak into the target's config.
        config = copy.deepcopy(config)

        quant_config = _mtp_quant_config(quant_config)

        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        RMSNorm_cls = GemmaRMSNorm
        self.pre_fc_norm_embedding = RMSNorm_cls(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = RMSNorm_cls(config.hidden_size, config.rms_norm_eps)
        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = 1
        mtp_config.full_attention_interval = 1
        self.model = Qwen3_5ForCausalLM(
            mtp_config,
            quant_config,
            prefix=add_prefix("mtp", prefix),
            is_nextn=True,
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )

        self.logits_processor = LogitsProcessor(config)
        self._local_argmax_enabled = False
        self._local_argmax_head_identity = None
        self._local_argmax_partitions = 512

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        if not self.config.tie_word_embeddings:
            del self.lm_head.weight

        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_lm_head_from_target(self, target_lm_head):
        if self.config.tie_word_embeddings:
            return

        self.lm_head = target_lm_head

    @torch.no_grad()
    def prepare_local_argmax(self) -> None:
        """Build the gfx1201 FP8 shortlist head used by greedy MTP decode.

        The original BF16 shard remains authoritative: the FP8 copy proposes
        one row per vocab partition and the refinement kernel rescans only
        those candidates with the original weights.
        """
        if not get_bool_env_var("SGLANG_ROCM_MTP_FP8_LM_HEAD", "0"):
            return
        weight = getattr(self.lm_head, "weight", None)
        if (
            not is_gfx1201_supported()
            or not isinstance(weight, torch.Tensor)
            or not weight.is_cuda
            or weight.dtype != torch.bfloat16
            or weight.ndim != 2
            or not weight.is_contiguous()
            or weight.shape[0] % 128 != 0
            or weight.shape[1] % 128 != 0
            or getattr(self.lm_head, "num_added_embeddings", 0) != 0
        ):
            logger.warning_once(
                "ROCm MTP local FP8 lm_head requested, but the bound head does "
                "not satisfy the gfx1201 BF16 128x128 contract; using the "
                "regular full-vocabulary path."
            )
            return
        if self._local_argmax_head_identity == id(weight):
            return

        # Quantize in modest row-block chunks.  A single vectorized conversion
        # would transiently materialize several GB of FP32 abs values for the
        # TP2 head and can make an otherwise fitting model fail at startup.
        rows, hidden = weight.shape
        weight_fp8 = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
        weight_scale = torch.empty(
            (rows // 128, hidden // 128), dtype=torch.float32, device=weight.device
        )
        row_blocks_per_chunk = 32
        for block_start in range(0, rows // 128, row_blocks_per_chunk):
            block_end = min(block_start + row_blocks_per_chunk, rows // 128)
            row_start, row_end = block_start * 128, block_end * 128
            view = weight[row_start:row_end].view(
                block_end - block_start, 128, hidden // 128, 128
            )
            scale = view.abs().float().amax(dim=(1, 3)).clamp_(min=1e-4) / 448.0
            quantized = (view / scale[:, None, :, None]).to(torch.float8_e4m3fn)
            weight_fp8[row_start:row_end].copy_(
                quantized.reshape(row_end - row_start, hidden)
            )
            weight_scale[block_start:block_end].copy_(scale)

        self.register_buffer(
            "_local_argmax_weight_fp8", weight_fp8, persistent=False
        )
        self.register_buffer(
            "_local_argmax_weight_scale", weight_scale, persistent=False
        )
        self._local_argmax_head_identity = id(weight)
        self._local_argmax_enabled = True
        torch.cuda.empty_cache()
        logger.info(
            "Prepared gfx1201 block-FP8 MTP lm_head shortlist (shape=%s, "
            "partitions=%d).",
            tuple(weight.shape),
            self._local_argmax_partitions,
        )

    def is_local_argmax_output(self, output: torch.Tensor) -> bool:
        return self._local_argmax_enabled and output.ndim == 2 and output.shape[-1] == 2

    def select_top_tokens_from_local_output(
        self, local_pair: torch.Tensor
    ) -> torch.Tensor:
        """Reduce shard-local ``(exact score, global token id)`` pairs."""
        gathered = get_tp_group().all_gather(local_pair, dim=-1)
        gathered = gathered.view(local_pair.shape[0], self.tp_size, 2)
        rank = gathered[..., 0].argmax(dim=-1, keepdim=True)
        return gathered[..., 1].gather(-1, rank).to(torch.int64)

    def _local_argmax_logits_output(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Optional[LogitsProcessorOutput]:
        if not self._local_argmax_enabled:
            return None
        metadata = LogitsMetadata.from_forward_batch(forward_batch)
        if metadata.extend_return_logprob:
            return None
        (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            *_unused,
        ) = self.logits_processor._get_pruned_states(
            hidden_states, None, None, metadata
        )
        # The local skinny head is deliberately specialized to concurrency 1.
        # Larger graph buckets and real batches retain the general BF16 path.
        if pruned_states.shape[0] != 1:
            return None
        hidden_states_to_store = self.logits_processor._get_hidden_states_to_store(
            hidden_states,
            None,
            None,
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            metadata,
        )
        head_input = (
            pruned_states[sample_indices]
            if sample_indices is not None
            else pruned_states
        ).contiguous()
        activation_fp8, activation_scale = per_token_group_quant_fp8(
            head_input, 128
        )
        approximate_logits = rdna4_fp8_blockwise_skinny(
            activation_fp8,
            self._local_argmax_weight_fp8,
            activation_scale.contiguous(),
            self._local_argmax_weight_scale,
        )
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            approximate_logits[..., -num_pad:] = -float("inf")
        local_values, local_indices = rdna4_fp8_lm_head_refine(
            approximate_logits,
            head_input,
            self.lm_head.weight,
            self._local_argmax_partitions,
        )
        global_indices = (
            local_indices + self.lm_head.shard_indices.org_vocab_start_index
        )
        local_pair = torch.stack(
            (local_values.float(), global_indices.float()), dim=-1
        )
        return LogitsProcessorOutput(
            next_token_logits=local_pair,
            hidden_states=hidden_states_to_store,
            mm_input_embeds=metadata.mm_input_embeds,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        exit_stack = ExitStack()
        # Match the target Qwen-VL wrapper: real multimodal requests consume
        # their three-axis MRoPE positions, while pure-text requests use the
        # equivalent (and faster) 1-D position path.
        if (
            forward_batch.mrope_positions is not None
            and forward_batch.contains_mm_inputs()
        ):
            positions = forward_batch.mrope_positions
        if (
            is_npu()
            and self.quant_config is None
            and get_model().quantization is not None
        ):
            # ascend mtp unquant
            exit_stack.enter_context(envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))
            exit_stack.enter_context(
                envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)
            )

        try:
            assert input_embeds is None
            input_embeds = forward_batch.mm_input_embeds
            if (
                forward_batch.forward_mode.is_extend()
                and forward_batch.contains_mm_inputs()
                and not forward_batch.forward_mode.is_draft_extend_v2()
            ):
                assert input_embeds is not None
                last_indices = (
                    forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1
                ).long()
                input_embeds[last_indices] = self.model.embed_tokens(
                    input_ids[last_indices]
                )

            if input_embeds is None:
                input_embeds = self.model.embed_tokens(input_ids)

            hidden_states = forward_batch.spec_info.hidden_states

            if not forward_batch.forward_mode.is_idle():
                input_embeds = self.pre_fc_norm_embedding(input_embeds)
                hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

            hidden_states = self.fc(hidden_states)

            with get_global_expert_distribution_recorder().disable_this_region():
                hidden_states = self.model(
                    input_ids,
                    positions,
                    forward_batch,
                    hidden_states,
                )
        finally:
            exit_stack.close()

        local_output = self._local_argmax_logits_output(
            input_ids, hidden_states, forward_batch
        )
        if local_output is not None:
            return local_output
        return self.logits_processor(input_ids, hidden_states, self.lm_head, forward_batch)

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for MoE experts (non-fused/fused)
        num_experts = getattr(self.config, "num_experts", None)
        if num_experts is not None:
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
            )
        else:
            expert_params_mapping = []

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        # fused experts: experts.w13_weight / experts.w2_weight
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            param = params_dict[name]
            weight_loader = param.weight_loader
            # Let EP MoE layer handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Only process MTP branch weights
            if "mtp" not in name:
                continue

            if name.startswith("mtp."):
                # Remove the mtp. prefix for processing
                name = name.replace("mtp.", "model.")

                name = name.replace("model.fc", "fc")
                name = name.replace("model.pre_fc", "pre_fc")

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            # 1) Process stacked parameters (q_proj/k_proj/v_proj & gate_proj/up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Check if this is a fused expert weight
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-matching weights
                if weight_name not in name:
                    continue

                # Skip MoE experts.* here, handled separately below
                if "mlp.experts" in name:
                    continue

                name_mapped = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if (
                    name_mapped.endswith(ignore_suffixes)
                    and name_mapped not in params_dict
                ):
                    continue

                if name_mapped not in params_dict:
                    continue

                param = params_dict[name_mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                name = name_mapped
                break
            else:
                # 2) Process MoE expert weights (including fused experts)
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)

                    # Fused experts: single checkpoint weight contains multiple experts
                    if is_fused_expert and num_experts is not None:
                        if "experts.gate_up_proj" in name:
                            # gate_up_proj fused: split into w1 / w3
                            loaded_w1, loaded_w3 = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_w1,
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_w3,
                                "w3",
                                num_experts,
                            )
                        else:
                            # down_proj fused: distribute entire weight
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Non-fused expert, load by expert_id/shard
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        if name_mapped not in params_dict:
                            break
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    # Skip expert weight if not handled by current rank
                    if is_expert_weight:
                        continue

                    # 3) Regular non-stacked / non-expert parameters, use default loader
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )

            loaded_params.add(name)
        return loaded_params


EntryClass = [Qwen3_5ForCausalLMMTP]
