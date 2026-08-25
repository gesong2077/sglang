#!/usr/bin/env bash
set -euo pipefail

readonly VENV=/opt/apps/sglang-combo/sglang-inference-env
readonly SGLANG=${VENV}/bin/sglang
readonly MODEL=/mnt/models/Qwen3.8-27B-FP8
readonly DFLASH_DRAFT_MODEL=/mnt/models/Qwen3.8-27B-DFlash2
readonly SGLANG_PYTHON=/opt/apps/sglang-combo/sglang/python
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly RCCL_TUNER_DIR=${SCRIPT_DIR}/rccl-tuner
readonly CHAT_TEMPLATE=${MODEL}/chat_template.jinja
# TP2 is the validated/default profile for this freeze. TP4 remains available
# through an explicit SGLANG_TP_SIZE=4 override.
readonly TP_SIZE="${SGLANG_TP_SIZE:-2}"

if [[ "${TP_SIZE}" == "2" ]]; then
  readonly DEFAULT_MAMBA_FULL_MEMORY_RATIO=6.08
  readonly DEFAULT_MAMBA_SSM_DTYPE=bfloat16
else
  readonly DEFAULT_MAMBA_FULL_MEMORY_RATIO=0.9
  readonly DEFAULT_MAMBA_SSM_DTYPE=bfloat16
fi

# Activation also selects the ROCm SDK bundled with this PyTorch wheel. That
# SDK supplies the matching hipcc, headers, and runtime libraries for JIT.
source "${VENV}/bin/activate"
export PYTHONPATH="${SGLANG_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Public-repository-safe development placeholder. Set a real value through the
# environment before exposing the listener outside a trusted local network.
export SGLANG_API_KEY="${SGLANG_API_KEY:-change-me-before-exposing-the-port}"
export SGLANG_MM_ATTENTION_BACKEND="${SGLANG_MM_ATTENTION_BACKEND:-triton_attn}"
export SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-triton}"
export SGLANG_PREFILL_ATTENTION_BACKEND="${SGLANG_PREFILL_ATTENTION_BACKEND:-triton}"
export SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-32}"
export SGLANG_TRITON_ATTENTION_NUM_KV_SPLITS="${SGLANG_TRITON_ATTENTION_NUM_KV_SPLITS:-32}"
export AITER_GFX1201_LONG_PREFILL_WAVES_PER_EU="${AITER_GFX1201_LONG_PREFILL_WAVES_PER_EU:-4}"
export AITER_GFX1201_DECODE_SPLIT_CU_MULTIPLIER="${AITER_GFX1201_DECODE_SPLIT_CU_MULTIPLIER:-1}"
export AITER_GFX1201_DECODE_3D_WARPS="${AITER_GFX1201_DECODE_3D_WARPS:-8}"
export AITER_GFX1201_DECODE_3D_WAVES_PER_EU="${AITER_GFX1201_DECODE_3D_WAVES_PER_EU:-2}"
export ROCM_QUICK_REDUCE_QUANTIZATION="${ROCM_QUICK_REDUCE_QUANTIZATION:-FP}"
export ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16="${ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16:-0}"
# Keep this first rollout decode-only. One MiB covers up to 102 Qwen hidden
# rows while preserving the already validated RCCL/AITER long-prefill path.
export ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB="${ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB:-1}"
export SGLANG_SPECULATIVE_ATTENTION_MODE="${SGLANG_SPECULATIVE_ATTENTION_MODE:-decode}"
export SGLANG_SPECULATIVE_DRAFT_ATTENTION_BACKEND="${SGLANG_SPECULATIVE_DRAFT_ATTENTION_BACKEND:-aiter}"
export SGLANG_ROCM_MTP_FP8_LM_HEAD="${SGLANG_ROCM_MTP_FP8_LM_HEAD:-1}"
export SGLANG_SPECULATIVE_NUM_STEPS="${SGLANG_SPECULATIVE_NUM_STEPS:-3}"
export SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS="${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-4}"
export SGLANG_ENABLE_LINEAR_REPLAYSSM_SPEC="${SGLANG_ENABLE_LINEAR_REPLAYSSM_SPEC:-1}"
export SGLANG_DFLASH_DRAFT_MODEL_PATH="${SGLANG_DFLASH_DRAFT_MODEL_PATH:-${DFLASH_DRAFT_MODEL}}"
export SGLANG_DFLASH_NUM_DRAFT_TOKENS="${SGLANG_DFLASH_NUM_DRAFT_TOKENS:-8}"
export SGLANG_MAMBA_FULL_MEMORY_RATIO="${SGLANG_MAMBA_FULL_MEMORY_RATIO:-${DEFAULT_MAMBA_FULL_MEMORY_RATIO}}"
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-${DEFAULT_MAMBA_SSM_DTYPE}}"

if [[ -z "${SGLANG_ENABLE_DFLASH+x}" ]]; then
  if [[ "${TP_SIZE}" == "2" ]]; then
    SGLANG_ENABLE_DFLASH=1
  else
    SGLANG_ENABLE_DFLASH=0
  fi
fi
if [[ "${SGLANG_ENABLE_DFLASH}" == "1" && "${SGLANG_ENABLE_MTP:-0}" == "1" ]]; then
  echo "SGLANG_ENABLE_DFLASH and SGLANG_ENABLE_MTP cannot both be enabled" >&2
  exit 1
fi

# AITER Unified materially improves Qwen3.8 MTP target verification on gfx1201,
# while the validated non-speculative and DFlash profiles retain Triton decode.
# Preserve an explicit caller selection for controlled A/B testing.
if [[ -z "${SGLANG_DECODE_ATTENTION_BACKEND+x}" ]]; then
  if [[ "${SGLANG_ENABLE_MTP:-0}" == "1" ]]; then
    SGLANG_DECODE_ATTENTION_BACKEND=aiter
  else
    SGLANG_DECODE_ATTENTION_BACKEND=triton
  fi
fi
export SGLANG_DECODE_ATTENTION_BACKEND

SPECULATIVE_ARGS=()
if [[ "${SGLANG_ENABLE_DFLASH}" == "1" ]]; then
  if [[ ! -r "${SGLANG_DFLASH_DRAFT_MODEL_PATH}/config.json" ]]; then
    echo "DFLASH draft model is missing: ${SGLANG_DFLASH_DRAFT_MODEL_PATH}" >&2
    exit 1
  fi
  SPECULATIVE_ARGS=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "${SGLANG_DFLASH_DRAFT_MODEL_PATH}"
    --speculative-num-draft-tokens "${SGLANG_DFLASH_NUM_DRAFT_TOKENS}"
    --speculative-draft-attention-backend triton
  )
elif [[ "${SGLANG_ENABLE_MTP:-0}" == "1" ]]; then
  SPECULATIVE_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps "${SGLANG_SPECULATIVE_NUM_STEPS}"
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens "${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS}"
    --speculative-attention-mode "${SGLANG_SPECULATIVE_ATTENTION_MODE}"
    --speculative-draft-attention-backend "${SGLANG_SPECULATIVE_DRAFT_ATTENTION_BACKEND}"
  )
  if [[ "${SGLANG_ENABLE_LINEAR_REPLAYSSM_SPEC}" == "1" ]]; then
    SPECULATIVE_ARGS+=(--enable-linear-replayssm-spec)
  fi
fi

CUSTOM_ALL_REDUCE_ARGS=()
if [[ -z "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE+x}" ]]; then
  if [[ "${TP_SIZE}" == "2" ]]; then
    export SGLANG_DISABLE_CUSTOM_ALL_REDUCE=0
  else
    # QuickReduce is restricted to TP2 on this PCIe topology. Preserve the
    # validated TP4 RCCL path unless the caller explicitly opts in.
    export SGLANG_DISABLE_CUSTOM_ALL_REDUCE=1
  fi
fi
if [[ "${SGLANG_DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  CUSTOM_ALL_REDUCE_ARGS=(--disable-custom-all-reduce)
fi

if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
  case "${TP_SIZE}" in
    2) export HIP_VISIBLE_DEVICES=0,1 ;;
    4) export HIP_VISIBLE_DEVICES=0,1,2,3 ;;
    *)
      echo "Set HIP_VISIBLE_DEVICES explicitly for SGLANG_TP_SIZE=${TP_SIZE}" >&2
      exit 1
      ;;
  esac
fi

# Keep the broad model-kernel switch and custom/one-stage AllReduce paths off by
# default.  Preserve explicit caller overrides so the AITER fused-AR path can be
# benchmarked independently without editing this launcher between A/B runs.
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-0}"
unset SGLANG_USE_1STAGE_ALLREDUCE
export SGLANG_USE_AITER_AR="${SGLANG_USE_AITER_AR:-0}"
export SGLANG_USE_AITER_UNIFIED_ATTN="${SGLANG_USE_AITER_UNIFIED_ATTN:-1}"
export SGLANG_AITER_UNIFIED_NATIVE_GQA="${SGLANG_AITER_UNIFIED_NATIVE_GQA:-1}"

# Keep Triton for the first 8192-token prefill chunk, then route chunks with an
# existing KV prefix through AITER Unified.  The first chunk is faster in
# Triton, while AITER wins once attention also reads prior chunks.  Both paths
# retain decode CUDA-graph capture.
export SGLANG_TRITON_USE_AITER_UNIFIED_PREFILL="${SGLANG_TRITON_USE_AITER_UNIFIED_PREFILL:-1}"
export SGLANG_AITER_UNIFIED_PREFILL_MIN_KV_LEN="${SGLANG_AITER_UNIFIED_PREFILL_MIN_KV_LEN:-8193}"

# The model issues 129 45-MiB TP all-reduces for a 4608-token prefill, while
# decode uses 10-KiB collectives. The tuner selects LL through 64 KiB and
# Simple above it, so both workloads can use their faster protocol. Eight Ring
# channels preserve the best measured prefill without sacrificing decode.
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_PROTO="${NCCL_PROTO:-LL,Simple}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-8}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-8}"
export NCCL_P2P_READ_ENABLE="${NCCL_P2P_READ_ENABLE:-1}"
if [[ "${TP_SIZE}" == 4 ]]; then
  export NCCL_TUNER_PLUGIN="${NCCL_TUNER_PLUGIN:-${RCCL_TUNER_DIR}/libnccl-tuner-r9700.so}"
  export NCCL_TUNER_CONFIG_FILE="${NCCL_TUNER_CONFIG_FILE:-${RCCL_TUNER_DIR}/r9700-tp4.conf}"
  if [[ ! -r "${NCCL_TUNER_PLUGIN}" || ! -r "${NCCL_TUNER_CONFIG_FILE}" ]]; then
    echo "RCCL tuner plugin or config is missing; rebuild it from ${RCCL_TUNER_DIR}/README.md" >&2
    exit 1
  fi
else
  # The checked-in tuner table is explicitly constrained to four ranks.
  # Let RCCL select TP2 algorithms unless the caller supplied a TP2 table.
  if [[ -z "${NCCL_TUNER_CONFIG_FILE:-}" ]]; then
    unset NCCL_TUNER_PLUGIN NCCL_TUNER_CONFIG_FILE
  fi
fi

# Use the supported CLI entrypoint; exec keeps the server in the foreground.
exec "${SGLANG}" serve "${MODEL}" \
  --tokenizer-path "${MODEL}" \
  --chat-template "${CHAT_TEMPLATE}" \
  --served-model-name Qwen3.8-27B \
  --context-length 262144 \
  --tp-size "${TP_SIZE}" \
  --attention-backend "${SGLANG_ATTENTION_BACKEND}" \
  --prefill-attention-backend "${SGLANG_PREFILL_ATTENTION_BACKEND}" \
  --decode-attention-backend "${SGLANG_DECODE_ATTENTION_BACKEND}" \
  --page-size "${SGLANG_PAGE_SIZE}" \
  --triton-attention-num-kv-splits "${SGLANG_TRITON_ATTENTION_NUM_KV_SPLITS}" \
  "${SPECULATIVE_ARGS[@]}" \
  --enable-multimodal \
  --mm-attention-backend "${SGLANG_MM_ATTENTION_BACKEND}" \
  --mm-preprocess-cache-size-mb 256 \
  --mm-feature-transport cpu \
  --limit-mm-data-per-request '{"video":0}' \
  --fp8-gemm-backend aiter \
  "${CUSTOM_ALL_REDUCE_ARGS[@]}" \
  --mem-fraction-static 0.80 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 8192 \
  --max-running-requests 16 \
  --mamba-full-memory-ratio "${SGLANG_MAMBA_FULL_MEMORY_RATIO}" \
  --mamba-radix-cache-strategy extra_buffer \
  --mamba-ssm-dtype "${SGLANG_MAMBA_SSM_DTYPE}" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --api-key "${SGLANG_API_KEY}" \
  --host 0.0.0.0 \
  --port 30000 \
  "$@"
