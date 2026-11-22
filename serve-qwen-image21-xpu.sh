#!/usr/bin/env bash
set -euo pipefail
venv="$HOME/sgl-xpu"
original_model="$HOME/models/Qwen-Image-2.1"
model_path="$original_model"
if [[ "${1:-}" == "--model-path" ]]; then
  model_path="${2:?missing model directory}"
  shift 2
fi
# Triton compiles SYCL launchers with the installed GCC 15 C++ headers.
export PATH="/opt/intel/oneapi/compiler/2026.1/bin:$PATH"
export CCC_OVERRIDE_OPTIONS="+--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/15"
export PYTHONPATH="$(dirname "$(realpath "$0")")/python${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ "$model_path" == "$original_model" || "$model_path" == "$original_model-Turbo" ]]; then
  export CCL_SYCL_COPY_ENGINE=1
else
  export SGLANG_XPU_W8A8_ONEDNN=1
  export SGLANG_XPU_W8A8_ONEDNN_LIB="$venv/lib/sglang-w8a8/libonednn_w8a8.so"
fi
exec "$venv/bin/sglang" serve \
  --model-path "$model_path" \
  --num-gpus 2 --tp-size 2 --ulysses-degree 1 \
  --attention-backend fa --performance-mode speed --vae-config.parallel-decode-mode spatial_shard \
  --host 0.0.0.0 --port 30010 "$@"
