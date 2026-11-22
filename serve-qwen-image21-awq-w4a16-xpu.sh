#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$(realpath "$0")")/serve-qwen-image21-int8-xpu.sh" \
  --model-path "$HOME/models/Qwen-Image-2.1-Turbo-DiT-W8A8-TE-W4A16-LLMC" "$@"
