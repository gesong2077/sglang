#!/usr/bin/env bash
set -euo pipefail
exec "$(dirname "$(realpath "$0")")/serve-qwen-image21-xpu.sh" "$@"
