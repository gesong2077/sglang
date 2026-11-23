#!/usr/bin/env bash
# Use an icpx matching the libsycl loaded by the target XPU PyTorch wheel.
set -eu
pkg=$(cd "$(dirname "$0")" && pwd)
compiler=${ICPX:-/opt/intel/oneapi/compiler/latest/bin/icpx}
out=$(mktemp "$pkg/.libflex_tp2_sum.XXXXXXXX")
trap 'rm -f "$out"' EXIT
"$compiler" -O3 -std=c++17 -fsycl -ffp-model=precise -fPIC -shared \
    --gcc-install-dir="$(dirname "$(gcc -print-libgcc-file-name)")" \
    "$pkg/flex_tp2_sum.cpp" -lze_loader -o "$out"
chmod 755 "$out"
mv -f "$out" "$pkg/libflex_tp2_sum.so"
