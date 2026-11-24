#!/usr/bin/env bash
# Build the optional prepacked Qwen BF16 gate/up bridge for Flex 170.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
pkg="$root/python/sglang/srt/hardware_backend/xpu/kernels"
dnnl_root=${DNNL_ROOT:-/opt/intel/oneapi/dnnl/latest}
tbb_root=${TBBROOT:-/opt/intel/oneapi/tbb/latest}
compiler=${ICPX:-/opt/intel/oneapi/compiler/latest/bin/icpx}
compiler_lib=$(cd "$(dirname "$compiler")/../lib" && pwd)
out=$(mktemp "$pkg/.libflex_bf16_mlp.XXXXXXXX")
trap 'rm -f "$out"' EXIT
"$compiler" -std=c++17 -O2 -fsycl -shared -fPIC \
    --gcc-install-dir="$(dirname "$(gcc -print-libgcc-file-name)")" \
    "$pkg/flex_bf16_mlp.cpp" -I"$dnnl_root/include" -L"$dnnl_root/lib" \
    '-Wl,-rpath,$ORIGIN:$ORIGIN/../../../../../../..' \
    "-Wl,-rpath,$dnnl_root/lib:$compiler_lib:$tbb_root/lib" -ldnnl -o "$out"
chmod 755 "$out"
mv -f "$out" "$pkg/libflex_bf16_mlp.so"
