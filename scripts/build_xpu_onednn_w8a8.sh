#!/usr/bin/env bash
# Build the optional SYCL/oneDNN W8A8 host bridge before building the XPU wheel.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
pkg="$root/python/sglang/srt/hardware_backend/xpu/quantization"
dnnl_root=${DNNL_ROOT:-/opt/intel/oneapi/dnnl/latest}
tbb_root=${TBBROOT:-/opt/intel/oneapi/tbb/latest}
compiler=${ICPX:-/opt/intel/oneapi/compiler/latest/bin/icpx}
compiler_lib=$(cd "$(dirname "$compiler")/../lib" && pwd)
out=$(mktemp "$pkg/.libonednn_w8a8.XXXXXXXX")
trap 'rm -f "$out"' EXIT
"$compiler" -std=c++17 -O2 -fsycl -shared -fPIC "$pkg/onednn_w8a8.cpp" \
  -I"$dnnl_root/include" -L"$dnnl_root/lib" -Wl,--disable-new-dtags \
  '-Wl,-rpath,$ORIGIN:$ORIGIN/../../../../../../..' \
  "-Wl,-rpath,$dnnl_root/lib:$compiler_lib:$tbb_root/lib" -ldnnl -o "$out"
chmod 755 "$out"
mv -f "$out" "$pkg/libonednn_w8a8.so"

# Group-256 ESIMD rotation is an optional ACM-G10 device image in the same
# wheel. Keep both native bridges linked to the active SYCL runtime.
out=$(mktemp "$pkg/.libflex_convrot.XXXXXXXX")
"$compiler" -std=c++20 -O2 -fp-model=precise -ffp-contract=off -fPIC -shared \
  -fsycl -fsycl-targets=intel_gpu_acm_g10 -Xs "-options '-doubleGRF'" \
  "$pkg/flex_convrot.cpp" -Wl,--disable-new-dtags \
  '-Wl,-rpath,$ORIGIN:$ORIGIN/../../../../../../..' \
  "-Wl,-rpath,$compiler_lib" -o "$out"
chmod 755 "$out"
mv -f "$out" "$pkg/libflex_convrot.so"
