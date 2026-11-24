// SPDX-License-Identifier: Apache-2.0
// Flex 170 rowwise group-256 ConvRot for the Qwen TP2 token layout.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <cstdint>

namespace es = sycl::ext::intel::esimd;
using bf16 = sycl::ext::oneapi::bfloat16;
using es::simd;

namespace {
// The first H4 group in the validated Triton kernel rounds the four initial
// sums/differences toward zero. FastTwoSum recovers the rounding residual
// from an RN sum, so only a result rounded away from zero moves one ULP.
template<int N>
inline simd<float, N> add_rz(simd<float, N> a, simd<float, N> b) {
  auto use_b = es::abs(b) > es::abs(a);
  simd<float, N> large = a, small = b;
  large.merge(b, use_b);
  small.merge(a, use_b);
  simd<float, N> sum = a + b;
  simd<float, N> tail = small - (sum - large);
  auto bits = sum.template bit_cast_view<uint32_t>().read();
  auto rem = tail.template bit_cast_view<uint32_t>().read();
  auto adjust = (tail != 0.f) & (((bits ^ rem) >> 31) != 0) &
                (es::abs(large) < sycl::bit_cast<float>(0x7f800000U));
  bits.merge(bits - 1U, adjust);
  return bits.template bit_cast_view<float>().read();
}

inline void first(simd<float, 256>& v) {
  simd<float, 64> a = v.select<64, 4>(0).read(), b = v.select<64, 4>(1).read();
  simd<float, 64> c = v.select<64, 4>(2).read(), d = v.select<64, 4>(3).read();
  simd<float, 64> ab = add_rz(a, b), cd = add_rz(c, d);
  simd<float, 64> dab = add_rz<64>(a, -b), dcd = add_rz<64>(c, -d);
  v.select<64, 4>(0) = ab + dcd;
  v.select<64, 4>(1) = ab - dcd;
  v.select<64, 4>(2) = cd + dab;
  v.select<64, 4>(3) = cd - dab;
}

template<int Stride>
inline void butterfly(simd<float, 256>& v) {
#pragma unroll
  for (int i = 0; i < 256 / (4 * Stride); ++i) {
    int start = i * 4 * Stride;
    simd<float, Stride> a = v.template select<Stride, 1>(start);
    simd<float, Stride> b = v.template select<Stride, 1>(start + Stride);
    simd<float, Stride> c = v.template select<Stride, 1>(start + 2 * Stride);
    simd<float, Stride> d = v.template select<Stride, 1>(start + 3 * Stride);
    simd<float, Stride> ab = a + b, cd = c + d;
    v.template select<Stride, 1>(start) = ab + (c - d);
    v.template select<Stride, 1>(start + Stride) = ab + (d - c);
    v.template select<Stride, 1>(start + 2 * Stride) = cd + (a - b);
    v.template select<Stride, 1>(start + 3 * Stride) = cd + (b - a);
  }
}

// Exponent limits keep all four H4 stages in normal FP32 arithmetic. Zeros
// are common and exact. Re-evaluate an entire row with the original kernel
// for subnormals, nonfinite data, and extreme products/modulation factors.
inline auto outside_domain(simd<uint16_t, 256> bits) {
  auto magnitude = bits & 0x7fffU;
  return (magnitude >= 0x5f80U) | ((magnitude != 0) & (magnitude < 0x1f80U));
}

template<int Width>
void launch(sycl::queue& queue, const uint16_t* x, const uint16_t* auxiliary,
            int8_t* values, float* scales, uint8_t* repair) {
  constexpr bool modulated = Width == 4096;
  constexpr bool paired = Width == 6144;
  constexpr int banks = paired ? 32 : Width / 256;
  constexpr int rows = paired ? 4096 : 2048;
  queue.parallel_for(sycl::nd_range<1>(rows * banks, banks),
      [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
    es::slm_init<banks * sizeof(uint32_t)>();
    int group = item.get_local_linear_id(), row = item.get_global_linear_id() / banks;
    bool valid = !paired || group < 24;
    int col = group * 256;
    simd<float, 256> v = 0;
    bool retry = false;
    if (valid) {
      int base = paired ? (row % 2048) * 24576 + (row / 2048) * 12288 + col : row * Width + col;
      auto bits = es::block_load<uint16_t, 256>(x + base);
      auto unusual = outside_domain(bits);
      if constexpr (modulated) {
        auto s = es::block_load<uint16_t, 256>(auxiliary + col);
        unusual |= outside_domain(s);
        auto mod = es::convert<float>(s.template bit_cast_view<bf16>().read());
        auto factor = es::convert<bf16>(mod + 1.f);
        auto gate = es::convert<float>(bits.template bit_cast_view<bf16>().read());
        auto activated = es::convert<bf16>(gate * es::convert<float>(factor));
        bits = activated.template bit_cast_view<uint16_t>().read();
      } else if constexpr (paired) {
        simd<uint16_t, 256> lookup;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          auto offsets = es::convert<uint32_t>(bits.template select<32, 1>(32 * i).read()) * 2U;
          lookup.template select<32, 1>(32 * i) = es::gather<uint16_t, 32>(auxiliary, offsets);
        }
        auto up = es::block_load<uint16_t, 256>(x + base + Width);
        unusual |= outside_domain(lookup) | outside_domain(up);
        auto gate = es::convert<float>(lookup.template bit_cast_view<bf16>().read());
        auto activated = es::convert<bf16>(gate * es::convert<float>(up.template bit_cast_view<bf16>().read()));
        bits = activated.template bit_cast_view<uint16_t>().read();
      }
      unusual |= outside_domain(bits);
      retry = unusual.any();
      v = es::convert<float>(bits.template bit_cast_view<bf16>().read());
      first(v);
      butterfly<4>(v);
      butterfly<16>(v);
      butterfly<64>(v);
      v *= 0.0625f;
    }
    auto magnitude = v.template bit_cast_view<uint32_t>().read() & 0x7fffffffU;
    auto high = es::hmax<uint32_t>(magnitude) | (retry ? 0x80000000U : 0);
    es::slm_scalar_store<uint32_t>(group * 4, high);
    es::barrier();
    if (valid) {
      auto peaks = es::slm_block_load<uint32_t, banks>(0);
      unsigned maxbits = es::hmax<uint32_t>(peaks);
      if (group == 0) repair[row] = maxbits >> 31;
      if (maxbits >> 31) return;
      float peak = sycl::bit_cast<float>(maxbits);
      float s = (peak < 1e-10f ? 1e-10f : peak) * (1.f / 127);
      if (group == 0) scales[row] = s;
      simd<float, 256> quotient = v * (1.f / s);
      auto distance = es::abs(quotient - es::rnde<float>(quotient));
      if ((distance > 0.49997f).any()) quotient = es::div_ieee(v, s);
      auto codes = es::convert<int8_t>(es::rnde<float>(quotient));
      es::block_store(values + row * Width + col, codes);
    }
  });
}
}  // namespace

extern "C" int flex_convrot_pack(void* stream, const uint16_t* x, const uint16_t* auxiliary,
                                 int8_t* values, float* scales, uint8_t* repair, int width) noexcept {
  if (!stream || !x || !values || !scales || !repair || (width != 2048 && !auxiliary)) return 1;
  try {
    auto& q = *static_cast<sycl::queue*>(stream);
    switch (width) {
      case 2048: launch<2048>(q, x, auxiliary, values, scales, repair); break;
      case 4096: launch<4096>(q, x, auxiliary, values, scales, repair); break;
      case 6144: launch<6144>(q, x, auxiliary, values, scales, repair); break;
      default: return 1;
    }
    return 0;
  } catch (const sycl::exception&) {
    return 1;
  }
}
