// Copyright 2026 SGLang Team.
// SPDX-License-Identifier: Apache-2.0
//
// Batch-one 128x128 block-scaled FP8 GEMV for RDNA4.  The implementation is
// derived from the gfx12 skinny-GEMM strategy used by vLLM's ROCm backend,
// specialized here to the only contract SGLang needs: [1,K] @ [N,K]^T with
// OCP E4M3 inputs, FP32 block scales, and BF16 output.

#pragma once

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

namespace sglang {

namespace rdna4_fp8_skinny_detail {

constexpr int kWaveSize = 32;
constexpr int kWavesPerBlock = 16;
constexpr int kRowsPerWave = 2;
constexpr int kChunkBytes = 16;
constexpr int kUnroll = 2;
constexpr int kMaxActivationBytes = 64 * 1024;

// The MTP lm_head uses the FP8 GEMV above only to build a shortlist.  These
// three kernels reproduce the exact-candidate refinement used by the local
// vLLM ROCm backend: take one approximate winner from each balanced vocab
// partition, rescore those rows with the original BF16 head, then reduce the
// exact scores to one shard-local winner.  This avoids scanning the full BF16
// head while retaining far better token agreement than a raw FP8 argmax.
constexpr int kRefinePartitionThreads = 256;
constexpr int kRefineDotThreads = 512;
constexpr int kRefineDotWaves = kRefineDotThreads / kWaveSize;
constexpr int kRefineMaxThreads = 512;

struct RefineMaxPair {
  float value;
  int index;
};

__device__ __forceinline__ bool refine_pair_is_better(
    const RefineMaxPair& candidate, const RefineMaxPair& current) {
  return candidate.value > current.value ||
         (candidate.value == current.value && candidate.index < current.index);
}

__device__ __forceinline__ RefineMaxPair refine_wave_reduce_max_pair(
    RefineMaxPair pair) {
#pragma unroll
  for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
    RefineMaxPair other{__shfl_down(pair.value, offset),
                        __shfl_down(pair.index, offset)};
    if (refine_pair_is_better(other, pair)) pair = other;
  }
  return pair;
}

__global__ __launch_bounds__(kRefinePartitionThreads)
void rdna4_fp8_lm_head_partition_top1_kernel(
    const __hip_bfloat16* __restrict__ logits,
    int* __restrict__ candidate_indices,
    int vocab_size,
    int partitions) {
  const int partition = blockIdx.x;
  const int batch = blockIdx.y;
  const int start = static_cast<int>(
      (static_cast<int64_t>(partition) * vocab_size) / partitions);
  const int end = static_cast<int>(
      (static_cast<int64_t>(partition + 1) * vocab_size) / partitions);

  RefineMaxPair pair{-INFINITY, INT_MAX};
  for (int row = start + threadIdx.x; row < end; row += blockDim.x) {
    RefineMaxPair candidate{
        __bfloat162float(logits[static_cast<int64_t>(batch) * vocab_size + row]),
        row};
    if (refine_pair_is_better(candidate, pair)) pair = candidate;
  }

  pair = refine_wave_reduce_max_pair(pair);
  __shared__ float wave_values[kRefinePartitionThreads / kWaveSize];
  __shared__ int wave_indices[kRefinePartitionThreads / kWaveSize];
  const int lane = threadIdx.x % kWaveSize;
  const int wave = threadIdx.x / kWaveSize;
  if (lane == 0) {
    wave_values[wave] = pair.value;
    wave_indices[wave] = pair.index;
  }
  __syncthreads();

  if (wave == 0) {
    constexpr int kWaves = kRefinePartitionThreads / kWaveSize;
    pair = lane < kWaves
               ? RefineMaxPair{wave_values[lane], wave_indices[lane]}
               : RefineMaxPair{-INFINITY, INT_MAX};
    pair = refine_wave_reduce_max_pair(pair);
    if (lane == 0) {
      candidate_indices[static_cast<int64_t>(batch) * partitions + partition] =
          pair.index;
    }
  }
}

__global__ __launch_bounds__(kRefineDotThreads)
void rdna4_fp8_lm_head_candidate_dot_kernel(
    const __hip_bfloat16* __restrict__ hidden,
    const __hip_bfloat16* __restrict__ weight,
    const int* __restrict__ candidate_indices,
    __hip_bfloat16* __restrict__ candidate_logits,
    int hidden_size,
    int partitions) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* shared_hidden = reinterpret_cast<__hip_bfloat16*>(shared_bytes);
  const int batch = blockIdx.y;
  for (int k = threadIdx.x; k < hidden_size; k += blockDim.x) {
    shared_hidden[k] = hidden[static_cast<int64_t>(batch) * hidden_size + k];
  }
  __syncthreads();

  const int lane = threadIdx.x % kWaveSize;
  const int wave = threadIdx.x / kWaveSize;
  const int partition = blockIdx.x * kRefineDotWaves + wave;
  if (partition >= partitions) return;

  const int candidate_row =
      candidate_indices[static_cast<int64_t>(batch) * partitions + partition];
  const auto* candidate_weight =
      weight + static_cast<int64_t>(candidate_row) * hidden_size;
  float sum = 0.0f;
  for (int k = lane; k < hidden_size; k += kWaveSize) {
    sum += __bfloat162float(shared_hidden[k]) *
           __bfloat162float(candidate_weight[k]);
  }
#pragma unroll
  for (int offset = kWaveSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down(sum, offset);
  }
  if (lane == 0) {
    candidate_logits[static_cast<int64_t>(batch) * partitions + partition] =
        __float2bfloat16(sum);
  }
}

__global__ __launch_bounds__(kRefineMaxThreads)
void rdna4_fp8_lm_head_candidate_top1_kernel(
    const __hip_bfloat16* __restrict__ candidate_logits,
    const int* __restrict__ candidate_indices,
    __hip_bfloat16* __restrict__ out_values,
    int64_t* __restrict__ out_indices,
    int partitions) {
  const int batch = blockIdx.x;
  RefineMaxPair pair{-INFINITY, INT_MAX};
  for (int partition = threadIdx.x; partition < partitions;
       partition += blockDim.x) {
    const int offset = batch * partitions + partition;
    RefineMaxPair candidate{__bfloat162float(candidate_logits[offset]),
                            candidate_indices[offset]};
    if (refine_pair_is_better(candidate, pair)) pair = candidate;
  }

  pair = refine_wave_reduce_max_pair(pair);
  __shared__ float wave_values[kRefineMaxThreads / kWaveSize];
  __shared__ int wave_indices[kRefineMaxThreads / kWaveSize];
  const int lane = threadIdx.x % kWaveSize;
  const int wave = threadIdx.x / kWaveSize;
  if (lane == 0) {
    wave_values[wave] = pair.value;
    wave_indices[wave] = pair.index;
  }
  __syncthreads();

  if (wave == 0) {
    constexpr int kWaves = kRefineMaxThreads / kWaveSize;
    pair = lane < kWaves
               ? RefineMaxPair{wave_values[lane], wave_indices[lane]}
               : RefineMaxPair{-INFINITY, INT_MAX};
    pair = refine_wave_reduce_max_pair(pair);
    if (lane == 0) {
      out_values[batch] = __float2bfloat16(pair.value);
      out_indices[batch] = static_cast<int64_t>(pair.index);
    }
  }
}

using int4_vec = int __attribute__((__vector_size__(4 * sizeof(int))));
using float4_vec = float __attribute__((__vector_size__(4 * sizeof(float))));

union Pack16 {
  unsigned char bytes[kChunkBytes];
  int words[kChunkBytes / sizeof(int)];
  int4_vec vec;
};

template <typename T>
__device__ __forceinline__ T load_nontemporal(const T* ptr) {
  return __builtin_nontemporal_load(ptr);
}

__device__ __forceinline__ float reduce_wave32(float value) {
  asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
      : "=v"(value)
      : "0"(value), "v"(value), "v"(value));
  asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
      : "=v"(value)
      : "0"(value), "v"(value), "v"(value));
  asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
      : "=v"(value)
      : "0"(value), "v"(value), "v"(value));
  asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:1 bound_ctrl:0 "
      : "=v"(value)
      : "0"(value), "v"(value), "v"(value));
  return value + __shfl_xor(value, 16);
}

__device__ __forceinline__ float bf16_dot2(float lhs, float rhs) {
  const auto lhs2 = *reinterpret_cast<const __hip_bfloat162*>(&lhs);
  const auto rhs2 = *reinterpret_cast<const __hip_bfloat162*>(&rhs);
  const float2 lhs_f32 = __bfloat1622float2(lhs2);
  const float2 rhs_f32 = __bfloat1622float2(rhs2);
  return lhs_f32.x * rhs_f32.x + lhs_f32.y * rhs_f32.y;
}

// Find the smallest active wave count that preserves the number of output
// rounds produced by the full 16-wave launch.
inline int active_waves_for_rows(int rows, int cu_count) {
  const int rows_per_full_round = cu_count * kRowsPerWave * kWavesPerBlock;
  const int rounds = (rows + rows_per_full_round - 1) / rows_per_full_round;
  for (int waves = 1; waves <= kWavesPerBlock; ++waves) {
    const int candidate = cu_count * kRowsPerWave * waves;
    if ((rows + candidate - 1) / candidate == rounds) {
      return waves;
    }
  }
  return kWavesPerBlock;
}

__global__ __launch_bounds__(kWaveSize * kWavesPerBlock)
void rdna4_fp8_blockwise_skinny_kernel(
    const unsigned char* __restrict__ activation,
    const unsigned char* __restrict__ weight,
    const float* __restrict__ activation_scale,
    const float* __restrict__ weight_scale,
    __hip_bfloat16* __restrict__ output,
    int rows,
    int hidden,
    int active_waves,
    int cu_count) {
  __shared__ __align__(16) unsigned char activation_smem[kMaxActivationBytes];

  const int linear_thread = threadIdx.y * kWaveSize + threadIdx.x;
  const int block_threads = kWaveSize * kWavesPerBlock;
  for (int k = linear_thread * kChunkBytes; k < hidden; k += block_threads * kChunkBytes) {
    *reinterpret_cast<int4_vec*>(&activation_smem[k]) =
        *reinterpret_cast<const int4_vec*>(&activation[k]);
  }
  __syncthreads();

  if (threadIdx.y >= active_waves) {
    return;
  }

  int row = (blockIdx.x * active_waves + threadIdx.y) * kRowsPerWave;
  const int row_stride = cu_count * active_waves * kRowsPerWave;
  const int hidden_blocks = hidden / 128;

  while (row < rows) {
    float sums[kRowsPerWave] = {};

    for (int k_base = 0;
         k_base < hidden;
         k_base += kWaveSize * kChunkBytes * kUnroll) {
      Pack16 weights[kRowsPerWave][kUnroll];
      Pack16 activations[kUnroll] = {};

#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int k = k_base + u * kWaveSize * kChunkBytes + threadIdx.x * kChunkBytes;
        const int logical_k = min(k, hidden - kChunkBytes);
#pragma unroll
        for (int y = 0; y < kRowsPerWave; ++y) {
          const int logical_row = min(row + y, rows - 1);
          const auto* ptr = weight + static_cast<int64_t>(logical_row) * hidden + logical_k;
          weights[y][u].vec = load_nontemporal(reinterpret_cast<const int4_vec*>(ptr));
        }
        if (k < hidden) {
          activations[u].vec =
              *reinterpret_cast<const int4_vec*>(&activation_smem[k]);
        }
      }

#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int k = k_base + u * kWaveSize * kChunkBytes + threadIdx.x * kChunkBytes;
        const int scale_k = k / 128;
        if (scale_k < hidden_blocks) {
          const float a_scale = activation_scale[scale_k];
          // row is even and 128 is divisible by two, so both rows share the
          // same weight-scale row.
          const int scale_row = row / 128;
#pragma unroll
          for (int y = 0; y < kRowsPerWave; ++y) {
            const float scale = a_scale * weight_scale[scale_row * hidden_blocks + scale_k];
#pragma unroll
            for (int i = 0; i < kChunkBytes / 4; ++i) {
              const float dot = __builtin_amdgcn_dot4_f32_fp8_fp8(
                  activations[u].words[i], weights[y][u].words[i], 0.0f);
              sums[y] += dot * scale;
            }
          }
        }
      }
    }

#pragma unroll
    for (int y = 0; y < kRowsPerWave; ++y) {
      asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:8 bound_ctrl:0 "
          : "=v"(sums[y])
          : "0"(sums[y]), "v"(sums[y]), "v"(sums[y]));
      asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:4 bound_ctrl:0 "
          : "=v"(sums[y])
          : "0"(sums[y]), "v"(sums[y]), "v"(sums[y]));
      asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:2 bound_ctrl:0 "
          : "=v"(sums[y])
          : "0"(sums[y]), "v"(sums[y]), "v"(sums[y]));
      asm("s_nop 0\n\tv_add_f32 %0, %2, %3 row_shr:1 bound_ctrl:0 "
          : "=v"(sums[y])
          : "0"(sums[y]), "v"(sums[y]), "v"(sums[y]));
      sums[y] += __shfl_xor(sums[y], 16);
    }

    if (threadIdx.x == kWaveSize - 1) {
#pragma unroll
      for (int y = 0; y < kRowsPerWave; ++y) {
        if (row + y < rows) {
          output[row + y] = __float2bfloat16(sums[y]);
        }
      }
    }

    row += row_stride;
  }
}

// Exact Qwen3.5/3.8-27B TP2/TP4 M=1 GDN input projection. The FP8 qkvz and
// BF16 ba projections consume the same hidden state. Keeping them in one
// launch lets the small ba projection reuse the launch residency and LDS
// allocation of the qkvz projection. This is intentionally a separate,
// exact-shape kernel so generic GEMV dispatch remains unchanged.
constexpr int kGdnHidden = 5120;
constexpr int kGdnTp4QkvzRows = 4096;
constexpr int kGdnTp4BaRows = 24;
constexpr int kGdnTp2QkvzRows = 8192;
constexpr int kGdnTp2BaRows = 48;
constexpr int kGdnGrid = 32;
constexpr int kBaChunkElements = 8;
constexpr int kBaUnroll = 4;

__global__ __launch_bounds__(kWaveSize * kWavesPerBlock)
void rdna4_gdn_dual_projection_kernel(
    const unsigned char* __restrict__ q_activation,
    const unsigned char* __restrict__ q_weight,
    const float* __restrict__ q_activation_scale,
    const float* __restrict__ q_weight_scale,
    const __hip_bfloat16* __restrict__ hidden,
    const __hip_bfloat16* __restrict__ ba_weight,
    __hip_bfloat16* __restrict__ q_output,
    __hip_bfloat16* __restrict__ ba_output,
    int q_rows,
    int ba_rows) {
  __shared__ __align__(16) unsigned char shared[kMaxActivationBytes];

  const int lane = threadIdx.x;
  const int wave = threadIdx.y;
  const int linear_thread = wave * kWaveSize + lane;

  const int activation_offset = linear_thread * kChunkBytes;
  if (activation_offset < kGdnHidden) {
    *reinterpret_cast<int4_vec*>(&shared[activation_offset]) =
        *reinterpret_cast<const int4_vec*>(&q_activation[activation_offset]);
  }
  __syncthreads();

  int row = (blockIdx.x * kWavesPerBlock + wave) * kRowsPerWave;
  constexpr int kRowStride = kGdnGrid * kWavesPerBlock * kRowsPerWave;
  constexpr int kHiddenBlocks = kGdnHidden / 128;
  while (row < q_rows) {
    float sums[kRowsPerWave] = {};

#pragma unroll
    for (int k_base = 0;
         k_base < kGdnHidden;
         k_base += kWaveSize * kChunkBytes * kUnroll) {
      Pack16 weights[kRowsPerWave][kUnroll];
      Pack16 activations[kUnroll];

#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int k = k_base + u * kWaveSize * kChunkBytes + lane * kChunkBytes;
        activations[u].vec =
            *reinterpret_cast<const int4_vec*>(&shared[k]);
#pragma unroll
        for (int y = 0; y < kRowsPerWave; ++y) {
          weights[y][u].vec = load_nontemporal(
              reinterpret_cast<const int4_vec*>(
                  q_weight + static_cast<int64_t>(row + y) * kGdnHidden + k));
        }
      }

#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const int k = k_base + u * kWaveSize * kChunkBytes + lane * kChunkBytes;
        const int scale_k = k / 128;
        const float a_scale = q_activation_scale[scale_k];
#pragma unroll
        for (int y = 0; y < kRowsPerWave; ++y) {
          const float scale =
              a_scale * q_weight_scale[(row / 128) * kHiddenBlocks + scale_k];
#pragma unroll
          for (int packed = 0; packed < kChunkBytes / 4; ++packed) {
            sums[y] += __builtin_amdgcn_dot4_f32_fp8_fp8(
                           activations[u].words[packed],
                           weights[y][u].words[packed], 0.0f) *
                       scale;
          }
        }
      }
    }

#pragma unroll
    for (int y = 0; y < kRowsPerWave; ++y) {
      sums[y] = reduce_wave32(sums[y]);
      if (lane == kWaveSize - 1) {
        q_output[row + y] = __float2bfloat16(sums[y]);
      }
    }
    row += kRowStride;
  }

  // All waves must reach this barrier before the shared allocation is reused.
  __syncthreads();
  auto* shared_bf16 = reinterpret_cast<__hip_bfloat16*>(shared);
  constexpr int kBlockThreads = kWaveSize * kWavesPerBlock;
  for (int k = linear_thread * kBaChunkElements; k < kGdnHidden;
       k += kBlockThreads * kBaChunkElements) {
    *reinterpret_cast<int4_vec*>(&shared_bf16[k]) =
        *reinterpret_cast<const int4_vec*>(&hidden[k]);
  }
  __syncthreads();

  // TP4 uses wave 0 for 24 rows. TP2 also uses wave 1 in the first 16
  // workgroups, covering 48 rows without adding another launch.
  if (wave >= 2) {
    return;
  }
  const int ba_row = blockIdx.x + wave * kGdnGrid;
  if (ba_row >= ba_rows) {
    return;
  }
  float ba_sum = 0.0f;
#pragma unroll
  for (int k_base = 0;
       k_base < kGdnHidden;
       k_base += kWaveSize * kBaChunkElements * kBaUnroll) {
#pragma unroll
    for (int u = 0; u < kBaUnroll; ++u) {
      const int k =
          k_base + u * kWaveSize * kBaChunkElements + lane * kBaChunkElements;
      const float4_vec activation =
          *reinterpret_cast<const float4_vec*>(&shared_bf16[k]);
      const float4_vec weight = load_nontemporal(
          reinterpret_cast<const float4_vec*>(
              &ba_weight[static_cast<int64_t>(ba_row) * kGdnHidden + k]));
#pragma unroll
      for (int packed = 0; packed < kBaChunkElements / 2; ++packed) {
        ba_sum += bf16_dot2(activation[packed], weight[packed]);
      }
    }
  }
  ba_sum = reduce_wave32(ba_sum);
  if (lane == kWaveSize - 1) {
    ba_output[ba_row] = __float2bfloat16(ba_sum);
  }
}

}  // namespace rdna4_fp8_skinny_detail

inline void rdna4_fp8_blockwise_skinny(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView activation_scale,
    tvm::ffi::TensorView weight_scale) {
  using namespace host;
  using namespace rdna4_fp8_skinny_detail;

  auto one = SymbolicSize{"batch"};
  auto rows = SymbolicSize{"rows"};
  auto hidden = SymbolicSize{"hidden"};
  auto hidden_blocks = SymbolicSize{"hidden_blocks"};
  auto row_blocks = SymbolicSize{"row_blocks"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  // HIP's shared TensorMatcher does not yet define an FP8 type trait, so the
  // FP8 tensors are shape/device checked here and dtype checked explicitly.
  TensorMatcher({one, hidden}).with_device(device).verify(activation);
  TensorMatcher({rows, hidden}).with_device(device).verify(weight);
  TensorMatcher({one, rows}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({one, hidden_blocks}).with_dtype<float>().with_device(device).verify(activation_scale);
  TensorMatcher({row_blocks, hidden_blocks}).with_dtype<float>().with_device(device).verify(weight_scale);

  CHECK_HOST(one.unwrap() == 1) << "rdna4_fp8_blockwise_skinny requires batch size 1";
  CHECK_HOST(rows.unwrap() > 0 && rows.unwrap() % 128 == 0)
      << "output rows must be positive and divisible by 128";
  CHECK_HOST(hidden.unwrap() > 0 && hidden.unwrap() <= kMaxActivationBytes && hidden.unwrap() % 128 == 0)
      << "hidden size must be positive, <= 65536, and divisible by 128";
  CHECK_HOST(hidden_blocks.unwrap() == hidden.unwrap() / 128)
      << "activation scale shape must be [1, hidden/128]";
  CHECK_HOST(row_blocks.unwrap() == rows.unwrap() / 128)
      << "weight scale shape must be [rows/128, hidden/128]";
  CHECK_HOST(activation.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && activation.dtype().bits == 8)
      << "activation must be OCP float8_e4m3fn";
  CHECK_HOST(weight.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn && weight.dtype().bits == 8)
      << "weight must be OCP float8_e4m3fn";

  const int row_count = static_cast<int>(rows.unwrap());
  const int hidden_size = static_cast<int>(hidden.unwrap());
  const int cu_count = static_cast<int>(host::runtime::get_sm_count(device.unwrap().device_id));
  const int active_waves = active_waves_for_rows(row_count, cu_count);

  host::LaunchKernel(dim3(cu_count), dim3(kWaveSize, kWavesPerBlock), device.unwrap())(
      rdna4_fp8_blockwise_skinny_kernel,
      static_cast<const unsigned char*>(activation.data_ptr()),
      static_cast<const unsigned char*>(weight.data_ptr()),
      static_cast<const float*>(activation_scale.data_ptr()),
      static_cast<const float*>(weight_scale.data_ptr()),
      static_cast<__hip_bfloat16*>(output.data_ptr()),
      row_count,
      hidden_size,
      active_waves,
      cu_count);
}

inline void rdna4_gdn_dual_projection(
    tvm::ffi::TensorView q_output,
    tvm::ffi::TensorView ba_output,
    tvm::ffi::TensorView hidden,
    tvm::ffi::TensorView q_activation,
    tvm::ffi::TensorView q_weight,
    tvm::ffi::TensorView q_activation_scale,
    tvm::ffi::TensorView q_weight_scale,
    tvm::ffi::TensorView ba_weight) {
  using namespace host;
  using namespace rdna4_fp8_skinny_detail;

  auto one = SymbolicSize{"batch"};
  auto hidden_size = SymbolicSize{"hidden"};
  auto q_rows = SymbolicSize{"q_rows"};
  auto ba_rows = SymbolicSize{"ba_rows"};
  auto hidden_blocks = SymbolicSize{"hidden_blocks"};
  auto q_row_blocks = SymbolicSize{"q_row_blocks"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({one, q_rows}).with_dtype<bf16_t>().with_device(device).verify(q_output);
  TensorMatcher({one, ba_rows}).with_dtype<bf16_t>().with_device(device).verify(ba_output);
  TensorMatcher({one, hidden_size}).with_dtype<bf16_t>().with_device(device).verify(hidden);
  TensorMatcher({one, hidden_size}).with_device(device).verify(q_activation);
  TensorMatcher({q_rows, hidden_size}).with_device(device).verify(q_weight);
  TensorMatcher({one, hidden_blocks}).with_dtype<float>().with_device(device).verify(q_activation_scale);
  TensorMatcher({q_row_blocks, hidden_blocks}).with_dtype<float>().with_device(device).verify(q_weight_scale);
  TensorMatcher({ba_rows, hidden_size}).with_dtype<bf16_t>().with_device(device).verify(ba_weight);

  const bool tp4_shape = q_rows.unwrap() == kGdnTp4QkvzRows &&
                         ba_rows.unwrap() == kGdnTp4BaRows;
  const bool tp2_shape = q_rows.unwrap() == kGdnTp2QkvzRows &&
                         ba_rows.unwrap() == kGdnTp2BaRows;
  CHECK_HOST(one.unwrap() == 1 && hidden_size.unwrap() == kGdnHidden &&
             (tp4_shape || tp2_shape))
      << "rdna4_gdn_dual_projection requires [1,5120] with qkvz/ba rows [4096,24] or [8192,48]";
  CHECK_HOST(hidden_blocks.unwrap() == kGdnHidden / 128 &&
             q_row_blocks.unwrap() == q_rows.unwrap() / 128)
      << "invalid block-scale shapes for rdna4_gdn_dual_projection";
  CHECK_HOST(q_activation.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
             q_activation.dtype().bits == 8 &&
             q_weight.dtype().code == DLDataTypeCode::kDLFloat8_e4m3fn &&
             q_weight.dtype().bits == 8)
      << "q activation and weight must be OCP float8_e4m3fn";
  CHECK_HOST(host::runtime::get_sm_count(device.unwrap().device_id) == kGdnGrid)
      << "rdna4_gdn_dual_projection requires a 32-CU gfx1201 GPU";

  host::LaunchKernel(dim3(kGdnGrid), dim3(kWaveSize, kWavesPerBlock), device.unwrap())(
      rdna4_gdn_dual_projection_kernel,
      static_cast<const unsigned char*>(q_activation.data_ptr()),
      static_cast<const unsigned char*>(q_weight.data_ptr()),
      static_cast<const float*>(q_activation_scale.data_ptr()),
      static_cast<const float*>(q_weight_scale.data_ptr()),
      static_cast<const __hip_bfloat16*>(hidden.data_ptr()),
      static_cast<const __hip_bfloat16*>(ba_weight.data_ptr()),
      static_cast<__hip_bfloat16*>(q_output.data_ptr()),
      static_cast<__hip_bfloat16*>(ba_output.data_ptr()),
      static_cast<int>(q_rows.unwrap()),
      static_cast<int>(ba_rows.unwrap()));
}

inline void rdna4_fp8_lm_head_refine(
    tvm::ffi::TensorView local_logits,
    tvm::ffi::TensorView hidden,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView candidate_indices,
    tvm::ffi::TensorView candidate_logits,
    tvm::ffi::TensorView out_values,
    tvm::ffi::TensorView out_indices) {
  using namespace host;
  using namespace rdna4_fp8_skinny_detail;

  auto batch = SymbolicSize{"batch"};
  auto vocab = SymbolicSize{"vocab"};
  auto hidden_size = SymbolicSize{"hidden"};
  auto partitions = SymbolicSize{"partitions"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  TensorMatcher({batch, vocab}).with_dtype<bf16_t>().with_device(device).verify(local_logits);
  TensorMatcher({batch, hidden_size}).with_dtype<bf16_t>().with_device(device).verify(hidden);
  TensorMatcher({vocab, hidden_size}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({batch, partitions}).with_dtype<int32_t>().with_device(device).verify(candidate_indices);
  TensorMatcher({batch, partitions}).with_dtype<bf16_t>().with_device(device).verify(candidate_logits);
  TensorMatcher({batch}).with_dtype<bf16_t>().with_device(device).verify(out_values);
  TensorMatcher({batch}).with_dtype<int64_t>().with_device(device).verify(out_indices);

  CHECK_HOST(batch.unwrap() >= 1 && batch.unwrap() <= 8)
      << "rdna4_fp8_lm_head_refine batch must be in [1,8]";
  CHECK_HOST(partitions.unwrap() >= 1 && partitions.unwrap() <= vocab.unwrap() &&
             partitions.unwrap() <= 4096)
      << "partitions must be in [1,min(vocab,4096)]";
  CHECK_HOST(hidden_size.unwrap() * sizeof(__hip_bfloat16) <= kMaxActivationBytes)
      << "hidden state is too large for the refinement shared-memory stage";
  CHECK_HOST(host::runtime::get_sm_count(device.unwrap().device_id) == 32)
      << "rdna4_fp8_lm_head_refine requires a 32-CU gfx1201 GPU";

  const int b = static_cast<int>(batch.unwrap());
  const int v = static_cast<int>(vocab.unwrap());
  const int h = static_cast<int>(hidden_size.unwrap());
  const int p = static_cast<int>(partitions.unwrap());

  LaunchKernel(dim3(p, b), kRefinePartitionThreads, device.unwrap())(
      rdna4_fp8_lm_head_partition_top1_kernel,
      static_cast<const __hip_bfloat16*>(local_logits.data_ptr()),
      static_cast<int*>(candidate_indices.data_ptr()), v, p);
  LaunchKernel(dim3((p + kRefineDotWaves - 1) / kRefineDotWaves, b),
               kRefineDotThreads, device.unwrap(),
               static_cast<size_t>(h) * sizeof(__hip_bfloat16))(
      rdna4_fp8_lm_head_candidate_dot_kernel,
      static_cast<const __hip_bfloat16*>(hidden.data_ptr()),
      static_cast<const __hip_bfloat16*>(weight.data_ptr()),
      static_cast<const int*>(candidate_indices.data_ptr()),
      static_cast<__hip_bfloat16*>(candidate_logits.data_ptr()), h, p);
  LaunchKernel(b, kRefineMaxThreads, device.unwrap())(
      rdna4_fp8_lm_head_candidate_top1_kernel,
      static_cast<const __hip_bfloat16*>(candidate_logits.data_ptr()),
      static_cast<const int*>(candidate_indices.data_ptr()),
      static_cast<__hip_bfloat16*>(out_values.data_ptr()),
      static_cast<int64_t*>(out_indices.data_ptr()), p);
}

}  // namespace sglang
