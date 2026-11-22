#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "MemoryAccess.h"
#include "Norm.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
#include <cutlass/util/GPU_Clock.hpp>
#endif

namespace at::native::xpu {
// Flatten tensor to 2D (M, N) for the kernel.  If the tensor is already 2D it
// is returned unchanged; 3D tensors are viewed as 2D.  Uses view() so that the
// returned tensor always shares storage with the original (no copy).
static inline Tensor flatten_to_2d(const Tensor& t, int64_t M, int64_t N) {
  if (t.dim() == 2) {
    return t;
  }
  return t.view({M, N});
}

// Describes how a flattened row index (0 .. M-1) maps to a byte offset on a
// 2D/3D/4D tensor without requiring a contiguous copy.  A row is decomposed
// into up to 3 nested indices (outer, inner0, inner1), from outermost to
// innermost:
//
//   outer  = r / (inner0_size * inner1_size)
//   inner0 = (r % (inner0_size * inner1_size)) / inner1_size
//   inner1 = r % inner1_size
//   offset(r) = outer * batch_stride + inner0 * inner0_stride + inner1 * inner1_stride
//
// For 2D and flattenable 3D tensors, inner0_size == inner1_size == 1 so the
// formula collapses to `offset(r) = r * batch_stride`.  For non-flattenable
// 3D tensors (e.g. a per-head slice of a packed QKV buffer reshaped to
// (tokens, heads, head_dim)) inner0_size/inner0_stride describe the second
// dim, with inner1_size == 1.  4D tensors additionally use inner1 when the
// leading dim (dim 0) is independent of dim 1 (e.g. a batched QKV slice),
// otherwise dim 0 folds away and only the 2-level form is needed.
struct RowStrides {
  int64_t batch_stride;
  int64_t inner0_size;
  int64_t inner0_stride;
  int64_t inner1_size;
  int64_t inner1_stride;
};

static inline RowStrides get_row_strides(const Tensor& t) {
  TORCH_CHECK(
      t.dim() == 2 || t.dim() == 3 || t.dim() == 4, "get_row_strides: expected a 2D/3D/4D tensor, got ", t.dim(), "D");
  if (t.dim() == 2) {
    return {t.stride(0), 1, 0, 1, 0};
  }

  // dim0/dim1 always participate in both the 3D and 4D shapes, so the fold
  // check between them is shared; dim2 only exists for 4D.
  int64_t d0_size = t.size(0);
  int64_t d0_stride = t.stride(0);
  int64_t d1_size = t.size(1);
  int64_t d1_stride = t.stride(1);
  bool d0_folds_into_d1 = (d0_size == 1) || (d0_stride == d1_size * d1_stride);

  if (t.dim() == 3) {
    if (d0_folds_into_d1) {
      return {d1_stride, 1, 0, 1, 0};
    }
    return {d0_stride, d1_size, d1_stride, 1, 0};
  }

  // t.dim() == 4: shape (d0, d1, d2, plane); row = d0*d1*d2.
  int64_t d2_size = t.size(2);
  int64_t d2_stride = t.stride(2);
  bool d1_folds_into_d2 = (d1_size == 1) || (d1_stride == d2_size * d2_stride);

  if (d0_folds_into_d1 && d1_folds_into_d2) {
    // Everything collapses into a single stride.
    return {d2_stride, 1, 0, 1, 0};
  }
  if (d0_folds_into_d1) {
    // dim0 contributes nothing; (d1, d2) alone need the 2-level form.
    return {d1_stride, d2_size, d2_stride, 1, 0};
  }
  // Fully independent: need the full 3-level (d0, d1, d2) decomposition.
  return {d0_stride, d1_size, d1_stride, d2_size, d2_stride};
}

// Shared 3-level row-offset computation used by every kernel site below; see
// the RowStrides comment for the (outer, inner0, inner1) decomposition.
template <typename index_t>
static inline index_t compute_row_offset(
    index_t row,
    index_t batch_stride,
    index_t inner0_size,
    index_t inner0_stride,
    index_t inner1_size,
    index_t inner1_stride) {
  index_t combined_inner = inner0_size * inner1_size;
  index_t outer_idx = row / combined_inner;
  index_t rem = row % combined_inner;
  index_t inner0_idx = rem / inner1_size;
  index_t inner1_idx = rem % inner1_size;
  return outer_idx * batch_stride + inner0_idx * inner0_stride + inner1_idx * inner1_stride;
}

template <typename scalar_t>
class NormNoRstdForward {
 public:
  using accscalar_t = acc_type<scalar_t>;
  NormNoRstdForward() = delete;
  NormNoRstdForward(scalar_t* X_data, scalar_t* Y_data, accscalar_t eps) : X_data(X_data), Y_data(Y_data), eps(eps) {}

  int get_update_vec_size(int Plane, int vec_size) const {
    return get_aligned_update_vec_size(Plane, vec_size, X_data, Y_data);
  }

 protected:
  template <typename... Args>
  int get_aligned_update_vec_size(int Plane, int vec_size, Args*... args) const {
    vec_size = get_min_vec_size(vec_size, args...);
    while (Plane % vec_size != 0) {
      vec_size = vec_size >> 1;
    }
    return vec_size;
  }

 public:
  scalar_t* X_data;
  scalar_t* Y_data;
  accscalar_t eps;
};

template <typename scalar_t, typename weight_t>
class RMSNormNoRstdForward : public NormNoRstdForward<scalar_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef NormNoRstdForward<scalar_t> NF;
  RMSNormNoRstdForward() = delete;
  RMSNormNoRstdForward(scalar_t* X_data, scalar_t* Y_data, weight_t* gamma_data, accscalar_t eps)
      : NormNoRstdForward<scalar_t>(X_data, Y_data, eps), gamma_data(gamma_data) {}

  int get_update_vec_size(int Plane, int vec_size) const {
    return NF::get_aligned_update_vec_size(Plane, vec_size, NF::X_data, NF::Y_data, gamma_data);
  }

  // Per-element weight application, the single place the normalized value meets
  // gamma.  Shared by the cached and non-cached update() loops and by the
  // AddNoRstdForward mixin; GemmaRMSNormNoRstdForward hides it with the
  // `1 + gamma` variant.
  static inline scalar_t apply_weight(accscalar_t x, accscalar_t rstd, accscalar_t gamma) {
    return static_cast<scalar_t>(x * rstd * gamma);
  }

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename index_t, typename Cfg>
  void reduce_combine(
      sycl::nd_item<1> item_id,
      const Cfg& cfg,
      index_t x_group_offset,
      accscalar_t& sum_value,
      vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    const index_t lid = item_id.get_local_id(0);

    if constexpr (cache_inputs) {
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const index_t plane_offset = (static_cast<index_t>(it) * cfg.workgroup_size + lid) * vec_size;
        if (plane_offset < cfg.Plane) {
          vec_t x_val = *(reinterpret_cast<vec_t*>(NF::X_data + x_group_offset + plane_offset));
          reg[it] = x_val;
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            const accscalar_t x = static_cast<accscalar_t>(x_val[v]);
            sum_value += x * x;
          }
        }
      }
    } else {
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        vec_t x_val = *(reinterpret_cast<vec_t*>(NF::X_data + x_group_offset + plane_offset));
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          const accscalar_t x = static_cast<accscalar_t>(x_val[v]);
          sum_value += x * x;
        }
      }
    }
  }

  template <typename Cfg>
  accscalar_t reduce_project(sycl::nd_item<1> item_id, const Cfg& cfg, accscalar_t sum_value) const {
    sum_value = sycl::reduce_over_group(item_id.get_group(), sum_value, sycl::plus<accscalar_t>());
    sum_value = sum_value < static_cast<accscalar_t>(0) ? static_cast<accscalar_t>(0) : sum_value;
    return Numerics<accscalar_t>::rsqrt(
        sum_value / static_cast<accscalar_t>(cfg.Plane) + static_cast<accscalar_t>(NF::eps));
  }

  template <
      int vec_size,
      int ITERS,
      bool cache_inputs,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename Cfg>
  void update(
      sycl::nd_item<1> item_id,
      const Cfg& cfg,
      index_t x_group_offset,
      index_t y_group_offset,
      accscalar_t rstd,
      const vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    const index_t lid = item_id.get_local_id(0);

    if constexpr (cache_inputs) {
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const index_t plane_offset = (static_cast<index_t>(it) * cfg.workgroup_size + lid) * vec_size;
        if (plane_offset < cfg.Plane) {
          vec_t x_val = reg[it];
          weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(gamma_data + plane_offset));
          vec_t y_val;
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            y_val[v] = apply_weight(static_cast<accscalar_t>(x_val[v]), rstd, static_cast<accscalar_t>(gamma_val[v]));
          }
          *(reinterpret_cast<vec_t*>(NF::Y_data + y_group_offset + plane_offset)) = y_val;
        }
      }
    } else {
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        vec_t x_val = *(reinterpret_cast<vec_t*>(NF::X_data + x_group_offset + plane_offset));
        weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(gamma_data + plane_offset));
        vec_t y_val;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          y_val[v] = apply_weight(static_cast<accscalar_t>(x_val[v]), rstd, static_cast<accscalar_t>(gamma_val[v]));
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + y_group_offset + plane_offset)) = y_val;
      }
    }
  }

 public:
  weight_t* gamma_data;
};

// Residual-add mixin for the register-cached no-rstd norms.  `BaseNorm` supplies
// reduce_project() and the per-element weight application (apply_weight); this
// only adds `x += add` in front of the reduction, so the llama-style and the
// Gemma-style fused-add norms share one implementation.
//
// reduce_combine() never stores the sum back to X_data: it goes to add_data,
// plus reg[] when the inputs are cached.  The only write to the x buffer is
// update()'s store through Y_data (which for the in-place export *is* x, after
// the reduction).  So the non-cached update() must re-read the sum from
// add_data and NOT from X_data, which still holds the original, un-summed x.
template <typename BaseNorm, typename scalar_t, typename weight_t>
class AddNoRstdForward : public BaseNorm {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef BaseNorm Base;
  typedef NormNoRstdForward<scalar_t> NF;
  AddNoRstdForward() = delete;
  AddNoRstdForward(scalar_t* X_data, scalar_t* Y_data, weight_t* gamma_data, accscalar_t eps, scalar_t* add_data)
      : BaseNorm(X_data, Y_data, gamma_data, eps), add_data(add_data) {}

  int get_update_vec_size(int Plane, int vec_size) const {
    return NF::get_aligned_update_vec_size(Plane, vec_size, NF::X_data, NF::Y_data, Base::gamma_data, add_data);
  }

  template <int vec_size, int ITERS, bool cache_inputs, typename vec_t, typename index_t, typename Cfg>
  void reduce_combine(
      sycl::nd_item<1> item_id,
      const Cfg& cfg,
      index_t x_group_offset,
      accscalar_t& sum_value,
      vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    const index_t lid = item_id.get_local_id(0);

    if constexpr (cache_inputs) {
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const index_t plane_offset = (static_cast<index_t>(it) * cfg.workgroup_size + lid) * vec_size;
        if (plane_offset < cfg.Plane) {
          vec_t x_val = *(reinterpret_cast<vec_t*>(NF::X_data + x_group_offset + plane_offset));
          vec_t add_val = *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset));
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            x_val[v] = static_cast<scalar_t>(static_cast<accscalar_t>(x_val[v]) + static_cast<accscalar_t>(add_val[v]));
          }
          *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset)) = x_val;
          reg[it] = x_val;
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            const accscalar_t x = static_cast<accscalar_t>(x_val[v]);
            sum_value += x * x;
          }
        }
      }
    } else {
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        vec_t x_val = *(reinterpret_cast<vec_t*>(NF::X_data + x_group_offset + plane_offset));
        vec_t add_val = *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset));
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          x_val[v] = static_cast<scalar_t>(static_cast<accscalar_t>(x_val[v]) + static_cast<accscalar_t>(add_val[v]));
        }
        *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset)) = x_val;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          const accscalar_t x = static_cast<accscalar_t>(x_val[v]);
          sum_value += x * x;
        }
      }
    }
  }

  template <
      int vec_size,
      int ITERS,
      bool cache_inputs,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename Cfg>
  void update(
      sycl::nd_item<1> item_id,
      const Cfg& cfg,
      index_t x_group_offset,
      index_t y_group_offset,
      accscalar_t rstd,
      const vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    if constexpr (cache_inputs) {
      Base::template update<vec_size, ITERS, cache_inputs, vec_t, weight_vec_t, index_t>(
          item_id, cfg, x_group_offset, y_group_offset, rstd, reg);
    } else {
      const index_t lid = item_id.get_local_id(0);
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        // add_data, not X_data: the residual sum was stored there by
        // reduce_combine, while X_data still holds the un-summed x.
        vec_t x_val = *(reinterpret_cast<vec_t*>(add_data + x_group_offset + plane_offset));
        weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(Base::gamma_data + plane_offset));
        vec_t y_val;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          y_val[v] =
              Base::apply_weight(static_cast<accscalar_t>(x_val[v]), rstd, static_cast<accscalar_t>(gamma_val[v]));
        }
        *(reinterpret_cast<vec_t*>(NF::Y_data + y_group_offset + plane_offset)) = y_val;
      }
    }
  }

 public:
  scalar_t* add_data;
};

template <typename scalar_t, typename weight_t>
using AddRMSNormNoRstdForward = AddNoRstdForward<RMSNormNoRstdForward<scalar_t, weight_t>, scalar_t, weight_t>;

template <typename scalar_t, typename weight_t>
class GemmaRMSNormNoRstdForward : public RMSNormNoRstdForward<scalar_t, weight_t> {
 public:
  using accscalar_t = acc_type<scalar_t>;
  typedef RMSNormNoRstdForward<scalar_t, weight_t> RNF;
  GemmaRMSNormNoRstdForward() = delete;
  using RNF::RNF;

  // Hides RMSNormNoRstdForward::apply_weight: Gemma stores gamma - 1.
  static inline scalar_t apply_weight(accscalar_t x, accscalar_t rstd, accscalar_t gamma) {
    return static_cast<scalar_t>(x * rstd * (static_cast<accscalar_t>(1.0) + gamma));
  }

  template <
      int vec_size,
      int ITERS,
      bool cache_inputs,
      typename vec_t,
      typename weight_vec_t,
      typename index_t,
      typename Cfg>
  void update(
      sycl::nd_item<1> item_id,
      const Cfg& cfg,
      index_t x_group_offset,
      index_t y_group_offset,
      accscalar_t rstd,
      const vec_t (&reg)[cache_inputs ? ITERS : 1]) const {
    const index_t lid = item_id.get_local_id(0);

    if constexpr (cache_inputs) {
#pragma unroll
      for (int it = 0; it < ITERS; ++it) {
        const index_t plane_offset = (static_cast<index_t>(it) * cfg.workgroup_size + lid) * vec_size;
        if (plane_offset < cfg.Plane) {
          vec_t x_val = reg[it];
          weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(RNF::gamma_data + plane_offset));
          vec_t y_val;
#pragma unroll
          for (int v = 0; v < vec_size; ++v) {
            y_val[v] = apply_weight(static_cast<accscalar_t>(x_val[v]), rstd, static_cast<accscalar_t>(gamma_val[v]));
          }
          *(reinterpret_cast<vec_t*>(RNF::Y_data + y_group_offset + plane_offset)) = y_val;
        }
      }
    } else {
      for (index_t plane_offset = lid * vec_size; plane_offset < cfg.Plane;
           plane_offset += cfg.workgroup_size * vec_size) {
        vec_t x_val = *(reinterpret_cast<vec_t*>(RNF::X_data + x_group_offset + plane_offset));
        weight_vec_t gamma_val = *(reinterpret_cast<weight_vec_t*>(RNF::gamma_data + plane_offset));
        vec_t y_val;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          y_val[v] = apply_weight(static_cast<accscalar_t>(x_val[v]), rstd, static_cast<accscalar_t>(gamma_val[v]));
        }
        *(reinterpret_cast<vec_t*>(RNF::Y_data + y_group_offset + plane_offset)) = y_val;
      }
    }
  }
};

template <typename scalar_t, typename weight_t>
using GemmaAddRMSNormNoRstdForward =
    AddNoRstdForward<GemmaRMSNormNoRstdForward<scalar_t, weight_t>, scalar_t, weight_t>;

template <
    typename scalar_t,
    typename weight_t,
    int vec_size,
    int ITERS,
    typename Norm,
    bool cache_inputs,
    typename index_t = uint32_t,
    typename Cfg = NormConfig>
struct RMSNormNoRstdKernelFunctor {
  using accscalar_t = acc_type<scalar_t>;
  using vec_t = aligned_vector_loop<scalar_t, vec_size>;
  using weight_vec_t = aligned_vector_loop<weight_t, vec_size>;

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item_id) const {
    const index_t row = item_id.get_group(0);
    const index_t x_group_offset = compute_row_offset<index_t>(
        row,
        cfg.input_batch_stride,
        cfg.input_inner0_size,
        cfg.input_inner0_stride,
        cfg.input_inner1_size,
        cfg.input_inner1_stride);
    const index_t y_group_offset = compute_row_offset<index_t>(
        row,
        cfg.output_batch_stride,
        cfg.output_inner0_size,
        cfg.output_inner0_stride,
        cfg.output_inner1_size,
        cfg.output_inner1_stride);

    accscalar_t sum_value = 0;
    vec_t reg[cache_inputs ? ITERS : 1];
    norm.template reduce_combine<vec_size, ITERS, cache_inputs, vec_t, index_t>(
        item_id, cfg, x_group_offset, sum_value, reg);
    const accscalar_t rstd = norm.reduce_project(item_id, cfg, sum_value);
    norm.template update<vec_size, ITERS, cache_inputs, vec_t, weight_vec_t, index_t>(
        item_id, cfg, x_group_offset, y_group_offset, rstd, reg);
  }

  RMSNormNoRstdKernelFunctor(Norm norm_, Cfg cfg_) : norm(norm_), cfg(cfg_) {}

 private:
  Norm norm;
  const Cfg cfg;
};

template <typename scalar_t, typename weight_t, int vec_size, int ITERS, typename Norm, bool cache_inputs>
void rmsnorm_no_rstd_kernel(Norm& norm, const NormConfig& config) {
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  using KernelFunctor = RMSNormNoRstdKernelFunctor<scalar_t, weight_t, vec_size, ITERS, Norm, cache_inputs>;

  KernelFunctor kfn(norm, config);
  sycl::range<1> local_range{static_cast<size_t>(config.workgroup_size)};
  sycl::range<1> global_range{static_cast<size_t>(config.workgroup_num * config.workgroup_size)};
  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

template <typename scalar_t, typename weight_t, int vec_size, typename Norm>
void dispatch_rmsnorm_no_rstd_iters(int iters, Norm& norm, const NormConfig& config) {
#define RMSNORM_NO_RSTD_CACHED_CASE(ITERS) \
  rmsnorm_no_rstd_kernel<scalar_t, weight_t, vec_size, ITERS, Norm, true>(norm, config)

  if (iters <= 1) {
    RMSNORM_NO_RSTD_CACHED_CASE(1);
  } else if (iters == 2) {
    RMSNORM_NO_RSTD_CACHED_CASE(2);
  } else if (iters == 3) {
    RMSNORM_NO_RSTD_CACHED_CASE(3);
  } else if (iters == 4) {
    RMSNORM_NO_RSTD_CACHED_CASE(4);
  } else if (iters == 5) {
    RMSNORM_NO_RSTD_CACHED_CASE(5);
  } else if (iters == 6) {
    RMSNORM_NO_RSTD_CACHED_CASE(6);
  } else if (iters == 7) {
    RMSNORM_NO_RSTD_CACHED_CASE(7);
  } else if (iters == 8) {
    RMSNORM_NO_RSTD_CACHED_CASE(8);
  } else {
    rmsnorm_no_rstd_kernel<scalar_t, weight_t, vec_size, 1, Norm, false>(norm, config);
  }
#undef RMSNORM_NO_RSTD_CACHED_CASE
}

template <typename scalar_t, typename weight_t, typename Norm>
void launch_vectorized_rmsnorm_no_rstd_kernel(Norm& norm, const NormConfig& config) {
  const int vec_size = config.update_vec_size;
  const int iters = (config.WGPlane + config.workgroup_size * vec_size - 1) / (config.workgroup_size * vec_size);

#define DISPATCH_RMSNORM_NO_RSTD_VEC(VEC_SIZE) \
  dispatch_rmsnorm_no_rstd_iters<scalar_t, weight_t, VEC_SIZE>(iters, norm, config)

  switch (vec_size) {
    case 8: {
      DISPATCH_RMSNORM_NO_RSTD_VEC(8);
      break;
    }
    case 4: {
      DISPATCH_RMSNORM_NO_RSTD_VEC(4);
      break;
    }
    case 2: {
      DISPATCH_RMSNORM_NO_RSTD_VEC(2);
      break;
    }
    default: {
      DISPATCH_RMSNORM_NO_RSTD_VEC(1);
      break;
    }
  }
#undef DISPATCH_RMSNORM_NO_RSTD_VEC
}

// Launch for contiguous rows built with NormConfig(FastRowTag): NormFastParams makes every inner
// size/stride a compile-time constant, so compute_row_offset folds its integer div/mod (Xe has no
// hardware divide) to row * batch_stride. 1, 2, 4 or 8 passes per row are instantiated; any other
// count takes the general launcher.
template <typename scalar_t, typename weight_t, int vec_size, typename Norm>
void launch_fast_rmsnorm_no_rstd_kernel(Norm& norm, const NormConfig& config) {
  const int iters = (config.WGPlane + config.workgroup_size * vec_size - 1) / (config.workgroup_size * vec_size);
  auto launch = [&](auto iters_tag) {
    using KernelFunctor = RMSNormNoRstdKernelFunctor<
        scalar_t,
        weight_t,
        vec_size,
        decltype(iters_tag)::value,
        Norm,
        /*cache_inputs=*/true,
        uint32_t,
        NormFastParams>;
    const NormFastParams fast_config{
        config.Plane, config.workgroup_size, config.input_batch_stride, config.output_batch_stride};
    const sycl::range<1> local_range{static_cast<size_t>(config.workgroup_size)};
    const sycl::range<1> global_range{
        static_cast<size_t>(config.workgroup_num) * static_cast<size_t>(config.workgroup_size)};
    sycl_kernel_submit(
        global_range, local_range, at::xpu::getCurrentXPUStream().queue(), KernelFunctor(norm, fast_config));
  };
  if (iters == 1) {
    launch(std::integral_constant<int, 1>{});
  } else if (iters == 2) {
    launch(std::integral_constant<int, 2>{});
  } else if (iters == 4) {
    launch(std::integral_constant<int, 4>{});
  } else if (iters == 8) {
    launch(std::integral_constant<int, 8>{});
  } else {
    launch_vectorized_rmsnorm_no_rstd_kernel<scalar_t, weight_t>(norm, config);
  }
}


}
