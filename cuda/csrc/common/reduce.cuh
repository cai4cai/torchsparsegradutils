#pragma once
// Warp reduction helpers + online log-sum-exp accumulator (architecture.md
// §5). Header-only device functions; used for real by kernel commits
// (Phase 3, e.g. the seglse family — kernels.md Family 2) and merely parsed
// (and instantiated once, on scratch values) by the smoke TU
// (cuda/csrc/kernels/_smoke/_smoke.cu) in this bring-up commit.

#include <cuda_runtime.h>

namespace tsgu {

// Full-warp sum reduction: every lane active in `mask` ends up holding the
// masked lanes' total. `mask` defaults to a full 32-lane warp.
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val, unsigned mask = 0xffffffffU) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}

// Full-warp max reduction, same masking convention as warp_reduce_sum.
template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_max(scalar_t val, unsigned mask = 0xffffffffU) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    scalar_t other = __shfl_down_sync(mask, val, offset);
    val = val > other ? val : other;
  }
  return val;
}

// Online (streaming) log-sum-exp accumulator: folds one value at a time into
// a running (max, sum-of-exp-shifted-by-max) pair without ever needing every
// value materialised at once (kernels.md Family 2's "no recompute" forward
// pass). combine() merges two partial accumulators (e.g. across warps).
template <typename scalar_t>
struct OnlineLogSumExp {
  scalar_t max_val;
  scalar_t sum_exp;

  __device__ __forceinline__ static OnlineLogSumExp identity() { return OnlineLogSumExp{-INFINITY, scalar_t(0)}; }

  __device__ __forceinline__ void update(scalar_t value) {
    if (value <= max_val) {
      sum_exp += exp(value - max_val);
      return;
    }
    sum_exp = sum_exp * exp(max_val - value) + scalar_t(1);
    max_val = value;
  }

  __device__ __forceinline__ static OnlineLogSumExp combine(OnlineLogSumExp a, OnlineLogSumExp b) {
    if (a.max_val == b.max_val) {
      return OnlineLogSumExp{a.max_val, a.sum_exp + b.sum_exp};
    }
    if (a.max_val > b.max_val) {
      return OnlineLogSumExp{a.max_val, a.sum_exp + (b.sum_exp * exp(b.max_val - a.max_val))};
    }
    return OnlineLogSumExp{b.max_val, b.sum_exp + (a.sum_exp * exp(a.max_val - b.max_val))};
  }

  __device__ __forceinline__ scalar_t log_sum_exp() const { return max_val + log(sum_exp); }
};

}  // namespace tsgu
