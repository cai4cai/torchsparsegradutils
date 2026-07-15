// NVBench target for tsgu::seglse (spec/commit.md Phase 3 commit 12, T3;
// kernels.md Family 2). Axes are the kernel's tuning knobs
// (cuda/csrc/kernels/logsumexp/seglse.cu): total number of segments,
// average segment length (nse per segment, i.e. sparsity), and
// `include_zeros` (whether the structural-zero fold-in branch is taken).
// Standalone CMake + FetchContent, same pattern as bench_smoke.cu
// (deliberately independent of the kernel-builder Nix build).

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <random>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

struct OnlineLSE {
  float max_val;
  float sum_exp;

  __device__ __forceinline__ static OnlineLSE identity() { return OnlineLSE{-INFINITY, 0.0f}; }

  __device__ __forceinline__ void update(float value) {
    if (value <= max_val) {
      sum_exp += expf(value - max_val);
      return;
    }
    sum_exp = sum_exp * expf(max_val - value) + 1.0f;
    max_val = value;
  }

  __device__ __forceinline__ static OnlineLSE combine(OnlineLSE a, OnlineLSE b) {
    if (a.max_val == b.max_val) {
      return OnlineLSE{a.max_val, a.sum_exp + b.sum_exp};
    }
    if (a.max_val > b.max_val) {
      return OnlineLSE{a.max_val, a.sum_exp + (b.sum_exp * expf(b.max_val - a.max_val))};
    }
    return OnlineLSE{b.max_val, b.sum_exp + (a.sum_exp * expf(a.max_val - b.max_val))};
  }

  __device__ __forceinline__ float log_sum_exp() const { return max_val + logf(sum_exp); }
};

__global__ void seglse_fwd_kernel(float *__restrict__ lse, float const *__restrict__ vals,
                                   int32_t const *__restrict__ rowptr, int64_t total_segs, int64_t m,
                                   bool include_zeros) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t seg = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (seg >= total_segs) {
    return;
  }

  int32_t start = rowptr[seg];
  int32_t end = rowptr[seg + 1];

  OnlineLSE acc = OnlineLSE::identity();
  for (int32_t i = start + static_cast<int32_t>(lane); i < end; i += kWarpSize) {
    acc.update(vals[i]);
  }

#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    float other_max = __shfl_down_sync(0xffffffffU, acc.max_val, offset);
    float other_sum = __shfl_down_sync(0xffffffffU, acc.sum_exp, offset);
    acc = OnlineLSE::combine(acc, OnlineLSE{other_max, other_sum});
  }

  if (lane == 0) {
    if (include_zeros) {
      float z = static_cast<float>(m - static_cast<int64_t>(end - start));
      if (z > 0.0f) {
        acc = OnlineLSE::combine(acc, OnlineLSE{0.0f, z});
      }
    }
    lse[seg] = acc.log_sum_exp();
  }
}

// Builds a synthetic CSR-like `rowptr` for `num_segments` segments each with
// `nse_per_segment` specified entries (the benchmark's density knob) and
// fills `vals` with random floats — mirrors benchmarks.md's synthetic tier.
void make_synthetic(int64_t num_segments, int64_t nse_per_segment, thrust::host_vector<int32_t> &rowptr_h,
                     thrust::host_vector<float> &vals_h) {
  rowptr_h.resize(num_segments + 1);
  int32_t offset = 0;
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  for (int64_t s = 0; s < num_segments; ++s) {
    rowptr_h[s] = offset;
    offset += static_cast<int32_t>(nse_per_segment);
  }
  rowptr_h[num_segments] = offset;
  vals_h.resize(offset);
  for (auto &v : vals_h) {
    v = dist(rng);
  }
}

void bench_seglse_fwd(nvbench::state &state) {
  int64_t const num_segments = state.get_int64("num_segments");
  int64_t const nse_per_segment = state.get_int64("nse_per_segment");
  bool const include_zeros = state.get_int64("include_zeros") != 0;
  int64_t const m = nse_per_segment * 4;  // full axis size (m >= nse so zero-fold branch is exercised)

  thrust::host_vector<int32_t> rowptr_h;
  thrust::host_vector<float> vals_h;
  make_synthetic(num_segments, nse_per_segment, rowptr_h, vals_h);

  thrust::device_vector<int32_t> rowptr = rowptr_h;
  thrust::device_vector<float> vals = vals_h;
  thrust::device_vector<float> lse(num_segments);

  state.exec([&](nvbench::launch &launch) {
    int64_t blocks = (num_segments + kWarpsPerBlock - 1) / kWarpsPerBlock;
    seglse_fwd_kernel<<<blocks, kThreadsPerBlock, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(lse.data()), thrust::raw_pointer_cast(vals.data()),
        thrust::raw_pointer_cast(rowptr.data()), num_segments, m, include_zeros);
  });
}
}  // namespace

NVBENCH_BENCH(bench_seglse_fwd)
    .add_int64_axis("num_segments", {1 << 10, 1 << 14, 1 << 18})
    .add_int64_axis("nse_per_segment", {8, 64, 512})
    .add_int64_axis("include_zeros", {0, 1});
