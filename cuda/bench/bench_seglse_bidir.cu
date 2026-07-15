// NVBench target for tsgu::seglse_bidir (spec/commit.md Phase 3 commit 13,
// T3; kernels.md Family 2). Axes are the kernel's tuning knobs
// (cuda/csrc/kernels/logsumexp/seglse_bidir.cu): number of rows, average
// nse per row (sparsity), and n_cols (column-axis width, which drives
// spinlock contention on the atomic column path). Standalone CMake +
// FetchContent, same pattern as bench_seglse.cu (deliberately independent
// of the kernel-builder Nix build).

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <random>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kFlatThreads = 256;

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

__device__ __forceinline__ void atomic_update_lse(int *lock, float *max_arr, float *sum_arr, int32_t *count_arr,
                                                    int64_t idx, float value) {
  while (atomicCAS(&lock[idx], 0, 1) != 0) {
  }
  __threadfence();
  float m = max_arr[idx];
  float s = sum_arr[idx];
  if (value <= m) {
    s += expf(value - m);
  } else {
    s = s * expf(m - value) + 1.0f;
    m = value;
  }
  max_arr[idx] = m;
  sum_arr[idx] = s;
  count_arr[idx] += 1;
  __threadfence();
  atomicExch(&lock[idx], 0);
}

__global__ void bidir_init_kernel(float *padded, int64_t padded_n, float *col_max, float *col_sum,
                                   int32_t *col_count, int *lock, int64_t col_n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < padded_n) {
    padded[idx] = -INFINITY;
  }
  if (idx < col_n) {
    col_max[idx] = -INFINITY;
    col_sum[idx] = 0.0f;
    col_count[idx] = 0;
    lock[idx] = 0;
  }
}

__global__ void bidir_fwd_kernel(float *padded, float const *vals, int32_t const *rowptr, int32_t const *col,
                                  float *col_max, float *col_sum, int32_t *col_count, int *lock, int64_t total_segs,
                                  int64_t n, int64_t m, int64_t G, int64_t B, bool include_zeros) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t seg = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (seg >= total_segs) {
    return;
  }
  int64_t b = seg / n;
  int64_t r = seg % n;
  int32_t start = rowptr[seg];
  int32_t end = rowptr[seg + 1];

  OnlineLSE acc = OnlineLSE::identity();
  for (int32_t i = start + static_cast<int32_t>(lane); i < end; i += kWarpSize) {
    float v = vals[i];
    acc.update(v);
    int64_t c = static_cast<int64_t>(col[i]);
    atomic_update_lse(lock, col_max, col_sum, col_count, b * m + c, v);
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
    padded[1 * B * G + b * G + r] = acc.log_sum_exp();
  }
}

__global__ void bidir_col_finalize_kernel(float *padded, float const *col_max, float const *col_sum,
                                           int32_t const *col_count, int64_t B, int64_t n, int64_t m, int64_t G,
                                           bool include_zeros) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= B * m) {
    return;
  }
  int64_t b = idx / m;
  int64_t c = idx % m;
  OnlineLSE acc{col_max[idx], col_sum[idx]};
  if (include_zeros) {
    float z = static_cast<float>(n - static_cast<int64_t>(col_count[idx]));
    if (z > 0.0f) {
      acc = OnlineLSE::combine(acc, OnlineLSE{0.0f, z});
    }
  }
  padded[0 * B * G + b * G + c] = acc.log_sum_exp();
}

void make_synthetic(int64_t num_rows, int64_t nse_per_row, int64_t n_cols, thrust::host_vector<int32_t> &rowptr_h,
                     thrust::host_vector<int32_t> &col_h, thrust::host_vector<float> &vals_h) {
  rowptr_h.resize(num_rows + 1);
  int32_t offset = 0;
  std::mt19937 rng(0);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  std::uniform_int_distribution<int32_t> coldist(0, static_cast<int32_t>(n_cols - 1));
  for (int64_t s = 0; s < num_rows; ++s) {
    rowptr_h[s] = offset;
    offset += static_cast<int32_t>(nse_per_row);
  }
  rowptr_h[num_rows] = offset;
  vals_h.resize(offset);
  col_h.resize(offset);
  for (auto &v : vals_h) {
    v = dist(rng);
  }
  for (auto &c : col_h) {
    c = coldist(rng);
  }
}

void bench_seglse_bidir_fwd(nvbench::state &state) {
  int64_t const num_rows = state.get_int64("num_rows");
  int64_t const nse_per_row = state.get_int64("nse_per_row");
  int64_t const n_cols = state.get_int64("n_cols");
  bool const include_zeros = state.get_int64("include_zeros") != 0;
  int64_t const B = 1;
  int64_t const G = std::max(num_rows, n_cols);

  thrust::host_vector<int32_t> rowptr_h, col_h;
  thrust::host_vector<float> vals_h;
  make_synthetic(num_rows, nse_per_row, n_cols, rowptr_h, col_h, vals_h);

  thrust::device_vector<int32_t> rowptr = rowptr_h;
  thrust::device_vector<int32_t> col = col_h;
  thrust::device_vector<float> vals = vals_h;
  thrust::device_vector<float> padded(2 * B * G);
  int64_t col_n = B * n_cols;
  thrust::device_vector<float> col_max(col_n), col_sum(col_n);
  thrust::device_vector<int32_t> col_count(col_n);
  thrust::device_vector<int> lock(col_n);

  state.exec([&](nvbench::launch &launch) {
    int64_t init_blocks = (std::max(2 * B * G, col_n) + kFlatThreads - 1) / kFlatThreads;
    bidir_init_kernel<<<init_blocks, kFlatThreads, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(padded.data()), 2 * B * G, thrust::raw_pointer_cast(col_max.data()),
        thrust::raw_pointer_cast(col_sum.data()), thrust::raw_pointer_cast(col_count.data()),
        thrust::raw_pointer_cast(lock.data()), col_n);

    int64_t blocks = (num_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    bidir_fwd_kernel<<<blocks, kThreadsPerBlock, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(padded.data()), thrust::raw_pointer_cast(vals.data()),
        thrust::raw_pointer_cast(rowptr.data()), thrust::raw_pointer_cast(col.data()),
        thrust::raw_pointer_cast(col_max.data()), thrust::raw_pointer_cast(col_sum.data()),
        thrust::raw_pointer_cast(col_count.data()), thrust::raw_pointer_cast(lock.data()), num_rows, num_rows, n_cols,
        G, B, include_zeros);

    int64_t fin_blocks = (col_n + kFlatThreads - 1) / kFlatThreads;
    bidir_col_finalize_kernel<<<fin_blocks, kFlatThreads, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(padded.data()), thrust::raw_pointer_cast(col_max.data()),
        thrust::raw_pointer_cast(col_sum.data()), thrust::raw_pointer_cast(col_count.data()), B, num_rows, n_cols, G,
        include_zeros);
  });
}
}  // namespace

NVBENCH_BENCH(bench_seglse_bidir_fwd)
    .add_int64_axis("num_rows", {1 << 10, 1 << 14, 1 << 18})
    .add_int64_axis("nse_per_row", {8, 64, 512})
    .add_int64_axis("n_cols", {1 << 10, 1 << 14})
    .add_int64_axis("include_zeros", {0, 1});
