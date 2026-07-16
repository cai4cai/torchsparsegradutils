// NVBench target for tsgu::sddmm (spec/commit.md Phase 3 commit 14, T3;
// kernels.md Family 1). Axes are the kernel's tuning knobs
// (cuda/csrc/kernels/sddmm/sddmm.cu): total specified entries (`nse_total`),
// the dense inner dimension `p` (each warp's reduction length), batch size
// `B` (affects `total_rows = B * n`), and `path` — 0 = warp-per-entry
// fallback, 1 = row-tiled hot path (G row staged in shared memory; the
// launcher's production heuristic picks it when avg entries/row >= 2 and the
// staging buffer fits 48KB). Standalone CMake + FetchContent, same pattern
// as bench_seglse.cu (deliberately independent of the kernel-builder Nix
// build).
//
// A vendor-baseline (cuSPARSE cusparseSDDMM, CSR, unbatched) row is *not*
// duplicated here: benchmarks.md's acceptance layer is the op-level Python
// harness (benchmarks/bench_sddmm.py), which measures it via
// torch.sparse.sampled_addmm (documented to dispatch to cusparseSDDMM for
// CUDA CSR inputs) -- the same vendor primitive, without hand-rolling a
// second cuSPARSE handle/descriptor lifecycle in this standalone NVBench
// binary. This target is for kernel-level tuning/regression-bisection only
// (benchmarks.md §1: "NVBench is for development and regression bisection
// ... never gates").

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <random>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffffU, val, offset);
  }
  return val;
}

template <typename index_t>
__device__ __forceinline__ int64_t find_row(index_t const *__restrict__ rowptr, int64_t total_rows, int64_t k) {
  int64_t lo = 0;
  int64_t hi = total_rows;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (static_cast<int64_t>(rowptr[mid + 1]) <= k) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__global__ void sddmm_kernel(float *__restrict__ out, int32_t const *__restrict__ rowptr,
                              int32_t const *__restrict__ col, float const *__restrict__ g,
                              float const *__restrict__ mat, int64_t total_rows, int64_t nse_total, int64_t n,
                              int64_t m, int64_t p) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t k = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (k >= nse_total) {
    return;
  }

  int64_t row_g = 0;
  if (lane == 0) {
    row_g = find_row<int32_t>(rowptr, total_rows, k);
  }
  row_g = __shfl_sync(0xffffffffU, row_g, 0);

  int64_t b = row_g / n;
  int64_t local_row = row_g % n;
  int64_t col_k = static_cast<int64_t>(col[k]);

  float const *g_row = g + (b * n + local_row) * p;
  float const *mat_row = mat + (b * m + col_k) * p;

  float partial = 0.0f;
  for (int64_t j = lane; j < p; j += kWarpSize) {
    partial += g_row[j] * mat_row[j];
  }
  partial = warp_reduce_sum(partial);

  if (lane == 0) {
    out[k] = partial;
  }
}

// Row-tiled hot path — mirrors csrc/kernels/sddmm/sddmm.cu's
// sddmm_row_kernel: one warp per folded row, G row staged once through
// dynamic shared memory, one warp-reduced dot per entry of the row.
__global__ void sddmm_row_kernel(float *__restrict__ out, int32_t const *__restrict__ rowptr,
                                  int32_t const *__restrict__ col, float const *__restrict__ g,
                                  float const *__restrict__ mat, int64_t total_rows, int64_t n, int64_t m,
                                  int64_t p) {
  extern __shared__ float g_smem_block[];

  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t row_g = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (row_g >= total_rows) {
    return;
  }

  int64_t start = static_cast<int64_t>(rowptr[row_g]);
  int64_t end = static_cast<int64_t>(rowptr[row_g + 1]);
  if (start == end) {
    return;
  }

  float *g_smem = g_smem_block + warp_in_block * p;
  float const *g_row = g + row_g * p;
  for (int64_t j = lane; j < p; j += kWarpSize) {
    g_smem[j] = g_row[j];
  }
  __syncwarp();

  int64_t b = row_g / n;
  float const *mat_b = mat + b * m * p;

  for (int64_t k = start; k < end; ++k) {
    float const *mat_row = mat_b + static_cast<int64_t>(col[k]) * p;
    float partial = 0.0f;
    for (int64_t j = lane; j < p; j += kWarpSize) {
      partial += g_smem[j] * mat_row[j];
    }
    partial = warp_reduce_sum(partial);
    if (lane == 0) {
      out[k] = partial;
    }
  }
}

// Builds a synthetic CSR-like `rowptr` (average `nse_per_row` specified
// entries per folded row, `total_rows = B * n`) and dense `g`/`mat` operands
// -- mirrors benchmarks.md's synthetic tier (bench_seglse.cu's
// `make_synthetic` sibling for this op's shape).
void make_synthetic(int64_t total_rows, int64_t nse_total, int64_t m, thrust::host_vector<int32_t> &rowptr_h,
                     thrust::host_vector<int32_t> &col_h) {
  rowptr_h.resize(total_rows + 1);
  col_h.resize(nse_total);
  std::mt19937 rng(0);
  std::uniform_int_distribution<int32_t> col_dist(0, static_cast<int32_t>(m - 1));

  int64_t base = nse_total / total_rows;
  int64_t remainder = nse_total % total_rows;
  int64_t offset = 0;
  for (int64_t r = 0; r < total_rows; ++r) {
    rowptr_h[r] = static_cast<int32_t>(offset);
    int64_t this_row_nse = base + (r < remainder ? 1 : 0);
    for (int64_t i = 0; i < this_row_nse; ++i) {
      col_h[offset + i] = col_dist(rng);
    }
    offset += this_row_nse;
  }
  rowptr_h[total_rows] = static_cast<int32_t>(offset);
}

void bench_sddmm_fwd(nvbench::state &state) {
  int64_t const nse_total = state.get_int64("nse_total");
  int64_t const p = state.get_int64("p");
  int64_t const B = state.get_int64("B");
  bool const row_path = state.get_int64("path") == 1;

  size_t const row_smem = static_cast<size_t>(kWarpsPerBlock) * static_cast<size_t>(p) * sizeof(float);
  if (row_path && row_smem > 48 * 1024) {
    state.skip("row path exceeds 48KB shared-memory window at this p");
    return;
  }

  // ~8 specified entries/row on average (DLMC-shaped sparsity), split evenly
  // across B batch items.
  int64_t total_rows = std::max<int64_t>(B, nse_total / 8);
  total_rows -= total_rows % B;  // n = total_rows / B must be exact
  if (total_rows < B) {
    total_rows = B;
  }
  int64_t n = total_rows / B;
  int64_t m = n;  // square pattern

  thrust::host_vector<int32_t> rowptr_h, col_h;
  make_synthetic(total_rows, nse_total, m, rowptr_h, col_h);
  int64_t actual_nse = rowptr_h[total_rows];

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  thrust::host_vector<float> g_h(B * n * p), mat_h(B * m * p);
  for (auto &v : g_h) v = dist(rng);
  for (auto &v : mat_h) v = dist(rng);

  thrust::device_vector<int32_t> rowptr = rowptr_h;
  thrust::device_vector<int32_t> col = col_h;
  thrust::device_vector<float> g = g_h;
  thrust::device_vector<float> mat = mat_h;
  thrust::device_vector<float> out(actual_nse);

  state.exec([&](nvbench::launch &launch) {
    if (row_path) {
      int64_t blocks = (total_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
      sddmm_row_kernel<<<blocks, kThreadsPerBlock, row_smem, launch.get_stream()>>>(
          thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(rowptr.data()),
          thrust::raw_pointer_cast(col.data()), thrust::raw_pointer_cast(g.data()),
          thrust::raw_pointer_cast(mat.data()), total_rows, n, m, p);
    } else {
      int64_t blocks = (actual_nse + kWarpsPerBlock - 1) / kWarpsPerBlock;
      sddmm_kernel<<<blocks, kThreadsPerBlock, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(rowptr.data()),
          thrust::raw_pointer_cast(col.data()), thrust::raw_pointer_cast(g.data()),
          thrust::raw_pointer_cast(mat.data()), total_rows, actual_nse, n, m, p);
    }
  });
}
}  // namespace

NVBENCH_BENCH(bench_sddmm_fwd)
    .add_int64_axis("nse_total", {1 << 14, 1 << 17, 1 << 20})
    .add_int64_axis("p", {1, 8, 32, 128, 512})
    .add_int64_axis("B", {1, 8, 64})
    .add_int64_axis("path", {0, 1});
