// NVBench target for tsgu::spmm (spec/commit.md Phase 3 commit 15, T3;
// kernels.md Family 3 SpMM row; architecture.md §5: "warp-per-folded-row,
// column-tiled"). Axes are the kernel's tuning knobs
// (cuda/csrc/kernels/spmm/spmm.cu): total specified entries (`nse_total`),
// the dense rhs width `p` (includes `p=1`, the SpMV case), and batch size
// `B`. Standalone CMake + FetchContent, same pattern as bench_sddmm.cu
// (deliberately independent of the kernel-builder Nix build).
//
// A vendor-baseline (cuSPARSE cusparseSpMM/SpMV, CSR, unbatched) row is not
// duplicated here: benchmarks.md's acceptance layer is the op-level Python
// harness (benchmarks/bench_spmm.py), which measures it via
// torch.sparse.mm / torch.matmul on a CUDA sparse_csr tensor (documented to
// dispatch to cusparseSpMM/SpMV) — the same vendor primitive, without
// hand-rolling a second cuSPARSE handle/descriptor lifecycle in this
// standalone NVBench binary. This target is for kernel-level tuning/
// regression-bisection only (benchmarks.md §1: "NVBench is for development
// and regression bisection ... never gates").

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <random>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kMaxRegTiles = 16;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffffU, val, offset);
  }
  return val;
}

// Mirrors csrc/kernels/spmm/spmm.cu's spmm_wide_kernel (p >= kWarpSize):
// column-tiled, one warp per folded row, row's entries walked once per tile.
__global__ void spmm_wide_kernel(float *__restrict__ out, int32_t const *__restrict__ rowptr,
                                  int32_t const *__restrict__ col, float const *__restrict__ vals,
                                  float const *__restrict__ dense, int64_t total_rows, int64_t n, int64_t m,
                                  int64_t p) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t row_g = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (row_g >= total_rows) {
    return;
  }

  int64_t b = row_g / n;
  int64_t start = static_cast<int64_t>(rowptr[row_g]);
  int64_t end = static_cast<int64_t>(rowptr[row_g + 1]);

  float *out_row = out + row_g * p;
  float const *dense_b = dense + b * m * p;
  int64_t const tile_width = static_cast<int64_t>(kWarpSize) * kMaxRegTiles;

  for (int64_t col_base = 0; col_base < p; col_base += tile_width) {
    float acc[kMaxRegTiles];
#pragma unroll
    for (int t = 0; t < kMaxRegTiles; ++t) acc[t] = 0.0f;

    for (int64_t k = start; k < end; ++k) {
      float v = vals[k];
      float const *mat_row = dense_b + static_cast<int64_t>(col[k]) * p + col_base;
#pragma unroll
      for (int t = 0; t < kMaxRegTiles; ++t) {
        int64_t j = static_cast<int64_t>(lane) + static_cast<int64_t>(t) * kWarpSize;
        if (col_base + j < p) acc[t] += v * mat_row[j];
      }
    }

#pragma unroll
    for (int t = 0; t < kMaxRegTiles; ++t) {
      int64_t j = static_cast<int64_t>(lane) + static_cast<int64_t>(t) * kWarpSize;
      if (col_base + j < p) out_row[col_base + j] = acc[t];
    }
  }
}

// Mirrors spmm_narrow_kernel (p < kWarpSize, the SpMV-shaped regime):
// entry-parallel, warp_reduce_sum per output column.
__global__ void spmm_narrow_kernel(float *__restrict__ out, int32_t const *__restrict__ rowptr,
                                    int32_t const *__restrict__ col, float const *__restrict__ vals,
                                    float const *__restrict__ dense, int64_t total_rows, int64_t n, int64_t m,
                                    int64_t p) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t row_g = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (row_g >= total_rows) {
    return;
  }

  int64_t b = row_g / n;
  int64_t start = static_cast<int64_t>(rowptr[row_g]);
  int64_t end = static_cast<int64_t>(rowptr[row_g + 1]);
  float const *dense_b = dense + b * m * p;

  float acc[kWarpSize];
#pragma unroll
  for (int j = 0; j < kWarpSize; ++j) acc[j] = 0.0f;

  for (int64_t k = start + lane; k < end; k += kWarpSize) {
    float v = vals[k];
    float const *mat_row = dense_b + static_cast<int64_t>(col[k]) * p;
#pragma unroll
    for (int j = 0; j < kWarpSize; ++j) {
      if (j < p) acc[j] += v * mat_row[j];
    }
  }

  float *out_row = out + row_g * p;
#pragma unroll
  for (int j = 0; j < kWarpSize; ++j) {
    if (j < p) {
      float reduced = warp_reduce_sum(acc[j]);
      if (lane == 0) out_row[j] = reduced;
    }
  }
}

// Builds a synthetic folded-CSR `(rowptr, col)` pattern (average
// `nse_total / total_rows` specified entries per folded row) plus dense
// `vals`/`dense` operands — mirrors bench_sddmm.cu's `make_synthetic`.
void make_synthetic(int64_t total_rows, int64_t nse_total, int64_t m, thrust::host_vector<int32_t> &rowptr_h,
                     thrust::host_vector<int32_t> &col_h, thrust::host_vector<float> &vals_h) {
  rowptr_h.resize(total_rows + 1);
  col_h.resize(nse_total);
  vals_h.resize(nse_total);
  std::mt19937 rng(0);
  std::uniform_int_distribution<int32_t> col_dist(0, static_cast<int32_t>(m - 1));
  std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

  int64_t base = nse_total / total_rows;
  int64_t remainder = nse_total % total_rows;
  int64_t offset = 0;
  for (int64_t r = 0; r < total_rows; ++r) {
    rowptr_h[r] = static_cast<int32_t>(offset);
    int64_t this_row_nse = base + (r < remainder ? 1 : 0);
    for (int64_t i = 0; i < this_row_nse; ++i) {
      col_h[offset + i] = col_dist(rng);
      vals_h[offset + i] = val_dist(rng);
    }
    offset += this_row_nse;
  }
  rowptr_h[total_rows] = static_cast<int32_t>(offset);
}

void bench_spmm_fwd(nvbench::state &state) {
  int64_t const nse_total = state.get_int64("nse_total");
  int64_t const p = state.get_int64("p");
  int64_t const B = state.get_int64("B");

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
  thrust::host_vector<float> vals_h;
  make_synthetic(total_rows, nse_total, m, rowptr_h, col_h, vals_h);
  int64_t actual_nse = rowptr_h[total_rows];

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  thrust::host_vector<float> dense_h(B * m * p);
  for (auto &v : dense_h) v = dist(rng);

  thrust::device_vector<int32_t> rowptr = rowptr_h;
  thrust::device_vector<int32_t> col = col_h;
  thrust::device_vector<float> vals = vals_h;
  thrust::device_vector<float> dense = dense_h;
  thrust::device_vector<float> out(B * n * p);

  state.exec([&](nvbench::launch &launch) {
    int64_t blocks = (total_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
    if (p >= kWarpSize) {
      spmm_wide_kernel<<<blocks, kThreadsPerBlock, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(rowptr.data()),
          thrust::raw_pointer_cast(col.data()), thrust::raw_pointer_cast(vals.data()),
          thrust::raw_pointer_cast(dense.data()), total_rows, n, m, p);
    } else {
      spmm_narrow_kernel<<<blocks, kThreadsPerBlock, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(rowptr.data()),
          thrust::raw_pointer_cast(col.data()), thrust::raw_pointer_cast(vals.data()),
          thrust::raw_pointer_cast(dense.data()), total_rows, n, m, p);
    }
  });
  (void)actual_nse;
}
}  // namespace

NVBENCH_BENCH(bench_spmm_fwd)
    .add_int64_axis("nse_total", {1 << 14, 1 << 17, 1 << 20})
    .add_int64_axis("p", {1, 8, 32, 128, 512})
    .add_int64_axis("B", {1, 8, 64});
