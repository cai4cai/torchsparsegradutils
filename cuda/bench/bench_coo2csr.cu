// NVBench target for tsgu::coo2csr (spec/commit.md Phase 3 commit 19, T3;
// kernels.md Family 3 coo2csr row: baseline "cuSPARSE `Xcoo2csr` + thrust
// sort", our edge "Fused sort+compress, int32 native"). Axes are the
// kernel's tuning knobs (cuda/csrc/kernels/convert/coo2csr.cu): specified
// entries `nse`, batch size `B`, rows-per-item `n` (col range fixed to n);
// the two index widths are the i32/i64 benchmarks below. Standalone CMake +
// FetchContent, same pattern as bench_grouped_gemm.cu: the kernels are
// mirrored inline rather than #including the torch-dependent .cu (CUB is
// used directly here — it ships with the toolkit).
//
// Like bench_grouped_gemm.cu, the vendor baseline IS duplicated here:
// kernels.md's named recipe — a thrust::sort_by_key over 64-bit packed
// `(row_global << 32) | col` keys carrying an iota permutation, an unpack
// pass, then `cusparseXcoo2csr` to compress the sorted row indices — runs in
// this same binary as the reference row. `Xcoo2csr` is int32-only, so the
// vendor row exists only at i32 (the i64 comparison is ours-vs-ours across
// widths). Both pipelines start each iteration from the same raw
// batch/row/col device arrays and produce [rowptr, col_sorted, permutation].
// This target remains for kernel-level tuning/regression-bisection only
// (benchmarks.md §1: "NVBench ... never gates").
//
// VRAM guard: this machine's GPU has 4 GB — any config whose tensors exceed
// 1.5 GB total is skipped via state.skip (the listed axes all fit; the guard
// protects against ad-hoc -a overrides).

#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <nvbench/nvbench.cuh>
#include <random>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 256;

// --- Mirrors of csrc/kernels/convert/coo2csr.cu (templated index_t) --------

// Mirror of coo2csr_fold_iota_kernel — keep in sync with
// csrc/kernels/convert/coo2csr.cu.
template <typename index_t>
__global__ void coo2csr_fold_iota_kernel(index_t* __restrict__ fold, index_t* __restrict__ iota,
                                         index_t const* __restrict__ batch, index_t const* __restrict__ row, int64_t n,
                                         int64_t nse_total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < nse_total) {
    fold[i] = static_cast<index_t>(static_cast<int64_t>(batch[i]) * n + static_cast<int64_t>(row[i]));
    iota[i] = static_cast<index_t>(i);
  }
}

// Mirror of coo2csr_gather_kernel — keep in sync with
// csrc/kernels/convert/coo2csr.cu.
template <typename index_t>
__global__ void coo2csr_gather_kernel(index_t* __restrict__ out, index_t const* __restrict__ src,
                                      index_t const* __restrict__ perm, int64_t nse_total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < nse_total) {
    out[i] = src[static_cast<int64_t>(perm[i])];
  }
}

// Mirror of coo2csr_compress_kernel — keep in sync with
// csrc/kernels/convert/coo2csr.cu.
template <typename index_t>
__global__ void coo2csr_compress_kernel(index_t* __restrict__ rowptr, index_t const* __restrict__ row_g_sorted,
                                        int64_t nse_total, int64_t total_rows) {
  int64_t r = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r <= total_rows) {
    int64_t lo = 0;
    int64_t hi = nse_total;
    while (lo < hi) {
      int64_t mid = lo + (hi - lo) / 2;
      if (static_cast<int64_t>(row_g_sorted[mid]) < r) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    rowptr[r] = static_cast<index_t>(lo);
  }
}

// --- Vendor-recipe helpers (packed-key thrust sort + Xcoo2csr) --------------

// Pack `(row_global, col)` into one 64-bit key (row_global in the high 32
// bits) and emit the identity permutation — the "thrust sort" half of
// kernels.md's vendor recipe sorts these lexicographically in one pass.
__global__ void vendor_pack_kernel(uint64_t* __restrict__ keys, int32_t* __restrict__ iota,
                                   int32_t const* __restrict__ batch, int32_t const* __restrict__ row,
                                   int32_t const* __restrict__ col, int64_t n, int64_t nse_total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < nse_total) {
    uint64_t rg = static_cast<uint64_t>(static_cast<int64_t>(batch[i]) * n + static_cast<int64_t>(row[i]));
    keys[i] = (rg << 32) | static_cast<uint32_t>(col[i]);
    iota[i] = static_cast<int32_t>(i);
  }
}

// Unpack the sorted keys back into the int32 sorted row indices Xcoo2csr
// consumes plus the sorted local columns.
__global__ void vendor_unpack_kernel(int32_t* __restrict__ rowg_sorted, int32_t* __restrict__ col_sorted,
                                     uint64_t const* __restrict__ keys, int64_t nse_total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < nse_total) {
    rowg_sorted[i] = static_cast<int32_t>(keys[i] >> 32);
    col_sorted[i] = static_cast<int32_t>(keys[i] & 0xffffffffULL);
  }
}

// --- Shared setup helpers ---------------------------------------------------

// 1.5 GB tensor-footprint cap (module comment: 4 GB VRAM machine).
constexpr double kMaxBytes = 1.5 * 1024.0 * 1024.0 * 1024.0;

bool skip_if_oversized(nvbench::state& state, double bytes) {
  if (bytes > kMaxBytes) {
    state.skip("tensor footprint exceeds 1.5 GB VRAM guard");
    return true;
  }
  return false;
}

// Uniform-random COO coordinates: batch in [0, B), row/col in [0, n).
// Duplicates may occur — irrelevant for throughput (no dedup happens in
// either pipeline).
template <typename index_t>
void make_coords(std::vector<index_t>& batch, std::vector<index_t>& row, std::vector<index_t>& col, int64_t nse,
                 int64_t B, int64_t n) {
  batch.resize(nse);
  row.resize(nse);
  col.resize(nse);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int64_t> bdist(0, B - 1);
  std::uniform_int_distribution<int64_t> ndist(0, n - 1);
  for (int64_t i = 0; i < nse; ++i) {
    batch[i] = static_cast<index_t>(bdist(rng));
    row[i] = static_cast<index_t>(ndist(rng));
    col[i] = static_cast<index_t>(ndist(rng));
  }
}

// --- tsgu fused sort+compress benchmark -------------------------------------

template <typename index_t>
void bench_tsgu(nvbench::state& state) {
  int64_t const nse = state.get_int64("nse");
  int64_t const B = state.get_int64("B");
  int64_t const n = state.get_int64("n");
  int64_t const total_rows = B * n;

  // 3 inputs + 3 outputs (rowptr ~ total_rows) + 6 workspaces, all index_t.
  double const bytes =
      static_cast<double>(sizeof(index_t)) * (11.0 * static_cast<double>(nse) + static_cast<double>(total_rows + 1));
  if (skip_if_oversized(state, bytes)) {
    return;
  }

  std::vector<index_t> batch_h, row_h, col_h;
  make_coords<index_t>(batch_h, row_h, col_h, nse, B, n);
  thrust::device_vector<index_t> batch(batch_h), row(row_h), col(col_h);
  thrust::device_vector<index_t> rowptr(total_rows + 1), col_sorted(nse), perm(nse);
  thrust::device_vector<index_t> fold(nse), iota(nse), keys1_out(nse), perm1(nse), keys2(nse), rowg_sorted(nse);

  int constexpr kFullBits = static_cast<int>(sizeof(index_t) * 8);
  int end_bit_pass2 = 1;
  while (end_bit_pass2 < kFullBits && (int64_t(1) << end_bit_pass2) < total_rows) {
    ++end_bit_pass2;
  }

  size_t temp_bytes1 = 0;
  size_t temp_bytes2 = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes1, thrust::raw_pointer_cast(col.data()),
                                  thrust::raw_pointer_cast(keys1_out.data()), thrust::raw_pointer_cast(iota.data()),
                                  thrust::raw_pointer_cast(perm1.data()), nse, 0, kFullBits);
  cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes2, thrust::raw_pointer_cast(keys2.data()),
                                  thrust::raw_pointer_cast(rowg_sorted.data()), thrust::raw_pointer_cast(perm1.data()),
                                  thrust::raw_pointer_cast(perm.data()), nse, 0, end_bit_pass2);
  size_t const temp_capacity = std::max(std::max(temp_bytes1, temp_bytes2), size_t(1));
  thrust::device_vector<uint8_t> temp(temp_capacity);

  state.add_global_memory_reads<index_t>(3 * nse);
  state.add_global_memory_writes<index_t>(2 * nse + total_rows + 1);

  int64_t const entry_blocks = (nse + kThreadsPerBlock - 1) / kThreadsPerBlock;
  int64_t const rowptr_blocks = (total_rows + 1 + kThreadsPerBlock - 1) / kThreadsPerBlock;

  state.exec([&](nvbench::launch& launch) {
    cudaStream_t stream = launch.get_stream();
    coo2csr_fold_iota_kernel<index_t><<<entry_blocks, kThreadsPerBlock, 0, stream>>>(
        thrust::raw_pointer_cast(fold.data()), thrust::raw_pointer_cast(iota.data()),
        thrust::raw_pointer_cast(batch.data()), thrust::raw_pointer_cast(row.data()), n, nse);
    size_t temp_used = temp_capacity;
    cub::DeviceRadixSort::SortPairs(thrust::raw_pointer_cast(temp.data()), temp_used,
                                    thrust::raw_pointer_cast(col.data()), thrust::raw_pointer_cast(keys1_out.data()),
                                    thrust::raw_pointer_cast(iota.data()), thrust::raw_pointer_cast(perm1.data()), nse,
                                    0, kFullBits, stream);
    coo2csr_gather_kernel<index_t><<<entry_blocks, kThreadsPerBlock, 0, stream>>>(
        thrust::raw_pointer_cast(keys2.data()), thrust::raw_pointer_cast(fold.data()),
        thrust::raw_pointer_cast(perm1.data()), nse);
    temp_used = temp_capacity;
    cub::DeviceRadixSort::SortPairs(
        thrust::raw_pointer_cast(temp.data()), temp_used, thrust::raw_pointer_cast(keys2.data()),
        thrust::raw_pointer_cast(rowg_sorted.data()), thrust::raw_pointer_cast(perm1.data()),
        thrust::raw_pointer_cast(perm.data()), nse, 0, end_bit_pass2, stream);
    coo2csr_gather_kernel<index_t><<<entry_blocks, kThreadsPerBlock, 0, stream>>>(
        thrust::raw_pointer_cast(col_sorted.data()), thrust::raw_pointer_cast(col.data()),
        thrust::raw_pointer_cast(perm.data()), nse);
    coo2csr_compress_kernel<index_t><<<rowptr_blocks, kThreadsPerBlock, 0, stream>>>(
        thrust::raw_pointer_cast(rowptr.data()), thrust::raw_pointer_cast(rowg_sorted.data()), nse, total_rows);
  });
}

void bench_coo2csr_i32(nvbench::state& state) { bench_tsgu<int32_t>(state); }
void bench_coo2csr_i64(nvbench::state& state) { bench_tsgu<int64_t>(state); }

// --- vendor reference (the bar, kernels.md Family 3 coo2csr row) ------------

// thrust::sort_by_key over packed 64-bit keys + cusparseXcoo2csr. int32
// coordinates only — Xcoo2csr has no 64-bit variant. thrust's sort allocates
// its own temporaries per call (and may sync); that is part of the recipe's
// real cost, so the exec loop is measured with exec_tag::sync.
void bench_coo2csr_vendor_i32(nvbench::state& state) {
  int64_t const nse = state.get_int64("nse");
  int64_t const B = state.get_int64("B");
  int64_t const n = state.get_int64("n");
  int64_t const total_rows = B * n;

  // 3 inputs + keys (8B) + iota/perm + unpacked rowg/col + rowptr, plus
  // thrust's internal sort double-buffer (~another 12B/entry).
  double const bytes = 4.0 * (5.0 * static_cast<double>(nse) + static_cast<double>(total_rows + 1)) +
                       2.0 * 8.0 * static_cast<double>(nse) + 4.0 * static_cast<double>(nse);
  if (skip_if_oversized(state, bytes)) {
    return;
  }

  std::vector<int32_t> batch_h, row_h, col_h;
  make_coords<int32_t>(batch_h, row_h, col_h, nse, B, n);
  thrust::device_vector<int32_t> batch(batch_h), row(row_h), col(col_h);
  thrust::device_vector<uint64_t> keys(nse);
  thrust::device_vector<int32_t> perm(nse), rowg_sorted(nse), col_sorted(nse), rowptr(total_rows + 1);

  cusparseHandle_t handle = nullptr;
  if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS) {
    state.skip("cusparseCreate failed");
    return;
  }

  state.add_global_memory_reads<int32_t>(3 * nse);
  state.add_global_memory_writes<int32_t>(2 * nse + total_rows + 1);

  int64_t const entry_blocks = (nse + kThreadsPerBlock - 1) / kThreadsPerBlock;

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudaStream_t stream = launch.get_stream();
    cusparseSetStream(handle, stream);
    vendor_pack_kernel<<<entry_blocks, kThreadsPerBlock, 0, stream>>>(
        thrust::raw_pointer_cast(keys.data()), thrust::raw_pointer_cast(perm.data()),
        thrust::raw_pointer_cast(batch.data()), thrust::raw_pointer_cast(row.data()),
        thrust::raw_pointer_cast(col.data()), n, nse);
    thrust::sort_by_key(thrust::cuda::par.on(stream), keys.begin(), keys.end(), perm.begin());
    vendor_unpack_kernel<<<entry_blocks, kThreadsPerBlock, 0, stream>>>(thrust::raw_pointer_cast(rowg_sorted.data()),
                                                                        thrust::raw_pointer_cast(col_sorted.data()),
                                                                        thrust::raw_pointer_cast(keys.data()), nse);
    cusparseXcoo2csr(handle, thrust::raw_pointer_cast(rowg_sorted.data()), static_cast<int>(nse),
                     static_cast<int>(total_rows), thrust::raw_pointer_cast(rowptr.data()), CUSPARSE_INDEX_BASE_ZERO);
  });

  // NVBench keeps the state alive until after exec's measurement loop, so
  // destroying here is safe.
  cusparseDestroy(handle);
}

}  // namespace

NVBENCH_BENCH(bench_coo2csr_i32)
    .add_int64_axis("nse", {100000, 1000000, 5000000})
    .add_int64_axis("B", {1, 16})
    .add_int64_axis("n", {4096, 65536});

NVBENCH_BENCH(bench_coo2csr_i64)
    .add_int64_axis("nse", {100000, 1000000, 5000000})
    .add_int64_axis("B", {1, 16})
    .add_int64_axis("n", {4096, 65536});

NVBENCH_BENCH(bench_coo2csr_vendor_i32)
    .add_int64_axis("nse", {100000, 1000000, 5000000})
    .add_int64_axis("B", {1, 16})
    .add_int64_axis("n", {4096, 65536});
