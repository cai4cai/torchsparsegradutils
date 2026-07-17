// tsgu::coo2csr — Family 3 "Vendor-baseline forwards" coo2csr row (kernels.md
// Family 3: baseline "cuSPARSE `Xcoo2csr` + thrust sort", our edge "Fused
// sort+compress, int32 native"; spec/commit.md Phase 3 commit 19).
//
// Fused sort+compress of batched COO coordinates `(batch, row, col)` into
// folded CSR `[rowptr, col_sorted, permutation]` (naming.md §2: folded row
// `row_global = b * n + r`, local column `col`, absolute `rowptr` of length
// `B * n + 1`). Serves `convert_coo_to_csr*` and `BatchedCSR.from_torch`'s
// COO path — the kernel replacement for `_batched.py::_fold_coo_to_csr`'s
// pure-torch internals. The schema (op def, fake kernel) already landed in
// torchsparsegradutils/utils/convert.py (commit 9) and is not touched here;
// per map.md's routing ("— (index-only, no grad)") the op has no autograd —
// it only rearranges integer index tensors.
//
// Algorithm — four deterministic stages on the current stream, no atomics:
//
// 1. Fold: a trivial kernel computes `row_global[i] = batch[i] * n + row[i]`
//    and an iota `0..nse_total-1` into index workspaces.
// 2. TWO-PASS STABLE radix sort (cub::DeviceRadixSort::SortPairs — radix
//    sort is stable): pass 1 sorts keys=`col` carrying values=iota; pass 2
//    sorts keys=`row_global` gathered through the pass-1 permutation,
//    carrying that permutation as values. Secondary key first, then primary
//    — the standard multi-key stable-sort composition, the exact composition
//    `_fold_coo_to_csr` performs with two stable `torch.argsort`s — so the
//    final order is the stable lexicographic order by `(row_global, col)`
//    and the pass-2 value output IS the final `permutation`. Pass 2 sorts
//    only the `ceil(log2(B * n))` key bits that `row_global < B * n` can
//    occupy (fewer radix sweeps); pass 1 sorts full-width (the schema does
//    not bounds-check `col` against n_cols — kernels.md's coo2csr row).
// 3. Gather: `col_sorted[i] = col[permutation[i]]`.
// 4. Compress: `rowptr[r] = lower_bound(row_global_sorted, r)`, one thread
//    per rowptr entry via binary search over the sorted keys — empty rows,
//    empty batch items, and the leading/trailing entries all fall out of the
//    lower-bound definition with no special cases.
//
// Determinism: radix sort is stable, the fold/gather/compress kernels write
// each output element from exactly one thread, and nothing atomics-reduces —
// the whole pipeline is bitwise deterministic (run-twice gate holds), so no
// separate path is needed under `torch.use_deterministic_algorithms(True)`
// (same resolution as every other kernel in this codebase, kernels.md
// resolved determinism question).
//
// Duplicates: duplicate coordinates are NOT deduplicated — the same caller
// guarantee `_fold_coo_to_csr` documents (a valid CSR/CSC pattern has none;
// dedup via torch.unique is what made the legacy path data-dependent in
// size). With no duplicates present, sort+compress is exact.
//
// Workspace: O(nse_total) — six index workspaces (fold, iota, pass-1 sorted
// keys [discarded], pass-1 permutation, gathered pass-2 keys, sorted
// row_global) plus cub's temp storage, all allocated with
// torch::stable::new_empty on the input's device (never cudaMalloc — it
// syncs); cub's two-phase size-query/run calls both take the current stream
// (tsgu::StreamGuard).
//
// int32 native (kernels.md "our edge"): int32 inputs sort 32-bit keys — no
// widening to int64 pairs. The launcher rejects int32 inputs whose folded
// key space or nse would overflow int32 rather than silently truncating.
//
// Index-only op: no value dtype dispatch — TSGU_DISPATCH_INDEX alone
// (i32/i64), matching architecture.md §3's index-dtype policy.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cstdint>
#include <cub/device/device_radix_sort.cuh>
#include <limits>
#include <vector>

#include "../../common/dispatch.cuh"
#include "../../common/stream.cuh"

namespace {

constexpr int kThreadsPerBlock = 256;

// Stage 1 (module comment): folded keys `row_global[i] = batch[i] * n +
// row[i]` and the identity permutation, one thread per specified entry.
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

// Stages 2b/3 (module comment): `out[i] = src[perm[i]]`, one thread per
// entry — used to route the folded keys through the pass-1 permutation and
// to produce `col_sorted` from the final permutation.
template <typename index_t>
__global__ void coo2csr_gather_kernel(index_t* __restrict__ out, index_t const* __restrict__ src,
                                      index_t const* __restrict__ perm, int64_t nse_total) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < nse_total) {
    out[i] = src[static_cast<int64_t>(perm[i])];
  }
}

// Stage 4 (module comment): `rowptr[r] = lower_bound(row_global_sorted, r)`
// — the first sorted position whose folded row is >= r — one thread per
// rowptr entry, binary search over the `nse_total` sorted keys. Yields
// `rowptr[0] == 0` and `rowptr[B * n] == nse_total` with no special-casing.
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

}  // namespace

std::vector<torch::stable::Tensor> tsgu_coo2csr_launch(torch::stable::Tensor const& batch,
                                                       torch::stable::Tensor const& row,
                                                       torch::stable::Tensor const& col, int64_t B, int64_t n) {
  STD_TORCH_CHECK(batch.is_cuda() && row.is_cuda() && col.is_cuda(),
                  "tsgu::coo2csr expects CUDA tensors; got is_cuda=(", batch.is_cuda(), ", ", row.is_cuda(), ", ",
                  col.is_cuda(), ") for (batch, row, col).");
  STD_TORCH_CHECK(batch.dim() == 1 && row.dim() == 1 && col.dim() == 1,
                  "tsgu::coo2csr expects batch, row, and col each of shape (nse_total,); got dims=(", batch.dim(), ", ",
                  row.dim(), ", ", col.dim(), ").");
  STD_TORCH_CHECK(batch.size(0) == row.size(0) && row.size(0) == col.size(0),
                  "tsgu::coo2csr expects batch, row, and col to share one length nse_total; got lengths=(",
                  batch.size(0), ", ", row.size(0), ", ", col.size(0), ").");
  STD_TORCH_CHECK(batch.scalar_type() == row.scalar_type() && row.scalar_type() == col.scalar_type(),
                  "tsgu::coo2csr expects batch, row, and col to share one index dtype (torch.int32 or torch.int64); "
                  "got dtype ids (",
                  static_cast<int>(batch.scalar_type()), ", ", static_cast<int>(row.scalar_type()), ", ",
                  static_cast<int>(col.scalar_type()), ").");
  STD_TORCH_CHECK(B >= 0 && n >= 0, "tsgu::coo2csr expects B >= 0 and n >= 0; got B=", B, ", n=", n, ".");

  int64_t nse_total = col.size(0);
  int64_t total_rows = B * n;

  // int32 native (module comment) — refuse rather than silently truncate:
  // folded keys reach total_rows - 1 and rowptr values reach nse_total, both
  // stored at the input's own 32-bit width.
  if (row.scalar_type() == torch::headeronly::ScalarType::Int) {
    STD_TORCH_CHECK(
        total_rows <= std::numeric_limits<int32_t>::max() && nse_total <= std::numeric_limits<int32_t>::max(),
        "tsgu::coo2csr with int32 indices expects B * n and nse_total to fit int32; got B * n=", total_rows,
        ", nse_total=", nse_total, " — pass int64 indices instead.");
  }

  torch::stable::Tensor col_sorted = torch::stable::new_empty(col, {nse_total});
  torch::stable::Tensor permutation = torch::stable::new_empty(row, {nse_total});

  if (nse_total == 0) {
    // All rows empty: rowptr is all zeros; the two entry-aligned outputs are
    // empty but still carry the right dtype/device.
    torch::stable::Tensor rowptr_zero = torch::stable::new_zeros(row, {total_rows + 1});
    return {rowptr_zero, col_sorted, permutation};
  }

  torch::stable::Tensor rowptr = torch::stable::new_empty(row, {total_rows + 1});
  tsgu::StreamGuard guard(row);

  TSGU_DISPATCH_INDEX(row.scalar_type(), "tsgu::coo2csr", [&] {
    // Workspaces (module comment: O(nse_total), never cudaMalloc).
    torch::stable::Tensor ws_fold = torch::stable::new_empty(row, {nse_total});
    torch::stable::Tensor ws_iota = torch::stable::new_empty(row, {nse_total});
    torch::stable::Tensor ws_keys1_out = torch::stable::new_empty(row, {nse_total});  // sorted col — discarded
    torch::stable::Tensor ws_perm1 = torch::stable::new_empty(row, {nse_total});
    torch::stable::Tensor ws_keys2 = torch::stable::new_empty(row, {nse_total});
    torch::stable::Tensor ws_rowg_sorted = torch::stable::new_empty(row, {nse_total});

    auto* fold_ptr = static_cast<index_t*>(ws_fold.mutable_data_ptr());
    auto* iota_ptr = static_cast<index_t*>(ws_iota.mutable_data_ptr());
    auto* keys1_out_ptr = static_cast<index_t*>(ws_keys1_out.mutable_data_ptr());
    auto* perm1_ptr = static_cast<index_t*>(ws_perm1.mutable_data_ptr());
    auto* keys2_ptr = static_cast<index_t*>(ws_keys2.mutable_data_ptr());
    auto* rowg_sorted_ptr = static_cast<index_t*>(ws_rowg_sorted.mutable_data_ptr());
    auto* perm_out_ptr = static_cast<index_t*>(permutation.mutable_data_ptr());

    int constexpr kFullBits = static_cast<int>(sizeof(index_t) * 8);
    // Pass 2 sorts only the bits `row_global < B * n` can occupy (module
    // comment) — keys are non-negative, so truncating the untouched high
    // bits (incl. the sign bit) preserves the order.
    int end_bit_pass2 = 1;
    while (end_bit_pass2 < kFullBits && (int64_t(1) << end_bit_pass2) < total_rows) {
      ++end_bit_pass2;
    }

    int64_t entry_blocks = (nse_total + kThreadsPerBlock - 1) / kThreadsPerBlock;

    // Stage 1: folded keys + iota.
    coo2csr_fold_iota_kernel<index_t><<<entry_blocks, kThreadsPerBlock, 0, guard.stream()>>>(
        fold_ptr, iota_ptr, static_cast<index_t const*>(batch.data_ptr()), static_cast<index_t const*>(row.data_ptr()),
        n, nse_total);

    // cub two-phase: one temp-size query per pass (end_bit differs), one
    // Byte workspace sized to the max; both phases on the current stream.
    size_t temp_bytes1 = 0;
    size_t temp_bytes2 = 0;
    cudaError_t err =
        cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes1, static_cast<index_t const*>(col.data_ptr()),
                                        keys1_out_ptr, iota_ptr, perm1_ptr, nse_total, 0, kFullBits, guard.stream());
    STD_TORCH_CHECK(err == cudaSuccess,
                    "tsgu::coo2csr: cub radix-sort temp-size query (pass 1) failed: ", cudaGetErrorString(err));
    err = cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes2, keys2_ptr, rowg_sorted_ptr, perm1_ptr, perm_out_ptr,
                                          nse_total, 0, end_bit_pass2, guard.stream());
    STD_TORCH_CHECK(err == cudaSuccess,
                    "tsgu::coo2csr: cub radix-sort temp-size query (pass 2) failed: ", cudaGetErrorString(err));
    int64_t temp_bytes = static_cast<int64_t>(temp_bytes1 > temp_bytes2 ? temp_bytes1 : temp_bytes2);
    torch::stable::Tensor ws_temp =
        torch::stable::new_empty(row, {temp_bytes > 0 ? temp_bytes : 1}, torch::headeronly::ScalarType::Byte);
    void* temp_ptr = ws_temp.mutable_data_ptr();

    // Stage 2, pass 1: stable sort by the SECONDARY key `col`, carrying the
    // identity permutation.
    size_t temp_capacity = static_cast<size_t>(temp_bytes > 0 ? temp_bytes : 1);
    size_t temp_used = temp_capacity;
    err = cub::DeviceRadixSort::SortPairs(temp_ptr, temp_used, static_cast<index_t const*>(col.data_ptr()),
                                          keys1_out_ptr, iota_ptr, perm1_ptr, nse_total, 0, kFullBits, guard.stream());
    STD_TORCH_CHECK(err == cudaSuccess, "tsgu::coo2csr: cub radix-sort pass 1 (col) failed: ", cudaGetErrorString(err));

    // Stage 2, between passes: route the folded keys through the pass-1
    // permutation so pass 2 sees them in col-sorted order.
    coo2csr_gather_kernel<index_t>
        <<<entry_blocks, kThreadsPerBlock, 0, guard.stream()>>>(keys2_ptr, fold_ptr, perm1_ptr, nse_total);

    // Stage 2, pass 2: stable sort by the PRIMARY key `row_global`; the
    // carried values land as the final lexicographic permutation.
    temp_used = temp_capacity;
    err = cub::DeviceRadixSort::SortPairs(temp_ptr, temp_used, keys2_ptr, rowg_sorted_ptr, perm1_ptr, perm_out_ptr,
                                          nse_total, 0, end_bit_pass2, guard.stream());
    STD_TORCH_CHECK(err == cudaSuccess,
                    "tsgu::coo2csr: cub radix-sort pass 2 (row_global) failed: ", cudaGetErrorString(err));

    // Stage 3: col through the final permutation.
    coo2csr_gather_kernel<index_t><<<entry_blocks, kThreadsPerBlock, 0, guard.stream()>>>(
        static_cast<index_t*>(col_sorted.mutable_data_ptr()), static_cast<index_t const*>(col.data_ptr()), perm_out_ptr,
        nse_total);

    // Stage 4: compress the sorted folded rows into rowptr.
    int64_t rowptr_blocks = (total_rows + 1 + kThreadsPerBlock - 1) / kThreadsPerBlock;
    coo2csr_compress_kernel<index_t><<<rowptr_blocks, kThreadsPerBlock, 0, guard.stream()>>>(
        static_cast<index_t*>(rowptr.mutable_data_ptr()), rowg_sorted_ptr, nse_total, total_rows);
  });

  return {rowptr, col_sorted, permutation};
}
