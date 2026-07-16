// tsgu::spmm — Family 3 "Vendor-baseline forwards", SpMM row (kernels.md
// Family 3; spec/commit.md Phase 3 commit 15; architecture.md §5: "warp-per-
// folded-row, column-tiled").
//
// Batched sparse-CSR @ dense matmul: for every folded row `row_g` (naming.md
// §2: `row_g = b * n + r`), `out[row_g, :] = sum_k vals[k] * dense[b, col[k],
// :]` over the row's specified entries `k`. Serves three call sites (map.md
// routing + this op's own schema docstring in torchsparsegradutils/ops/
// matmul.py, commit 9): `sparse_mm` forward, its `gradB` (called on the
// cached transpose, BatchedCSC, with `n`/`m` swapped), and `spmv` (`p = 1`,
// no separate op — map.md's own routing note).
//
// Design (architecture.md §5's own annotation, followed verbatim): **one
// warp per folded row**, the row's entries walked exactly once, with the `p`
// (dense inner/rhs) axis column-tiled across the warp's 32 lanes so
// `dense[b, col[k], :]` reads are coalesced (consecutive lanes -> consecutive
// addresses) and the output tile accumulates in registers, written once per
// tile — no re-reads of `out`, no atomics (every folded row is owned by
// exactly one warp, so no other warp ever touches the same output row) ->
// deterministic by construction, same resolution as sddmm.cu/seglse.cu for
// kernels.md's open Q1 (no separate path needed under
// torch.use_deterministic_algorithms(True)).
//
// This directly avoids the mistake commit 14 shipped and commit
// f69990d had to fix for sddmm: a naive one-warp/one-thread-*per-specified-
// entry* design here would need every entry of a row to atomically accumulate
// into the same `out[row, :]` (multiple entries share a row), forcing either
// atomics or a second reduction pass. Warp-per-row sidesteps that entirely by
// construction, not as a follow-up perf fix.
//
// Two kernels, chosen host-side by `p`:
//
// 1. `spmm_wide_kernel` (p >= kWarpSize): each lane owns a strided slice of
//    the `p` output columns (`j = lane, lane + 32, lane + 64, ...`), and
//    walks the row's `nse_row` entries ONCE, accumulating
//    `acc[t] += vals[k] * dense[b, col[k], lane + t*32]` for each of its
//    owned columns into small per-lane registers (`kMaxRegTiles` = 16,
//    covering p up to 512 -- benchmarks.md's own p sweep ceiling -- in one
//    pass over the row; wider p re-walks the row once per extra 512-wide
//    tile, re-reading only the cheap `col`/`vals` arrays, never re-reading
//    `out` or repeating any `dense` read). Empty rows (`start == end`) fall
//    straight through to the unconditional zero-write below -- naming.md/
//    this commit's own instruction: "out is dense -- every row must be
//    initialized", never skipped like sddmm's nse-aligned output is.
//
// 2. `spmm_narrow_kernel` (p < kWarpSize, the SpMV-shaped regime, `p = 1`
//    included): column-tiling one lane per output column would waste
//    31/32 lanes when p=1 -- kernels.md/this commit's own guidance ("must
//    not waste a full warp's lanes on one column"). Instead this path
//    parallelises over the row's ENTRIES (the classic CSR "vector" SpMV
//    kernel): each lane strides over a disjoint subset of the row's
//    specified entries (`k = start + lane; k < end; k += 32`), accumulating
//    all `p` (< 32) output columns for its own entries in registers, then a
//    per-column `warp_reduce_sum` combines the 32 lanes' partial sums before
//    lane 0 writes the row's final `p` values -- still exactly one warp
//    owning one row, still no atomics.
//
// Every `dense[b, col[k], :]` element is read exactly once across the whole
// kernel regardless of path (nse_total * p total dense-operand traffic,
// the same bound cuSPARSE SpMM pays) -- the two paths differ only in which
// axis (columns vs entries) the warp's 32 lanes divide, not in total memory
// traffic.
//
// f32/f64 x i32/i64 via common/dispatch.cuh, matching every other kernel.
// Dense-operand contiguity: `dense` is read through flat strided-free
// pointers (row-major `(B, m, p)`), so the launcher makes a contiguous copy
// host-side via torch::stable::contiguous() when needed -- same convention
// as sddmm.cu, so opcheck's non-contiguous-operand cases exercise real,
// correct output rather than an error path.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <string>

#include "../../common/dispatch.cuh"
#include "../../common/reduce.cuh"
#include "../../common/stream.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;  // 256 threads/block — one warp per folded row
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
// Register tile count for the wide-p path: kWarpSize * kMaxRegTiles = 512
// covers benchmarks.md's p sweep {1, 8, 32, 128, 512} in a single pass over
// the row; larger p re-walks the row once per extra 512-wide tile (module
// comment above).
constexpr int kMaxRegTiles = 16;

// Wide-p path: column-tiled, one warp per folded row (module comment §1).
template <typename scalar_t, typename index_t>
__global__ void spmm_wide_kernel(scalar_t *__restrict__ out, index_t const *__restrict__ rowptr,
                                  index_t const *__restrict__ col, scalar_t const *__restrict__ vals,
                                  scalar_t const *__restrict__ dense, int64_t total_rows, int64_t n, int64_t m,
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

  scalar_t *out_row = out + row_g * p;
  scalar_t const *dense_b = dense + b * m * p;
  int64_t const tile_width = static_cast<int64_t>(kWarpSize) * kMaxRegTiles;

  for (int64_t col_base = 0; col_base < p; col_base += tile_width) {
    scalar_t acc[kMaxRegTiles];
#pragma unroll
    for (int t = 0; t < kMaxRegTiles; ++t) {
      acc[t] = scalar_t(0);
    }

    // Walk the row's entries exactly once per column tile — vals/col are
    // small (one scalar_t + one index_t per entry) and never re-read `out`;
    // the only cost of a second column tile is re-reading these, not dense.
    for (int64_t k = start; k < end; ++k) {
      scalar_t v = vals[k];
      scalar_t const *mat_row = dense_b + static_cast<int64_t>(col[k]) * p + col_base;
#pragma unroll
      for (int t = 0; t < kMaxRegTiles; ++t) {
        int64_t j = static_cast<int64_t>(lane) + static_cast<int64_t>(t) * kWarpSize;
        if (col_base + j < p) {
          acc[t] += v * mat_row[j];
        }
      }
    }

#pragma unroll
    for (int t = 0; t < kMaxRegTiles; ++t) {
      int64_t j = static_cast<int64_t>(lane) + static_cast<int64_t>(t) * kWarpSize;
      if (col_base + j < p) {
        // Unconditional write, even for an empty row (acc stays zero) — out
        // is dense and every row must be initialized (module comment §1).
        out_row[col_base + j] = acc[t];
      }
    }
  }
}

// Narrow-p (SpMV-shaped, p < kWarpSize) path: entry-parallel, one warp per
// folded row (module comment §2) — the classic CSR "vector" kernel shape,
// generalised to p < 32 output columns via one warp_reduce_sum per column.
template <typename scalar_t, typename index_t>
__global__ void spmm_narrow_kernel(scalar_t *__restrict__ out, index_t const *__restrict__ rowptr,
                                    index_t const *__restrict__ col, scalar_t const *__restrict__ vals,
                                    scalar_t const *__restrict__ dense, int64_t total_rows, int64_t n, int64_t m,
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
  scalar_t const *dense_b = dense + b * m * p;

  scalar_t acc[kWarpSize];  // p < kWarpSize, so this always fits.
#pragma unroll
  for (int j = 0; j < kWarpSize; ++j) {
    acc[j] = scalar_t(0);
  }

  for (int64_t k = start + lane; k < end; k += kWarpSize) {
    scalar_t v = vals[k];
    scalar_t const *mat_row = dense_b + static_cast<int64_t>(col[k]) * p;
#pragma unroll
    for (int j = 0; j < kWarpSize; ++j) {
      if (j < p) {
        acc[j] += v * mat_row[j];
      }
    }
  }

  scalar_t *out_row = out + row_g * p;
#pragma unroll
  for (int j = 0; j < kWarpSize; ++j) {
    if (j < p) {
      scalar_t reduced = tsgu::warp_reduce_sum<scalar_t>(acc[j]);
      if (lane == 0) {
        // Unconditional write (module comment §1) — reduced is exactly zero
        // for an empty row, since every lane's acc[j] stayed zero.
        out_row[j] = reduced;
      }
    }
  }
}

}  // namespace

torch::stable::Tensor tsgu_spmm_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                        torch::stable::Tensor const &col, torch::stable::Tensor const &dense,
                                        int64_t B, int64_t n, int64_t m) {
  STD_TORCH_CHECK(vals.is_cuda() && rowptr.is_cuda() && col.is_cuda() && dense.is_cuda(),
                   "tsgu::spmm expects CUDA tensors; got is_cuda=(", vals.is_cuda(), ", ", rowptr.is_cuda(), ", ",
                   col.is_cuda(), ", ", dense.is_cuda(), ") for (vals, rowptr, col, dense).");
  STD_TORCH_CHECK(rowptr.dim() == 1 && rowptr.size(0) == B * n + 1,
                   "tsgu::spmm expects rowptr of shape (B * n + 1,) = (", B * n + 1, ",) for B=", B, ", n=", n,
                   "; got shape (", (rowptr.dim() == 1 ? rowptr.size(0) : -1), ",) with dim=", rowptr.dim(), ".");
  STD_TORCH_CHECK(col.dim() == 1, "tsgu::spmm expects col of shape (nse_total,); got dim=", col.dim(), ".");
  STD_TORCH_CHECK(vals.dim() == 1 && vals.size(0) == col.size(0),
                   "tsgu::spmm expects vals of shape (nse_total,) matching col's length (", col.size(0),
                   ",); got shape with dim=", vals.dim(), (vals.dim() == 1 ? ", (" : ""),
                   (vals.dim() == 1 ? vals.size(0) : -1), (vals.dim() == 1 ? ",)" : ""), ".");
  STD_TORCH_CHECK(rowptr.scalar_type() == col.scalar_type(),
                   "tsgu::spmm expects rowptr and col to share one index dtype (torch.int32 or torch.int64); got "
                   "rowptr dtype id ",
                   static_cast<int>(rowptr.scalar_type()), " vs col dtype id ", static_cast<int>(col.scalar_type()),
                   ".");
  STD_TORCH_CHECK(dense.dim() == 3 && dense.size(0) == B && dense.size(1) == m,
                   "tsgu::spmm expects dense of shape (B, m, p) = (", B, ", ", m,
                   ", p); got shape with dim=", dense.dim(), (dense.dim() == 3 ? ", (" : ""),
                   (dense.dim() == 3 ? dense.size(0) : -1), (dense.dim() == 3 ? ", " : ""),
                   (dense.dim() == 3 ? dense.size(1) : -1), (dense.dim() == 3 ? ", *)" : ""), ".");
  STD_TORCH_CHECK(vals.scalar_type() == dense.scalar_type(),
                   "tsgu::spmm expects vals and dense to share one value dtype (torch.float32 or torch.float64); "
                   "got vals dtype id ",
                   static_cast<int>(vals.scalar_type()), " vs dense dtype id ", static_cast<int>(dense.scalar_type()),
                   ".");

  // Dense-operand contiguity (module comment above): make a host-side copy
  // rather than reject, so the flat-pointer indexing in the kernel is always
  // valid regardless of the caller's strides.
  torch::stable::Tensor dense_c = dense.is_contiguous() ? dense : torch::stable::contiguous(dense);

  int64_t p = dense.size(2);
  torch::stable::Tensor out = torch::stable::new_empty(dense, {B, n, p});
  tsgu::StreamGuard guard(vals);

  int64_t total_rows = B * n;
  if (total_rows == 0) {
    return out;
  }

  TSGU_DISPATCH_VALUE(vals.scalar_type(), "tsgu::spmm", [&] {
    TSGU_DISPATCH_INDEX(rowptr.scalar_type(), "tsgu::spmm", [&] {
      int64_t blocks = (total_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
      if (p == 0) {
        // Nothing to write — `out` is already correctly shaped/empty (0
        // columns); no kernel launch needed.
        return;
      }
      if (p >= kWarpSize) {
        spmm_wide_kernel<scalar_t, index_t><<<blocks, kThreadsPerBlock, 0, guard.stream()>>>(
            static_cast<scalar_t *>(out.mutable_data_ptr()), static_cast<index_t const *>(rowptr.data_ptr()),
            static_cast<index_t const *>(col.data_ptr()), static_cast<scalar_t const *>(vals.data_ptr()),
            static_cast<scalar_t const *>(dense_c.data_ptr()), total_rows, n, m, p);
      } else {
        spmm_narrow_kernel<scalar_t, index_t><<<blocks, kThreadsPerBlock, 0, guard.stream()>>>(
            static_cast<scalar_t *>(out.mutable_data_ptr()), static_cast<index_t const *>(rowptr.data_ptr()),
            static_cast<index_t const *>(col.data_ptr()), static_cast<scalar_t const *>(vals.data_ptr()),
            static_cast<scalar_t const *>(dense_c.data_ptr()), total_rows, n, m, p);
      }
    });
  });

  return out;
}
