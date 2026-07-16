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
// Two kernels, chosen host-side by `p`, both templated on the column-tile
// count so the hot loop's unroll width matches the actual `p` at COMPILE
// time (this file's first cut instantiated one fixed 16-tile/32-column
// unroll and predicated the unused iterations away at runtime — at p=128
// only 4 of 16 wide-path iterations did work and at p=1 only 1 of 32
// narrow-path iterations did, a 3-30x instruction-issue overhead per entry
// that put every acceptance row at 0.63-0.74x of its baseline; the
// templated dispatch below removes the dead issue slots outright):
//
// 1. `spmm_wide_kernel<kTiles>` (p > 16; kTiles = ceil(p/32) rounded
//    up to {1,2,4,8,16}): each lane owns a strided slice of the `p` output
//    columns (`j = lane, lane + t*32`), and walks the row's entries ONCE,
//    accumulating `acc[t] += vals[k] * dense[b, col[k], j]` in registers.
//    The row's `col`/`vals` entries are staged CHUNK-WISE through warp
//    shuffles: each lane loads one entry of a 32-entry chunk (coalesced
//    reads of both arrays), then the chunk is replayed to all lanes via
//    `__shfl_sync` broadcast — no lane ever issues the serial per-entry
//    scalar loads the first cut paid for. p beyond kTiles*32 (i.e. > 512)
//    re-walks the row once per extra 512-wide tile in the kTiles=16
//    instantiation, re-reading only `col`/`vals`, never re-reading `out` or
//    repeating any `dense` read. Empty rows fall through to the
//    unconditional zero-write (out is dense — every row initialized, never
//    skipped like sddmm's nse-aligned output).
//
// 2. `spmm_narrow_kernel<kP>` (p <= 16, the SpMV-shaped regime, p=1
//    included; kP = p rounded up to {1,2,4,8,16}): column-tiling one lane
//    per output column would waste 31/32 lanes at p=1 — instead this path
//    parallelises over the row's ENTRIES (classic CSR "vector" SpMV): each
//    lane strides the row's entries (`k = start + lane; k += 32`),
//    accumulating all `p` output columns for its own entries in `acc[kP]`
//    registers, then one `warp_reduce_sum` per column combines the lanes
//    before lane 0 writes — still exactly one warp owning one row, still no
//    atomics.
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
// Largest wide-path instantiation: kWarpSize * kMaxRegTiles = 512 columns
// per pass covers benchmarks.md's p sweep {1, 8, 32, 128, 512} in a single
// pass over the row; larger p re-walks the row once per extra 512-wide tile
// (module comment above).
constexpr int kMaxRegTiles = 16;
constexpr unsigned kFullMask = 0xffffffffU;

// Wide-p path: column-tiled, one warp per folded row (module comment §1).
// kTiles is the compile-time unroll width: the launcher instantiates
// {1,2,4,8,16} and picks the smallest with kTiles*32 >= p, so the per-entry
// FMA loop carries no dead predicated iterations.
template <typename scalar_t, typename index_t, int kTiles>
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
  int64_t const tile_width = static_cast<int64_t>(kWarpSize) * kTiles;

  for (int64_t col_base = 0; col_base < p; col_base += tile_width) {
    scalar_t acc[kTiles];
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
      acc[t] = scalar_t(0);
    }

    // Per-lane column validity is loop-invariant across the row's entries —
    // hoisted here so the entry loop's guard is a stable predicate (and
    // vanishes entirely when p == kTiles*32, the exact-fit instantiations).
    bool valid[kTiles];
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
      valid[t] = col_base + static_cast<int64_t>(lane) + static_cast<int64_t>(t) * kWarpSize < p;
    }

    // Walk the row's entries once per column tile, staged in 32-entry
    // chunks: each lane loads one (col, val) pair — coalesced reads of both
    // arrays — then the chunk is replayed to the whole warp via shuffle
    // broadcast (module comment §1). vals/col are cheap to re-read on a
    // second column tile; `dense` and `out` never are.
    for (int64_t chunk = start; chunk < end; chunk += kWarpSize) {
      int64_t kk = chunk + lane;
      scalar_t v_lane = kk < end ? vals[kk] : scalar_t(0);
      int64_t c_lane = kk < end ? static_cast<int64_t>(col[kk]) : 0;
      int chunk_len = static_cast<int>(min(static_cast<int64_t>(kWarpSize), end - chunk));

      for (int i = 0; i < chunk_len; ++i) {
        scalar_t v = __shfl_sync(kFullMask, v_lane, i);
        int64_t c = __shfl_sync(kFullMask, c_lane, i);
        scalar_t const *mat_row = dense_b + c * p + col_base + lane;
#pragma unroll
        for (int t = 0; t < kTiles; ++t) {
          if (valid[t]) {
            acc[t] += v * mat_row[static_cast<int64_t>(t) * kWarpSize];
          }
        }
      }
    }

#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
      if (valid[t]) {
        // Unconditional write, even for an empty row (acc stays zero) — out
        // is dense and every row must be initialized (module comment §1).
        out_row[col_base + static_cast<int64_t>(lane) + static_cast<int64_t>(t) * kWarpSize] = acc[t];
      }
    }
  }
}

// Narrow-p (SpMV-shaped, p < kWarpSize) path: entry-parallel, one warp per
// folded row (module comment §2) — the classic CSR "vector" kernel shape,
// generalised to p < 32 output columns via one warp_reduce_sum per column.
// kP is p rounded up to {1,2,4,8,16} so the register array and unrolled
// loops are sized to the actual workload, not a fixed 32.
template <typename scalar_t, typename index_t, int kP>
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

  scalar_t acc[kP];  // kP >= p by dispatch, so this always fits.
#pragma unroll
  for (int j = 0; j < kP; ++j) {
    acc[j] = scalar_t(0);
  }

  for (int64_t k = start + lane; k < end; k += kWarpSize) {
    scalar_t v = vals[k];
    scalar_t const *mat_row = dense_b + static_cast<int64_t>(col[k]) * p;
#pragma unroll
    for (int j = 0; j < kP; ++j) {
      if (j < p) {
        acc[j] += v * mat_row[j];
      }
    }
  }

  scalar_t *out_row = out + row_g * p;
#pragma unroll
  for (int j = 0; j < kP; ++j) {
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

// Launch helpers: pick the smallest instantiation covering p (wide) / the
// next power-of-two register count (narrow) so unroll width == workload.
template <typename scalar_t, typename index_t>
void launch_wide(scalar_t *out, index_t const *rowptr, index_t const *col, scalar_t const *vals,
                 scalar_t const *dense, int64_t total_rows, int64_t n, int64_t m, int64_t p, int64_t blocks,
                 cudaStream_t stream) {
  int64_t tiles = (p + kWarpSize - 1) / kWarpSize;
  if (tiles <= 1) {
    spmm_wide_kernel<scalar_t, index_t, 1>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(out, rowptr, col, vals, dense, total_rows, n, m, p);
  } else if (tiles <= 2) {
    spmm_wide_kernel<scalar_t, index_t, 2>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(out, rowptr, col, vals, dense, total_rows, n, m, p);
  } else if (tiles <= 4) {
    spmm_wide_kernel<scalar_t, index_t, 4>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(out, rowptr, col, vals, dense, total_rows, n, m, p);
  } else if (tiles <= 8) {
    spmm_wide_kernel<scalar_t, index_t, 8>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(out, rowptr, col, vals, dense, total_rows, n, m, p);
  } else {
    spmm_wide_kernel<scalar_t, index_t, kMaxRegTiles>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(out, rowptr, col, vals, dense, total_rows, n, m, p);
  }
}

template <typename scalar_t, typename index_t>
void launch_narrow(scalar_t *out, index_t const *rowptr, index_t const *col, scalar_t const *vals,
                   scalar_t const *dense, int64_t total_rows, int64_t n, int64_t m, int64_t p, int64_t blocks,
                   cudaStream_t stream) {
  if (p <= 1) {
    spmm_narrow_kernel<scalar_t, index_t, 1>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(out, rowptr, col, vals, dense, total_rows, n, m, p);
  } else if (p <= 2) {
    spmm_narrow_kernel<scalar_t, index_t, 2>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(out, rowptr, col, vals, dense, total_rows, n, m, p);
  } else if (p <= 4) {
    spmm_narrow_kernel<scalar_t, index_t, 4>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(out, rowptr, col, vals, dense, total_rows, n, m, p);
  } else if (p <= 8) {
    spmm_narrow_kernel<scalar_t, index_t, 8>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(out, rowptr, col, vals, dense, total_rows, n, m, p);
  } else {
    // p in (8, 16] — the launcher routes p > 16 to the wide path, so kP=16
    // always covers p here.
    spmm_narrow_kernel<scalar_t, index_t, 16>
        <<<blocks, kThreadsPerBlock, 0, stream>>>(out, rowptr, col, vals, dense, total_rows, n, m, p);
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
      // p > 16 goes wide (kTiles=1 covers 17..32 with its hoisted lane
      // guard); p <= 16 goes narrow (entry-parallel, kP = next pow2 >= p).
      if (p > 16) {
        launch_wide<scalar_t, index_t>(
            static_cast<scalar_t *>(out.mutable_data_ptr()), static_cast<index_t const *>(rowptr.data_ptr()),
            static_cast<index_t const *>(col.data_ptr()), static_cast<scalar_t const *>(vals.data_ptr()),
            static_cast<scalar_t const *>(dense_c.data_ptr()), total_rows, n, m, p, blocks, guard.stream());
      } else {
        launch_narrow<scalar_t, index_t>(
            static_cast<scalar_t *>(out.mutable_data_ptr()), static_cast<index_t const *>(rowptr.data_ptr()),
            static_cast<index_t const *>(col.data_ptr()), static_cast<scalar_t const *>(vals.data_ptr()),
            static_cast<scalar_t const *>(dense_c.data_ptr()), total_rows, n, m, p, blocks, guard.stream());
      }
    });
  });

  return out;
}
