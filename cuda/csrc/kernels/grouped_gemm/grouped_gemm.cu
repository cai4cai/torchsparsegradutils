// tsgu::grouped_gemm — Family 3 "Vendor-baseline forwards", Grouped GEMM row
// (kernels.md Family 3; spec/commit.md Phase 3 commit 18; architecture.md §5:
// "segment_mm/gather_mm, gather fused in prologue"). Baseline: cuBLAS
// `cublasGemmGroupedBatched`; our edge (kernels.md Family 3 table): "fuse the
// gather into the GEMM prologue (`gather_mm`); skip materialising gathered
// operand".
//
// One op, two modes (schema + autograd already landed in
// torchsparsegradutils/ops/indexed_matmul.py, commit 9 — this file supplies
// the one CUDA implementation that schema has been missing; both `segment_mm`
// and `gather_mm` route their forward AND both gradients through it, map.md
// routing "transposed operands"; per architecture.md §6 the op bypasses
// BatchedCSR entirely — nothing sparse touches it):
//
// 1. GATHER mode (`reduce = false`) — forward of both public ops and each
//    op's gradA. `a` is a dense matrix with shape `(N, D1)`, `b` a bank of
//    `num_groups` dense matrices with shape `(num_groups, D1, D2)`, `idx` a
//    per-row group index with shape `(N,)`; `out[i, :] = a[i, :] @
//    b[idx[i], :, :]`, shape `(N, D2)`.
//
//    Classic tiled GEMM: each block owns a 32(rows)x32(cols) output tile
//    (256 threads as a (32, 8) layout, 4 output rows per thread), walking
//    the shared inner dimension D1 in 32-wide chunks with the `a` tile
//    staged through shared memory every chunk. The gather is fused into the
//    GEMM prologue via an ADAPTIVE per-block `b`-operand path, decided by a
//    runtime block vote after the tile's 32 `idx` values are staged:
//
//    - UNIFORM tile (all valid rows share one group — always true inside a
//      `segment_mm` segment longer than the 32-row tile, and for tail rows
//      past N, which are excluded from the vote): the `b` k-tile is staged
//      through shared memory too, and the inner loop is the fully-tiled GEMM
//      loop — cuBLAS-shaped arithmetic intensity, no per-row indirection
//      left at all.
//    - MIXED tile (`gather_mm`'s arbitrary `idx`, or a tile straddling a
//      segment boundary): each thread reads its own rows' `b` elements
//      straight from global memory, coalesced across the 32 column lanes. A
//      gathered copy of `b` is never materialised (kernels.md: "skip
//      materialising gathered operand") — `b` is `(num_groups, D1, D2)` and
//      small relative to `a`, so L2 catches the cross-row reuse.
//
//    Every output element is written by exactly one thread — no atomics →
//    deterministic by construction; no separate path is needed under
//    `torch.use_deterministic_algorithms(True)` (same resolution of
//    kernels.md's determinism policy as sddmm.cu/spmm.cu).
//
// 2. SCATTER-REDUCE mode (`reduce = true`) — each op's gradB, the adjoint of
//    gather mode's linear map in `b`. `a` with shape `(N, D1)`, `b` with
//    shape `(N, D2)` (both row-indexed by `idx` here); `out[k, :, :] =
//    sum_{i : idx[i] == k} outer(a[i, :], b[i, :])`, shape
//    `(num_groups, D1, D2)`.
//
//    ** CONTRACT: `idx` MUST be sorted non-decreasing in this mode. ** The
//    op's Python autograd guarantees it (a stable argsort of `idx` is applied
//    to `a`/`b`/`idx` before any reduce=true call; `segment_mm`'s
//    repeat-interleaved `idx` is sorted by construction), so the kernel
//    recovers each group's contiguous row range `[lo, hi)` with two binary
//    searches instead of scattering with atomics. Not verified on device
//    (that would need a sync or an extra pass — kernels.md shared rules: no
//    host syncs in the hot path); an unsorted `idx` silently drops the
//    out-of-place rows from their groups. Debug builds may check host-side
//    in the Python wrapper; the kernel trusts the contract.
//
//    Grid `(num_groups, ceil(D1/32), ceil(D2/32))`: each block owns one
//    group's 32x32 tile of the `(D1, D2)` outer-product accumulator, binary-
//    searches `idx` for its group's row range once, then accumulates over
//    rows `lo..hi` IN ASCENDING ORDER, staging 32-row chunks of `a` and `b`
//    through shared memory and accumulating outer products in registers.
//    Sequential ordered accumulation, exactly one block owning every output
//    element, no atomics → bitwise deterministic. Empty groups (and
//    `N == 0`) fall through to the unconditional zero-write — the register
//    accumulator starts at zero and every in-range output element is written
//    exactly once, so `new_empty` output is safe.
//
// Shared-memory budget: two 32x33 tiles (the +1 pads bank conflicts away)
// = 2 * 32 * 33 * 8 B ≈ 16.5 KB at f64 — comfortably inside the 48 KB
// default window, no dynamic shared memory or opt-in needed.
//
// `idx` values are trusted to lie in `[0, num_groups)` — an out-of-range
// value is undefined behaviour (out-of-bounds read), same trust the schema's
// docstring states; validating on device would need a sync.
//
// f32/f64 x i32/i64 via common/dispatch.cuh, matching every other kernel.
// Dense-operand contiguity: `a`/`b`/`idx` are read through flat strided-free
// pointers, so the launcher makes contiguous copies host-side via
// torch::stable::contiguous() when needed — same convention as
// sddmm.cu/spmm.cu, so opcheck's non-contiguous-operand cases exercise real,
// correct output rather than an error path.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <string>

#include "../../common/dispatch.cuh"
#include "../../common/stream.cuh"

namespace {

constexpr int kTile = 32;      // output-tile edge and inner-dimension chunk width
constexpr int kBlockRows = 8;  // thread-block y extent; each thread owns kTile / kBlockRows = 4 output rows
constexpr int kThreadsPerBlock = kTile * kBlockRows;  // 256
constexpr int kRowsPerThread = kTile / kBlockRows;    // 4

// First position in the sorted `idx` array whose value is >= g (lower bound)
// / > g (upper bound): standard binary searches over the non-decreasing
// per-row group indices, giving group g's contiguous row range [lo, hi).
template <typename index_t>
__device__ __forceinline__ int64_t lower_bound_idx(index_t const* __restrict__ idx, int64_t n, int64_t g) {
  int64_t lo = 0;
  int64_t hi = n;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (static_cast<int64_t>(idx[mid]) < g) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename index_t>
__device__ __forceinline__ int64_t upper_bound_idx(index_t const* __restrict__ idx, int64_t n, int64_t g) {
  int64_t lo = 0;
  int64_t hi = n;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (static_cast<int64_t>(idx[mid]) <= g) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Gather mode (module comment §1): out[i, :] = a[i, :] @ b[idx[i], :, :].
// Block = one 32x32 output tile; threads (32, 8), 4 output rows each; the
// per-block uniform/mixed vote picks the b-operand path.
template <typename scalar_t, typename index_t>
__global__ void grouped_gemm_gather_kernel(scalar_t* __restrict__ out, scalar_t const* __restrict__ a,
                                           scalar_t const* __restrict__ b, index_t const* __restrict__ idx, int64_t N,
                                           int64_t D1, int64_t D2) {
  __shared__ scalar_t a_s[kTile][kTile + 1];
  __shared__ scalar_t b_s[kTile][kTile + 1];
  __shared__ int64_t idx_s[kTile];
  __shared__ int uniform_s;

  int64_t const row_base = static_cast<int64_t>(blockIdx.x) * kTile;
  int64_t const col_base = static_cast<int64_t>(blockIdx.y) * kTile;
  int const tid = static_cast<int>(threadIdx.y) * kTile + static_cast<int>(threadIdx.x);

  // Stage the tile's idx values once, then block-vote uniformity: rows past
  // N carry the sentinel -1 and never break the vote (a tail tile of one
  // long segment still takes the uniform path). row_base < N always holds
  // (grid is sized from N), so idx_s[0] is a real group.
  if (tid == 0) {
    uniform_s = 1;
  }
  if (tid < kTile) {
    int64_t row = row_base + tid;
    idx_s[tid] = row < N ? static_cast<int64_t>(idx[row]) : int64_t(-1);
  }
  __syncthreads();
  int64_t const g0 = idx_s[0];
  if (tid < kTile && idx_s[tid] != g0 && idx_s[tid] != int64_t(-1)) {
    uniform_s = 0;  // benign write race: every writer stores the same 0
  }
  __syncthreads();
  bool const uniform = uniform_s != 0;

  scalar_t acc[kRowsPerThread];
#pragma unroll
  for (int rr = 0; rr < kRowsPerThread; ++rr) {
    acc[rr] = scalar_t(0);
  }

  scalar_t const* b_g0 = b + g0 * D1 * D2;
  int64_t const col = col_base + static_cast<int64_t>(threadIdx.x);

  for (int64_t kb = 0; kb < D1; kb += kTile) {
    // Stage the a tile (32 rows x 32 k), zero-filling past N/D1 so the inner
    // loop needs no bounds arithmetic; the linear->(r, kk) split keeps kk
    // fastest so consecutive threads read consecutive a addresses.
    for (int e = tid; e < kTile * kTile; e += kThreadsPerBlock) {
      int r = e / kTile;
      int kk = e % kTile;
      int64_t row = row_base + r;
      int64_t k = kb + kk;
      a_s[r][kk] = (row < N && k < D1) ? a[row * D1 + k] : scalar_t(0);
    }
    if (uniform) {
      // Uniform tile: stage the single group's b k-tile too (module comment
      // §1) — the fully-tiled GEMM inner loop, c fastest for coalescing.
      for (int e = tid; e < kTile * kTile; e += kThreadsPerBlock) {
        int kk = e / kTile;
        int c = e % kTile;
        int64_t k = kb + kk;
        int64_t cc = col_base + c;
        b_s[kk][c] = (k < D1 && cc < D2) ? b_g0[k * D2 + cc] : scalar_t(0);
      }
    }
    __syncthreads();

    if (uniform) {
#pragma unroll
      for (int kk = 0; kk < kTile; ++kk) {
        scalar_t bv = b_s[kk][threadIdx.x];
#pragma unroll
        for (int rr = 0; rr < kRowsPerThread; ++rr) {
          acc[rr] += a_s[threadIdx.y + rr * kBlockRows][kk] * bv;
        }
      }
    } else if (col < D2) {
      // Mixed tile: b straight from global, per row group (module comment
      // §1) — coalesced across the 32 column lanes for every (rr, kk); a
      // gathered b copy is never materialised.
      int64_t const kmax = min(static_cast<int64_t>(kTile), D1 - kb);
#pragma unroll
      for (int rr = 0; rr < kRowsPerThread; ++rr) {
        int r = static_cast<int>(threadIdx.y) + rr * kBlockRows;
        int64_t g = idx_s[r];
        if (g >= 0) {
          scalar_t const* b_col = b + g * D1 * D2 + kb * D2 + col;
          for (int64_t kk = 0; kk < kmax; ++kk) {
            acc[rr] += a_s[r][kk] * b_col[kk * D2];
          }
        }
      }
    }
    __syncthreads();
  }

  if (col < D2) {
#pragma unroll
    for (int rr = 0; rr < kRowsPerThread; ++rr) {
      int64_t row = row_base + threadIdx.y + rr * kBlockRows;
      if (row < N) {
        out[row * D2 + col] = acc[rr];
      }
    }
  }
}

// Scatter-reduce mode (module comment §2): out[g, :, :] = sum over the
// sorted-idx row range [lo, hi) of outer(a[i, :], b[i, :]). Block = one
// group's 32x32 tile of the (D1, D2) accumulator; rows accumulated in
// ascending order — bitwise deterministic, no atomics.
template <typename scalar_t, typename index_t>
__global__ void grouped_gemm_scatter_kernel(scalar_t* __restrict__ out, scalar_t const* __restrict__ a,
                                            scalar_t const* __restrict__ b, index_t const* __restrict__ idx, int64_t N,
                                            int64_t D1, int64_t D2) {
  __shared__ scalar_t a_s[kTile][kTile + 1];
  __shared__ scalar_t b_s[kTile][kTile + 1];

  int64_t const g = static_cast<int64_t>(blockIdx.x);
  int64_t const i_base = static_cast<int64_t>(blockIdx.y) * kTile;
  int64_t const j_base = static_cast<int64_t>(blockIdx.z) * kTile;
  int const tid = static_cast<int>(threadIdx.y) * kTile + static_cast<int>(threadIdx.x);

  // Per-thread binary search (all threads land on the same [lo, hi); two
  // O(log N) probes are noise next to the row walk, so no lane-0+broadcast
  // dance is needed).
  int64_t const lo = lower_bound_idx<index_t>(idx, N, g);
  int64_t const hi = upper_bound_idx<index_t>(idx, N, g);

  scalar_t acc[kRowsPerThread];
#pragma unroll
  for (int rr = 0; rr < kRowsPerThread; ++rr) {
    acc[rr] = scalar_t(0);
  }

  int64_t const j = j_base + static_cast<int64_t>(threadIdx.x);

  for (int64_t r0 = lo; r0 < hi; r0 += kTile) {
    // Stage 32-row chunks of a (columns i_base..) and b (columns j_base..),
    // zero-filled past hi/D1/D2 — zero rows contribute nothing.
    for (int e = tid; e < kTile * kTile; e += kThreadsPerBlock) {
      int r = e / kTile;
      int c = e % kTile;
      int64_t row = r0 + r;
      int64_t i = i_base + c;
      a_s[r][c] = (row < hi && i < D1) ? a[row * D1 + i] : scalar_t(0);
      int64_t jj = j_base + c;
      b_s[r][c] = (row < hi && jj < D2) ? b[row * D2 + jj] : scalar_t(0);
    }
    __syncthreads();

    // Ascending, order-fixed accumulation over the chunk's rows (module
    // comment §2's determinism story); zero-filled tail rows add 0.
#pragma unroll 4
    for (int r = 0; r < kTile; ++r) {
      scalar_t bv = b_s[r][threadIdx.x];
#pragma unroll
      for (int rr = 0; rr < kRowsPerThread; ++rr) {
        acc[rr] += a_s[r][threadIdx.y + rr * kBlockRows] * bv;
      }
    }
    __syncthreads();
  }

  // Unconditional write of the block's in-range tile — empty groups
  // (lo == hi) write zeros, which is what makes new_empty output safe.
  if (j < D2) {
    scalar_t* out_g = out + g * D1 * D2;
#pragma unroll
    for (int rr = 0; rr < kRowsPerThread; ++rr) {
      int64_t i = i_base + threadIdx.y + rr * kBlockRows;
      if (i < D1) {
        out_g[i * D2 + j] = acc[rr];
      }
    }
  }
}

}  // namespace

// Launcher for tsgu::grouped_gemm (schema: torchsparsegradutils/ops/
// indexed_matmul.py — `grouped_gemm(a, b, idx, num_groups, reduce)`).
// reduce=false: gather mode, a (N, D1), b (num_groups, D1, D2), idx (N,) →
// out (N, D2). reduce=true: scatter-reduce mode, a (N, D1), b (N, D2),
// idx (N,) SORTED NON-DECREASING (module comment §2 — the Python autograd
// guarantees it; unsorted idx silently misassigns rows) → out
// (num_groups, D1, D2).
torch::stable::Tensor tsgu_grouped_gemm_launch(torch::stable::Tensor const& a, torch::stable::Tensor const& b,
                                               torch::stable::Tensor const& idx, int64_t num_groups, bool reduce) {
  STD_TORCH_CHECK(a.is_cuda() && b.is_cuda() && idx.is_cuda(), "tsgu::grouped_gemm expects CUDA tensors; got is_cuda=(",
                  a.is_cuda(), ", ", b.is_cuda(), ", ", idx.is_cuda(), ") for (a, b, idx).");
  STD_TORCH_CHECK(num_groups >= 0, "tsgu::grouped_gemm expects num_groups >= 0; got ", num_groups, ".");
  STD_TORCH_CHECK(a.dim() == 2, "tsgu::grouped_gemm expects a of shape (N, D1); got dim=", a.dim(), ".");
  STD_TORCH_CHECK(idx.dim() == 1 && idx.size(0) == a.size(0), "tsgu::grouped_gemm expects idx of shape (N,) = (",
                  a.size(0), ",) matching a's leading dimension; got shape with dim=", idx.dim(),
                  (idx.dim() == 1 ? ", (" : ""), (idx.dim() == 1 ? idx.size(0) : -1), (idx.dim() == 1 ? ",)" : ""),
                  ".");
  STD_TORCH_CHECK(idx.scalar_type() == torch::headeronly::ScalarType::Int ||
                      idx.scalar_type() == torch::headeronly::ScalarType::Long,
                  "tsgu::grouped_gemm expects idx of an index dtype (torch.int32 or torch.int64); got dtype id ",
                  static_cast<int>(idx.scalar_type()), ".");
  STD_TORCH_CHECK(a.scalar_type() == b.scalar_type(),
                  "tsgu::grouped_gemm expects a and b to share one value dtype (torch.float32 or torch.float64); "
                  "got a dtype id ",
                  static_cast<int>(a.scalar_type()), " vs b dtype id ", static_cast<int>(b.scalar_type()), ".");
  if (!reduce) {
    STD_TORCH_CHECK(b.dim() == 3 && b.size(0) == num_groups && b.size(1) == a.size(1),
                    "tsgu::grouped_gemm (gather mode, reduce=False) expects b of shape (num_groups, D1, D2) = (",
                    num_groups, ", ", a.size(1), ", D2) matching a's shape (N, D1); got shape with dim=", b.dim(),
                    (b.dim() == 3 ? ", (" : ""), (b.dim() == 3 ? b.size(0) : -1), (b.dim() == 3 ? ", " : ""),
                    (b.dim() == 3 ? b.size(1) : -1), (b.dim() == 3 ? ", *)" : ""), ".");
  } else {
    STD_TORCH_CHECK(b.dim() == 2 && b.size(0) == a.size(0),
                    "tsgu::grouped_gemm (scatter-reduce mode, reduce=True) expects b of shape (N, D2) = (", a.size(0),
                    ", D2) matching a's shape (N, D1); got shape with dim=", b.dim(), (b.dim() == 2 ? ", (" : ""),
                    (b.dim() == 2 ? b.size(0) : -1), (b.dim() == 2 ? ", *)" : ""), ".");
  }

  // Operand contiguity (module comment above): make host-side copies rather
  // than reject, so the flat-pointer indexing in the kernels is always valid
  // regardless of the caller's strides.
  torch::stable::Tensor a_c = a.is_contiguous() ? a : torch::stable::contiguous(a);
  torch::stable::Tensor b_c = b.is_contiguous() ? b : torch::stable::contiguous(b);
  torch::stable::Tensor idx_c = idx.is_contiguous() ? idx : torch::stable::contiguous(idx);

  int64_t const N = a.size(0);
  int64_t const D1 = a.size(1);
  int64_t const D2 = b.size(reduce ? 1 : 2);

  torch::stable::Tensor out =
      reduce ? torch::stable::new_empty(a, {num_groups, D1, D2}) : torch::stable::new_empty(a, {N, D2});
  tsgu::StreamGuard guard(a);

  // Degenerate extents where no block would have anything to write: gather's
  // out is (N, D2) — empty when either is 0 (D1 == 0 still launches: zeros
  // must be written); scatter's out is (num_groups, D1, D2) — empty when any
  // is 0 (N == 0 still launches: every group is empty, zeros must be
  // written).
  if (!reduce && (N == 0 || D2 == 0)) {
    return out;
  }
  if (reduce && (num_groups == 0 || D1 == 0 || D2 == 0)) {
    return out;
  }

  TSGU_DISPATCH_VALUE(a.scalar_type(), "tsgu::grouped_gemm", [&] {
    TSGU_DISPATCH_INDEX(idx.scalar_type(), "tsgu::grouped_gemm", [&] {
      dim3 const block(kTile, kBlockRows);
      if (!reduce) {
        dim3 const grid(static_cast<unsigned>((N + kTile - 1) / kTile),
                        static_cast<unsigned>((D2 + kTile - 1) / kTile));
        grouped_gemm_gather_kernel<scalar_t, index_t><<<grid, block, 0, guard.stream()>>>(
            static_cast<scalar_t*>(out.mutable_data_ptr()), static_cast<scalar_t const*>(a_c.data_ptr()),
            static_cast<scalar_t const*>(b_c.data_ptr()), static_cast<index_t const*>(idx_c.data_ptr()), N, D1, D2);
      } else {
        dim3 const grid(static_cast<unsigned>(num_groups), static_cast<unsigned>((D1 + kTile - 1) / kTile),
                        static_cast<unsigned>((D2 + kTile - 1) / kTile));
        grouped_gemm_scatter_kernel<scalar_t, index_t><<<grid, block, 0, guard.stream()>>>(
            static_cast<scalar_t*>(out.mutable_data_ptr()), static_cast<scalar_t const*>(a_c.data_ptr()),
            static_cast<scalar_t const*>(b_c.data_ptr()), static_cast<index_t const*>(idx_c.data_ptr()), N, D1, D2);
      }
    });
  });

  return out;
}
