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
//    Classic register-blocked tiled GEMM: each block owns a 64(rows)x64(cols)
//    output tile (256 threads as a logical (16, 16) layout, each owning a
//    4x4 register tile strided 16 apart), walking the shared inner dimension
//    D1 in 16-wide chunks with the `a` tile staged (transposed) through
//    shared memory every chunk — 16 FMAs per 8 shared loads, the geometry
//    the first cut's 32x32/4-per-thread shape lacked (see the constants
//    block). The gather is fused into the GEMM prologue via an ADAPTIVE
//    per-block `b`-operand path, decided by a runtime block vote after the
//    tile's 64 `idx` values are staged:
//
//    - UNIFORM tile (all valid rows share one group — always true inside a
//      `segment_mm` segment longer than the 64-row tile, and for tail rows
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
//    Grid `(num_groups, ceil(D1/64), ceil(D2/64))`: each block owns one
//    group's 64x64 tile of the `(D1, D2)` outer-product accumulator, binary-
//    searches `idx` for its group's row range once, then accumulates over
//    rows `lo..hi` IN ASCENDING ORDER, staging 16-row chunks of `a` and `b`
//    through shared memory and accumulating outer products in 4x4 register
//    tiles (same thread geometry as gather mode).
//    Sequential ordered accumulation, exactly one block owning every output
//    element, no atomics → bitwise deterministic. Empty groups (and
//    `N == 0`) fall through to the unconditional zero-write — the register
//    accumulator starts at zero and every in-range output element is written
//    exactly once, so `new_empty` output is safe.
//
// Shared-memory budget: two 16x65 tiles (the +1 pads bank conflicts away)
// = 2 * 16 * 65 * 8 B ≈ 16.6 KB at f64 — comfortably inside the 48 KB
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

// Register-blocked tile geometry (perf rewrite, spec/commit.md commit 18
// follow-up): the first cut's 32x32 block tile with 4 outputs per thread ran
// at ~6% of f32 peak (0.17x of cuBLAS at D=256) because its inner loop did 4
// FMAs per 5 shared-memory loads — shared-bandwidth-bound, exactly the
// register-tile deficit the sddmm/spmm perf commits fixed before it. This
// geometry is the classic CUDA-core SGEMM shape, templated (BM, BN, BK, TM,
// TN) with two instantiated configs picked by the launcher:
//   64x64x16 / 4x4  — 16 FMAs per 8 shared loads; the default, and the only
//                     config for f64 (an 8x8 f64 accumulator block would
//                     spill registers) and narrow shapes.
//   128x128x8 / 8x8 — 64 FMAs per 16 shared loads (the cuBLAS-class 1:4
//                     ratio); f32 wide shapes (D >= 128), where the
//                     vendor-parity bar lives.
// Both run 256 threads as a logical (BM/TM, BN/TN) = (16, 16) lane grid,
// register tiles strided LY/LX apart so shared reads stay
// conflict-free/broadcast.
constexpr int kBM = 64;                // block-tile rows (gather: output rows; scatter: D1 extent)
constexpr int kBN = 64;                // block-tile cols (D2 extent)
constexpr int kBK = 16;                // inner-dimension chunk (gather: D1; scatter: row chunk)
constexpr int kThreadsPerBlock = 256;  // logical (16, 16)
constexpr int kTM = 4;                 // register-tile rows per thread (kBM / 16)
constexpr int kTN = 4;                 // register-tile cols per thread (kBN / 16)

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
// Block = one 64x64 output tile; 256 threads as logical (16, 16), each
// owning a 4x4 register tile strided 16 apart; the per-block uniform/mixed
// vote picks the b-operand path.
template <typename scalar_t, typename index_t, int BM, int BN, int BK, int TM, int TN>
__global__ void grouped_gemm_gather_kernel(scalar_t* __restrict__ out, scalar_t const* __restrict__ a,
                                           scalar_t const* __restrict__ b, index_t const* __restrict__ idx, int64_t N,
                                           int64_t D1, int64_t D2) {
  // a staged transposed (a_s[kk][row]) so the inner loop's per-thread row
  // reads are broadcast/conflict-free; +1 pads away bank conflicts on the
  // transposed store.
  __shared__ scalar_t a_s[BK][BM + 1];
  __shared__ scalar_t b_s[BK][BN + 1];
  __shared__ int64_t idx_s[BM];
  __shared__ int uniform_s;

  int64_t const row_base = static_cast<int64_t>(blockIdx.x) * BM;
  int64_t const col_base = static_cast<int64_t>(blockIdx.y) * BN;
  int const tid = static_cast<int>(threadIdx.x);
  constexpr int LX = BN / TN;  // column-lane count (thread grid is (LY, LX) = 256)
  constexpr int LY = BM / TM;  // row-lane count
  int const tx = tid % LX;     // register-tile column lane
  int const ty = tid / LX;     // register-tile row lane

  // Stage the tile's idx values once, then block-vote uniformity: rows past
  // N carry the sentinel -1 and never break the vote (a tail tile of one
  // long segment still takes the uniform path). row_base < N always holds
  // (grid is sized from N), so idx_s[0] is a real group.
  if (tid == 0) {
    uniform_s = 1;
  }
  if (tid < BM) {
    int64_t row = row_base + tid;
    idx_s[tid] = row < N ? static_cast<int64_t>(idx[row]) : int64_t(-1);
  }
  __syncthreads();
  int64_t const g0 = idx_s[0];
  if (tid < BM && idx_s[tid] != g0 && idx_s[tid] != int64_t(-1)) {
    uniform_s = 0;  // benign write race: every writer stores the same 0
  }
  __syncthreads();
  bool const uniform = uniform_s != 0;

  scalar_t acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = scalar_t(0);
    }
  }

  scalar_t const* b_g0 = b + g0 * D1 * D2;

  for (int64_t kb = 0; kb < D1; kb += BK) {
    // Stage the a tile (BM rows x BK k, stored transposed), zero-filling
    // past N/D1 so the inner loop needs no bounds arithmetic; the
    // linear->(r, kk) split keeps kk fastest so consecutive threads read
    // consecutive a addresses.
#pragma unroll
    for (int e = tid; e < BM * BK; e += kThreadsPerBlock) {
      int r = e / BK;
      int kk = e % BK;
      int64_t row = row_base + r;
      int64_t k = kb + kk;
      a_s[kk][r] = (row < N && k < D1) ? a[row * D1 + k] : scalar_t(0);
    }
    if (uniform) {
      // Uniform tile: stage the single group's b k-tile too (module comment
      // §1) — the fully-tiled GEMM inner loop, c fastest for coalescing.
#pragma unroll
      for (int e = tid; e < BK * BN; e += kThreadsPerBlock) {
        int kk = e / BN;
        int c = e % BN;
        int64_t k = kb + kk;
        int64_t cc = col_base + c;
        b_s[kk][c] = (k < D1 && cc < D2) ? b_g0[k * D2 + cc] : scalar_t(0);
      }
    }
    __syncthreads();

    if (uniform) {
#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        scalar_t a_frag[TM];
        scalar_t b_frag[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          a_frag[i] = a_s[kk][ty + i * LY];
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          b_frag[j] = b_s[kk][tx + j * LX];
        }
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] += a_frag[i] * b_frag[j];
          }
        }
      }
    } else {
      // Mixed tile: b straight from global, per row group (module comment
      // §1) — for each (row, kk) the four column reads are coalesced across
      // the 16 adjacent tx lanes; a gathered b copy is never materialised.
      int const kmax = static_cast<int>(min(static_cast<int64_t>(BK), D1 - kb));
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int r = ty + i * LY;
        int64_t g = idx_s[r];
        if (g >= 0) {
          scalar_t const* b_gk = b + g * D1 * D2 + kb * D2;
          for (int kk = 0; kk < kmax; ++kk) {
            scalar_t const a_v = a_s[kk][r];
            scalar_t const* b_row = b_gk + static_cast<int64_t>(kk) * D2;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
              int64_t cc = col_base + tx + j * LX;
              if (cc < D2) {
                acc[i][j] += a_v * b_row[cc];
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int64_t row = row_base + ty + i * LY;
    if (row < N) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int64_t cc = col_base + tx + j * LX;
        if (cc < D2) {
          out[row * D2 + cc] = acc[i][j];
        }
      }
    }
  }
}

// 16-byte vector type per value dtype for LDS.128 fragment loads
// (float4 for f32, double2 for f64); count = TM/TN elements per 16B.
template <typename T>
struct Vec16;
template <>
struct Vec16<float> {
  using type = float4;
  static constexpr int kElems = 4;
};
template <>
struct Vec16<double> {
  using type = double2;
  static constexpr int kElems = 2;
};

// Scatter-reduce mode (module comment §2): out[g, :, :] = sum over the
// sorted-idx row range [lo, hi) of outer(a[i, :], b[i, :]). Block = one
// group's 64x64 tile of the (D1, D2) accumulator (256 threads, 4x4 register
// tiles, same geometry as gather mode); rows accumulated in ascending order
// in 16-row chunks — bitwise deterministic, no atomics.
template <typename scalar_t, typename index_t, int BM, int BN, int BK, int TM, int TN>
__global__ void grouped_gemm_scatter_kernel(scalar_t* __restrict__ out, scalar_t const* __restrict__ a,
                                            scalar_t const* __restrict__ b, index_t const* __restrict__ idx, int64_t N,
                                            int64_t D1, int64_t D2) {
  // Unpadded: both tiles stage row-contiguously (conflict-free writes), and
  // 16-byte-aligned row bases enable vectorised LDS.128 fragment reads.
  __shared__ scalar_t a_s[BK][BM];
  __shared__ scalar_t b_s[BK][BN];

  int64_t const g = static_cast<int64_t>(blockIdx.x);
  int64_t const i_base = static_cast<int64_t>(blockIdx.y) * BM;
  int64_t const j_base = static_cast<int64_t>(blockIdx.z) * BN;
  int const tid = static_cast<int>(threadIdx.x);
  constexpr int LX = BN / TN;  // column-lane count (thread grid is (LY, LX) = 256)
  constexpr int LY = BM / TM;  // row-lane count
  int const tx = tid % LX;
  int const ty = tid / LX;

  // Per-thread binary search (all threads land on the same [lo, hi); two
  // O(log N) probes are noise next to the row walk, so no lane-0+broadcast
  // dance is needed).
  int64_t const lo = lower_bound_idx<index_t>(idx, N, g);
  int64_t const hi = upper_bound_idx<index_t>(idx, N, g);

  scalar_t acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = scalar_t(0);
    }
  }

  for (int64_t r0 = lo; r0 < hi; r0 += BK) {
    // Stage 16-row chunks of a (columns i_base..i_base+63) and b (columns
    // j_base..j_base+63), zero-filled past hi/D1/D2 — zero rows contribute
    // nothing. c-fastest split keeps the global reads coalesced.
#pragma unroll
    for (int e = tid; e < BK * BM; e += kThreadsPerBlock) {
      int r = e / BM;
      int c = e % BM;
      int64_t row = r0 + r;
      int64_t i = i_base + c;
      a_s[r][c] = (row < hi && i < D1) ? a[row * D1 + i] : scalar_t(0);
      int64_t jj = j_base + c;
      b_s[r][c] = (row < hi && jj < D2) ? b[row * D2 + jj] : scalar_t(0);
    }
    __syncthreads();

    // Ascending, order-fixed accumulation over the chunk's rows (module
    // comment §2's determinism story); zero-filled tail rows add 0.
#pragma unroll
    for (int r = 0; r < BK; ++r) {
      scalar_t a_frag[TM];
      scalar_t b_frag[TN];
      // Contiguous per-thread fragments read as 16-byte vectors (LDS.128) —
      // cuts the shared-load instruction count 4x (f32) vs the strided
      // scalar mapping; measured 1.15 -> 0.99 ms at N=16384/D=256/R=16
      // (kernel_best_practices.md "Grouped GEMM" §7; siboehm SGEMM worklog).
      using vec_t = typename Vec16<scalar_t>::type;
      constexpr int kVE = Vec16<scalar_t>::kElems;
      vec_t const* a_vec = reinterpret_cast<vec_t const*>(&a_s[r][ty * TM]);
      vec_t const* b_vec = reinterpret_cast<vec_t const*>(&b_s[r][tx * TN]);
#pragma unroll
      for (int v = 0; v < TM / kVE; ++v) {
        reinterpret_cast<vec_t*>(a_frag)[v] = a_vec[v];
      }
#pragma unroll
      for (int v = 0; v < TN / kVE; ++v) {
        reinterpret_cast<vec_t*>(b_frag)[v] = b_vec[v];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }
    __syncthreads();
  }

  // Unconditional write of the block's in-range tile — empty groups
  // (lo == hi) write zeros, which is what makes new_empty output safe.
  scalar_t* out_g = out + g * D1 * D2;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int64_t ii = i_base + ty * TM + i;
    if (ii < D1) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int64_t jj = j_base + tx * TN + j;
        if (jj < D2) {
          out_g[ii * D2 + jj] = acc[i][j];
        }
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
      dim3 const block(kThreadsPerBlock);
      scalar_t* out_p = static_cast<scalar_t*>(out.mutable_data_ptr());
      scalar_t const* a_p = static_cast<scalar_t const*>(a_c.data_ptr());
      scalar_t const* b_p = static_cast<scalar_t const*>(b_c.data_ptr());
      index_t const* idx_p = static_cast<index_t const*>(idx_c.data_ptr());
      // Tile-config selection (constants block above): the 128x128x8 config's
      // 8x8 register tiles reach the cuBLAS-class 1:4 shared-load:FFMA ratio
      // but roughly double the accumulator registers — used for the wide f32
      // shapes where the vendor-parity bar lives. f64 (64 f64 accumulators
      // would spill) and narrow shapes stay on 64x64x16. `if constexpr`
      // keeps the fat config uninstantiated for f64 entirely.
      if (!reduce) {
        bool big = false;
        if constexpr (sizeof(scalar_t) == 4) {
          big = D2 >= 128 && N >= 128;
          if (big) {
            dim3 const grid(static_cast<unsigned>((N + 127) / 128), static_cast<unsigned>((D2 + 127) / 128));
            grouped_gemm_gather_kernel<scalar_t, index_t, 128, 128, 16, 8, 8>
                <<<grid, block, 0, guard.stream()>>>(out_p, a_p, b_p, idx_p, N, D1, D2);
          }
        }
        if (!big) {
          dim3 const grid(static_cast<unsigned>((N + kBM - 1) / kBM), static_cast<unsigned>((D2 + kBN - 1) / kBN));
          grouped_gemm_gather_kernel<scalar_t, index_t, kBM, kBN, kBK, kTM, kTN>
              <<<grid, block, 0, guard.stream()>>>(out_p, a_p, b_p, idx_p, N, D1, D2);
        }
      } else {
        bool big = false;
        if constexpr (sizeof(scalar_t) == 4) {
          big = D1 >= 128 && D2 >= 128;
          if (big) {
            dim3 const grid(static_cast<unsigned>(num_groups), static_cast<unsigned>((D1 + 127) / 128),
                            static_cast<unsigned>((D2 + 127) / 128));
            grouped_gemm_scatter_kernel<scalar_t, index_t, 128, 128, 16, 8, 8>
                <<<grid, block, 0, guard.stream()>>>(out_p, a_p, b_p, idx_p, N, D1, D2);
          }
        }
        if (!big) {
          dim3 const grid(static_cast<unsigned>(num_groups), static_cast<unsigned>((D1 + kBM - 1) / kBM),
                          static_cast<unsigned>((D2 + kBN - 1) / kBN));
          grouped_gemm_scatter_kernel<scalar_t, index_t, kBM, kBN, kBK, kTM, kTN>
              <<<grid, block, 0, guard.stream()>>>(out_p, a_p, b_p, idx_p, N, D1, D2);
        }
      }
    });
  });

  return out;
}
