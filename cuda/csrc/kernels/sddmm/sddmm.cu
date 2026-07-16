// tsgu::sddmm — Family 1 "SDDMM (the shared backward)" (kernels.md Family 1;
// spec/commit.md Phase 3 commit 14).
//
// Sampled dense-dense matmul at a fixed sparsity pattern:
// `out[k] = dot(g[b(k), row(k), :], mat[b(k), col[k], :])` for every
// specified entry `k` of the pattern given by `rowptr`/`col` — never a dense
// materialisation, and `out` reuses the pattern's own index arrays (no index
// allocation, per kernels.md). Serves the shared backward of four ops
// (`sparse_mm`, `sparse_triangular_solve`, `sparse_generic_solve`,
// `sparse_generic_lstsq` — map.md routing); the schema itself (op def, fake
// kernel) already landed in torchsparsegradutils/ops/matmul.py (commit 9)
// and is not touched here — this file supplies the one CUDA implementation
// that schema has been missing.
//
// Parallelisation — two kernels, chosen host-side by the launcher:
//
// 1. `sddmm_row_kernel` (the hot path): one warp per FOLDED ROW. The warp
//    stages `g[row, :]` into shared memory once, then walks the row's
//    entries doing one warp-reduced dot against `mat[col[k], :]` per entry.
//    This is kernels.md Family 1's "tile G rows through shared memory for
//    large p" — with ~hundreds of entries per row (the benchmark corpus,
//    and any real gradient pattern of similar density), the warp-per-entry
//    design re-reads the same G row once PER ENTRY, which put commit 14's
//    first cut at 0.73x of cuSPARSE (11.81ms vs 8.66ms, 4096^2/1.68M/p=128):
//    it was already at the no-reuse roofline (~150 GB/s of ~176 GB/s peak),
//    so the only way forward was to stop paying for G. Row-tiling reads G
//    exactly once per row (n*p total instead of nse*p) — a ~2x traffic cut
//    at that density. Row lookup also becomes free (row = warp id), killing
//    the per-entry binary search.
//
// 2. `sddmm_kernel` (fallback): one warp per SPECIFIED ENTRY with a per-warp
//    binary search for the row — kernels.md's "good for small p" baseline
//    design, kept for the patterns where row-tiling has nothing to reuse or
//    can't run: near-diagonal patterns (avg < 2 entries/row — the G-row
//    staging would be pure overhead and warp-per-row underparallelises),
//    and p too large for the shared-memory budget (row of f64 at
//    kWarpsPerBlock rows/block must fit the 48KB default smem window).
//    It is also perfectly load-balanced, which the row kernel is not —
//    acceptable because skewed-row load balancing is spmm's problem
//    (merge-path, kernels.md Family 3), not the backward's.
//
// Both kernels write each output element from exactly one warp — no atomics,
// both deterministic, so the heuristic switch is invisible to the
// determinism contract (gate 5 run-twice holds regardless of path).
//
// Row lookup: the schema gives each entry only `col[k]`, not its row —
// `rowptr` is monotonic non-decreasing over folded rows (naming.md §2:
// `row_global = b * n + r`), so `row(k)` is recovered with a per-warp binary
// search (`find_row` below, run once by lane 0, broadcast via warp shuffle)
// rather than a second precomputed index array — the same technique CSR
// SDDMM/SpMV implementations generally use to avoid uncompressing `rowptr`
// into a full per-entry row array. `batch(k) = row(k) / n`, matching
// common/batched_csr.cuh's `BatchedCSRView::batch_of`.
//
// Fusion: `negate` (kernels.md: "negate-and-scale folded in for the solve
// backwards, avoid a second pass over nse") is applied as a sign flip on the
// warp's own reduced value before the single write to `out[k]` — no second
// kernel, no extra pass over `vals`/`col` (there is no `vals` here at all;
// this op samples `g`/`mat`, not the pattern's stored values).
//
// No atomics anywhere (each warp owns exactly one output element) →
// deterministic by construction; no separate path is needed under
// `torch.use_deterministic_algorithms(True)` (kernels.md open Q1, same
// resolution as seglse.cu).
//
// f32/f64 x i32/i64 via common/dispatch.cuh, matching every other kernel in
// this codebase (architecture.md §3).
//
// Dense-operand contiguity: `g`/`mat` are read through flat strided-free
// pointers inside the kernel (row-major `(B, n/m, p)`), so a non-contiguous
// caller tensor would alias incorrectly through that flat indexing — exactly
// the stride-0 broadcast-grad OOB lesson from commit 12 (seglse_bwd). Unlike
// seglse_bwd (whose Python wrapper `.contiguous()`s before the call), this
// op has no wrapper in this commit (T4 is skipped — "no wrapper switch"), so
// the launcher itself makes `g`/`mat` contiguous host-side via
// `torch::stable::contiguous()` before ever computing a pointer into them —
// the "or reject" alternative kernels.md/this commit's instructions allow is
// not taken here so that opcheck's non-contiguous-operand cases (testing.md
// Pillar 2) exercise real, correct output rather than an error path.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <string>

#include "../../common/dispatch.cuh"
#include "../../common/reduce.cuh"
#include "../../common/stream.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;  // 256 threads/block — one warp per specified entry
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

// Finds the (folded) row `row_g` such that `rowptr[row_g] <= k <
// rowptr[row_g + 1]`, i.e. the segment owning entry index `k`, via binary
// search over the `total_rows + 1`-length monotonic `rowptr` array. Standard
// upper-bound-style CSR "row of nnz index" search — empty rows
// (`rowptr[i] == rowptr[i + 1]`) are correctly skipped since the search
// condition only advances past a row once its end pointer exceeds `k`.
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

template <typename scalar_t, typename index_t>
__global__ void sddmm_kernel(scalar_t *__restrict__ out, index_t const *__restrict__ rowptr,
                              index_t const *__restrict__ col, scalar_t const *__restrict__ g,
                              scalar_t const *__restrict__ mat, int64_t total_rows, int64_t nse_total, int64_t n,
                              int64_t m, int64_t p, bool negate) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t k = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (k >= nse_total) {
    return;
  }

  int64_t row_g = 0;
  if (lane == 0) {
    row_g = find_row<index_t>(rowptr, total_rows, k);
  }
  row_g = __shfl_sync(0xffffffffU, row_g, 0);

  int64_t b = row_g / n;
  int64_t local_row = row_g % n;
  int64_t col_k = static_cast<int64_t>(col[k]);

  scalar_t const *g_row = g + (b * n + local_row) * p;
  scalar_t const *mat_row = mat + (b * m + col_k) * p;

  scalar_t partial = scalar_t(0);
  for (int64_t j = lane; j < p; j += kWarpSize) {
    partial += g_row[j] * mat_row[j];
  }
  partial = tsgu::warp_reduce_sum<scalar_t>(partial);

  if (lane == 0) {
    out[k] = negate ? -partial : partial;
  }
}

// Row-tiled hot path (module comment §1): one warp per folded row, G row
// staged through dynamic shared memory (`kWarpsPerBlock * p * sizeof(scalar)`
// per block — the launcher only selects this kernel when that fits the 48KB
// default window). Empty rows return immediately; `out[k]` is written by the
// single warp owning row(k), so no atomics and bitwise-deterministic output.
template <typename scalar_t, typename index_t>
__global__ void sddmm_row_kernel(scalar_t *__restrict__ out, index_t const *__restrict__ rowptr,
                                  index_t const *__restrict__ col, scalar_t const *__restrict__ g,
                                  scalar_t const *__restrict__ mat, int64_t total_rows, int64_t n, int64_t m,
                                  int64_t p, bool negate) {
  extern __shared__ unsigned char smem_raw[];
  scalar_t *g_smem_block = reinterpret_cast<scalar_t *>(smem_raw);

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

  // Stage this warp's G row once; every entry of the row reuses it.
  scalar_t *g_smem = g_smem_block + warp_in_block * p;
  scalar_t const *g_row = g + row_g * p;  // (b * n + local_row) * p == row_g * p
  for (int64_t j = lane; j < p; j += kWarpSize) {
    g_smem[j] = g_row[j];
  }
  __syncwarp();

  int64_t b = row_g / n;
  scalar_t const *mat_b = mat + b * m * p;

  for (int64_t k = start; k < end; ++k) {
    scalar_t const *mat_row = mat_b + static_cast<int64_t>(col[k]) * p;
    scalar_t partial = scalar_t(0);
    for (int64_t j = lane; j < p; j += kWarpSize) {
      partial += g_smem[j] * mat_row[j];
    }
    partial = tsgu::warp_reduce_sum<scalar_t>(partial);
    if (lane == 0) {
      out[k] = negate ? -partial : partial;
    }
  }
}

}  // namespace

torch::stable::Tensor tsgu_sddmm_launch(torch::stable::Tensor const &rowptr, torch::stable::Tensor const &col,
                                         torch::stable::Tensor const &g, torch::stable::Tensor const &mat, int64_t B,
                                         int64_t n, int64_t m, bool negate) {
  STD_TORCH_CHECK(rowptr.is_cuda() && col.is_cuda() && g.is_cuda() && mat.is_cuda(),
                   "tsgu::sddmm expects CUDA tensors; got is_cuda=(", rowptr.is_cuda(), ", ", col.is_cuda(), ", ",
                   g.is_cuda(), ", ", mat.is_cuda(), ") for (rowptr, col, g, mat).");
  STD_TORCH_CHECK(rowptr.dim() == 1 && rowptr.size(0) == B * n + 1,
                   "tsgu::sddmm expects rowptr of shape (B * n + 1,) = (", B * n + 1, ",) for B=", B, ", n=", n,
                   "; got shape (", (rowptr.dim() == 1 ? rowptr.size(0) : -1), ",) with dim=", rowptr.dim(), ".");
  STD_TORCH_CHECK(col.dim() == 1, "tsgu::sddmm expects col of shape (nse_total,); got dim=", col.dim(), ".");
  STD_TORCH_CHECK(rowptr.scalar_type() == col.scalar_type(),
                   "tsgu::sddmm expects rowptr and col to share one index dtype (torch.int32 or torch.int64); got "
                   "rowptr dtype id ",
                   static_cast<int>(rowptr.scalar_type()), " vs col dtype id ", static_cast<int>(col.scalar_type()),
                   ".");
  STD_TORCH_CHECK(g.dim() == 3 && g.size(0) == B && g.size(1) == n,
                   "tsgu::sddmm expects g of shape (B, n, p) = (", B, ", ", n,
                   ", p); got shape with dim=", g.dim(), (g.dim() == 3 ? ", (" : ""),
                   (g.dim() == 3 ? g.size(0) : -1), (g.dim() == 3 ? ", " : ""), (g.dim() == 3 ? g.size(1) : -1),
                   (g.dim() == 3 ? ", *)" : ""), ".");
  STD_TORCH_CHECK(mat.dim() == 3 && mat.size(0) == B && mat.size(1) == m,
                   "tsgu::sddmm expects mat of shape (B, m, p) = (", B, ", ", m,
                   ", p); got shape with dim=", mat.dim(), (mat.dim() == 3 ? ", (" : ""),
                   (mat.dim() == 3 ? mat.size(0) : -1), (mat.dim() == 3 ? ", " : ""), (mat.dim() == 3 ? mat.size(1) : -1),
                   (mat.dim() == 3 ? ", *)" : ""), ".");
  STD_TORCH_CHECK(g.size(2) == mat.size(2),
                   "tsgu::sddmm expects g and mat to share their trailing dimension p; got g[..., ", g.size(2),
                   "] vs mat[..., ", mat.size(2), "].");
  STD_TORCH_CHECK(g.scalar_type() == mat.scalar_type(),
                   "tsgu::sddmm expects g and mat to share one value dtype (torch.float32 or torch.float64); got g "
                   "dtype id ",
                   static_cast<int>(g.scalar_type()), " vs mat dtype id ", static_cast<int>(mat.scalar_type()), ".");

  // Dense-operand contiguity (module comment above): make host-side copies
  // rather than reject, so the flat-pointer indexing in the kernel is always
  // valid regardless of the caller's strides.
  torch::stable::Tensor g_c = g.is_contiguous() ? g : torch::stable::contiguous(g);
  torch::stable::Tensor mat_c = mat.is_contiguous() ? mat : torch::stable::contiguous(mat);

  int64_t nse_total = col.size(0);
  int64_t p = g.size(2);

  torch::stable::Tensor out = torch::stable::new_empty(g, {nse_total});
  tsgu::StreamGuard guard(rowptr);

  if (nse_total == 0) {
    return out;
  }

  int64_t total_rows = B * n;

  TSGU_DISPATCH_VALUE(g.scalar_type(), "tsgu::sddmm", [&] {
    TSGU_DISPATCH_INDEX(rowptr.scalar_type(), "tsgu::sddmm", [&] {
      // Path choice (module comment): row-tiling pays only when rows have
      // entries to reuse the staged G row on (avg >= 2 entries/row) and the
      // per-block staging buffer fits the 48KB default shared-memory window.
      size_t row_smem = static_cast<size_t>(kWarpsPerBlock) * static_cast<size_t>(p) * sizeof(scalar_t);
      bool use_row_path = nse_total >= 2 * total_rows && row_smem <= 48 * 1024;
      if (use_row_path) {
        int64_t blocks = (total_rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
        sddmm_row_kernel<scalar_t, index_t><<<blocks, kThreadsPerBlock, row_smem, guard.stream()>>>(
            static_cast<scalar_t *>(out.mutable_data_ptr()), static_cast<index_t const *>(rowptr.data_ptr()),
            static_cast<index_t const *>(col.data_ptr()), static_cast<scalar_t const *>(g_c.data_ptr()),
            static_cast<scalar_t const *>(mat_c.data_ptr()), total_rows, n, m, p, negate);
      } else {
        int64_t blocks = (nse_total + kWarpsPerBlock - 1) / kWarpsPerBlock;
        sddmm_kernel<scalar_t, index_t><<<blocks, kThreadsPerBlock, 0, guard.stream()>>>(
            static_cast<scalar_t *>(out.mutable_data_ptr()), static_cast<index_t const *>(rowptr.data_ptr()),
            static_cast<index_t const *>(col.data_ptr()), static_cast<scalar_t const *>(g_c.data_ptr()),
            static_cast<scalar_t const *>(mat_c.data_ptr()), total_rows, nse_total, n, m, p, negate);
      }
    });
  });

  return out;
}
