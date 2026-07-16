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
// Parallelisation (kernels.md Family 1: "one warp per specified entry,
// warp-reduce over p"): one warp per `k` in `[0, nse_total)`, not one warp
// per row as in the seglse family — the reduction axis here is `p` (the
// dense inner dimension), not the segment's member count, so each lane
// strides its slice of `p` and a warp-shuffle sum-reduction (reduce.cuh's
// warp_reduce_sum) finishes it. Shared-memory G-row tiling for large `p`
// (kernels.md's "optional optimisation") is not pursued here — the plain
// warp path is exercised by the benchmark (bench_sddmm.cu) up to p=512 and
// is not "clearly poor" at that width, so the optional path is skipped per
// the family section's own guidance.
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
  int64_t blocks = (nse_total + kWarpsPerBlock - 1) / kWarpsPerBlock;

  TSGU_DISPATCH_VALUE(g.scalar_type(), "tsgu::sddmm", [&] {
    TSGU_DISPATCH_INDEX(rowptr.scalar_type(), "tsgu::sddmm", [&] {
      sddmm_kernel<scalar_t, index_t><<<blocks, kThreadsPerBlock, 0, guard.stream()>>>(
          static_cast<scalar_t *>(out.mutable_data_ptr()), static_cast<index_t const *>(rowptr.data_ptr()),
          static_cast<index_t const *>(col.data_ptr()), static_cast<scalar_t const *>(g_c.data_ptr()),
          static_cast<scalar_t const *>(mat_c.data_ptr()), total_rows, nse_total, n, m, p, negate);
    });
  });

  return out;
}
