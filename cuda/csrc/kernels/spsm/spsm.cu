// tsgu::spsm — Family 3 "Vendor-baseline forwards", Triangular SpSM row
// (kernels.md Family 3; spec/commit.md Phase 3 commit 16). Provenance:
// **custom**, not vendor-scaffold — goal.md's scaffold rule permits a
// cuSPARSE-wrapping v1, but this implementation is a from-scratch,
// level-scheduled SpTRSV; no cuSPARSE handle is opened anywhere in this
// file (advisor decision recorded in this commit's brief: linking cuSPARSE
// into the stable-ABI extension is an unforced build risk, and the
// post-migration backlog's "custom SpTRSV v2 only if warm SpSM loses"
// applies to a LATER rework, not this one).
//
// Solves, per folded row/owner (naming.md §2), a triangular system: for
// `!transpose`, `A[b] x[b] = rhs[b]`; for `transpose`, `A[b]^T x[b] =
// rhs[b]` — both with `A` triangular (`upper`/`unitriangular` flags kept
// verbatim, map.md contract). Also serves this same op's own `gradB`
// (map.md routing: "transposed plan" — the same op with `transpose`
// flipped, torchsparsegradutils/ops/triangular_solve.py's
// `_spsm_backward`, commit 9).
//
// Design (kernels.md Family 3 SpSM row, expanded to implementation level in
// this commit's brief) — two phases:
//
// 1. ANALYSIS (plan.h/plan.cpp): host-side, cached, NOT the hot path. Every
//    call reaches this same C++ entry point (the frozen schema has no plan-
//    tensor argument — plan.h's own module comment has the full "why"), so
//    the plan cache lives HERE, keyed on rowptr/col tensor identity,
//    realizing architecture.md §3's "the plan cache lives on the
//    descriptor" via the fact that a Python BatchedCSR holds the same
//    rowptr/col tensors for its whole lifetime. Produces a level schedule:
//    `level[owner] = 1 + max(level[dep])` over the owner's strict off-
//    diagonal dependencies, rows/columns grouped into `row_order` by level.
//
// 2. SOLVE (this file): one kernel LAUNCH PER LEVEL (kernels.md: "one
//    launch per level is the simple, correct v1"), all on the current
//    stream — CUDA's own same-stream launch ordering is what guarantees a
//    level's dependencies (all in strictly earlier levels, by the level
//    recurrence) are fully written before this level's kernel begins, with
//    NO host sync anywhere in this function (the analysis's one host
//    round-trip already happened, cached, before any solve kernel ever
//    launches — genuinely amortised on a cache hit). Within a level, ONE
//    WARP PER OWNER (mirroring spmm.cu's "one warp per folded row" — same
//    reasoning: every output row is owned by exactly one warp, so there
//    are NO ATOMICS anywhere in this kernel -> deterministic by
//    construction, same resolution as every other kernel in this codebase
//    for kernels.md's open Q1 (no separate path needed under
//    torch.use_deterministic_algorithms(True)) -- determinism additionally
//    depends on the level ORDER being fixed across calls, which it is: the
//    plan is built once and reused, and even a freshly-rebuilt plan for an
//    unchanged pattern recomputes the identical level assignment (the
//    analysis has no randomness or races of its own).
//
// The solve loop is intentionally the SIMPLEST correct shape (one register
// accumulator, entries re-walked once per 32-wide column stride) rather
// than spmm.cu/sddmm.cu's templated column-tile dispatch: SpSM's cost is
// dominated by the LEVEL COUNT (launch/dependency latency, especially for a
// deep chain — kernels.md's own worst case), not by per-row FMA throughput,
// so the wins spmm/sddmm's perf-fix commits (f69990d, a28fd95) chased don't
// apply the same way here; "cleanliness first" (kernels.md Family 3's own
// closing line) is the right v1 shape, with the plan cache doing the
// actual heavy lifting the acceptance bar (benchmarks.md §3: "win warm")
// asks for.
//
// f32/f64 only, BY POLICY (kernels.md: "half-precision triangular/iterative
// solves are numerically meaningless; documented as out of scope, not
// missing") — TSGU_DISPATCH_VALUE already only offers float/double, so this
// falls out for free, no extra guard needed.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <string>

#include "../../common/dispatch.cuh"
#include "../../common/stream.cuh"
#include "plan.h"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;  // 256 threads/block — one warp per owner in the level
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

// One level's worth of owners, one warp per owner (module comment above).
// `eff_ptr`/`eff_dep`/`eff_val_idx`/`diag_val_idx`/`row_order` are the
// plan's own int64 arrays (plan.h) — never the caller's original rowptr/
// col, which the plan has already fully absorbed at analysis time.
template <typename scalar_t>
__global__ void spsm_level_kernel(scalar_t *__restrict__ x, int64_t const *__restrict__ eff_ptr,
                                   int64_t const *__restrict__ eff_dep, int64_t const *__restrict__ eff_val_idx,
                                   int64_t const *__restrict__ diag_val_idx, int64_t const *__restrict__ row_order,
                                   int64_t level_start, int64_t level_size, scalar_t const *__restrict__ vals,
                                   scalar_t const *__restrict__ rhs, int64_t p, bool unitriangular) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (idx >= level_size) {
    return;
  }
  int64_t owner = row_order[level_start + idx];

  int64_t start = eff_ptr[owner];
  int64_t end = eff_ptr[owner + 1];

  // Diagonal value: analysis (plan.cpp) already validated existence per
  // the unitriangular contract (map.md invariant 7), so diag_val_idx[owner]
  // is a real index whenever !unitriangular — no -1 guard needed here.
  scalar_t diag = scalar_t(1);
  if (!unitriangular) {
    diag = vals[diag_val_idx[owner]];
  }

  scalar_t *x_row = x + owner * p;
  scalar_t const *rhs_row = rhs + owner * p;

  // Column-strided across the warp's 32 lanes; entries re-walked once per
  // stride group (module comment: cleanliness-first v1, level count/launch
  // latency dominates SpSM cost, not this loop's FMA throughput).
  for (int64_t j = lane; j < p; j += kWarpSize) {
    scalar_t acc = scalar_t(0);
    for (int64_t k = start; k < end; ++k) {
      int64_t dep = eff_dep[k];
      if (dep == owner) {
        continue;  // diagonal entry, not a dependency
      }
      // dep is always in a strictly earlier level than owner (the level
      // recurrence guarantees it), so x[dep, :] was written by a PRIOR
      // kernel launch on this same stream — safe to read with no sync here.
      acc += vals[eff_val_idx[k]] * x[dep * p + j];
    }
    scalar_t result = rhs_row[j] - acc;
    if (!unitriangular) {
      result = result / diag;
    }
    x_row[j] = result;  // owned by exactly this warp — no atomics anywhere.
  }
}

}  // namespace

torch::stable::Tensor tsgu_spsm_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                        torch::stable::Tensor const &col, torch::stable::Tensor const &rhs, int64_t B,
                                        int64_t n, bool upper, bool unitriangular, bool transpose) {
  STD_TORCH_CHECK(vals.is_cuda() && rowptr.is_cuda() && col.is_cuda() && rhs.is_cuda(),
                   "tsgu::spsm expects CUDA tensors; got is_cuda=(", vals.is_cuda(), ", ", rowptr.is_cuda(), ", ",
                   col.is_cuda(), ", ", rhs.is_cuda(), ") for (vals, rowptr, col, rhs).");
  STD_TORCH_CHECK(rowptr.dim() == 1 && rowptr.size(0) == B * n + 1,
                   "tsgu::spsm expects rowptr of shape (B * n + 1,) = (", B * n + 1, ",) for B=", B, ", n=", n,
                   "; got shape (", (rowptr.dim() == 1 ? rowptr.size(0) : -1), ",) with dim=", rowptr.dim(), ".");
  STD_TORCH_CHECK(col.dim() == 1, "tsgu::spsm expects col of shape (nse_total,); got dim=", col.dim(), ".");
  STD_TORCH_CHECK(vals.dim() == 1 && vals.size(0) == col.size(0),
                   "tsgu::spsm expects vals of shape (nse_total,) matching col's length (", col.size(0),
                   ",); got shape with dim=", vals.dim(), (vals.dim() == 1 ? ", (" : ""),
                   (vals.dim() == 1 ? vals.size(0) : -1), (vals.dim() == 1 ? ",)" : ""), ".");
  STD_TORCH_CHECK(rowptr.scalar_type() == col.scalar_type(),
                   "tsgu::spsm expects rowptr and col to share one index dtype (torch.int32 or torch.int64); got "
                   "rowptr dtype id ",
                   static_cast<int>(rowptr.scalar_type()), " vs col dtype id ", static_cast<int>(col.scalar_type()),
                   ".");
  STD_TORCH_CHECK(rowptr.scalar_type() == torch::headeronly::ScalarType::Int ||
                       rowptr.scalar_type() == torch::headeronly::ScalarType::Long,
                   "tsgu::spsm expects rowptr/col index dtype to be torch.int32 or torch.int64; got dtype id ",
                   static_cast<int>(rowptr.scalar_type()), ".");
  STD_TORCH_CHECK(rhs.dim() == 3 && rhs.size(0) == B && rhs.size(1) == n,
                   "tsgu::spsm expects rhs of shape (B, n, p) = (", B, ", ", n, ", p); got shape with dim=", rhs.dim(),
                   (rhs.dim() == 3 ? ", (" : ""), (rhs.dim() == 3 ? rhs.size(0) : -1), (rhs.dim() == 3 ? ", " : ""),
                   (rhs.dim() == 3 ? rhs.size(1) : -1), (rhs.dim() == 3 ? ", *)" : ""), ".");
  STD_TORCH_CHECK(vals.scalar_type() == rhs.scalar_type(),
                   "tsgu::spsm expects vals and rhs to share one value dtype (torch.float32 or torch.float64 — "
                   "kernels.md: SpSM is f32/f64 only, by policy); got vals dtype id ",
                   static_cast<int>(vals.scalar_type()), " vs rhs dtype id ", static_cast<int>(rhs.scalar_type()),
                   ".");

  // Dense-operand contiguity (matching spmm.cu/sddmm.cu's convention): a
  // host-side copy rather than a reject, so opcheck's non-contiguous-rhs
  // cases exercise real, correct output.
  torch::stable::Tensor rhs_c = rhs.is_contiguous() ? rhs : torch::stable::contiguous(rhs);

  int64_t p = rhs.size(2);
  torch::stable::Tensor x = torch::stable::new_empty(rhs, {B, n, p});
  tsgu::StreamGuard guard(vals);

  int64_t total_rows = B * n;
  if (total_rows == 0 || p == 0) {
    // Nothing to solve / nothing to write — x is already correctly shaped
    // (and empty in the relevant dimension), matching spmm.cu's p==0
    // convention.
    return x;
  }

  // Analysis (plan.h/plan.cpp) — cache hit on any repeat call built from
  // the same BatchedCSR's rowptr/col (architecture.md §3's "plan cache
  // lives on the descriptor", realized via tensor-identity keying — see
  // plan.h's module comment for the full design).
  std::shared_ptr<tsgu::SpsmPlan const> plan = tsgu::get_or_build_plan(rowptr, col, B, n, upper, unitriangular, transpose);

  TSGU_DISPATCH_VALUE(vals.scalar_type(), "tsgu::spsm", [&] {
    int64_t const *eff_ptr = static_cast<int64_t const *>(plan->eff_ptr.data_ptr());
    int64_t const *eff_dep = static_cast<int64_t const *>(plan->eff_dep.data_ptr());
    int64_t const *eff_val_idx = static_cast<int64_t const *>(plan->eff_val_idx.data_ptr());
    int64_t const *diag_val_idx = static_cast<int64_t const *>(plan->diag_val_idx.data_ptr());
    int64_t const *row_order = static_cast<int64_t const *>(plan->row_order.data_ptr());
    scalar_t *x_ptr = static_cast<scalar_t *>(x.mutable_data_ptr());
    scalar_t const *vals_ptr = static_cast<scalar_t const *>(vals.data_ptr());
    scalar_t const *rhs_ptr = static_cast<scalar_t const *>(rhs_c.data_ptr());

    int64_t const n_levels = plan->n_levels();
    for (int64_t lvl = 0; lvl < n_levels; ++lvl) {
      int64_t level_start = plan->level_ptr[static_cast<size_t>(lvl)];
      int64_t level_size = plan->level_ptr[static_cast<size_t>(lvl) + 1] - level_start;
      if (level_size == 0) {
        continue;
      }
      int64_t blocks = (level_size + kWarpsPerBlock - 1) / kWarpsPerBlock;
      // One launch per level (module comment) — same stream throughout, so
      // level lvl+1 never begins until level lvl's writes to `x` are
      // complete; no host sync anywhere in this loop.
      spsm_level_kernel<scalar_t><<<blocks, kThreadsPerBlock, 0, guard.stream()>>>(
          x_ptr, eff_ptr, eff_dep, eff_val_idx, diag_val_idx, row_order, level_start, level_size, vals_ptr, rhs_ptr,
          p, unitriangular);
    }
  });

  return x;
}

// tsgu::_spsm_plan_cache_stats — test/introspection-only (torchsparsegradutils/
// ops/triangular_solve.py's own comment has the full "why"): exposes
// plan.cpp's process-wide (builds, hits) counters as a 2-element int64
// tensor, so a Python test can assert "same BatchedCSR solved twice ->
// analysis computed once" (spec/commit.md Phase 3 commit 16 T5).
torch::stable::Tensor tsgu_spsm_plan_cache_stats_launch(torch::stable::Tensor const &anchor) {
  tsgu::SpsmPlanCacheStats stats = tsgu::spsm_plan_cache_stats();
  torch::stable::Tensor out = torch::stable::new_empty(anchor, {2}, torch::headeronly::ScalarType::Long);
  int64_t host[2] = {stats.builds, stats.hits};
  cudaMemcpy(out.mutable_data_ptr(), host, sizeof(host), cudaMemcpyHostToDevice);
  return out;
}
