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
// 2. SOLVE (this file): ONE kernel launch total — a SYNC-FREE persistent-
//    warp solve (the standard sync-free SpTRSV technique, à la Liu et al.),
//    replacing commit 16's one-launch-per-level v1. The per-level shape was
//    correct but its cost was the LEVEL COUNT: a banded matrix (the
//    acceptance row: n=4096, bandwidth 8) has one level per row, so the
//    "solve" was 4096 back-to-back kernel launches at ~3.6us each — ~15ms
//    of pure launch latency for ~100us of arithmetic, which is why warm
//    only tied cuSPARSE (1.04x) and cold lost (0.84x).
//
//    Shape: a fixed grid of persistent warps; each warp repeatedly claims
//    the next unprocessed position in `row_order` via an atomic ticket
//    counter, waits (spin on a per-row done-flag) for the claimed row's
//    dependencies to be solved, solves it, publishes its done-flag. Two
//    properties make this safe and exact:
//
//    - NO DEADLOCK, regardless of block scheduling: `row_order` is
//      topological (level-sorted, plan.cpp), so a row's dependencies sit
//      strictly earlier in `row_order` and their tickets were claimed
//      before this warp claimed its own. Inductively the minimal
//      unfinished ticket always has every dependency finished, so some
//      resident warp always progresses — no reliance on CUDA scheduling
//      blocks in blockIdx order (the assumption published sync-free
//      SpTRSVs lean on; the ticket counter removes it).
//
//    - STILL DETERMINISTIC: the flag atomics only gate VISIBILITY (when a
//      row may be read), never participate in a value. Each x row is
//      written by exactly one warp, and each lane's accumulator walks the
//      row's entries in the same fixed order every run — bitwise-identical
//      output across runs, same resolution as every other kernel here for
//      kernels.md's open Q1 (no separate path needed under
//      torch.use_deterministic_algorithms(True)).
//
//    Memory ordering: the producer __threadfence()s after writing its x
//    row, then sets its flag with an atomic; the consumer spins on the
//    flag with an atomic read and __threadfence()s once before reading any
//    x rows — the standard release/acquire pairing for CUDA global memory.
//
// The per-row solve loop keeps the SIMPLEST correct shape (one register
// accumulator, entries re-walked once per 32-wide column stride) rather
// than spmm.cu/sddmm.cu's templated column-tile dispatch: with launches
// eliminated, remaining cost is dependency-chain latency, not per-row FMA
// throughput ("cleanliness first", kernels.md Family 3's closing line).
//
// f32/f64 only, BY POLICY (kernels.md: "half-precision triangular/iterative
// solves are numerically meaningless; documented as out of scope, not
// missing") — TSGU_DISPATCH_VALUE already only offers float/double, so this
// falls out for free, no extra guard needed.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <algorithm>
#include <string>

#include "../../common/dispatch.cuh"
#include "../../common/stream.cuh"
#include "plan.h"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;  // 256 threads/block
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
// Consecutive row_order positions solved per claimed ticket (kernel comment:
// the serial-chain latency fix — intra-chunk dependencies never pay the
// inter-warp flag handoff). 8 balances chain latency against parallel width
// on wide levels (total_rows/8 concurrent warps is still far above resident
// capacity for any acceptance-size problem).
constexpr int kRowsPerTicket = 8;

// Sync-free persistent-warp solve (module comment §2). `eff_ptr`/`eff_dep`/
// `eff_val_idx`/`diag_val_idx`/`row_order` are the plan's own int64 arrays
// (plan.h) — never the caller's original rowptr/col, which the plan has
// already fully absorbed at analysis time. `work` is the per-call int32
// workspace: work[0] = ticket counter, work[1 + row] = done-flag per folded
// row, all zeroed before launch.
template <typename scalar_t>
__global__ void spsm_syncfree_kernel(scalar_t *__restrict__ x, int64_t const *__restrict__ eff_ptr,
                                     int64_t const *__restrict__ eff_dep, int64_t const *__restrict__ eff_val_idx,
                                     int64_t const *__restrict__ diag_val_idx, int64_t const *__restrict__ row_order,
                                     int64_t total_rows, scalar_t const *__restrict__ vals,
                                     scalar_t const *__restrict__ rhs, int64_t p, bool unitriangular,
                                     int *__restrict__ work) {
  int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  int *ticket_counter = work;
  int *done = work + 1;

  while (true) {
    // Claim the next CHUNK of row_order positions, warp-wide. Chunking is
    // the serial-chain latency fix: consecutive positions in a dependency
    // chain land in the same warp, whose own writes are visible to itself
    // immediately — the expensive inter-warp flag handoff (spin + fences
    // through L2) happens once per chunk instead of once per row.
    int ticket = 0;
    if (lane == 0) {
      ticket = atomicAdd(ticket_counter, 1);
    }
    ticket = __shfl_sync(0xffffffffU, ticket, 0);
    int64_t chunk_base = static_cast<int64_t>(ticket) * kRowsPerTicket;
    if (chunk_base >= total_rows) {
      return;
    }
    int64_t chunk_end = min(chunk_base + kRowsPerTicket, total_rows);

    for (int64_t pos = chunk_base; pos < chunk_end; ++pos) {
    int64_t owner = row_order[pos];

    int64_t start = eff_ptr[owner];
    int64_t end = eff_ptr[owner + 1];

    // Wait phase: spin until every dependency's done-flag is set. Lanes
    // split the entry list so independent flags are polled concurrently;
    // the atomicAdd(...,0) read bypasses L1 so a producer's flag store is
    // always observed.
    for (int64_t k = start + lane; k < end; k += kWarpSize) {
      int64_t dep = eff_dep[k];
      if (dep == owner) {
        continue;  // diagonal entry, not a dependency
      }
      while (atomicAdd(done + dep, 0) == 0) {
#if __CUDA_ARCH__ >= 700
        __nanosleep(40);
#endif
      }
    }
    __syncwarp();
    __threadfence();  // acquire: dep flags seen -> dep x rows readable below.

    // Diagonal value: analysis (plan.cpp) already validated existence per
    // the unitriangular contract (map.md invariant 7), so diag_val_idx[owner]
    // is a real index whenever !unitriangular — no -1 guard needed here.
    scalar_t diag = scalar_t(1);
    if (!unitriangular) {
      diag = vals[diag_val_idx[owner]];
    }

    scalar_t *x_row = x + owner * p;
    scalar_t const *rhs_row = rhs + owner * p;

    if (p <= 16) {
      // Small-p path (the solve-shaped regime): the row solve sits on the
      // serial critical path of the dependency chain, so its LATENCY — a
      // chain of dependent global loads — is what matters, not throughput.
      // Split the warp into G = 32/p_pad entry-groups × p_pad columns: each
      // group accumulates a disjoint entry subset concurrently (dependent-
      // load depth drops G-fold), then a fixed shuffle tree sums the groups
      // — fixed combine order, still deterministic (module comment §2).
      int p_pad = 1;
      while (p_pad < static_cast<int>(p)) {
        p_pad <<= 1;
      }
      int groups = kWarpSize / p_pad;
      int group = lane / p_pad;
      int j = lane % p_pad;

      scalar_t acc = scalar_t(0);
      if (j < p) {
        for (int64_t k = start + group; k < end; k += groups) {
          int64_t dep = eff_dep[k];
          if (dep == owner) {
            continue;  // diagonal entry, not a dependency
          }
          acc += vals[eff_val_idx[k]] * x[dep * p + j];
        }
      }
#pragma unroll
      for (int offset = kWarpSize / 2; offset >= 1; offset >>= 1) {
        if (offset >= p_pad) {
          acc += __shfl_down_sync(0xffffffffU, acc, offset);
        }
      }
      if (group == 0 && j < p) {
        scalar_t result = rhs_row[j] - acc;
        if (!unitriangular) {
          result = result / diag;
        }
        x_row[j] = result;  // owned by exactly this warp — no data atomics.
      }
    } else {
      // Wide-p path: column-strided across the warp's 32 lanes; entries
      // re-walked once per stride group, each lane's accumulation order
      // fixed -> deterministic (module comment §2).
      for (int64_t j = lane; j < p; j += kWarpSize) {
        scalar_t acc = scalar_t(0);
        for (int64_t k = start; k < end; ++k) {
          int64_t dep = eff_dep[k];
          if (dep == owner) {
            continue;  // diagonal entry, not a dependency
          }
          acc += vals[eff_val_idx[k]] * x[dep * p + j];
        }
        scalar_t result = rhs_row[j] - acc;
        if (!unitriangular) {
          result = result / diag;
        }
        x_row[j] = result;  // owned by exactly this warp — no data atomics.
      }
    }

    // Publish: release-fence the x row, then set the done-flag.
    __syncwarp();
    __threadfence();
    if (lane == 0) {
      atomicExch(done + owner, 1);
    }
    }  // chunk loop
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

  // Per-call int32 workspace: [0] = ticket counter, [1..total_rows] =
  // done-flags. O(n_rows), within kernels.md's workspace bound; zeroed
  // asynchronously on the same stream (no host sync).
  torch::stable::Tensor work =
      torch::stable::new_empty(rowptr, {total_rows + 1}, torch::headeronly::ScalarType::Int);
  cudaMemsetAsync(work.mutable_data_ptr(), 0, static_cast<size_t>(total_rows + 1) * sizeof(int), guard.stream());

  TSGU_DISPATCH_VALUE(vals.scalar_type(), "tsgu::spsm", [&] {
    int64_t const *eff_ptr = static_cast<int64_t const *>(plan->eff_ptr.data_ptr());
    int64_t const *eff_dep = static_cast<int64_t const *>(plan->eff_dep.data_ptr());
    int64_t const *eff_val_idx = static_cast<int64_t const *>(plan->eff_val_idx.data_ptr());
    int64_t const *diag_val_idx = static_cast<int64_t const *>(plan->diag_val_idx.data_ptr());
    int64_t const *row_order = static_cast<int64_t const *>(plan->row_order.data_ptr());
    scalar_t *x_ptr = static_cast<scalar_t *>(x.mutable_data_ptr());
    scalar_t const *vals_ptr = static_cast<scalar_t const *>(vals.data_ptr());
    scalar_t const *rhs_ptr = static_cast<scalar_t const *>(rhs_c.data_ptr());

    // Persistent grid: enough blocks to cover the work, capped at what the
    // device can keep resident simultaneously (the ticket loop makes any
    // grid size correct; the cap just avoids launching blocks that would
    // only queue behind spinning ones).
    int device = 0;
    cudaGetDevice(&device);
    int sm_count = 1;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    int blocks_per_sm = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, spsm_syncfree_kernel<scalar_t>, kThreadsPerBlock,
                                                  0);
    int64_t max_resident = static_cast<int64_t>(sm_count) * std::max(blocks_per_sm, 1);
    int64_t tickets = (total_rows + kRowsPerTicket - 1) / kRowsPerTicket;
    int64_t needed = (tickets + kWarpsPerBlock - 1) / kWarpsPerBlock;
    int64_t blocks = std::min(needed, max_resident);

    // Single launch, current stream (module comment §2) — the done-flag
    // protocol inside the kernel replaces the per-level launch ordering.
    spsm_syncfree_kernel<scalar_t><<<blocks, kThreadsPerBlock, 0, guard.stream()>>>(
        x_ptr, eff_ptr, eff_dep, eff_val_idx, diag_val_idx, row_order, total_rows, vals_ptr, rhs_ptr, p,
        unitriangular, static_cast<int *>(work.mutable_data_ptr()));
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
