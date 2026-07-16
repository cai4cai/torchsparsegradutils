#pragma once
// tsgu::spsm analysis-plan cache (architecture.md §3: "SpSM analysis
// plans... the plan cache lives on the descriptor, so its lifetime and
// invalidation are the descriptor's lifetime; no global cache, no hashing";
// kernels.md Family 3 SpSM row: "Analysis-phase reuse: cache level-schedule
// per sparsity pattern... amortised wins in iterative/rsample loops").
//
// The Python-facing tsgu::spsm schema (torchsparsegradutils/ops/
// triangular_solve.py, commit 9) is frozen at
// (vals, rowptr, col, rhs, B, n, upper, unitriangular, transpose) -- there
// is no room in that schema for separate plan-tensor arguments, and
// spsm.register_autograd's own gradB call invokes torch.ops.tsgu.spsm
// directly (bypassing any Python-level wrapper orchestration), so a
// Python-side plan object cannot be threaded through to the kernel via the
// op boundary at all: every call, forward or backward, reaches this same
// C++ entry point with only the raw pattern arrays.
//
// Resolution (this commit): the plan lives here, C++-side, keyed on the
// *tensor identity* (data pointer) of the caller's rowptr/col -- which is
// exactly how "the plan cache lives on the descriptor" is realized in
// practice: torchsparsegradutils._batched.BatchedCSR is a frozen dataclass
// that holds the SAME rowptr/col Tensor objects for its whole lifetime, so
// repeated tsgu::spsm calls built from one BatchedCSR (the common case --
// an rsample loop, a CG solve reusing one preconditioner, this op's own
// gradB call reusing the SAME vals/rowptr/col the forward call used) hit
// this cache; a fresh BatchedCSR (new tensors) misses and rebuilds. No
// content hashing of index arrays (kernels.md open Q3's "weak-ref on index
// tensors? / hashing?" question) -- lookup is a bounded linear scan over
// pointer-equality keys (an equality comparison, not a hash function), and
// the cache is capacity-bounded with LRU eviction rather than global/
// unbounded.
//
// Pointer-identity keying is only safe if the pointer can't be silently
// reused by an UNRELATED tensor while a stale cache entry still claims it
// (CUDA's caching allocator reuses freed addresses eagerly — this was
// caught live by this commit's own gate-5 determinism suite: a later test's
// freshly-allocated rowptr/col landed at the same address a former test's
// had just freed, and the cache handed back that former pattern's plan).
// The fix: each cache SLOT holds its own strong reference (a
// torch::stable::Tensor copy, cheap — shared_ptr semantics) to the exact
// rowptr/col it was built from, keeping their storage alive for as long as
// the slot claims to represent them. A slot's storage — and therefore its
// pointers' trustworthiness — is only released on eviction, at which point
// the key stops being consulted at all. This is what makes "the plan
// cache lives on the descriptor" true in practice even though the cache
// itself lives in this C++ TU, not in the Python BatchedCSR object.

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <torch/csrc/stable/tensor.h>

namespace tsgu {

// One analysis result: a level schedule over the EFFECTIVE dependency
// problem (kernels.md Family 3 SpSM row, expanded in this commit's design):
//
//   level[owner] = 1 + max(level[dep] : dep a strict off-diagonal
//                          dependency of owner), else 0 if none.
//
// "owner"/"dep" are FOLDED indices (naming.md §2: row_global = b*n+r) over
// the EFFECTIVE problem being solved:
//   - !transpose: owner = the folded row, dep = its entries' folded column
//     (col[k] is local; folded via the entry's own batch b). eff_ptr/
//     eff_dep/eff_val_idx are then simply the caller's own rowptr/col/
//     identity-permutation, upcast to int64 -- no CSC construction needed.
//   - transpose: A^T's row is A's column, so owner = the folded COLUMN and
//     dep = the folded ROW of each entry -- a CSC-shaped reindexing of the
//     SAME (rowptr, col, vals) triple, built once here and cached. This is
//     a private C++ implementation detail of the plan, never exposed back
//     to Python as a BatchedCSC (the schema-note's "no separately-
//     transposed (CSC) pattern is needed" is about the op's ARGUMENTS).
//
// Both cases are strictly triangular DAGs, so dependencies are monotonic in
// folded-owner-index: for effective_upper = (upper != transpose), off-
// diagonal deps satisfy dep > owner (else dep < owner) -- level[] is
// therefore computable in ONE monotonic host-side pass over owners (highest
// index first when effective_upper, else lowest first), no iterative
// fixed-point/BFS needed. A dep found on the wrong side of that monotonic
// order means the caller's pattern isn't actually triangular as claimed --
// STD_TORCH_CHECK raises rather than silently reading an unwritten level
// (map.md invariant 7).
struct SpsmPlan {
  bool transpose = false;

  // Effective adjacency (module comment above). Always int64, regardless of
  // the caller's rowptr/col index dtype -- the solve kernel (spsm.cu) reads
  // only these, never the original rowptr/col.
  torch::stable::Tensor eff_ptr;       // (B * n + 1,) int64 — owner pointer
  torch::stable::Tensor eff_dep;       // (nse_total,) int64 — dependency's folded index (== owner for the diagonal entry)
  torch::stable::Tensor eff_val_idx;   // (nse_total,) int64 — index into the ORIGINAL vals array
  torch::stable::Tensor diag_val_idx;  // (B * n,) int64 — index into vals of owner's diagonal entry, or -1

  // Level schedule: row_order groups folded owner indices contiguously by
  // level (kernels.md: "rows within a level are independent -> solved in
  // parallel; levels execute in sequence"); level_ptr (host, tiny) slices it.
  torch::stable::Tensor row_order;  // (B * n,) int64
  std::vector<int64_t> level_ptr;   // (n_levels + 1,) host

  int64_t n_levels() const { return static_cast<int64_t>(level_ptr.size()) - 1; }
};

// Returns a plan for (rowptr, col, B, n, upper, unitriangular, transpose),
// building it (STD_TORCH_CHECK-validating the diagonal per map.md
// invariant 7: "raise, never silently accept") on a cache miss, else
// returning the cached one via a shared_ptr (safe against a concurrent
// call evicting the same cache slot -- see module comment: this cache is
// not a hard real-time structure, just bounded and simple). `rowptr`/`col`
// must be CUDA tensors already validated by the caller (spsm.cu's own
// STD_TORCH_CHECKs run first).
std::shared_ptr<SpsmPlan const> get_or_build_plan(torch::stable::Tensor const &rowptr,
                                                   torch::stable::Tensor const &col, int64_t B, int64_t n, bool upper,
                                                   bool unitriangular, bool transpose);

// Test/introspection-only (spec/commit.md Phase 3 commit 16 T5: "plan-cache
// tests: same BatchedCSR solved twice -> analysis computed once (assert via
// the lazy member's identity/a counter)"). Monotonically increasing counts
// of cache misses (an analysis actually ran) and hits, process-wide.
struct SpsmPlanCacheStats {
  int64_t builds;
  int64_t hits;
};
SpsmPlanCacheStats spsm_plan_cache_stats();

}  // namespace tsgu
