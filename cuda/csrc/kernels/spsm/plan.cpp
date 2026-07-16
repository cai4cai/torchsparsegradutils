// tsgu::spsm analysis-plan cache — implementation (plan.h's module comment
// carries the full design rationale; this file is "just" the host-side
// level-set analysis (kernels.md Family 3 SpSM row) plus the bounded LRU
// cache architecture.md §3 calls for).
//
// Analysis is explicitly NOT the hot path (kernels.md: "host-side pure-
// torch analysis is acceptable for v1... it is NOT the hot path; it's
// amortised by the cache") -- this file does plain host (CPU) loops over
// std::vector, with synchronous cudaMemcpy D2H/H2D at the boundary. The
// solve kernel (spsm.cu) never runs any of this; it only reads the plan's
// already-built device tensors.

#include "plan.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "../../common/dispatch.cuh"

namespace tsgu {

namespace {

// --- host-side copy helpers -------------------------------------------------

// Reads a (possibly int32 or int64) CUDA index tensor of length `count` into
// a freshly-allocated host int64 vector.
std::vector<int64_t> to_host_int64(torch::stable::Tensor const &t, int64_t count) {
  std::vector<int64_t> host(static_cast<size_t>(count));
  if (count == 0) {
    return host;
  }
  TSGU_DISPATCH_INDEX(t.scalar_type(), "tsgu::spsm plan analysis", [&] {
    std::vector<index_t> tmp(static_cast<size_t>(count));
    cudaMemcpy(tmp.data(), t.data_ptr(), static_cast<size_t>(count) * sizeof(index_t), cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < count; ++i) {
      host[static_cast<size_t>(i)] = static_cast<int64_t>(tmp[static_cast<size_t>(i)]);
    }
  });
  return host;
}

// Allocates a fresh int64 CUDA tensor (device/inherited from `reference`)
// of the given length and fills it from a host int64 vector.
torch::stable::Tensor to_device_int64(torch::stable::Tensor const &reference, std::vector<int64_t> const &host) {
  torch::stable::Tensor out =
      torch::stable::new_empty(reference, {static_cast<int64_t>(host.size())}, torch::headeronly::ScalarType::Long);
  if (!host.empty()) {
    cudaMemcpy(out.mutable_data_ptr(), host.data(), host.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  }
  return out;
}

// --- analysis (kernels.md Family 3 SpSM row, expanded design) --------------

SpsmPlan build_plan(torch::stable::Tensor const &rowptr, torch::stable::Tensor const &col, int64_t B, int64_t n,
                     bool upper, bool unitriangular, bool transpose) {
  int64_t const total_rows = B * n;
  int64_t const nse_total = col.size(0);

  std::vector<int64_t> rowptr_h = to_host_int64(rowptr, total_rows + 1);
  std::vector<int64_t> col_h = to_host_int64(col, nse_total);

  // Uncompress rowptr -> row(k) (naming.md §1: "expanded/repeat-
  // interleaved", never "decompressed").
  std::vector<int64_t> row_of_entry(static_cast<size_t>(nse_total));
  for (int64_t r = 0; r < total_rows; ++r) {
    for (int64_t k = rowptr_h[static_cast<size_t>(r)]; k < rowptr_h[static_cast<size_t>(r) + 1]; ++k) {
      row_of_entry[static_cast<size_t>(k)] = r;
    }
  }

  // Effective adjacency (plan.h module comment): owner/dep both live in the
  // shared folded [0, B*n) index space.
  std::vector<int64_t> eff_ptr_h;
  std::vector<int64_t> eff_dep_h(static_cast<size_t>(nse_total));
  std::vector<int64_t> eff_val_idx_h(static_cast<size_t>(nse_total));

  if (!transpose) {
    eff_ptr_h = rowptr_h;
    for (int64_t k = 0; k < nse_total; ++k) {
      int64_t r = row_of_entry[static_cast<size_t>(k)];
      int64_t batch = r / n;
      eff_dep_h[static_cast<size_t>(k)] = batch * n + col_h[static_cast<size_t>(k)];
      eff_val_idx_h[static_cast<size_t>(k)] = k;
    }
  } else {
    // CSC-shaped reindexing of (rowptr, col) — a counting sort bucketing
    // entries by their folded "owner" (= fold(batch_of(row(k)), col[k])),
    // the standard CSR<->CSC bucket-construction technique (same shape as
    // torchsparsegradutils/_batched.py's _fold_coo_to_csr, done here in
    // plain C++ since this is a private plan-cache detail, never a
    // Python-visible BatchedCSC).
    std::vector<int64_t> owner_of_entry(static_cast<size_t>(nse_total));
    for (int64_t k = 0; k < nse_total; ++k) {
      int64_t r = row_of_entry[static_cast<size_t>(k)];
      int64_t batch = r / n;
      owner_of_entry[static_cast<size_t>(k)] = batch * n + col_h[static_cast<size_t>(k)];
    }
    eff_ptr_h.assign(static_cast<size_t>(total_rows) + 1, 0);
    for (int64_t k = 0; k < nse_total; ++k) {
      eff_ptr_h[static_cast<size_t>(owner_of_entry[static_cast<size_t>(k)]) + 1] += 1;
    }
    for (int64_t r = 0; r < total_rows; ++r) {
      eff_ptr_h[static_cast<size_t>(r) + 1] += eff_ptr_h[static_cast<size_t>(r)];
    }
    std::vector<int64_t> cursor = eff_ptr_h;
    for (int64_t k = 0; k < nse_total; ++k) {
      int64_t o = owner_of_entry[static_cast<size_t>(k)];
      int64_t pos = cursor[static_cast<size_t>(o)]++;
      eff_dep_h[static_cast<size_t>(pos)] = row_of_entry[static_cast<size_t>(k)];
      eff_val_idx_h[static_cast<size_t>(pos)] = k;
    }
  }

  // Diagonal lookup + existence contract (map.md invariant 7: "raise, never
  // silently accept"; this commit's brief: "for non-unit diagonal the
  // diagonal entry must exist in the pattern — if missing, that's singular
  // input: raise... analysis-time check host-side is fine and best").
  std::vector<int64_t> diag_val_idx_h(static_cast<size_t>(total_rows), -1);
  bool any_diag_present = false;
  bool any_diag_missing = false;
  for (int64_t owner = 0; owner < total_rows; ++owner) {
    for (int64_t k = eff_ptr_h[static_cast<size_t>(owner)]; k < eff_ptr_h[static_cast<size_t>(owner) + 1]; ++k) {
      if (eff_dep_h[static_cast<size_t>(k)] == owner) {
        diag_val_idx_h[static_cast<size_t>(owner)] = eff_val_idx_h[static_cast<size_t>(k)];
      }
    }
    if (diag_val_idx_h[static_cast<size_t>(owner)] != -1) {
      any_diag_present = true;
    } else {
      any_diag_missing = true;
    }
  }
  STD_TORCH_CHECK(!(unitriangular && any_diag_present),
                  "tsgu::spsm expects a strictly triangular pattern when unitriangular=True (unit diagonal is "
                  "implicit — the stored matrix must have no explicit diagonal entries); got at least one row with "
                  "an explicit diagonal entry.");
  STD_TORCH_CHECK(!(!unitriangular && any_diag_missing),
                  "tsgu::spsm expects an explicit diagonal entry in every row when unitriangular=False; got at "
                  "least one row missing its diagonal entry (singular/incomplete triangular input).");

  // Level computation (plan.h module comment): strictly monotonic in folded
  // owner index for a genuinely triangular pattern -- one pass suffices,
  // and a dependency found on the wrong side of that monotonic order means
  // the input was not actually triangular as claimed (raise, don't
  // silently read an unwritten level).
  bool const effective_upper = (upper != transpose);
  std::vector<int64_t> level(static_cast<size_t>(total_rows), 0);
  auto process_owner = [&](int64_t owner) {
    int64_t lvl = 0;
    for (int64_t k = eff_ptr_h[static_cast<size_t>(owner)]; k < eff_ptr_h[static_cast<size_t>(owner) + 1]; ++k) {
      int64_t dep = eff_dep_h[static_cast<size_t>(k)];
      if (dep == owner) {
        continue;  // diagonal, not a dependency
      }
      bool ordered_correctly = effective_upper ? (dep > owner) : (dep < owner);
      STD_TORCH_CHECK(ordered_correctly,
                      "tsgu::spsm expects A to be genuinely ", (upper ? "upper" : "lower"),
                      "-triangular (the `upper` flag matches the stored pattern) — got an off-diagonal entry on "
                      "the wrong side of the diagonal for owner index ", owner, ", dependency index ", dep, ".");
      lvl = std::max(lvl, level[static_cast<size_t>(dep)] + 1);
    }
    level[static_cast<size_t>(owner)] = lvl;
  };
  if (effective_upper) {
    for (int64_t owner = total_rows - 1; owner >= 0; --owner) {
      process_owner(owner);
    }
  } else {
    for (int64_t owner = 0; owner < total_rows; ++owner) {
      process_owner(owner);
    }
  }

  // Group folded owner indices by level (counting sort by level value) —
  // kernels.md: "rows within a level are independent -> solved in
  // parallel; levels execute in sequence".
  int64_t n_levels = 0;
  for (int64_t owner = 0; owner < total_rows; ++owner) {
    n_levels = std::max(n_levels, level[static_cast<size_t>(owner)] + 1);
  }
  std::vector<int64_t> level_ptr(static_cast<size_t>(n_levels) + 1, 0);
  for (int64_t owner = 0; owner < total_rows; ++owner) {
    level_ptr[static_cast<size_t>(level[static_cast<size_t>(owner)]) + 1] += 1;
  }
  for (int64_t l = 0; l < n_levels; ++l) {
    level_ptr[static_cast<size_t>(l) + 1] += level_ptr[static_cast<size_t>(l)];
  }
  std::vector<int64_t> row_order_h(static_cast<size_t>(total_rows));
  std::vector<int64_t> level_cursor = level_ptr;
  for (int64_t owner = 0; owner < total_rows; ++owner) {
    int64_t l = level[static_cast<size_t>(owner)];
    int64_t pos = level_cursor[static_cast<size_t>(l)]++;
    row_order_h[static_cast<size_t>(pos)] = owner;
  }

  SpsmPlan plan;
  plan.transpose = transpose;
  plan.eff_ptr = to_device_int64(rowptr, eff_ptr_h);
  plan.eff_dep = to_device_int64(rowptr, eff_dep_h);
  plan.eff_val_idx = to_device_int64(rowptr, eff_val_idx_h);
  plan.diag_val_idx = to_device_int64(rowptr, diag_val_idx_h);
  plan.row_order = to_device_int64(rowptr, row_order_h);
  plan.level_ptr = std::move(level_ptr);
  return plan;
}

// --- bounded LRU cache (plan.h module comment: pointer-identity key, no
// content hashing, capacity-bounded rather than unbounded/global) ----------

struct PlanKey {
  void const *rowptr_ptr;
  void const *col_ptr;
  int64_t B;
  int64_t n;
  bool upper;
  bool unitriangular;
  bool transpose;

  bool operator==(PlanKey const &o) const {
    return rowptr_ptr == o.rowptr_ptr && col_ptr == o.col_ptr && B == o.B && n == o.n && upper == o.upper &&
           unitriangular == o.unitriangular && transpose == o.transpose;
  }
};

struct CacheEntry {
  PlanKey key{};
  // Strong references to the SAME rowptr/col tensors the key's pointers were
  // read from -- this is what makes pointer-identity keying actually safe:
  // without it, a call's rowptr/col Tensor could be destroyed and its CUDA
  // allocation reused by an UNRELATED tensor of the same size (the caching
  // allocator does this eagerly), and a later call with that unrelated
  // tensor would collide on the stale key and silently reuse the WRONG
  // plan. Holding these keeps the underlying storage alive for exactly as
  // long as this cache slot claims to represent it (released on eviction,
  // at which point the key's pointers stop being trustworthy and a
  // subsequent collision is a correct cache MISS instead of a silent hit,
  // because the slot's `valid` flag / key no longer matches).
  torch::stable::Tensor rowptr_keepalive;
  torch::stable::Tensor col_keepalive;
  std::shared_ptr<SpsmPlan const> plan;
  int64_t last_used = 0;
  bool valid = false;
};

constexpr size_t kPlanCacheCapacity = 16;

std::mutex &plan_cache_mutex() {
  static std::mutex m;
  return m;
}
std::vector<CacheEntry> &plan_cache() {
  static std::vector<CacheEntry> cache(kPlanCacheCapacity);
  return cache;
}
int64_t &plan_cache_tick() {
  static int64_t tick = 0;
  return tick;
}
int64_t &plan_cache_builds() {
  static int64_t builds = 0;
  return builds;
}
int64_t &plan_cache_hits() {
  static int64_t hits = 0;
  return hits;
}

}  // namespace

std::shared_ptr<SpsmPlan const> get_or_build_plan(torch::stable::Tensor const &rowptr, torch::stable::Tensor const &col,
                                                   int64_t B, int64_t n, bool upper, bool unitriangular,
                                                   bool transpose) {
  PlanKey key{rowptr.data_ptr(), col.data_ptr(), B, n, upper, unitriangular, transpose};

  std::lock_guard<std::mutex> lock(plan_cache_mutex());
  auto &cache = plan_cache();
  for (auto &entry : cache) {
    if (entry.valid && entry.key == key) {
      entry.last_used = ++plan_cache_tick();
      ++plan_cache_hits();
      return entry.plan;
    }
  }

  // Miss: build (outside no lock-release needed -- analysis is host-side
  // and short enough relative to a kernel launch that holding the mutex
  // for its duration is the simplest correct thing to do; it is never the
  // hot path, per kernels.md).
  auto built = std::make_shared<SpsmPlan>(build_plan(rowptr, col, B, n, upper, unitriangular, transpose));
  ++plan_cache_builds();

  size_t victim = 0;
  int64_t min_used = std::numeric_limits<int64_t>::max();
  bool victim_is_empty_slot = false;
  for (size_t i = 0; i < cache.size() && !victim_is_empty_slot; ++i) {
    if (!cache[i].valid) {
      victim = i;
      victim_is_empty_slot = true;
    } else if (cache[i].last_used < min_used) {
      min_used = cache[i].last_used;
      victim = i;
    }
  }
  cache[victim] = CacheEntry{key, rowptr, col, built, ++plan_cache_tick(), true};
  return built;
}

SpsmPlanCacheStats spsm_plan_cache_stats() {
  std::lock_guard<std::mutex> lock(plan_cache_mutex());
  return SpsmPlanCacheStats{plan_cache_builds(), plan_cache_hits()};
}

}  // namespace tsgu
