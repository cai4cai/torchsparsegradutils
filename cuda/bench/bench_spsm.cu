// NVBench target for tsgu::spsm (spec/commit.md Phase 3 commit 16, T3;
// kernels.md Family 3 SpSM row). Axes are the SOLVE kernel's tuning knobs
// (cuda/csrc/kernels/spsm/spsm.cu's spsm_level_kernel): matrix size `n`,
// the dense rhs width `p`, and `n_levels` — the level-schedule DEPTH, this
// kernel's dominant cost axis (module comment in spsm.cu: level count/
// launch latency, not per-row FMA throughput, is what drives SpSM time; a
// deep chain means many small launches, a shallow/wide schedule means few
// large ones).
//
// This target benches the SOLVE kernel directly, with a level schedule
// already known (built host-side, by construction, from a synthetic
// lower-triangular BANDED + random-fill generator — see make_synthetic
// below) — i.e. this IS the "warm" path (plan already built; only the
// per-level kernel launches are timed). It deliberately does NOT reimplement
// plan.cpp's general analysis algorithm (that would just be timing this
// binary's own analysis code, not the kernel) — cold-vs-warm, and the
// analysis cost itself, is what benchmarks/bench_spsm.py measures at the
// op level (benchmarks.md §3: "SpSM... incl. its analysis cost
// amortisation... win warm (plan cached on descriptor)").
//
// Standalone CMake + FetchContent, same pattern as bench_spmm.cu
// (deliberately independent of the kernel-builder Nix build).

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <random>
#include <vector>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

// Mirrors csrc/kernels/spsm/spsm.cu's spsm_level_kernel exactly (float
// values, int64_t plan indices — the plan's arrays are always int64
// regardless of the caller's rowptr/col dtype, per plan.h).
__global__ void spsm_level_kernel(float *__restrict__ x, int64_t const *__restrict__ eff_ptr,
                                   int64_t const *__restrict__ eff_dep, int64_t const *__restrict__ eff_val_idx,
                                   int64_t const *__restrict__ diag_val_idx, int64_t const *__restrict__ row_order,
                                   int64_t level_start, int64_t level_size, float const *__restrict__ vals,
                                   float const *__restrict__ rhs, int64_t p, bool unitriangular) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t idx = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (idx >= level_size) return;
  int64_t owner = row_order[level_start + idx];

  int64_t start = eff_ptr[owner];
  int64_t end = eff_ptr[owner + 1];

  float diag = 1.0f;
  if (!unitriangular) diag = vals[diag_val_idx[owner]];

  float *x_row = x + owner * p;
  float const *rhs_row = rhs + owner * p;

  for (int64_t j = lane; j < p; j += kWarpSize) {
    float acc = 0.0f;
    for (int64_t k = start; k < end; ++k) {
      int64_t dep = eff_dep[k];
      if (dep == owner) continue;
      acc += vals[eff_val_idx[k]] * x[dep * p + j];
    }
    float result = rhs_row[j] - acc;
    if (!unitriangular) result = result / diag;
    x_row[j] = result;
  }
}

// Synthetic lower-triangular, BANDED + random-fill generator (this
// commit's brief: "a synthetic triangular generator — lower-triangular
// banded + random fill is fine"), built so the level schedule is known BY
// CONSTRUCTION (no analysis code duplicated in this bench binary): rows are
// partitioned into `n_levels` contiguous blocks of ~`n / n_levels` rows;
// row r in block L gets a diagonal entry plus up to `fill` random distinct
// columns drawn from strictly earlier blocks (columns < block * L) — every
// off-diagonal dependency is therefore in a strictly earlier block, so
// row_order = identity and level_ptr = the block boundaries, exactly.
struct Synthetic {
  thrust::host_vector<int64_t> eff_ptr, eff_dep, eff_val_idx, diag_val_idx, row_order;
  thrust::host_vector<float> vals;
  std::vector<int64_t> level_ptr;
};

Synthetic make_synthetic(int64_t n, int64_t n_levels, int64_t fill) {
  Synthetic s;
  int64_t block = std::max<int64_t>(1, n / std::max<int64_t>(1, n_levels));

  std::mt19937 rng(0);
  std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

  s.eff_ptr.resize(n + 1);
  s.diag_val_idx.assign(n, -1);
  s.row_order.resize(n);
  std::vector<int64_t> level_of(n);
  int64_t max_level = 0;
  for (int64_t r = 0; r < n; ++r) {
    level_of[r] = std::min(r / block, std::max<int64_t>(0, n_levels - 1));
    max_level = std::max(max_level, level_of[r]);
    s.row_order[r] = r;  // identity — rows already grouped by block/level.
  }
  s.level_ptr.assign(static_cast<size_t>(max_level) + 2, 0);
  for (int64_t r = 0; r < n; ++r) s.level_ptr[static_cast<size_t>(level_of[r]) + 1] += 1;
  for (size_t l = 0; l < s.level_ptr.size() - 1; ++l) s.level_ptr[l + 1] += s.level_ptr[l];

  std::vector<std::vector<int64_t>> row_cols(n);
  for (int64_t r = 0; r < n; ++r) {
    int64_t earlier_bound = std::min(r, level_of[r] * block);  // columns < this are strictly earlier-level
    int64_t k = std::min(fill, earlier_bound);
    if (k > 0) {
      std::uniform_int_distribution<int64_t> col_dist(0, earlier_bound - 1);
      std::vector<int64_t> chosen;
      chosen.reserve(static_cast<size_t>(k));
      for (int64_t i = 0; i < k; ++i) chosen.push_back(col_dist(rng));
      std::sort(chosen.begin(), chosen.end());
      chosen.erase(std::unique(chosen.begin(), chosen.end()), chosen.end());
      row_cols[r] = chosen;
    }
    row_cols[r].push_back(r);  // diagonal, always last (r is the max column for this row).
  }

  int64_t nse = 0;
  for (int64_t r = 0; r < n; ++r) nse += static_cast<int64_t>(row_cols[r].size());
  s.eff_dep.resize(nse);
  s.eff_val_idx.resize(nse);
  s.vals.resize(nse);
  int64_t offset = 0;
  for (int64_t r = 0; r < n; ++r) {
    s.eff_ptr[r] = offset;
    for (int64_t c : row_cols[r]) {
      s.eff_dep[offset] = c;
      s.eff_val_idx[offset] = offset;  // identity — vals indexed directly, no separate original array here.
      s.vals[offset] = val_dist(rng);
      if (c == r) s.diag_val_idx[r] = offset;
      ++offset;
    }
  }
  s.eff_ptr[n] = offset;
  return s;
}

void bench_spsm_solve(nvbench::state &state) {
  int64_t const n = state.get_int64("n");
  int64_t const p = state.get_int64("p");
  int64_t const n_levels = state.get_int64("n_levels");
  int64_t const fill = 4;  // fixed off-diagonal fill per row (DLMC-shaped low-degree triangular rows)

  Synthetic s = make_synthetic(n, n_levels, fill);

  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  thrust::host_vector<float> rhs_h(n * p);
  for (auto &v : rhs_h) v = dist(rng);

  thrust::device_vector<int64_t> eff_ptr = s.eff_ptr;
  thrust::device_vector<int64_t> eff_dep = s.eff_dep;
  thrust::device_vector<int64_t> eff_val_idx = s.eff_val_idx;
  thrust::device_vector<int64_t> diag_val_idx = s.diag_val_idx;
  thrust::device_vector<int64_t> row_order = s.row_order;
  thrust::device_vector<float> vals = s.vals;
  thrust::device_vector<float> rhs = rhs_h;
  thrust::device_vector<float> x(n * p);

  state.exec([&](nvbench::launch &launch) {
    for (size_t l = 0; l + 1 < s.level_ptr.size(); ++l) {
      int64_t level_start = s.level_ptr[l];
      int64_t level_size = s.level_ptr[l + 1] - level_start;
      if (level_size == 0) continue;
      int64_t blocks = (level_size + kWarpsPerBlock - 1) / kWarpsPerBlock;
      spsm_level_kernel<<<blocks, kThreadsPerBlock, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(x.data()), thrust::raw_pointer_cast(eff_ptr.data()),
          thrust::raw_pointer_cast(eff_dep.data()), thrust::raw_pointer_cast(eff_val_idx.data()),
          thrust::raw_pointer_cast(diag_val_idx.data()), thrust::raw_pointer_cast(row_order.data()), level_start,
          level_size, thrust::raw_pointer_cast(vals.data()), thrust::raw_pointer_cast(rhs.data()), p,
          /*unitriangular=*/false);
    }
  });
}
}  // namespace

NVBENCH_BENCH(bench_spsm_solve)
    .add_int64_axis("n", {1 << 12, 1 << 16, 1 << 18})
    .add_int64_axis("p", {1, 8, 32, 128})
    .add_int64_axis("n_levels", {8, 64, 1024});
