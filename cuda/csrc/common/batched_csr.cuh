#pragma once
// Device-side BatchedCSR descriptor accessor (architecture.md §3, §5;
// naming.md §2 short names). Mirrors torchsparsegradutils/_batched.py's
// BatchedCSR field semantics — vals/rowptr/col, folded rows, local columns —
// as a plain pointer-based, pass-by-value view for kernels. Header-only;
// parsed (and instantiated once, on scratch values) by the smoke TU
// (cuda/csrc/kernels/_smoke/_smoke.cu) until kernel commits (Phase 3)
// construct one from real op arguments.

#include <cuda_runtime.h>

namespace tsgu {

// A non-owning view of one BatchedCSR descriptor's storage
// (torchsparsegradutils/_batched.py), templated over the value/index dtypes
// TSGU_DISPATCH_VALUE/TSGU_DISPATCH_INDEX (dispatch.cuh) resolve to.
template <typename scalar_t, typename index_t>
struct BatchedCSRView {
  scalar_t const *vals;   // (nse_total,) — stored values
  index_t const *rowptr;  // (B * n + 1,) — absolute over folded rows
  index_t const *col;     // (nse_total,) — local columns in [0, m)
  int64_t B, n, m;        // batch_size, n_rows, n_cols (naming.md §2)

  // Batch of an entry, recovered from its folded row (architecture.md §3:
  // "batch of an entry recovered as row_global / n").
  __device__ __forceinline__ int64_t batch_of(index_t row_g) const { return row_g / n; }

  // Local (unfolded) row within its batch item.
  __device__ __forceinline__ int64_t local_row_of(index_t row_g) const { return row_g % n; }

  // Number of specified entries in folded row `row_g` (that row's
  // nse_per_item slice, naming.md §2).
  __device__ __forceinline__ index_t row_nse(index_t row_g) const { return rowptr[row_g + 1] - rowptr[row_g]; }
};

}  // namespace tsgu
