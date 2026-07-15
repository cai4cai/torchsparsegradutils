// tsgu::seglse_bidir + tsgu::seglse_bidir_bwd — Family 2 "Segmented
// logsumexp" fused bidirectional forward and backward (kernels.md Family 2
// "Bidirectional"; spec/commit.md Phase 3 commit 13).
//
// A batched matrix's *specified entries* are folded row-major: segment `seg`
// = folded row `b * n + r` (naming.md §2), and `rowptr` gives its members as
// the contiguous range `vals[rowptr[seg] : rowptr[seg + 1]]` — exactly as in
// tsgu::seglse (seglse.cu). What's new here: every entry also carries a
// *local column* `col[i] in [0, m)` (naming.md §2), and this op reduces
// BOTH axes from that single `(rowptr, col, vals)` traversal, instead of the
// two independent seglse calls sparse_bidir_logsumexp would otherwise need
// (kernels.md: "the entire point is a single traversal updating row and
// column accumulators together — one read of values/indices instead of
// two").
//
// Row reduction: identical to seglse's forward — one warp per folded row,
// each lane strides its slice, online log-sum-exp folded via a warp-shuffle
// combine, no atomics, deterministic, written once per warp straight into
// the output. Free to compute mid-traversal because a row's members are
// already contiguous.
//
// Column reduction: a row's entries land in *scattered* columns (no
// contiguity to exploit), so this is exactly the "one-pass online, via
// atomics" candidate kernels.md's own forward section describes for the
// single-axis case — applied here to the column axis only, while the row
// axis takes the atomics-free path in the very same kernel launch. Each
// entry's value is folded into its column's running (max, sum-of-shifted-
// exp) accumulator under a per-column spinlock (there is no atomic
// generalisation of the online-log-sum-exp recurrence itself, only of the
// critical section guarding it). This is the non-deterministic path
// kernels.md's "Determinism policy" flags for atomics-based kernels: run
// order across rows/warps affects float rounding in the column sums (not
// their mathematical value). No deterministic column fallback is
// implemented in this commit (open per kernels.md's resolution: a kernel
// "may" ship an atomic fast path; this one does, documented here rather
// than silently).
//
// include_zeros: the row side folds its structural-zero count (m minus the
// row's specified-entry count) in directly, same as seglse. The column
// side's structural-zero count (n minus that column's specified-entry
// count) isn't known until every row has been visited, so it is folded in
// by a small second kernel after the traversal — still only one pass over
// `vals` (kernels.md's target), the second kernel only ever touches the
// O(B*m) column accumulators, never `vals` again.
//
// Output layout: (2, B, G) with G = max(n, m) — index 0 is the column
// reduction (padded to G with -inf beyond m), index 1 is the row reduction
// (padded to G with -inf beyond n). This is the op's native buffer; the
// tuple/nested layouts sparse_bidir_logsumexp exposes are assembled
// host-side from it (ops/logsumexp.py), never inside this kernel.
//
// Backward: embarrassingly parallel per specified entry, same launch shape
// as the forward (one warp per folded row so `rowptr`/row-lookup is read
// once per row, not once per entry): every entry receives a gradient
// contribution from *both* directions it participates in —
// `gradA_val = exp(v - col_lse[col]) * gout[0, col] + exp(v - row_lse[row])
// * gout[1, row]` — using the `padded` buffer saved from the forward (no
// recompute, kernels.md).
//
// fp16/bf16 storage: not implemented in this commit, same deferral as
// seglse.cu (common/dispatch.cuh's TSGU_DISPATCH_VALUE only switches on
// float32/float64 today).

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "../../common/dispatch.cuh"
#include "../../common/reduce.cuh"
#include "../../common/stream.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;  // 256 threads/block — one warp per row segment
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kFlatThreads = 256;

template <typename scalar_t>
__device__ __forceinline__ tsgu::OnlineLogSumExp<scalar_t> warp_reduce_lse(tsgu::OnlineLogSumExp<scalar_t> acc) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    scalar_t other_max = __shfl_down_sync(0xffffffffU, acc.max_val, offset);
    scalar_t other_sum = __shfl_down_sync(0xffffffffU, acc.sum_exp, offset);
    acc = tsgu::OnlineLogSumExp<scalar_t>::combine(acc, tsgu::OnlineLogSumExp<scalar_t>{other_max, other_sum});
  }
  return acc;
}

// Critical-section merge of `value` into column accumulator `idx`'s running
// online-log-sum-exp pair, guarded by a per-column spinlock (see file
// header: no atomic generalisation of the recurrence itself exists). Also
// bumps that column's specified-entry counter for the deferred
// include_zeros fold.
template <typename scalar_t>
__device__ __forceinline__ void atomic_update_lse(int *__restrict__ lock, scalar_t *__restrict__ max_arr,
                                                    scalar_t *__restrict__ sum_arr, int32_t *__restrict__ count_arr,
                                                    int64_t idx, scalar_t value) {
  while (atomicCAS(&lock[idx], 0, 1) != 0) {
    // spin
  }
  __threadfence();
  scalar_t m = max_arr[idx];
  scalar_t s = sum_arr[idx];
  if (value <= m) {
    s += exp(value - m);
  } else {
    s = s * exp(m - value) + scalar_t(1);
    m = value;
  }
  max_arr[idx] = m;
  sum_arr[idx] = s;
  count_arr[idx] += 1;
  __threadfence();
  atomicExch(&lock[idx], 0);
}

// Sets `padded` (2 * B * G) to -inf and the column scratch accumulators
// (B * m each) to the online-log-sum-exp identity / zero.
template <typename scalar_t>
__global__ void bidir_init_kernel(scalar_t *__restrict__ padded, int64_t padded_n, scalar_t *__restrict__ col_max,
                                   scalar_t *__restrict__ col_sum, int32_t *__restrict__ col_count, int *__restrict__ lock,
                                   int64_t col_n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < padded_n) {
    padded[idx] = -INFINITY;
  }
  if (idx < col_n) {
    col_max[idx] = -INFINITY;
    col_sum[idx] = scalar_t(0);
    col_count[idx] = 0;
    lock[idx] = 0;
  }
}

template <typename scalar_t, typename index_t>
__global__ void bidir_fwd_kernel(scalar_t *__restrict__ padded, scalar_t const *__restrict__ vals,
                                  index_t const *__restrict__ rowptr, index_t const *__restrict__ col,
                                  scalar_t *__restrict__ col_max, scalar_t *__restrict__ col_sum,
                                  int32_t *__restrict__ col_count, int *__restrict__ lock, int64_t total_segs,
                                  int64_t n, int64_t m, int64_t G, int64_t B, bool include_zeros) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t seg = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (seg >= total_segs) {
    return;
  }

  int64_t b = seg / n;
  int64_t r = seg % n;

  index_t start = rowptr[seg];
  index_t end = rowptr[seg + 1];

  auto acc = tsgu::OnlineLogSumExp<scalar_t>::identity();
  for (index_t i = start + static_cast<index_t>(lane); i < end; i += kWarpSize) {
    scalar_t v = vals[i];
    acc.update(v);
    int64_t c = static_cast<int64_t>(col[i]);
    atomic_update_lse<scalar_t>(lock, col_max, col_sum, col_count, b * m + c, v);
  }
  acc = warp_reduce_lse<scalar_t>(acc);

  if (lane == 0) {
    if (include_zeros) {
      scalar_t z = static_cast<scalar_t>(m - static_cast<int64_t>(end - start));
      if (z > scalar_t(0)) {
        acc = tsgu::OnlineLogSumExp<scalar_t>::combine(acc, tsgu::OnlineLogSumExp<scalar_t>{scalar_t(0), z});
      }
    }
    // Row reduction -> padded[1, b, r].
    padded[1 * B * G + b * G + r] = acc.log_sum_exp();
  }
}

// Folds each column's deferred structural-zero count in (include_zeros)
// and writes the finished column reduction into padded[0, b, c].
template <typename scalar_t>
__global__ void bidir_col_finalize_kernel(scalar_t *__restrict__ padded, scalar_t const *__restrict__ col_max,
                                           scalar_t const *__restrict__ col_sum, int32_t const *__restrict__ col_count,
                                           int64_t B, int64_t n, int64_t m, int64_t G, bool include_zeros) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= B * m) {
    return;
  }
  int64_t b = idx / m;
  int64_t c = idx % m;

  auto acc = tsgu::OnlineLogSumExp<scalar_t>{col_max[idx], col_sum[idx]};
  if (include_zeros) {
    scalar_t z = static_cast<scalar_t>(n - static_cast<int64_t>(col_count[idx]));
    if (z > scalar_t(0)) {
      acc = tsgu::OnlineLogSumExp<scalar_t>::combine(acc, tsgu::OnlineLogSumExp<scalar_t>{scalar_t(0), z});
    }
  }
  padded[0 * B * G + b * G + c] = acc.log_sum_exp();
}

template <typename scalar_t, typename index_t>
__global__ void bidir_bwd_kernel(scalar_t *__restrict__ grad_vals, scalar_t const *__restrict__ vals,
                                  index_t const *__restrict__ rowptr, index_t const *__restrict__ col,
                                  scalar_t const *__restrict__ padded, scalar_t const *__restrict__ gout,
                                  int64_t total_segs, int64_t n, int64_t m, int64_t G, int64_t B) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t seg = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (seg >= total_segs) {
    return;
  }

  int64_t b = seg / n;
  int64_t r = seg % n;

  index_t start = rowptr[seg];
  index_t end = rowptr[seg + 1];

  scalar_t row_lse = padded[1 * B * G + b * G + r];
  scalar_t row_gout = gout[1 * B * G + b * G + r];

  for (index_t i = start + static_cast<index_t>(lane); i < end; i += kWarpSize) {
    int64_t c = static_cast<int64_t>(col[i]);
    scalar_t col_lse = padded[0 * B * G + b * G + c];
    scalar_t col_gout = gout[0 * B * G + b * G + c];
    scalar_t v = vals[i];
    // kernels.md Family 2 backward, extended to both directions this op
    // fuses: every entry gets a contribution from the column reduction it
    // feeds AND the row reduction it feeds.
    grad_vals[i] = exp(v - col_lse) * col_gout + exp(v - row_lse) * row_gout;
  }
}

}  // namespace

torch::stable::Tensor tsgu_seglse_bidir_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                                torch::stable::Tensor const &col, int64_t B, int64_t n, int64_t m,
                                                bool include_zeros) {
  STD_TORCH_CHECK(vals.is_cuda() && rowptr.is_cuda() && col.is_cuda(), "tsgu::seglse_bidir expects CUDA tensors");
  STD_TORCH_CHECK(vals.dim() == 1, "tsgu::seglse_bidir expects vals of shape (nse_total,)");
  STD_TORCH_CHECK(rowptr.dim() == 1 && rowptr.size(0) == B * n + 1,
                   "tsgu::seglse_bidir expects rowptr of shape (B * n + 1,)");
  STD_TORCH_CHECK(col.dim() == 1 && col.size(0) == vals.size(0),
                   "tsgu::seglse_bidir expects col of shape (nse_total,) aligned with vals");

  int64_t G = std::max(n, m);
  torch::stable::Tensor padded = torch::stable::new_empty(vals, {2, B, G});
  tsgu::StreamGuard guard(vals);

  int64_t total_segs = B * n;
  int64_t padded_n = 2 * B * G;
  int64_t col_n = B * m;

  if (padded_n == 0) {
    return padded;
  }

  // Scratch column accumulators — allocated as vals-typed / int32 tensors so
  // dtype/device follow `vals` exactly (same pattern as seglse's `lse`).
  // col_count / lock are always int32 regardless of rowptr's index dtype —
  // the kernel reads them through a plain int32_t*/int* pointer, so an
  // int64 rowptr must not silently size these buffers as int64 (new_empty's
  // dtype override, not a same-dtype-as-rowptr copy).
  torch::stable::Tensor col_max = torch::stable::new_empty(vals, {std::max<int64_t>(col_n, 1)});
  torch::stable::Tensor col_sum = torch::stable::new_empty(vals, {std::max<int64_t>(col_n, 1)});
  torch::stable::Tensor col_count =
      torch::stable::new_empty(rowptr, {std::max<int64_t>(col_n, 1)}, torch::headeronly::ScalarType::Int);
  torch::stable::Tensor lock =
      torch::stable::new_empty(rowptr, {std::max<int64_t>(col_n, 1)}, torch::headeronly::ScalarType::Int);

  TSGU_DISPATCH_VALUE(vals.scalar_type(), "tsgu::seglse_bidir", [&] {
    int64_t init_blocks = (std::max(padded_n, col_n) + kFlatThreads - 1) / kFlatThreads;
    bidir_init_kernel<scalar_t><<<init_blocks, kFlatThreads, 0, guard.stream()>>>(
        static_cast<scalar_t *>(padded.mutable_data_ptr()), padded_n, static_cast<scalar_t *>(col_max.mutable_data_ptr()),
        static_cast<scalar_t *>(col_sum.mutable_data_ptr()), static_cast<int32_t *>(col_count.mutable_data_ptr()),
        static_cast<int *>(lock.mutable_data_ptr()), col_n);

    if (total_segs > 0) {
      // Launched even when vals.numel() == 0: rows still need their
      // include_zeros fold (bidir_fwd_kernel's start==end path handles an
      // empty per-row range the same way seglse_fwd_kernel does).
      TSGU_DISPATCH_INDEX(rowptr.scalar_type(), "tsgu::seglse_bidir", [&] {
        int64_t blocks = (total_segs + kWarpsPerBlock - 1) / kWarpsPerBlock;
        bidir_fwd_kernel<scalar_t, index_t><<<blocks, kThreadsPerBlock, 0, guard.stream()>>>(
            static_cast<scalar_t *>(padded.mutable_data_ptr()), static_cast<scalar_t const *>(vals.data_ptr()),
            static_cast<index_t const *>(rowptr.data_ptr()), static_cast<index_t const *>(col.data_ptr()),
            static_cast<scalar_t *>(col_max.mutable_data_ptr()), static_cast<scalar_t *>(col_sum.mutable_data_ptr()),
            static_cast<int32_t *>(col_count.mutable_data_ptr()), static_cast<int *>(lock.mutable_data_ptr()),
            total_segs, n, m, G, B, include_zeros);
      });
    }

    if (col_n > 0) {
      int64_t fin_blocks = (col_n + kFlatThreads - 1) / kFlatThreads;
      bidir_col_finalize_kernel<scalar_t><<<fin_blocks, kFlatThreads, 0, guard.stream()>>>(
          static_cast<scalar_t *>(padded.mutable_data_ptr()), static_cast<scalar_t const *>(col_max.data_ptr()),
          static_cast<scalar_t const *>(col_sum.data_ptr()), static_cast<int32_t const *>(col_count.data_ptr()), B, n,
          m, G, include_zeros);
    }
  });

  return padded;
}

torch::stable::Tensor tsgu_seglse_bidir_bwd_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                                    torch::stable::Tensor const &col, torch::stable::Tensor const &padded,
                                                    torch::stable::Tensor const &gout, int64_t B, int64_t n, int64_t m) {
  STD_TORCH_CHECK(vals.is_cuda() && rowptr.is_cuda() && col.is_cuda() && padded.is_cuda() && gout.is_cuda(),
                   "tsgu::seglse_bidir_bwd expects CUDA tensors");
  STD_TORCH_CHECK(rowptr.size(0) == B * n + 1, "tsgu::seglse_bidir_bwd expects rowptr of shape (B * n + 1,)");
  // Same landmine as tsgu::seglse_bwd (seglse.cu): gout may arrive as a
  // broadcast/expanded (stride-0) view from a reduction's backward and is
  // read through a plain contiguous pointer below — the Python autograd
  // wrapper (ops/logsumexp.py _seglse_bidir_backward) must .contiguous() it
  // first; this check turns a violation into a clear error, not a silent
  // OOB read.
  STD_TORCH_CHECK(gout.is_contiguous(), "tsgu::seglse_bidir_bwd expects a contiguous gout");
  STD_TORCH_CHECK(padded.is_contiguous(), "tsgu::seglse_bidir_bwd expects a contiguous padded");

  torch::stable::Tensor grad_vals = torch::stable::empty_like(vals);
  tsgu::StreamGuard guard(vals);

  int64_t total_segs = B * n;
  int64_t G = std::max(n, m);
  if (total_segs == 0 || vals.numel() == 0) {
    return grad_vals;
  }

  int64_t blocks = (total_segs + kWarpsPerBlock - 1) / kWarpsPerBlock;

  TSGU_DISPATCH_VALUE(vals.scalar_type(), "tsgu::seglse_bidir_bwd", [&] {
    TSGU_DISPATCH_INDEX(rowptr.scalar_type(), "tsgu::seglse_bidir_bwd", [&] {
      bidir_bwd_kernel<scalar_t, index_t><<<blocks, kThreadsPerBlock, 0, guard.stream()>>>(
          static_cast<scalar_t *>(grad_vals.mutable_data_ptr()), static_cast<scalar_t const *>(vals.data_ptr()),
          static_cast<index_t const *>(rowptr.data_ptr()), static_cast<index_t const *>(col.data_ptr()),
          static_cast<scalar_t const *>(padded.data_ptr()), static_cast<scalar_t const *>(gout.data_ptr()),
          total_segs, n, m, G, B);
    });
  });

  return grad_vals;
}
