// tsgu::seglse + tsgu::seglse_bwd — Family 2 "Segmented logsumexp" forward
// and backward (kernels.md Family 2; spec/commit.md Phase 3 commit 12).
//
// Segments are a batched matrix's *specified entries*, folded row-major:
// segment `seg` = folded row `b * n + r` (naming.md §2), and its members are
// exactly `vals[rowptr[seg] : rowptr[seg + 1]]` — CSR/CSC give this for free
// (kernels.md: "CSR rows are sorted for free, COO dim=0 / CSC give the other
// axis"), so the wrapper (torchsparsegradutils/ops/logsumexp.py) is
// responsible for handing this op A's own BatchedCSR (segments = rows) or
// its BatchedCSC (segments = columns) — this op only ever sees a generic
// folded-row/segment structure.
//
// Forward strategy (kernels.md's two candidates: two-pass segmented vs
// one-pass online-atomic): this kernel takes a *third*, simpler-and-safe
// option that kernels.md's own "CSR rows are sorted for free" observation
// makes available — segments are already contiguous ranges, so one warp
// owns one segment exclusively; no atomics anywhere, no second pass over
// global memory. Each lane strides through its slice of the segment folding
// values into a running (max, sum-of-shifted-exp) pair via the online
// log-sum-exp recurrence (common/reduce.cuh's OnlineLogSumExp — the same
// recurrence the "one-pass online" candidate uses, but applied per-lane
// instead of via atomics), then a warp shuffle-reduction (reduce.cuh's
// combine()) merges the 32 partial accumulators into the segment's answer.
// Deterministic (kernels.md open Q1: no atomic fast path here, so no
// separate deterministic path is needed either) and single-pass over `vals`
// (kernels.md Family 2 backward note: "no recompute" — this op *is* the
// only read of vals the forward needs).
//
// Backward: embarrassingly parallel per specified entry (kernels.md):
// `gradA_val = exp(v - lse[seg]) * gout[seg]`, using the `lse` saved from
// the forward (no recompute) — reuses the same one-warp-per-segment
// launch shape so the segment-of-an-entry lookup (`rowptr`) is read once
// per segment rather than once per entry.
//
// fp16/bf16 storage (kernels.md "seglse family: accepts fp16/bf16 storage,
// always reduces/accumulates in fp32"): NOT implemented in this commit —
// common/dispatch.cuh's TSGU_DISPATCH_VALUE macro (Phase 2 infra, commit 10)
// only switches on float32/float64 today; extending it to half/bfloat16
// storage-with-fp32-compute is left to a follow-up so as not to silently
// widen commit 10's shared dispatch infra inside this kernel commit. Flagged
// in this commit's message, not invented here.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "../../common/dispatch.cuh"
#include "../../common/reduce.cuh"
#include "../../common/stream.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;  // 256 threads/block — one warp per segment
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

// Warp-shuffle reduction of an OnlineLogSumExp accumulator held one-per-lane
// down to lane 0 (reduce.cuh provides the scalar warp_reduce_* helpers and
// the combine() recurrence; this loop is their pairwise-combine analogue).
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

template <typename scalar_t, typename index_t>
__global__ void seglse_fwd_kernel(scalar_t *__restrict__ lse, scalar_t const *__restrict__ vals,
                                   index_t const *__restrict__ rowptr, int64_t total_segs, int64_t m,
                                   bool include_zeros) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t seg = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (seg >= total_segs) {
    return;
  }

  index_t start = rowptr[seg];
  index_t end = rowptr[seg + 1];

  auto acc = tsgu::OnlineLogSumExp<scalar_t>::identity();
  for (index_t i = start + static_cast<index_t>(lane); i < end; i += kWarpSize) {
    acc.update(vals[i]);
  }
  acc = warp_reduce_lse<scalar_t>(acc);

  if (lane == 0) {
    if (include_zeros) {
      // Structural zeros in this segment: m - (specified entries) copies of
      // the value 0, folded in as one more accumulator (naming.md §2:
      // stored values vs structural zeros).
      scalar_t z = static_cast<scalar_t>(m - static_cast<int64_t>(end - start));
      if (z > scalar_t(0)) {
        acc = tsgu::OnlineLogSumExp<scalar_t>::combine(acc, tsgu::OnlineLogSumExp<scalar_t>{scalar_t(0), z});
      }
    }
    lse[seg] = acc.log_sum_exp();
  }
}

template <typename scalar_t, typename index_t>
__global__ void seglse_bwd_kernel(scalar_t *__restrict__ grad_vals, scalar_t const *__restrict__ vals,
                                   index_t const *__restrict__ rowptr, scalar_t const *__restrict__ lse,
                                   scalar_t const *__restrict__ gout, int64_t total_segs) {
  int64_t warp_in_block = threadIdx.x / kWarpSize;
  int64_t lane = threadIdx.x % kWarpSize;
  int64_t seg = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp_in_block;
  if (seg >= total_segs) {
    return;
  }

  index_t start = rowptr[seg];
  index_t end = rowptr[seg + 1];
  scalar_t seg_lse = lse[seg];
  scalar_t seg_gout = gout[seg];

  for (index_t i = start + static_cast<index_t>(lane); i < end; i += kWarpSize) {
    // gradA_val = exp(v - lse[seg]) * gout[seg] (kernels.md Family 2
    // backward, "Save lse from forward — no recompute").
    grad_vals[i] = exp(vals[i] - seg_lse) * seg_gout;
  }
}

}  // namespace

torch::stable::Tensor tsgu_seglse_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                          int64_t B, int64_t n, int64_t m, bool include_zeros) {
  STD_TORCH_CHECK(vals.is_cuda() && rowptr.is_cuda(), "tsgu::seglse expects CUDA tensors");
  STD_TORCH_CHECK(vals.dim() == 1, "tsgu::seglse expects vals of shape (nse_total,)");
  STD_TORCH_CHECK(rowptr.dim() == 1 && rowptr.size(0) == B * n + 1,
                   "tsgu::seglse expects rowptr of shape (B * n + 1,)");

  torch::stable::Tensor lse = torch::stable::new_empty(vals, {B, n});
  tsgu::StreamGuard guard(vals);

  int64_t total_segs = B * n;
  if (total_segs == 0) {
    return lse;
  }

  int64_t blocks = (total_segs + kWarpsPerBlock - 1) / kWarpsPerBlock;

  TSGU_DISPATCH_VALUE(vals.scalar_type(), "tsgu::seglse", [&] {
    TSGU_DISPATCH_INDEX(rowptr.scalar_type(), "tsgu::seglse", [&] {
      seglse_fwd_kernel<scalar_t, index_t><<<blocks, kThreadsPerBlock, 0, guard.stream()>>>(
          static_cast<scalar_t *>(lse.mutable_data_ptr()), static_cast<scalar_t const *>(vals.data_ptr()),
          static_cast<index_t const *>(rowptr.data_ptr()), total_segs, m, include_zeros);
    });
  });

  return lse;
}

torch::stable::Tensor tsgu_seglse_bwd_launch(torch::stable::Tensor const &vals, torch::stable::Tensor const &rowptr,
                                              torch::stable::Tensor const &lse, torch::stable::Tensor const &gout,
                                              int64_t B, int64_t n) {
  STD_TORCH_CHECK(vals.is_cuda() && rowptr.is_cuda() && lse.is_cuda() && gout.is_cuda(),
                   "tsgu::seglse_bwd expects CUDA tensors");
  STD_TORCH_CHECK(rowptr.size(0) == B * n + 1, "tsgu::seglse_bwd expects rowptr of shape (B * n + 1,)");
  // gout is read through a plain contiguous pointer below; a broadcast/
  // expanded (stride-0) grad_output -- e.g. from a .sum() reduction's
  // backward -- would alias out-of-bounds memory if passed through
  // uncontiguated. The Python autograd wrapper (ops/logsumexp.py
  // _seglse_backward) is responsible for calling .contiguous() before this
  // op; this check turns a violation into a clear error instead of a silent
  // OOB read.
  STD_TORCH_CHECK(gout.is_contiguous(), "tsgu::seglse_bwd expects a contiguous gout");

  torch::stable::Tensor grad_vals = torch::stable::empty_like(vals);
  tsgu::StreamGuard guard(vals);

  int64_t total_segs = B * n;
  if (total_segs == 0 || vals.numel() == 0) {
    return grad_vals;
  }

  int64_t blocks = (total_segs + kWarpsPerBlock - 1) / kWarpsPerBlock;

  TSGU_DISPATCH_VALUE(vals.scalar_type(), "tsgu::seglse_bwd", [&] {
    TSGU_DISPATCH_INDEX(rowptr.scalar_type(), "tsgu::seglse_bwd", [&] {
      seglse_bwd_kernel<scalar_t, index_t><<<blocks, kThreadsPerBlock, 0, guard.stream()>>>(
          static_cast<scalar_t *>(grad_vals.mutable_data_ptr()), static_cast<scalar_t const *>(vals.data_ptr()),
          static_cast<index_t const *>(rowptr.data_ptr()), static_cast<scalar_t const *>(lse.data_ptr()),
          static_cast<scalar_t const *>(gout.data_ptr()), total_segs);
    });
  });

  return grad_vals;
}
