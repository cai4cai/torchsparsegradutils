// tsgu::_smoke — CUDA kernel-builder bring-up (spec/commit.md Phase 2 #10).
//
// Trivial op: out = x + 1, computed by an actual CUDA kernel launch (not a
// torch:: call), to prove compile + load + dispatch end-to-end. The op's
// *schema* lives in the front package (torchsparsegradutils/_dispatch.py,
// guarded as private — architecture.md §2); this translation unit only
// provides the CUDA dispatch-key implementation, registered from
// csrc/registration.cpp.
//
// No real op routes through this kernel; it exists so the whole toolchain —
// and all four common/ headers (naming.md §2 short names, dispatch.cuh,
// reduce.cuh, batched_csr.cuh, stream.cuh) — compile together in one real
// translation unit before any kernel commit (Phase 3) depends on them. The
// reduce.cuh / batched_csr.cuh calls below run on scratch values that never
// feed `out`: this op's only contract is exact `x + 1`.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "../../common/batched_csr.cuh"
#include "../../common/dispatch.cuh"
#include "../../common/reduce.cuh"
#include "../../common/stream.cuh"

namespace {

template <typename scalar_t>
__global__ void smoke_add_one_kernel(scalar_t *__restrict__ out, scalar_t const *__restrict__ in, int64_t n) {
  int64_t idx = (static_cast<int64_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
  if (idx >= n) {
    return;
  }

  // Exercise reduce.cuh and batched_csr.cuh once each — on scratch state that
  // never touches `out` — so this TU actually instantiates their templates
  // (not just parses their syntax), without perturbing the exact x + 1
  // contract the smoke test checks.
  scalar_t scratch = tsgu::warp_reduce_sum<scalar_t>(in[idx], 1U << (threadIdx.x % warpSize));
  scratch = tsgu::warp_reduce_max<scalar_t>(scratch, 1U << (threadIdx.x % warpSize));
  auto acc = tsgu::OnlineLogSumExp<scalar_t>::identity();
  acc.update(scratch);
  tsgu::BatchedCSRView<scalar_t, int64_t> scratch_view{in, nullptr, nullptr, 1, n, n};
  scratch = tsgu::OnlineLogSumExp<scalar_t>::combine(acc, acc).log_sum_exp() * scalar_t(0) +
            static_cast<scalar_t>(scratch_view.batch_of(0)) * scalar_t(0);
  (void)scratch;

  out[idx] = in[idx] + scalar_t(1);
}

}  // namespace

torch::stable::Tensor tsgu_smoke_launch(torch::stable::Tensor const &x) {
  STD_TORCH_CHECK(x.is_cuda(), "tsgu::_smoke expects a CUDA tensor");
  STD_TORCH_CHECK(x.is_contiguous(), "tsgu::_smoke expects a contiguous tensor");

  torch::stable::Tensor out = torch::stable::empty_like(x);
  tsgu::StreamGuard guard(x);

  int64_t n = x.numel();
  constexpr int64_t threads = 256;
  int64_t blocks = (n + threads - 1) / threads;

  if (n == 0) {
    return out;
  }

  TSGU_DISPATCH_VALUE(x.scalar_type(), "tsgu::_smoke", [&] {
    smoke_add_one_kernel<scalar_t><<<blocks, threads, 0, guard.stream()>>>(
        static_cast<scalar_t *>(out.mutable_data_ptr()), static_cast<scalar_t const *>(x.data_ptr()), n);
  });

  return out;
}
