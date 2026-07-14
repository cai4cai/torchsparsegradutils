// Dummy NVBench target (spec/commit.md Phase 2 #10) — proves the NVBench
// toolchain (CMake + FetchContent + a runnable executable) is wired up
// before any kernel commit (Phase 3) adds a benchmark swept over its own
// tuning-knob axes (kernels.md). Deliberately standalone (no libtorch
// dependency, see CMakeLists.txt): benchmarks the same trivial add-one
// computation as tsgu::_smoke (csrc/kernels/_smoke/_smoke.cu), not the op
// itself — kernel commits benchmark their real kernels directly the same
// way, one target per kernel.

#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>

namespace {

__global__ void smoke_add_one_kernel(float *out, float const *in, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] + 1.0f;
  }
}

}  // namespace

void bench_smoke_add_one(nvbench::state &state) {
  int64_t const n = state.get_int64("N");

  thrust::device_vector<float> in(n, 1.0f);
  thrust::device_vector<float> out(n);

  state.exec([&](nvbench::launch &launch) {
    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    smoke_add_one_kernel<<<blocks, threads, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(in.data()), n);
  });
}
NVBENCH_BENCH(bench_smoke_add_one).add_int64_axis("N", {1 << 16, 1 << 20, 1 << 24});
