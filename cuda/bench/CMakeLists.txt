# cuda/bench — NVBench microbenchmarks (architecture.md §5: "one target per
# kernel, lands with the kernel"). This bring-up commit (spec/commit.md Phase
# 2 #10) adds exactly one dummy target proving the NVBench toolchain works;
# kernel commits (Phase 3) add their own target here alongside their kernel,
# per family (kernels.md), following this same standalone-CMake pattern.
#
# Deliberately standalone from the kernel-builder/Nix build (build.toml only
# builds the torch extension itself) — NVBench benchmarks measure kernel
# throughput directly and gain nothing from going through torch::Tensor /
# Python; build directly:
#   cmake -S cuda/bench -B cuda/bench/build -DCMAKE_CUDA_ARCHITECTURES=86
#   cmake --build cuda/bench/build
# (needs CMake >= 3.30 — NVBench's own floor; use `nix shell nixpkgs#cmake`
# if the system cmake is older, as it is on this machine's Ubuntu 22.04).
cmake_minimum_required(VERSION 3.30)
project(tsgu_cuda_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)  # this machine's RTX A1000 Laptop (sm_86)
endif()

include(FetchContent)
FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG main
)
FetchContent_MakeAvailable(nvbench)

add_executable(bench_smoke bench_smoke.cu)
target_link_libraries(bench_smoke PRIVATE nvbench::nvbench nvbench::main)

# tsgu::seglse (spec/commit.md Phase 3 commit 12) — kernels.md Family 2.
add_executable(bench_seglse bench_seglse.cu)
target_link_libraries(bench_seglse PRIVATE nvbench::nvbench nvbench::main)

# tsgu::seglse_bidir (spec/commit.md Phase 3 commit 13) — kernels.md Family 2
# fused bidirectional variant.
add_executable(bench_seglse_bidir bench_seglse_bidir.cu)
target_link_libraries(bench_seglse_bidir PRIVATE nvbench::nvbench nvbench::main)

# tsgu::sddmm (spec/commit.md Phase 3 commit 14) — kernels.md Family 1.
add_executable(bench_sddmm bench_sddmm.cu)
target_link_libraries(bench_sddmm PRIVATE nvbench::nvbench nvbench::main)

# tsgu::spmm (spec/commit.md Phase 3 commit 15) — kernels.md Family 3.
add_executable(bench_spmm bench_spmm.cu)
target_link_libraries(bench_spmm PRIVATE nvbench::nvbench nvbench::main)
