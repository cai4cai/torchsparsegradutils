# cuda/bench — NVBench microbenchmarks (architecture.md §5: "one target per
# kernel, lands with the kernel"). This bring-up commit (spec/commit.md Phase
# 2 #10) adds exactly one dummy target proving the NVBench toolchain works;
# kernel commits (Phase 3) add their own target here alongside their kernel,
# per family (kernels.md), following this same standalone-CMake pattern.
#
# Deliberately standalone from the kernel-builder/Nix build (build.toml only
# builds the torch extension itself) — NVBench benchmarks measure kernel
# throughput directly and gain nothing from going through torch::Tensor /
# Python; build directly (Ninja, so the RAM guard below is enforced):
#   cmake -S cuda/bench -B cuda/bench/build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86
#   cmake --build cuda/bench/build
# (needs CMake >= 3.30 — NVBench's own floor; use `nix shell nixpkgs#cmake
# nixpkgs#ninja` if the system cmake is older, as it is on this machine's
# Ubuntu 22.04).
cmake_minimum_required(VERSION 3.30)
project(tsgu_cuda_bench LANGUAGES CXX CUDA)

# Compile-parallelism guard. NVBench TUs are template-heavy: each cc1plus
# peaks at ~0.7–2.8 GB RSS, and Ninja's default of nproc+2 jobs (22 on a
# 20-core box) wants ~18 GB at once — enough to OOM a 30 GB laptop running a
# desktop session (observed 2026-07-16: global OOM, hard power-off). Bound
# the pool from *available* RAM at ~3 GB per job so a cold, wide build stays
# inside memory no matter how `cmake --build` is invoked. Job pools are a
# Ninja feature; the Makefile generator ignores them but defaults to -j1
# anyway. Override for a dedicated build box with -DTSGU_BENCH_COMPILE_JOBS=N.
if(NOT DEFINED TSGU_BENCH_COMPILE_JOBS)
  cmake_host_system_information(RESULT _tsgu_avail_mb QUERY AVAILABLE_PHYSICAL_MEMORY)
  cmake_host_system_information(RESULT _tsgu_ncores QUERY NUMBER_OF_LOGICAL_CORES)
  math(EXPR TSGU_BENCH_COMPILE_JOBS "${_tsgu_avail_mb} / 3072")
  if(TSGU_BENCH_COMPILE_JOBS LESS 2)
    set(TSGU_BENCH_COMPILE_JOBS 2)
  endif()
  if(TSGU_BENCH_COMPILE_JOBS GREATER ${_tsgu_ncores})
    set(TSGU_BENCH_COMPILE_JOBS ${_tsgu_ncores})
  endif()
endif()
message(STATUS "tsgu bench: capping concurrent compile jobs at ${TSGU_BENCH_COMPILE_JOBS}")
set_property(GLOBAL PROPERTY JOB_POOLS "compile_pool=${TSGU_BENCH_COMPILE_JOBS}")
set(CMAKE_JOB_POOL_COMPILE compile_pool)
set(CMAKE_JOB_POOL_LINK compile_pool)
if(NOT CMAKE_GENERATOR MATCHES "Ninja")
  message(WARNING "tsgu bench: generator is '${CMAKE_GENERATOR}', which ignores "
    "job pools — the RAM guard above is NOT enforced. Configure with -G Ninja, "
    "or never build with more than -j ${TSGU_BENCH_COMPILE_JOBS}.")
endif()

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

# tsgu::spsm (spec/commit.md Phase 3 commit 16) — kernels.md Family 3
# Triangular SpSM row; benches the level-scheduled solve kernel (the "warm"
# path — plan already built).
add_executable(bench_spsm bench_spsm.cu)
target_link_libraries(bench_spsm PRIVATE nvbench::nvbench nvbench::main)
