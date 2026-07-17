// NVBench target for tsgu::grouped_gemm (spec/commit.md Phase 3 commit 18,
// T3; kernels.md Family 3 Grouped GEMM row; architecture.md §5:
// "segment_mm/gather_mm, gather fused in prologue"). Axes are the kernel's
// tuning knobs (cuda/csrc/kernels/grouped_gemm/grouped_gemm.cu): row count
// `N`, matrix edge `D` (= D1 = D2, square group matrices), group count `R`;
// the three idx regimes are separate benchmarks below (uniform sorted
// segments = the segment_mm shape and the kernel's uniform-tile fast path;
// uniform-random idx = the gather_mm shape and the mixed-tile path;
// sorted idx = scatter-reduce mode / gradB). Standalone CMake + FetchContent,
// same pattern as bench_spmm.cu: the kernels are mirrored inline (f32/i32)
// rather than #including the torch-dependent .cu.
//
// Unlike bench_spmm.cu, the vendor baseline IS duplicated here:
// benchmarks.md §3's grouped-GEMM bar (">= parity" vs cuBLAS
// `cublasGemmGroupedBatched`, TF32 OFF) has no convenient op-level torch
// route (torch exposes no grouped-GEMM primitive), so the reference row is a
// cuBLAS run over the same uniform segment partition, in this same binary:
// `cublasSgemmGroupedBatched` when the toolkit provides it (CUDA >= 12.5),
// with a per-group `cublasSgemm` loop as the documented fallback. Math mode
// is CUBLAS_PEDANTIC_MATH — never TF32 (benchmarks.md §2: "TF32 off for
// every parity-relevant number"). This target remains for kernel-level
// tuning/regression-bisection only (benchmarks.md §1: "NVBench ... never
// gates").
//
// VRAM guard: this machine's GPU has 4 GB — any config whose tensors exceed
// 1.5 GB total is skipped via state.skip (the listed axes all fit; the guard
// protects against ad-hoc -a overrides).

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdint>
#include <nvbench/nvbench.cuh>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kBM = 64;                // block-tile rows (gather: output rows; scatter: D1 extent)
constexpr int kBN = 64;                // block-tile cols (D2 extent)
constexpr int kBK = 16;                // inner-dimension chunk (gather: D1; scatter: row chunk)
constexpr int kThreadsPerBlock = 256;  // logical (16, 16)
constexpr int kTM = 4;                 // register-tile rows per thread (kBM / 16)
constexpr int kTN = 4;                 // register-tile cols per thread (kBN / 16)

// --- Mirrors of csrc/kernels/grouped_gemm/grouped_gemm.cu (f32/i32) --------

__device__ __forceinline__ int64_t lower_bound_idx(int32_t const* __restrict__ idx, int64_t n, int64_t g) {
  int64_t lo = 0;
  int64_t hi = n;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (static_cast<int64_t>(idx[mid]) < g) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ __forceinline__ int64_t upper_bound_idx(int32_t const* __restrict__ idx, int64_t n, int64_t g) {
  int64_t lo = 0;
  int64_t hi = n;
  while (lo < hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (static_cast<int64_t>(idx[mid]) <= g) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Mirror of grouped_gemm_gather_kernel (f32/i32) — keep in sync with
// csrc/kernels/grouped_gemm/grouped_gemm.cu.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void grouped_gemm_gather_kernel(float* __restrict__ out, float const* __restrict__ a,
                                           float const* __restrict__ b, int32_t const* __restrict__ idx, int64_t N,
                                           int64_t D1, int64_t D2) {
  // a staged transposed (a_s[kk][row]) so the inner loop's per-thread row
  // reads are broadcast/conflict-free; +1 pads away bank conflicts on the
  // transposed store.
  __shared__ float a_s[BK][BM + 1];
  __shared__ float b_s[BK][BN + 1];
  __shared__ int64_t idx_s[BM];
  __shared__ int uniform_s;

  int64_t const row_base = static_cast<int64_t>(blockIdx.x) * BM;
  int64_t const col_base = static_cast<int64_t>(blockIdx.y) * BN;
  int const tid = static_cast<int>(threadIdx.x);
  constexpr int LX = BN / TN;  // column-lane count (thread grid is (LY, LX) = 256)
  constexpr int LY = BM / TM;  // row-lane count
  int const tx = tid % LX;     // register-tile column lane
  int const ty = tid / LX;     // register-tile row lane

  // Stage the tile's idx values once, then block-vote uniformity: rows past
  // N carry the sentinel -1 and never break the vote (a tail tile of one
  // long segment still takes the uniform path). row_base < N always holds
  // (grid is sized from N), so idx_s[0] is a real group.
  if (tid == 0) {
    uniform_s = 1;
  }
  if (tid < BM) {
    int64_t row = row_base + tid;
    idx_s[tid] = row < N ? static_cast<int64_t>(idx[row]) : int64_t(-1);
  }
  __syncthreads();
  int64_t const g0 = idx_s[0];
  if (tid < BM && idx_s[tid] != g0 && idx_s[tid] != int64_t(-1)) {
    uniform_s = 0;  // benign write race: every writer stores the same 0
  }
  __syncthreads();
  bool const uniform = uniform_s != 0;

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = float(0);
    }
  }

  float const* b_g0 = b + g0 * D1 * D2;

  for (int64_t kb = 0; kb < D1; kb += BK) {
    // Stage the a tile (BM rows x BK k, stored transposed), zero-filling
    // past N/D1 so the inner loop needs no bounds arithmetic; the
    // linear->(r, kk) split keeps kk fastest so consecutive threads read
    // consecutive a addresses.
#pragma unroll
    for (int e = tid; e < BM * BK; e += kThreadsPerBlock) {
      int r = e / BK;
      int kk = e % BK;
      int64_t row = row_base + r;
      int64_t k = kb + kk;
      a_s[kk][r] = (row < N && k < D1) ? a[row * D1 + k] : float(0);
    }
    if (uniform) {
      // Uniform tile: stage the single group's b k-tile too (module comment
      // §1) — the fully-tiled GEMM inner loop, c fastest for coalescing.
#pragma unroll
      for (int e = tid; e < BK * BN; e += kThreadsPerBlock) {
        int kk = e / BN;
        int c = e % BN;
        int64_t k = kb + kk;
        int64_t cc = col_base + c;
        b_s[kk][c] = (k < D1 && cc < D2) ? b_g0[k * D2 + cc] : float(0);
      }
    }
    __syncthreads();

    if (uniform) {
#pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        float a_frag[TM];
        float b_frag[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          a_frag[i] = a_s[kk][ty + i * LY];
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          b_frag[j] = b_s[kk][tx + j * LX];
        }
#pragma unroll
        for (int i = 0; i < TM; ++i) {
#pragma unroll
          for (int j = 0; j < TN; ++j) {
            acc[i][j] += a_frag[i] * b_frag[j];
          }
        }
      }
    } else {
      // Mixed tile: b straight from global, per row group (module comment
      // §1) — for each (row, kk) the four column reads are coalesced across
      // the 16 adjacent tx lanes; a gathered b copy is never materialised.
      int const kmax = static_cast<int>(min(static_cast<int64_t>(BK), D1 - kb));
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        int r = ty + i * LY;
        int64_t g = idx_s[r];
        if (g >= 0) {
          float const* b_gk = b + g * D1 * D2 + kb * D2;
          for (int kk = 0; kk < kmax; ++kk) {
            float const a_v = a_s[kk][r];
            float const* b_row = b_gk + static_cast<int64_t>(kk) * D2;
#pragma unroll
            for (int j = 0; j < TN; ++j) {
              int64_t cc = col_base + tx + j * LX;
              if (cc < D2) {
                acc[i][j] += a_v * b_row[cc];
              }
            }
          }
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int64_t row = row_base + ty + i * LY;
    if (row < N) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int64_t cc = col_base + tx + j * LX;
        if (cc < D2) {
          out[row * D2 + cc] = acc[i][j];
        }
      }
    }
  }
}

// Mirror of grouped_gemm_scatter_kernel (f32/i32) — keep in sync with
// csrc/kernels/grouped_gemm/grouped_gemm.cu.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void grouped_gemm_scatter_kernel(float* __restrict__ out, float const* __restrict__ a,
                                            float const* __restrict__ b, int32_t const* __restrict__ idx, int64_t N,
                                            int64_t D1, int64_t D2) {
  __shared__ float a_s[BK][BM];  // unpadded: row-contiguous staging is conflict-free, and 16B-aligned rows enable LDS.128 frags
  __shared__ float b_s[BK][BN];

  int64_t const g = static_cast<int64_t>(blockIdx.x);
  int64_t const i_base = static_cast<int64_t>(blockIdx.y) * BM;
  int64_t const j_base = static_cast<int64_t>(blockIdx.z) * BN;
  int const tid = static_cast<int>(threadIdx.x);
  constexpr int LX = BN / TN;  // column-lane count (thread grid is (LY, LX) = 256)
  constexpr int LY = BM / TM;  // row-lane count
  int const tx = tid % LX;
  int const ty = tid / LX;

  // Per-thread binary search (all threads land on the same [lo, hi); two
  // O(log N) probes are noise next to the row walk, so no lane-0+broadcast
  // dance is needed).
  int64_t const lo = lower_bound_idx(idx, N, g);
  int64_t const hi = upper_bound_idx(idx, N, g);

  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = float(0);
    }
  }

  for (int64_t r0 = lo; r0 < hi; r0 += BK) {
    // Stage 16-row chunks of a (columns i_base..i_base+63) and b (columns
    // j_base..j_base+63), zero-filled past hi/D1/D2 — zero rows contribute
    // nothing. c-fastest split keeps the global reads coalesced.
#pragma unroll
    for (int e = tid; e < BK * BM; e += kThreadsPerBlock) {
      int r = e / BM;
      int c = e % BM;
      int64_t row = r0 + r;
      int64_t i = i_base + c;
      a_s[r][c] = (row < hi && i < D1) ? a[row * D1 + i] : float(0);
      int64_t jj = j_base + c;
      b_s[r][c] = (row < hi && jj < D2) ? b[row * D2 + jj] : float(0);
    }
    __syncthreads();

    // Ascending, order-fixed accumulation over the chunk's rows (module
    // comment §2's determinism story); zero-filled tail rows add 0.
#pragma unroll
    for (int r = 0; r < BK; ++r) {
      float a_frag[TM];
      float b_frag[TN];
      // Contiguous per-thread fragments read as float4 (LDS.128) — cuts the
      // shared-load instruction count 4x vs the strided scalar mapping
      // (perf follow-up; see kernel_best_practices.md "Grouped GEMM" §7 and
      // siboehm's SGEMM worklog). Row bases are 16B-aligned (unpadded BM/BN
      // multiples of 4) and ty*TM / tx*TN offsets keep alignment.
      float4 const* a_vec = reinterpret_cast<float4 const*>(&a_s[r][ty * TM]);
      float4 const* b_vec = reinterpret_cast<float4 const*>(&b_s[r][tx * TN]);
#pragma unroll
      for (int v = 0; v < TM / 4; ++v) {
        reinterpret_cast<float4*>(a_frag)[v] = a_vec[v];
      }
#pragma unroll
      for (int v = 0; v < TN / 4; ++v) {
        reinterpret_cast<float4*>(b_frag)[v] = b_vec[v];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }
    __syncthreads();
  }

  // Unconditional write of the block's in-range tile — empty groups
  // (lo == hi) write zeros, which is what makes new_empty output safe.
  float* out_g = out + g * D1 * D2;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int64_t ii = i_base + ty * TM + i;
    if (ii < D1) {
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        int64_t jj = j_base + tx * TN + j;
        if (jj < D2) {
          out_g[ii * D2 + jj] = acc[i][j];
        }
      }
    }
  }
}

// --- Shared setup helpers ---------------------------------------------------

// 1.5 GB tensor-footprint cap (module comment: 4 GB VRAM machine).
constexpr double kMaxBytes = 1.5 * 1024.0 * 1024.0 * 1024.0;

bool skip_if_oversized(nvbench::state& state, double bytes) {
  if (bytes > kMaxBytes) {
    state.skip("tensor footprint exceeds 1.5 GB VRAM guard");
    return true;
  }
  return false;
}

// Uniform sorted segments: R segments of ~N/R rows each (the segment_mm
// shape; also scatter mode's sorted-idx input).
std::vector<int32_t> make_segment_idx(int64_t N, int64_t R) {
  std::vector<int32_t> idx(N);
  int64_t base = N / R;
  int64_t rem = N % R;
  int64_t off = 0;
  for (int64_t g = 0; g < R; ++g) {
    int64_t len = base + (g < rem ? 1 : 0);
    for (int64_t i = 0; i < len; ++i) {
      idx[off + i] = static_cast<int32_t>(g);
    }
    off += len;
  }
  return idx;
}

// Uniform-random idx (the gather_mm shape — defeats the uniform-tile vote).
std::vector<int32_t> make_random_idx(int64_t N, int64_t R) {
  std::vector<int32_t> idx(N);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(R - 1));
  for (auto& v : idx) {
    v = dist(rng);
  }
  return idx;
}

std::vector<float> make_uniform(int64_t count, uint32_t seed) {
  std::vector<float> v(count);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) {
    x = dist(rng);
  }
  return v;
}

// --- tsgu gather-mode benchmarks --------------------------------------------

void bench_gather(nvbench::state& state, std::vector<int32_t> const& idx_h) {
  int64_t const N = state.get_int64("N");
  int64_t const D = state.get_int64("D");
  int64_t const R = state.get_int64("R");

  double const bytes = 4.0 * (N * D /*a*/ + R * D * D /*b*/ + N * D /*out*/ + N /*idx (int32)*/);
  if (skip_if_oversized(state, bytes)) {
    return;
  }

  thrust::device_vector<float> a(make_uniform(N * D, 0));
  thrust::device_vector<float> b(make_uniform(R * D * D, 1));
  thrust::device_vector<int32_t> idx(idx_h);
  thrust::device_vector<float> out(N * D);

  state.add_global_memory_reads<float>(N * D + R * D * D);
  state.add_global_memory_writes<float>(N * D);

  state.exec([&](nvbench::launch& launch) {
    dim3 const block(kThreadsPerBlock);
    if (D >= 128 && N >= 128) {
      dim3 const grid(static_cast<unsigned>((N + 127) / 128), static_cast<unsigned>((D + 127) / 128));
      grouped_gemm_gather_kernel<128, 128, 16, 8, 8><<<grid, block, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()),
          thrust::raw_pointer_cast(idx.data()), N, D, D);
      return;
    }
    dim3 const grid(static_cast<unsigned>((N + kBM - 1) / kBM), static_cast<unsigned>((D + kBN - 1) / kBN));
    grouped_gemm_gather_kernel<kBM, kBN, kBK, kTM, kTN><<<grid, block, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(idx.data()), N, D, D);
  });
}

// segment_mm shape: sorted idx, equal-length segments — every full 32-row
// tile takes the uniform (fully-tiled GEMM) path.
void bench_grouped_gemm_gather_segments(nvbench::state& state) {
  bench_gather(state, make_segment_idx(state.get_int64("N"), state.get_int64("R")));
}

// gather_mm shape: uniform-random idx — nearly every tile takes the mixed
// (global-b, L2-served) path.
void bench_grouped_gemm_gather_random(nvbench::state& state) {
  bench_gather(state, make_random_idx(state.get_int64("N"), state.get_int64("R")));
}

// --- tsgu scatter-reduce benchmark ------------------------------------------

void bench_grouped_gemm_scatter_sorted(nvbench::state& state) {
  int64_t const N = state.get_int64("N");
  int64_t const D = state.get_int64("D");
  int64_t const R = state.get_int64("R");

  double const bytes = 4.0 * (N * D /*a*/ + N * D /*b*/ + R * D * D /*out*/ + N /*idx*/);
  if (skip_if_oversized(state, bytes)) {
    return;
  }

  thrust::device_vector<float> a(make_uniform(N * D, 0));
  thrust::device_vector<float> b(make_uniform(N * D, 1));
  thrust::device_vector<int32_t> idx(make_segment_idx(N, R));
  thrust::device_vector<float> out(R * D * D);

  state.add_global_memory_reads<float>(2 * N * D);
  state.add_global_memory_writes<float>(R * D * D);

  state.exec([&](nvbench::launch& launch) {
    dim3 const block(kThreadsPerBlock);
    if (D >= 128) {
      dim3 const grid(static_cast<unsigned>(R), static_cast<unsigned>((D + 127) / 128),
                      static_cast<unsigned>((D + 127) / 128));
      grouped_gemm_scatter_kernel<128, 128, 16, 8, 8><<<grid, block, 0, launch.get_stream()>>>(
          thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()),
          thrust::raw_pointer_cast(idx.data()), N, D, D);
      return;
    }
    dim3 const grid(static_cast<unsigned>(R), static_cast<unsigned>((D + kBM - 1) / kBM),
                    static_cast<unsigned>((D + kBN - 1) / kBN));
    grouped_gemm_scatter_kernel<kBM, kBN, kBK, kTM, kTN><<<grid, block, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(out.data()), thrust::raw_pointer_cast(a.data()), thrust::raw_pointer_cast(b.data()),
        thrust::raw_pointer_cast(idx.data()), N, D, D);
  });
}

// --- cuBLAS reference (the vendor bar, benchmarks.md §3) ---------------------

// Row-major segment GEMM via column-major cuBLAS: out_seg^T (D2 x n_g) =
// b_g^T (D2 x D1) @ a_seg^T (D1 x n_g), i.e. sgemm(N, N, m=D2, n=n_g, k=D1)
// with the row-major buffers passed as their column-major transposes.
void bench_grouped_gemm_cublas_segments(nvbench::state& state) {
  int64_t const N = state.get_int64("N");
  int64_t const D = state.get_int64("D");
  int64_t const R = state.get_int64("R");

  double const bytes = 4.0 * (N * D + R * D * D + N * D);
  if (skip_if_oversized(state, bytes)) {
    return;
  }

  thrust::device_vector<float> a(make_uniform(N * D, 0));
  thrust::device_vector<float> b(make_uniform(R * D * D, 1));
  thrust::device_vector<float> out(N * D);

  // Same uniform segment partition as the tsgu segment benchmark.
  std::vector<int64_t> seg_off(R + 1, 0);
  {
    int64_t base = N / R;
    int64_t rem = N % R;
    for (int64_t g = 0; g < R; ++g) {
      seg_off[g + 1] = seg_off[g] + base + (g < rem ? 1 : 0);
    }
  }

  cublasHandle_t handle = nullptr;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    state.skip("cublasCreate failed");
    return;
  }
  // TF32 must stay OFF for the parity bar (benchmarks.md §2); pedantic math
  // pins cuBLAS to plain CUDA-core f32.
  cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);

  int const group_count = static_cast<int>(R);
  std::vector<cublasOperation_t> transa(group_count, CUBLAS_OP_N);
  std::vector<cublasOperation_t> transb(group_count, CUBLAS_OP_N);
  std::vector<int> m_arr(group_count), n_arr(group_count), k_arr(group_count);
  std::vector<int> lda(group_count), ldb(group_count), ldc(group_count);
  std::vector<float> alpha(group_count, 1.0f), beta(group_count, 0.0f);
  std::vector<float const*> a_ptrs(group_count);
  std::vector<float const*> b_ptrs(group_count);
  std::vector<float*> c_ptrs(group_count);
  std::vector<int> group_size(group_count, 1);
  for (int g = 0; g < group_count; ++g) {
    int64_t n_g = seg_off[g + 1] - seg_off[g];
    m_arr[g] = static_cast<int>(D);    // rows of C^T = D2
    n_arr[g] = static_cast<int>(n_g);  // cols of C^T = segment length
    k_arr[g] = static_cast<int>(D);    // inner = D1
    lda[g] = static_cast<int>(D);      // b_g row-major (D1, D2) as cm (D2, D1)
    ldb[g] = static_cast<int>(D);      // a_seg row-major (n_g, D1) as cm (D1, n_g)
    ldc[g] = static_cast<int>(D);      // out_seg row-major (n_g, D2) as cm (D2, n_g)
    a_ptrs[g] = thrust::raw_pointer_cast(b.data()) + static_cast<int64_t>(g) * D * D;
    b_ptrs[g] = thrust::raw_pointer_cast(a.data()) + seg_off[g] * D;
    c_ptrs[g] = thrust::raw_pointer_cast(out.data()) + seg_off[g] * D;
  }
  // The grouped API takes the per-problem matrix pointer arrays in DEVICE
  // memory (like cublas<t>gemmBatched; the per-group scalar arrays stay
  // host-side) — NVIDIA's CUDALibrarySamples gemm_grouped_batched sample
  // copies them up exactly like this. Host-resident pointer arrays fault
  // with cudaErrorIllegalAddress.
  thrust::device_vector<float const*> a_ptrs_d(a_ptrs);
  thrust::device_vector<float const*> b_ptrs_d(b_ptrs);
  thrust::device_vector<float*> c_ptrs_d(c_ptrs);

  // Prefer the grouped API (the named vendor primitive); fall back to a
  // per-group sgemm loop if this toolkit/GPU combination rejects it.
  bool use_grouped = true;
#if CUBLAS_VERSION >= 120500
  {
    cublasStatus_t st = cublasSgemmGroupedBatched(
        handle, transa.data(), transb.data(), m_arr.data(), n_arr.data(), k_arr.data(), alpha.data(),
        thrust::raw_pointer_cast(a_ptrs_d.data()), lda.data(), thrust::raw_pointer_cast(b_ptrs_d.data()), ldb.data(),
        beta.data(), thrust::raw_pointer_cast(c_ptrs_d.data()), ldc.data(), group_count, group_size.data());
    use_grouped = st == CUBLAS_STATUS_SUCCESS && cudaDeviceSynchronize() == cudaSuccess;
  }
#else
  use_grouped = false;
#endif
  cudaDeviceSynchronize();
  {
    auto& summ = state.add_summary("tsgu/cublas_api");
    summ.set_string("description", "cuBLAS API used for the reference row");
    summ.set_string("name", "cublas_api");
    summ.set_string("value", use_grouped ? "SgemmGroupedBatched" : "Sgemm-loop");
  }

  state.add_global_memory_reads<float>(N * D + R * D * D);
  state.add_global_memory_writes<float>(N * D);

  state.exec([&](nvbench::launch& launch) {
    cublasSetStream(handle, launch.get_stream());
#if CUBLAS_VERSION >= 120500
    if (use_grouped) {
      cublasSgemmGroupedBatched(handle, transa.data(), transb.data(), m_arr.data(), n_arr.data(), k_arr.data(),
                                alpha.data(), thrust::raw_pointer_cast(a_ptrs_d.data()), lda.data(),
                                thrust::raw_pointer_cast(b_ptrs_d.data()), ldb.data(), beta.data(),
                                thrust::raw_pointer_cast(c_ptrs_d.data()), ldc.data(), group_count, group_size.data());
      return;
    }
#endif
    for (int g = 0; g < group_count; ++g) {
      if (n_arr[g] == 0) {
        continue;
      }
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m_arr[g], n_arr[g], k_arr[g], &alpha[g], a_ptrs[g], lda[g],
                  b_ptrs[g], ldb[g], &beta[g], c_ptrs[g], ldc[g]);
    }
  });

  // NVBench keeps the state alive until after exec's measurement loop, so
  // destroying here is safe.
  cublasDestroy(handle);
}

}  // namespace

NVBENCH_BENCH(bench_grouped_gemm_gather_segments)
    .add_int64_axis("N", {4096, 16384, 65536})
    .add_int64_axis("D", {64, 128, 256})
    .add_int64_axis("R", {4, 16, 64});

NVBENCH_BENCH(bench_grouped_gemm_gather_random)
    .add_int64_axis("N", {4096, 16384, 65536})
    .add_int64_axis("D", {64, 128, 256})
    .add_int64_axis("R", {4, 16, 64});

NVBENCH_BENCH(bench_grouped_gemm_scatter_sorted)
    .add_int64_axis("N", {4096, 16384, 65536})
    .add_int64_axis("D", {64, 128, 256})
    .add_int64_axis("R", {4, 16, 64});

NVBENCH_BENCH(bench_grouped_gemm_cublas_segments)
    .add_int64_axis("N", {4096, 16384, 65536})
    .add_int64_axis("D", {64, 128, 256})
    .add_int64_axis("R", {4, 16, 64});
