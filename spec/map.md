# MAP

Every public function/class in the package, its current implementation, and its
CUDA-rewrite classification. Baseline: `main` @ f19d7b4 (v0.2.3). Terminology
follows `docs/source/naming.rst`. Goal context: [goal.md](goal.md).

## Contract invariants (apply to every public API)

Stated once here; per-op specifics live in the **Contract** column of §1.

1. **Surviving signatures frozen at v0.2.3** — names, parameters, defaults, return
   types of every API that is not ⚫ RETIRE. Retired APIs are removed outright —
   this is a breaking release with no migration path or deprecation cycle.
2. **naming.rst is binding** for all new code, docs, and error messages.
3. **Gradient guarantee** — gradients w.r.t. a sparse input are sparse, at the
   input's sparsity pattern, in the input's layout. Never densified.
4. **Layout parity** — every op accepts what it accepts today (COO and/or CSR);
   output layouts unchanged.
5. **Batching** — one leading batch axis, exact match, no broadcasting. Batched
   COO items may have unequal nse; batched CSR requires equal nse per item.
   `_nnz()` semantics per layout (CSR: per batch item; COO: whole tensor) preserved.
6. **Index dtypes** — int32 and int64 supported where supported today; dtype
   preserved through outputs and gradients (upstream COO int64 coercion excepted).
7. **Errors** — invalid input raises (never silently accepts); messages state
   accepted logical shapes and the received shape.
8. **No legacy code** — the current pure-PyTorch implementations are not carried
   into the rewrite. Whether they serve as an *external* differential oracle
   (and what the CPU/non-CUDA story is) is deferred to the testing discussion.
9. **Coalescing** — COO semantics for uncoalesced input match current behaviour
   per op; compressed layouts never described as (un)coalesced.

**Legend**

| Tier | Meaning |
|------|---------|
| 🔴 KERNEL | Needs a custom CUDA kernel (no vendor primitive covers it) |
| 🟠 VENDOR | Custom-kernel target with a vendor primitive (cuSPARSE / cuBLAS / cuDSS) as the **baseline to beat**; vendor binding allowed only as bring-up scaffold + benchmark reference |
| 🟡 COMPOSITE | Host-side orchestration; becomes CUDA-native by calling the 🔴/🟠 ops above it |
| 🟢 HOST | Stays pure PyTorch/host — no CUDA work needed (setup, generation, stats, glue) |
| ⚫ RETIRE | Made obsolete by the rewrite (workaround for a gap that native kernels close) |

## 1. Core ops (`torchsparsegradutils/__init__.py` exports)

| API | Location | Current implementation | Tier | CUDA plan | Contract |
|-----|----------|------------------------|------|-----------|----------|
| `sparse_mm(A, B)` | `sparse_matmul.py:8` | `torch.sparse.mm` fwd; backward builds sparse gradA via `index_select` + row-dot | 🟠 fwd / 🔴 bwd | Fwd: native SpMM (COO/CSR, batched), baseline cuSPARSE SpMM. Bwd gradA is exactly **SDDMM** (sampled dense-dense matmul at A's sparsity pattern) — baseline cuSPARSE `SDDMM`, custom for batched + COO; gradB = SpMM(Aᵀ, G) | A: COO+CSR, `(n,m)`/`(b,n,m)`; B dense. Out dense. gradA sparse @ A's pattern+layout; gradB dense; index dtype preserved |
| `SparseMatMul` (autograd.Function) | `sparse_matmul.py:132` | block-diagonalises batch, saves A,B | ⚫ RETIRE (internal) | Replaced by native batched kernel dispatch; block-diag batching trick no longer needed | — internal |
| `sparse_triangular_solve(A, B, upper, unitriangular)` | `sparse_solve.py:9` | `torch.triangular_solve` on CSR (CPU path) / dense fallback paths | 🟠 fwd / 🔴 bwd | Fwd: native triangular SpSM, baseline cuSPARSE SpSM. Bwd: gradB = SpSM(Aᵀ); gradA = sampled outer product −x·gradBᵀ at A's pattern (SDDMM-shaped kernel) | A: COO+CSR triangular `(n,n)`/`(b,n,n)`; B dense rhs. `upper`/`unitriangular` flags kept. X dense = B's shape. gradA sparse @ A's pattern+layout; gradB dense |
| `SparseTriangularSolve` | `sparse_solve.py:150` | as above | ⚫ RETIRE (internal) | Folded into native op | — internal |
| `sparse_generic_solve(A, B, solve, transpose_solve)` | `sparse_solve.py:257` | pluggable iterative solver, pure PyTorch | 🟡 COMPOSITE | Loop stays host; speed comes from native SpMV/SpMM inside solvers. Bwd sparse gradA = SDDMM kernel. Direct factorisation (cuDSS-style) stays out of scope — served by `torch.sparse.linalg.spsolve` / torch-sla | A: COO+CSR `(n,n)`; B dense rhs. Pluggable `solve`/`transpose_solve` callables kept (user solvers must keep working). gradA sparse @ A's pattern+layout; gradB dense |
| `SparseGenericSolve` | `sparse_solve.py:429` | as above | ⚫ RETIRE (internal) | Folded into native op | — internal |
| `sparse_generic_lstsq(A, B, lstsq, transpose_lstsq)` | `sparse_lstsq.py:6` | LSMR-backed, pure PyTorch | 🟡 COMPOSITE | Same as generic_solve: host loop over native SpMV; SDDMM-shaped sparse gradient | A: COO+CSR `(n,m)` rectangular; B dense. Pluggable `lstsq`/`transpose_lstsq` kept. gradA sparse @ A's pattern+layout; gradB dense |
| `SparseGenericLstsq` | `sparse_lstsq.py:159` | as above | ⚫ RETIRE (internal) | Folded into native op | — internal |
| `sparse_logsumexp(A, dim, include_zeros)` | `sparse_logsumexp.py:246` | scatter-based max-shift reduction, pure PyTorch | 🔴 KERNEL | **Flagship custom kernel** — no vendor primitive, no maintained CUDA competitor operating on native sparse layouts (pytorch_scatter is index-array only, legacy). Segmented max + shifted exp-sum, fused; bwd = softmax-weighted scatter at A's pattern | A: COO+CSR+CSC, `(n,m)`/`(b,n,m)`. `dim`, `include_zeros` semantics exact (structural-zero handling per current docs). Out dense, matches `torch.logsumexp` shape rules. gradA sparse @ A's pattern+layout |
| `sparse_bidir_logsumexp(A, ...)` | `sparse_logsumexp.py:354` | fuses dim-0 and dim-1 passes into one batched scatter | 🔴 KERNEL | Fused row+col variant of the above — single traversal emitting both reductions; `tuple`/`padded`/`nested` output layouts assembled host-side | As above + must equal two single-dim calls exactly; `tuple`/`padded`/`nested` output layouts kept |
| `segment_mm(a, b, seglen_a)` | `indexed_matmul.py:12` | nested-tensor ops (torch ≥2.4) | 🟠 VENDOR | Native grouped GEMM, baseline cuBLAS `cublasGemmGroupedBatched`; matches DGL/pyg-lib territory — differentiator is staying dependency-light | Dense in/out; DGL `segment_mm` semantics exact (`a[off:off+len] @ b[i]`); both grads dense |
| `gather_mm(a, b, idx_b)` | `indexed_matmul.py:109` | nested-tensor ops | 🟠 VENDOR | Gather fused into grouped GEMM (custom); baseline gather + cuBLAS grouped GEMM | Dense in/out; DGL `gather_mm` semantics exact; both grads dense |

### Kernel routing (§1 ops → `tsgu::` ops → `csrc/kernels/` dir)

Names are final — [naming.md](naming.md) §2 governs; implementers copy them
verbatim, never invent. `spmv` = `spmm` with `p = 1` (no separate op).

| Public API | Forward | Backward | Kernel dir(s) | Status |
|------------|---------|----------|---------------|--------|
| `sparse_mm` | `tsgu::spmm` | gradA `tsgu::sddmm` · gradB `tsgu::spmm` (on cached CSC) | `spmm/`, `sddmm/` | ✅ live (commit 15) |
| `sparse_triangular_solve` | `tsgu::spsm` | gradB `tsgu::spsm` (transposed plan) · gradA `tsgu::sddmm` (negate epilogue) | `spsm/`, `sddmm/` | ✅ live (commit 16) |
| `sparse_generic_solve` | host loop → `tsgu::spmm` | host `transpose_solve` → gradA `tsgu::sddmm` | `spmm/`, `sddmm/` | ✅ live (commit 17) |
| `sparse_generic_lstsq` | host loop → `tsgu::spmm` (A and cached CSC) | host `transpose_lstsq` → gradA `tsgu::sddmm` | `spmm/`, `sddmm/` | ✅ live (commit 17) |
| `sparse_logsumexp` | `tsgu::seglse` | `tsgu::seglse_bwd` (uses saved `lse`) | `logsumexp/` | ✅ live (commit 12) |
| `sparse_bidir_logsumexp` | `tsgu::seglse_bidir` | `tsgu::seglse_bidir_bwd` | `logsumexp/` | ✅ live (commit 13) — correctness bar met, fusion perf bar NOT met (see commit message) |
| `segment_mm` | `tsgu::grouped_gemm` | both grads `tsgu::grouped_gemm` (transposed operands) | `grouped_gemm/` | ✅ live (commit 18) — correctness bars met; cuBLAS-parity bar NOT met (scatter/gradB 0.83× after vectorised LDS.128 frags; gather 0.5–0.78× — its mixed path needs the opposite lane mapping, so the follow-up is a uniform/mixed two-kernel split + warptiling, measured flat for double buffering; see kernel_best_practices.md) |
| `gather_mm` | `tsgu::grouped_gemm` (gather prologue) | both grads `tsgu::grouped_gemm` | `grouped_gemm/` | ✅ live (commit 18) — beats gather+per-group-cuBLAS 2.6–3.1× and legacy nested 3.5–6.9×; segment-shape parity pending with the above |
| `convert_coo_to_csr*` | `tsgu::coo2csr` | — (index-only, no grad) | `convert/` | ✅ live (commit 19) — 1.18× vs thrust+Xcoo2csr (NVBench), 1.9–3.7× vs pure-torch path |

Composites route through the table: iterative solvers (§2) call `tsgu::spmm`;
distributions/encoder (§6–7) call the §1 ops and never touch `tsgu::` directly.

## 2. Iterative solvers (`utils/`)

| API | Location | Current implementation | Tier | CUDA plan |
|-----|----------|------------------------|------|-----------|
| `linear_cg`, `LinearCGSettings` | `utils/linear_cg.py` | ported from linear_operator, pure PyTorch | 🟡 COMPOSITE | Host loop; hot path is SpMV (🟠) + dot/axpy. Optional 🔴 fused CG-iteration kernel later (dot+axpy fusion) — low priority, measure first |
| `bicgstab`, `BICGSTABSettings` | `utils/bicgstab.py` | ported from pykrylov | 🟡 COMPOSITE | Same |
| `lsmr` | `utils/lsmr.py` | ported from pytorch-minimize | 🟡 COMPOSITE | Same (needs SpMV with A and Aᵀ) |
| `minres`, `MINRESSettings` | `utils/minres.py` | ported from linear_operator | 🟡 COMPOSITE | Same |

## 3. Sparse utilities (`utils/utils.py`)

| API | Location | Current implementation | Tier | CUDA plan |
|-----|----------|------------------------|------|-----------|
| `stack_csr(tensors, dim)` | `utils/utils.py:6` | index concat/metadata | 🟢 HOST | Device-agnostic index manipulation; keep |
| `convert_coo_to_csr_indices_values(...)` | `utils/utils.py:236` | sort + bincount compress, pure PyTorch | 🟠 VENDOR | Native sort+compress kernel, baseline cuSPARSE `Xcoo2csr` |
| `convert_coo_to_csr(A)` | `utils/utils.py:349` | wraps the above | 🟡 COMPOSITE | Inherits |
| `sparse_block_diag(*tensors)` | `utils/utils.py:474` | index offsetting + concat | ⚫ RETIRE | Existed to fake batched SpMM/solve; native batched kernels remove the need. Dropped outright |
| `sparse_block_diag_split(T, *shapes)` | `utils/utils.py:648` | inverse of the above | ⚫ RETIRE | Dropped with it |
| `sparse_eye(size, layout, ...)` | `utils/utils.py:793` | arange-based construction | 🟢 HOST | Trivial on-device construction; keep |

## 4. Random sparse generators (`utils/random_sparse.py`) — test/benchmark support

| API | Tier | Notes |
|-----|------|-------|
| `rand_sparse`, `rand_sparse_tri` | 🟢 HOST | Generation logic; already device-parameterised |
| `generate_random_sparse_coo_matrix`, `..._csr_matrix` | 🟢 HOST | Same |
| `generate_random_sparse_strictly_triangular_coo/csr_matrix` | 🟢 HOST | Same |
| `generate_random_sparse_triangular_coo/csr_matrix` | 🟢 HOST | Same |
| `make_spd_sparse` | 🟢 HOST | Same |

## 5. Statistics helpers (`utils/dist_stats_helpers.py`) — validation support

| API | Tier | Notes |
|-----|------|-------|
| `mean_hotelling_t2_test` | 🟢 HOST | Dense small-matrix stats; no CUDA work |
| `cov_nagao_test` | 🟢 HOST | Same |

## 6. Distributions (`distributions/`)

| API | Location | Tier | CUDA plan |
|-----|----------|------|-----------|
| `SparseMultivariateNormal` (`rsample`, properties `loc/mean/mode/diagonal/scale_tril/precision_tril/is_ldlt_parameterization`) | `sparse_multivariate_normal.py:105` | 🟡 COMPOSITE | `rsample` = sparse_mm / triangular_solve chains → becomes CUDA-native for free once §1 lands. Revisit LL^T-precision gradient instability once native kernels expose better-conditioned paths |
| `SparseMultivariateNormalNative` (`rsample`, `log_prob`, `covariance_matrix`, `variance`, ...) | `sparse_multivariate_normal.py:392` | 🟡 COMPOSITE | Uses native `torch.sparse.mm` only; inherits |

## 7. Encoders (`encoders/`)

| API | Location | Tier | CUDA plan |
|-----|----------|------|-----------|
| `PairwiseEncoder` (nn.Module, `forward`, `device`) | `pairwise_encoder.py:562` | 🟡 COMPOSITE / 🔴 candidate | Index construction is precomputed host-side (fine); the per-call value gather + CSR permutation is the hot path and the source of the documented CSR backward memory blow-up — candidate for a custom gather/permute kernel with a sparse-aware backward |
| `calc_pairwise_coo_indices_nd` | `pairwise_encoder.py:383` | 🟢 HOST | One-off index computation, cacheable |
| `calc_pairwise_coo_indices` (deprecated) | `pairwise_encoder.py:508` | ⚫ RETIRE | Deleted |
| `PairwiseVoxelEncoder` (deprecated alias) | `pairwise_voxel_encoder.py:20` | ⚫ RETIRE | Deleted |

## 8. CuPy bridge (`cupy/`)

| API | Location | Tier | CUDA plan |
|-----|----------|------|-----------|
| `sparse_solve_c4t(A, B, solve, ...)` / `SparseSolveC4T` | `cupy_sparse_solve.py` | ⚫ RETIRE | Existed to borrow CUDA solvers we don't have; removed outright — breaking release, no deprecation cycle |
| `t2c_csr`, `c2t_csr`, `t2c_coo`, `c2t_coo` | `cupy_bindings.py` | ⚫ RETIRE | Removed with the bridge |

## 9. JAX bridge (`jax/`)

| API | Location | Tier | CUDA plan |
|-----|----------|------|-----------|
| `sparse_solve_j4t(A, B, solve, ...)` / `SparseSolveJ4T` | `jax_sparse_solve.py` | ⚫ RETIRE | Same as CuPy bridge — removed outright |
| `j2t`, `t2j`, `j2t_csr`, `t2j_csr`, `j2t_coo`, `t2j_coo`, `spmm_t4j` | `jax_bindings.py` | ⚫ RETIRE | Removed with the bridge |

## Summary

The rewrite reduces to **two custom kernel families plus vendor-baseline kernels**:

1. 🔴 **SDDMM-shaped sparse-gradient kernel** — the single backward primitive shared by
   `sparse_mm`, `sparse_triangular_solve`, `sparse_generic_solve`, `sparse_generic_lstsq`
   (gradA = sampled product of two dense matrices at A's sparsity pattern).
   cuSPARSE `SDDMM` is the unbatched-CSR baseline; batched + COO have no vendor
   primitive at all.
2. 🔴 **Segmented logsumexp kernel family** — `sparse_logsumexp` + fused bidirectional
   variant. No vendor primitive, no maintained competitor on native sparse layouts:
   the clearest differentiator.
3. 🟠 **Vendor-baseline kernels** — our own SpMM/SpSM/coo2csr and grouped GEMM,
   each benchmarked head-to-head against its cuSPARSE/cuBLAS counterpart with the
   explicit target of **beating it** (batched and COO cases are where vendor
   primitives are weakest and native batching gives us the edge). Exception:
   cuDSS-style *direct factorisation* solves stay out of scope — that space is
   already served by `torch.sparse.linalg.spsolve` / torch-sla, and our solvers
   are iterative.

Everything else is composite (inherits), host-side support code, or retired
workarounds (block-diag batching, CuPy/JAX solver bridges).

**Acceptance bar per kernel:** ≥ parity with the vendor primitive on the unbatched
SuiteSparse (Rothberg/cfd2) benchmarks, and a clear win on batched/COO paths where
the current block-diag workaround is the comparison point.
