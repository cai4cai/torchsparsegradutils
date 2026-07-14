# Architecture

Supersedes the 2026-07-13 plan where they differ (that plan kept CPU reference
implementations, bridges, and block-diag as public API — all dropped per
[map.md](map.md); its dispatch and format decisions are carried forward here).

## 1. Two packages, one monorepo

| Package | Contents | Ships as |
|---------|----------|----------|
| **`torchsparsegradutils`** | Pure Python: public API wrappers, op definitions, dispatch, host-side composites (Krylov loops, distributions, encoder, utils) | `py3-none-any` wheel — no compiler, ever |
| **`torchsparsegradutils-cuda`** | The kernels: CUDA C++ under `csrc/`, registered into the same ops at import | Prebuilt binaries — mechanism is build.md's decision (HF kernel-builder → Hub is front-runner; stable-ABI `cp310-abi3` wheels per CUDA major the alternative) |

- Dependency direction: `-cuda` depends on the front package, never the reverse.
- Version handshake: `-cuda` exposes `__backend_api_version__`; the front package
  probes at import (`_dispatch.py`), refuses on mismatch. Escape hatch:
  `TSGU_DISABLE_CUDA_BACKEND=1`.

## 2. Dispatch: `torch.library` custom ops

- Every kernel-backed op is a `tsgu::` custom op: `torch.library.custom_op` +
  `register_fake` + `register_autograd`. This replaces the four
  `autograd.Function` classes and buys `torch.compile` compatibility and
  `torch.library.opcheck` testability.
- **Ops take plain dense tensors** (values / rowptr / col / shape ints), never
  `torch.sparse_*` tensors — sparse layouts have poor FakeTensor/meta support.
  Public wrappers unwrap at the boundary and rewrap results in the layout the
  user passed (COO in → COO grad out, per map.md invariant 3).
- Fake/meta kernels must be value-independent: output shapes derive from index
  array lengths and shape args only.
- The CUDA package registers implementations for the CUDA dispatch key from C++
  (`STABLE_TORCH_LIBRARY_IMPL(tsgu, CUDA, m)`).

## 3. Canonical internal format: `BatchedCSR`

**[naming.md](naming.md) is the backbone here** — it carries `naming.rst`'s
conventions through the migration and defines this section's vocabulary
(*descriptor*, *folded row*, *local column*, `nse_total`/`nse_per_item`, the
kernel short-name mapping). Every identifier below is used in its naming.md
sense; new BatchedCSR-related terms get added there, not invented ad hoc.

One frozen descriptor (`_batched.py`) replaces the block-diag hack everywhere,
with **ragged nse per batch item as first-class** (COO batching's flexibility,
CSR's kernel-friendliness):

- `values (nse_total,)` · `rowptr (B·n+1,)` absolute over folded rows `b·n+r` ·
  `col (nse_total,)` **local** columns in `[0, m)` · `shape (B, n, m)`; `B=1`
  encodes unbatched.
- Local (unoffset) columns keep int32 viable and let kernels address
  `Bdense[b, col, :]` directly; batch of an entry recovered as `row_global / n`.
- Lazy cached members: uncompressed `row_indices` (SDDMM), transposed
  `BatchedCSC` (gradB, dim=-2 reductions), **SpSM analysis plans** — resolving
  kernels.md open Q3: the plan cache lives on the descriptor, so its lifetime
  and invalidation are the descriptor's lifetime; no global cache, no hashing.
- `from_torch()`: 2D COO/CSR (CSR zero-copy), 3D COO, batched CSR, list-of-CSR
  (ragged-native). `to_torch(like=...)` restores the caller's layout.
- int32 index path whenever `max(nse_total, B·n+1, m) < 2³¹`; kernels templated
  over `{f32,f64} × {i32,i64}`.

## 4. CPU / non-CUDA story — RESOLVED (with testing.md)

The old pure-PyTorch implementations do not ship (map.md invariant 8).
**Decision: CUDA-required at runtime + oracle outside the wheel** — the front
package raises a clear error at import-probe time without the backend, and the
old implementations live as the test-only parity oracle in `tests/oracle/`
(extracted from git history, never packaged). A `-cpu` backend package (same
`tsgu::` ops, C++) remains possible later without architectural change.

## 5. File hierarchy

```
torchsparsegradutils/                  # monorepo root
├── spec/                              # this spec (index, goal, map, commit, kernels, architecture, build)
├── pyproject.toml                     # front package (uv-managed, dependency-groups)
├── uv.lock
├── torchsparsegradutils/              # ── front package (pure Python) ──
│   ├── __init__.py                    # public API re-exports (map.md §1 surface)
│   ├── _dispatch.py                   # backend probe, version handshake, TSGU_DISABLE_CUDA_BACKEND
│   ├── _batched.py                    # BatchedCSR / BatchedCSC descriptors + layout (un)wrapping
│   ├── ops/                           # one module per public op: wrapper + tsgu:: op def + fake kernel
│   │   ├── __init__.py
│   │   ├── matmul.py                  # sparse_mm            → tsgu::spmm, tsgu::sddmm
│   │   ├── triangular_solve.py        # sparse_triangular_solve → tsgu::spsm (+ sddmm bwd)
│   │   ├── generic_solve.py           # sparse_generic_solve (host loop + sddmm bwd)
│   │   ├── lstsq.py                   # sparse_generic_lstsq
│   │   ├── logsumexp.py               # sparse_logsumexp, sparse_bidir_logsumexp → tsgu::seglse*
│   │   └── indexed_matmul.py          # segment_mm, gather_mm → tsgu::grouped_gemm
│   ├── solvers/                       # host-side Krylov loops, batched (B,n,p) iterates
│   │   ├── __init__.py
│   │   ├── cg.py
│   │   ├── bicgstab.py
│   │   ├── lsmr.py
│   │   └── minres.py
│   ├── distributions/
│   │   ├── __init__.py
│   │   └── sparse_multivariate_normal.py
│   ├── encoders/
│   │   ├── __init__.py
│   │   └── pairwise_encoder.py        # no deprecated aliases
│   └── utils/
│       ├── __init__.py
│       ├── convert.py                 # convert_coo_to_csr* → tsgu::coo2csr; stack_csr; sparse_eye
│       ├── random_sparse.py           # generators (host)
│       └── dist_stats_helpers.py      # Hotelling T², Nagao (host)
├── cuda/                              # ── torchsparsegradutils-cuda package ──
│   ├── build.toml                     # kernel-builder config (or pyproject.toml if wheels win — build.md)
│   ├── flake.nix
│   ├── torchsparsegradutils_cuda/
│   │   └── __init__.py                # loads the extension, __backend_api_version__
│   ├── bench/                         # NVBench microbenchmarks — one target per kernel, lands with the kernel (day-one)
│   └── csrc/
│       ├── registration.cpp           # STABLE_TORCH_LIBRARY_IMPL(tsgu, CUDA, …) — all ops
│       ├── common/                    # infra shared by every kernel — not a kernel family
│       │   ├── batched_csr.cuh        # descriptor accessors, folded-row/batch math
│       │   ├── reduce.cuh             # warp/block reductions, online-max helpers
│       │   ├── dispatch.cuh           # TSGU_DISPATCH_VALUE / TSGU_DISPATCH_INDEX macros
│       │   └── stream.cuh             # CUDAGuard + current-stream launch plumbing
│       └── kernels/                   # one dir per family/op — mirrors kernels.md
│           ├── sddmm/
│           │   └── sddmm.cu           # Family 1 (fused negate/scale epilogues)
│           ├── logsumexp/
│           │   ├── seglse.cu          # Family 2 forward + backward
│           │   └── seglse_bidir.cu    # fused row+col traversal
│           ├── spmm/
│           │   └── spmm.cu            # Family 3: warp-per-folded-row, column-tiled
│           ├── spsm/
│           │   ├── spsm.cu            # triangular solve (v1 may wrap cuSPARSE — goal.md scaffold rule)
│           │   └── plan.cpp           # analysis-plan object owned by BatchedCSR lazy member
│           ├── convert/
│           │   └── coo2csr.cu         # fused sort+compress
│           └── grouped_gemm/
│               └── grouped_gemm.cu    # segment_mm/gather_mm, gather fused in prologue
├── tests/                             # differential vs oracle, opcheck, gradcheck (testing.md)
├── benchmarks/                        # beat-cuSPARSE evidence (benchmarks.md)
└── docs/
```

Notable deltas vs today: `sparse_matmul.py`/`sparse_solve.py`/… → `ops/`;
`utils/utils.py` (912 lines) dissolves — block-diag pair deleted, the rest into
`utils/convert.py`; `cupy/`, `jax/`, `pairwise_voxel_encoder.py` deleted
(commit.md Commit 2); tests/benchmarks move to repo root (front package stays
import-light).

## 6. Open questions

None.

Resolved: §4 CPU story → CUDA-required + oracle outside the wheel (testing.md).
Distribution → Hub-only via kernel-builder for now; wheels parked
post-migration (build.md). `segment_mm`/`gather_mm` → bypass `BatchedCSR`
entirely — confirmed against DGL/pyg-lib signatures: dense `a (N,D1)`,
dense `b (R,D1,D2)`, integer `seglen`/`idx_b`; nothing sparse touches these
ops. Namespace → `tsgu::` confirmed.
