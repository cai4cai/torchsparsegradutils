# Kernel Design — the three families

One doc, three families. Split a family out only if implementation detail
outgrows its section. Contract per op: [map.md](map.md) §1. Goal: [goal.md](goal.md).

## Shared rules (all kernels)

- **Variant matrix:** COO + CSR (CSC only where the contract requires it:
  logsumexp) × unbatched/batched (one leading batch axis) × float32/float64 ×
  int32/int64 indices. Index dtype handled natively — never silently upcast
  (differentiator: torch COO can't hold int32, our CSR paths must exploit it).
- **No dense materialisation, ever.** Workspace bounded by O(nse) or O(n_rows).
- **Stream-safe:** run on the current CUDA stream; no host syncs or round-trips
  in the hot path.
- **Registration:** as torch ops via `torch.library` so autograd and
  `torch.compile` compose (mechanics in architecture.md).
- **Determinism policy (open):** atomics-based paths are non-deterministic;
  decide whether we honour `torch.use_deterministic_algorithms(True)` with a
  slower sorted/scan path per kernel, or document non-determinism.
- **Half precision (decided, phased):**
  - SpMM / SDDMM / grouped GEMM: fp16 + bf16 values with **fp32 accumulation**
    via tensor-core paths — **v2**, benchmark-gated, with its own acceptance bar
    against cuSPARSE's half/tensor-core paths (a CUDA-core half kernel would
    lose ~2–4× to TC-backed cuSPARSE; don't ship a losing dtype). v1 is f32/f64.
  - seglse family: accepts fp16/bf16 *storage*, always reduces/accumulates in
    fp32 — numerical stability is the op's purpose. Cheap (bandwidth-bound, no
    tensor cores), so this lands in v1.
  - SpSM + Krylov solvers: **f32/f64 only, by policy** — half-precision
    triangular/iterative solves are numerically meaningless; documented as out
    of scope, not missing.
  - Gradients match input dtype; parity/gradcheck runs use f64.

## Family 1 — SDDMM (the shared backward)

`gradA[i,j] = dot(G[i,:], B[j,:])` evaluated **only at A's specified entries**.
One kernel family serves the backward of four ops: `sparse_mm`,
`sparse_triangular_solve` (as −X·gradBᵀ sampled at pattern), `sparse_generic_solve`,
`sparse_generic_lstsq`.

- **Shape:** G `(n,p)` or `(b,n,p)`, B `(m,p)`; output values aligned with A's
  index arrays (layout + pattern reuse — zero index allocation).
- **Parallelisation:** one warp per specified entry, warp-reduce over `p`
  (good for small p); tile G rows through shared memory for large p. Row-major
  CSR gives G-row reuse for free; COO path must not assume sorted order unless
  coalesced input is guaranteed by the calling op.
- **Fusions:** negate-and-scale folded in for the solve backwards (avoid a
  second pass over nse); optional epilogue writing straight into the output
  values buffer of the grad tensor.
- **Baseline:** cuSPARSE `cusparseSDDMM` (CSR, unbatched). Batched and COO have
  **no vendor primitive** — the comparison there is the current block-diag
  workaround, and the win must be decisive.

## Family 2 — Segmented logsumexp (flagship)

`sparse_logsumexp` + `sparse_bidir_logsumexp`. No vendor primitive, no maintained
CUDA competitor on native sparse layouts. Semantics frozen by map.md §1:
`dim`, `include_zeros` (structural zeros contribute to the reduction per current
docs), output shape mirrors `torch.logsumexp`; bidir must equal two single-dim
calls exactly.

- **Forward, two candidate strategies (decide by benchmark):**
  1. *Two-pass segmented:* segmented max, then segmented sum of shifted exps.
     Deterministic given sorted segments; CSR rows are sorted for free, COO
     `dim=0` / CSC give the other axis.
  2. *One-pass online:* running max with rescale (online-softmax style) via
     atomics — fewer passes, non-deterministic, needs the float atomic-max trick.
- **Bidirectional:** the entire point is a **single traversal updating row and
  column accumulators together** — one read of values/indices instead of two.
  This is where the fusion win over two separate calls lives; `tuple`/`padded`/
  `nested` output layouts assembled host-side, outside the kernel.
- **Backward:** embarrassingly parallel per specified entry:
  `gradA_val = exp(v − lse[seg]) · gout[seg]`. Save `lse` from forward (no
  recompute); accumulate in fp32.
- **Baseline:** none exists — acceptance is (a) large multiple over the current
  pure-PyTorch scatter path, (b) beat `pytorch_scatter.scatter_logsumexp` on
  equivalent index-array inputs (it's the only comparable CUDA code, legacy or not).

## Family 3 — Vendor-baseline forwards

Our own kernels for ops where a vendor primitive exists and is the bar.

| Kernel | Serves | Baseline | Our edge |
|--------|--------|----------|----------|
| SpMM (+ SpMV as p=1) | `sparse_mm` fwd, gradB, all iterative solvers | cuSPARSE SpMM/SpMV | Native batched (vendor has none — beats block-diag), int32 COO, merge-path load balancing for skewed rows |
| Triangular SpSM | `sparse_triangular_solve` fwd + its backward solve | cuSPARSE SpSM | Analysis-phase reuse: cache level-schedule per sparsity pattern (keyed on index tensors) across calls — amortised wins in iterative/rsample loops; vendor API makes reuse awkward |
| coo2csr | `convert_coo_to_csr*`, internal layout switches | cuSPARSE `Xcoo2csr` + thrust sort | Fused sort+compress, int32 native |
| Grouped GEMM | `segment_mm`, `gather_mm` | cuBLAS `cublasGemmGroupedBatched` | Fuse the gather into the GEMM prologue (`gather_mm`); skip materialising gathered operand |

- **Acceptance (from map.md):** ≥ parity with the vendor primitive unbatched on
  the SuiteSparse suite; clear win on batched/COO paths.
- **SpSM analysis caching is the one stateful design point** — decide the cache
  key and invalidation story (weak-ref on index tensors?) in architecture.md.

## Open questions

None.

Resolved: fp16/bf16 policy → shared rules above. SpSM analysis cache → lazy
member on the `BatchedCSR` descriptor (architecture.md §3). Determinism →
**both**: a kernel may ship an atomic fast path (documented non-deterministic,
per kernel) *and* must engage a deterministic path under
`torch.use_deterministic_algorithms(True)`; run-twice tests cover each
(testing.md). SpMM sequencing → implementer's choice; what protects the goal is
benchmark provenance, not ordering — every benchmark row records
`custom` vs `vendor-scaffold` backend, and scaffold rows never count toward
beat-cuSPARSE claims (benchmarks.md). Family-3 "our edge" items → pursued
opportunistically, cleanliness first; each edge is claimed only when its
benchmark row exists.
