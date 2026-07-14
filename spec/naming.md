# Naming — the backbone

Derived from `docs/source/naming.rst` (the published conventions, which remain
authoritative for user-facing docs). This copy exists so the rules travel with
the migration — every spec doc, kernel comment, op definition, error message,
commit message, and review on this branch follows it — **plus the rewrite-specific
vocabulary in §2 that naming.rst doesn't cover yet**. When the rewrite ships,
§2 merges back into naming.rst.

## 1. The rules (condensed from naming.rst)

- **Matrix vs tensor** — "matrix" is the logical row/column object; "tensor" is
  the PyTorch object. State the logical object first, then its shape:
  "a batched sparse matrix with shape `(batch_size, n_rows, n_cols)`", never
  "a 3D sparse tensor". `2D`/`rank 2` describe axis count only.
- **One leading batch axis**; batch sizes match exactly; no broadcasting unless
  an API documents it.
- **Dimension words are not interchangeable** — tensor rank (`ndim`), batch
  dimension (indexes problem instances), sparse dimension (indexed by sparse
  indices), dense dimension of a sparse tensor (trailing per-value axis ≠ a
  dense strided tensor).
- **COO vs compressed batching differ** — COO batching is a third sparse dim
  *logically interpreted* as batch; items may have unequal nse (even zero).
  Batched CSR/CSC require equal nse per item — a storage constraint, not math.
  `_nnz()`: per batch item for CSR, whole tensor for COO.
- **Specified entries, not nonzeros** — prefer *specified entry* / *nse*;
  explicit zero (stored 0) ≠ structural zero (absent coordinate). `values()`
  are "stored values", not "nonzero values".
- **Layout names** — PyTorch's: sparse COO/CSR/CSC/BSR/BSC, strided. Say
  "layout", not "format". Index arrays named for the represented axis:
  COO `row_indices`/`col_indices`; CSR `crow_indices`/`col_indices`; CSC
  `ccol_indices`/`row_indices`. Compressed pointers are **uncompressed /
  expanded / repeat-interleaved** into coordinates — never "decompressed rows".
- **Shapes** — mm: `A (n_rows, n_inner) @ B (n_inner, n_cols)`; solve:
  `A (n, n)`, `B (n, n_rhs)` — B is the **right-hand side**, its last axis is
  n_rhs, not a batch. `A.mT` for batched transpose, never `A.T`.
- **Reductions** — name the reduced *and* retained axes; avoid bare
  "row-wise/column-wise".
- **Distributions** — `sample_shape + batch_shape + event_shape`; sample ≠
  batch; spatial/channel/batch axes distinct.
- **Memory words** — view (shared storage), copy (new storage), reshape
  (either). "Materialise dense" = construct a full dense tensor.
- **Coalescing is COO-only.** Coalesce before nonlinear ops on values unless
  duplicate handling is proven equivalent.
- **Errors** state the accepted logical forms and the received shape:
  `"A must be … (n_rows, n_cols) or … (batch_size, n_rows, n_cols); got {shape}."`

## 2. Rewrite extensions (new vocabulary this branch introduces)

### BatchedCSR descriptor ([architecture.md](architecture.md) §3)

| Term | Meaning |
|------|---------|
| **descriptor** | A `BatchedCSR`/`BatchedCSC` instance. Never call it a tensor — it holds tensors |
| **folded row** | `row_global = b · n_rows + r`; "folded" is the word — not "flattened", "blocked", or "stacked" |
| **local column** | column index in `[0, n_cols)`, *not* offset by batch — say "local" whenever ambiguity is possible |
| **`nse_total`** | specified entries summed over the whole batch (COO-style whole-tensor count) |
| **`nse_per_item`** | specified entries of one batch item; ragged: may differ per item |
| **`B = 1` encodes unbatched** | an unbatched matrix is a batch of one — code paths never branch on "has batch dim" |

### Kernel-side short names

naming.rst permits short local names in compact numerical code. In C++/CUDA and
op schemas the fixed mapping is:

| Full (Python-visible) | Kernel/schema short form |
|-----------------------|--------------------------|
| `crow_indices` (folded, absolute) | `rowptr` |
| `col_indices` (local) | `col` |
| `values` / stored values | `vals` |
| `batch_size, n_rows, n_cols, n_rhs` | `B, n, m, p` |
| folded row index | `row_g` |
| batch index of an entry | `b` |

The mapping is one-way and fixed: a kernel may use the short form, but any
Python-facing surface (public API, error messages, docs) uses the full names.

### `tsgu::` op names

`spmm`, `sddmm`, `spsm`, `seglse`, `seglse_bwd`, `seglse_bidir`,
`seglse_bidir_bwd`, `coo2csr`, `grouped_gemm` — snake_case, matching the
`csrc/kernels/` directory that owns them. An op name states the operation, not
the caller (`tsgu::sddmm`, not `tsgu::sparse_mm_backward`). A dedicated backward
kernel takes the `_bwd` suffix on its forward's name — but only when the
backward is genuinely its own kernel (`seglse_bwd`); backwards expressed through
existing ops reuse those ops (`sddmm`, transposed `spmm`), no alias names.
Full public-API → op routing: [map.md](map.md) "Kernel routing".

## 3. Enforcement through the migration

- Every kernel file header comment states the logical objects and shapes of its
  arguments in naming.rst form before any implementation detail.
- Shape-validation errors in `ops/` wrappers follow the §1 error template.
- Review checklist item (pre-PR): grep new code for "3D sparse", "nonzero
  values", "format", `.T` on batched matrices — all four are defect signals.
