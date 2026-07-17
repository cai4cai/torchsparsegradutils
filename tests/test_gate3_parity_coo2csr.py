"""Gate 3 — parity for tsgu::coo2csr against the pure-torch reference
(spec/testing.md "Gates & ordering": "parity"; spec/commit.md Phase 3 commit
19).

The reference implements the op's ordering contract independently
(torchsparsegradutils/_batched.py ``_fold_coo_to_csr``'s composition): a
two-pass **stable** argsort — secondary key ``col`` first, then primary key
``row_global = b * n + r`` — i.e. a stable lexicographic sort by
``(row_global, col)``, then a bincount+cumsum compress. No dedup: duplicate
coordinates would be kept in stable input order (valid patterns have none;
parity inputs here are duplicate-free).

Comparison policy: ``rowptr`` and ``col_sorted`` are compared EXACTLY. The
``permutation`` is validated semantically — ``values[permutation]`` must
match the reference's values ordering — rather than demanding bit-identity
with any particular sort's tie-breaking. (On duplicate-free input the
``(row_global, col)`` key is unique, so the permutation is in fact unique
too — but the semantic check is the contract, and it is also what the
legacy ``_sort_coo_indices``'s ``torch.unique``-based permutation cannot be
trusted to match under duplicates.)
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate3

_SKIP_REASON = (
    "tsgu::coo2csr has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)


def _coo2csr_cuda_ready() -> bool:
    # Dispatch-key check: skip (not error) until the cuda/ tree is rebuilt
    # with the convert kernel (commit 19 is developed in two concurrent lanes).
    return (
        torch.cuda.is_available()
        and backend_available()
        and torch._C._dispatch_has_kernel_for_dispatch_key("tsgu::coo2csr", "CUDA")
    )


requires_cuda_backend = pytest.mark.skipif(not _coo2csr_cuda_ready(), reason=_SKIP_REASON)

INDEX_DTYPES = (torch.int32, torch.int64)


def _reference_coo2csr(batch, row, col, B, n):
    """Pure-torch reference for the op's ordering contract (see module
    docstring). Computed in int64 throughout; callers cast for dtype
    comparisons."""
    row_global = batch.long() * n + row.long()
    permutation_secondary = torch.argsort(col.long(), stable=True)
    permutation_primary = torch.argsort(row_global[permutation_secondary], stable=True)
    permutation = permutation_secondary[permutation_primary]
    counts = torch.bincount(row_global[permutation], minlength=B * n) if row_global.numel() else None
    rowptr = torch.zeros(B * n + 1, dtype=torch.int64, device=batch.device)
    if counts is not None:
        rowptr[1:] = counts.cumsum(0)
    return rowptr, col.long()[permutation], permutation


def _check_parity(batch, row, col, B, n, index_dtype):
    values = torch.randn(row.shape[0], dtype=torch.float64, device="cuda")
    rowptr, col_sorted, permutation = torch.ops.tsgu.coo2csr(batch, row, col, B, n)
    ref_rowptr, ref_col, ref_perm = _reference_coo2csr(batch, row, col, B, n)

    # Metadata: dtype preserved (map.md invariant 6), device, shapes.
    assert rowptr.dtype == index_dtype and col_sorted.dtype == index_dtype and permutation.dtype == index_dtype
    assert rowptr.shape == (B * n + 1,)
    assert col_sorted.shape == col.shape and permutation.shape == col.shape

    # rowptr / col_sorted: exact.
    assert torch.equal(rowptr.long(), ref_rowptr)
    assert torch.equal(col_sorted.long(), ref_col)

    # permutation: semantic — values reordered by it match the reference
    # ordering (see module docstring).
    assert torch.equal(values.index_select(0, permutation.long()), values.index_select(0, ref_perm))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_parity_unbatched_shuffled(index_dtype):
    gen = torch.Generator().manual_seed(0)
    n, m = 7, 6
    flat = torch.randperm(n * m, generator=gen)[:20]
    row = (flat // m).to(index_dtype).cuda()
    col = (flat % m).to(index_dtype).cuda()
    batch = torch.zeros_like(row)
    _check_parity(batch, row, col, 1, n, index_dtype)


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_parity_batched_shuffled_ragged(index_dtype):
    """Ragged nse per item, scattered input order, some rows empty."""
    gen = torch.Generator().manual_seed(1)
    B, n, m = 3, 5, 4
    coords = []
    for b, n_item in enumerate((7, 2, 11)):
        flat = torch.randperm(n * m, generator=gen)[:n_item]
        coords.extend((b, int(f) // m, int(f) % m) for f in flat)
    order = torch.randperm(len(coords), generator=gen).tolist()
    coords = [coords[i] for i in order]
    batch = torch.tensor([c[0] for c in coords], dtype=index_dtype, device="cuda")
    row = torch.tensor([c[1] for c in coords], dtype=index_dtype, device="cuda")
    col = torch.tensor([c[2] for c in coords], dtype=index_dtype, device="cuda")
    _check_parity(batch, row, col, B, n, index_dtype)


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_parity_empty_batch_item_and_empty_rows(index_dtype):
    """Batch item 1 entirely empty; the occupied items leave empty rows."""
    coords = [(0, 2, 1), (0, 0, 3), (2, 4, 0), (2, 4, 2), (2, 0, 0)]
    batch = torch.tensor([c[0] for c in coords], dtype=index_dtype, device="cuda")
    row = torch.tensor([c[1] for c in coords], dtype=index_dtype, device="cuda")
    col = torch.tensor([c[2] for c in coords], dtype=index_dtype, device="cuda")
    _check_parity(batch, row, col, 3, 5, index_dtype)


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_parity_nse_zero(index_dtype):
    empty = torch.zeros(0, dtype=index_dtype, device="cuda")
    _check_parity(empty, empty.clone(), empty.clone(), 2, 4, index_dtype)


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_parity_single_entry(index_dtype):
    batch = torch.tensor([1], dtype=index_dtype, device="cuda")
    row = torch.tensor([2], dtype=index_dtype, device="cuda")
    col = torch.tensor([0], dtype=index_dtype, device="cuda")
    _check_parity(batch, row, col, 2, 3, index_dtype)


# ---------------------------------------------------------------------------
# Wrapper parity: convert_coo_to_csr_indices_values' kernel path vs its own
# pure-torch fallback (the frozen return convention must be identical).
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("batched", [False, True])
def test_wrapper_kernel_path_matches_fallback(monkeypatch, batched):
    import torchsparsegradutils.utils.convert as convert_mod
    from torchsparsegradutils.utils.convert import convert_coo_to_csr_indices_values

    gen = torch.Generator().manual_seed(2)
    if batched:
        B, n, m, nse_per_item = 3, 4, 5, 6  # equal nse per item (wrapper contract)
        coords = []
        for b in range(B):
            flat = torch.randperm(n * m, generator=gen)[:nse_per_item]
            coords.extend((b, int(f) // m, int(f) % m) for f in flat)
        indices = torch.tensor(coords, dtype=torch.int64, device="cuda").t().contiguous()
    else:
        n, m = 6, 5
        flat = torch.randperm(n * m, generator=gen)[:12]
        indices = torch.stack([flat // m, flat % m]).cuda()
    values = torch.randn(indices.shape[1], dtype=torch.float64, device="cuda")

    crow_op, col_op, vals_op = convert_coo_to_csr_indices_values(indices, n, values)

    monkeypatch.setattr(convert_mod, "_coo2csr_backend_ready", lambda _t: False)
    crow_ref, col_ref, vals_ref = convert_coo_to_csr_indices_values(indices, n, values)

    assert crow_op.shape == crow_ref.shape and crow_op.dtype == crow_ref.dtype
    assert torch.equal(crow_op, crow_ref)
    assert torch.equal(col_op, col_ref)
    assert torch.equal(vals_op, vals_ref)
