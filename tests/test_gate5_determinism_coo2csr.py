"""Gate 5 — determinism + adversarial structure for tsgu::coo2csr
(spec/testing.md "Gates & ordering": "property + adversarial + determinism";
spec/commit.md Phase 3 commit 19).

The kernel's contract is a *deterministic two-pass stable radix sort* into
``(row_global, col)`` lexicographic order (identical ordering to
_batched.py's ``_fold_coo_to_csr``) — no atomics-order dependence, so
kernels.md open Q1's policy requires: run-twice bitwise equality, and the
same under ``torch.use_deterministic_algorithms(True)`` (the only path
already qualifies; no separate slow path needed).

Stability is itself part of the contract and only observable under
duplicate coordinates (equal sort keys): duplicates are the *caller's*
responsibility semantically (no dedup is performed — a valid pattern has
none), but the kernel's ordering of them must still be the stable
input-order tie-break, bitwise reproducible across runs.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate5

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


def _random_scattered(index_dtype, *, seed, B=4, n=6, m=5, density=0.5):
    gen = torch.Generator().manual_seed(seed)
    coords = []
    for b in range(B):
        n_item = max(1, int(density * n * m))
        flat = torch.randperm(n * m, generator=gen)[:n_item]
        coords.extend((b, int(f) // m, int(f) % m) for f in flat)
    order = torch.randperm(len(coords), generator=gen).tolist()
    coords = [coords[i] for i in order]
    batch = torch.tensor([c[0] for c in coords], dtype=index_dtype, device="cuda")
    row = torch.tensor([c[1] for c in coords], dtype=index_dtype, device="cuda")
    col = torch.tensor([c[2] for c in coords], dtype=index_dtype, device="cuda")
    return batch, row, col, B, n


def _assert_run_twice_bitwise_equal(batch, row, col, B, n):
    out1 = torch.ops.tsgu.coo2csr(batch, row, col, B, n)
    out2 = torch.ops.tsgu.coo2csr(batch, row, col, B, n)
    for a, b, name in zip(out1, out2, ("rowptr", "col_sorted", "permutation")):
        assert torch.equal(a, b), f"tsgu::coo2csr {name} differs between identical runs (stable radix sort contract)"


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_determinism_run_twice_bitwise_equal(index_dtype):
    batch, row, col, B, n = _random_scattered(index_dtype, seed=42)
    _assert_run_twice_bitwise_equal(batch, row, col, B, n)


@requires_cuda_backend
def test_determinism_under_use_deterministic_algorithms():
    batch, row, col, B, n = _random_scattered(torch.int64, seed=43)
    torch.use_deterministic_algorithms(True)
    try:
        _assert_run_twice_bitwise_equal(batch, row, col, B, n)
    finally:
        torch.use_deterministic_algorithms(False)


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_stability_duplicate_coordinates_keep_input_order(index_dtype):
    """Equal (row_global, col) keys — a STABLE sort must keep them in input
    order, so the permutation is fully determined even here (no dedup: all
    duplicates survive into col_sorted/permutation)."""
    # Entries 1, 3, 4 are the same coordinate (b=0, r=1, c=2).
    batch = torch.tensor([0, 0, 1, 0, 0], dtype=index_dtype, device="cuda")
    row = torch.tensor([2, 1, 0, 1, 1], dtype=index_dtype, device="cuda")
    col = torch.tensor([0, 2, 1, 2, 2], dtype=index_dtype, device="cuda")
    rowptr, col_sorted, permutation = torch.ops.tsgu.coo2csr(batch, row, col, 2, 3)

    # No dedup: every entry (duplicates included) is present.
    assert col_sorted.shape == (5,)
    assert int(rowptr[-1]) == 5
    # Stable tie-break: the three duplicates appear in input order 1, 3, 4,
    # ahead of row 2's entry (0) within batch 0, and entry 2 is batch 1's.
    assert permutation.long().tolist() == [1, 3, 4, 0, 2]
    assert col_sorted.long().tolist() == [2, 2, 2, 0, 1]

    _assert_run_twice_bitwise_equal(batch, row, col, 2, 3)


@requires_cuda_backend
def test_determinism_adversarial_all_one_row():
    """Every entry in one folded row (single long segment) — the two-pass
    sort degenerates to a pure col sort; still bitwise reproducible."""
    gen = torch.Generator().manual_seed(44)
    m = 64
    col = torch.randperm(m, generator=gen).to(torch.int64).cuda()
    row = torch.full_like(col, 2)
    batch = torch.zeros_like(col)
    _assert_run_twice_bitwise_equal(batch, row, col, 1, 5)
