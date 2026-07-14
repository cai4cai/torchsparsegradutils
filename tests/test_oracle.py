"""Smoke test for tests/oracle (parity Oracle A, testing.md Pillar 1).

Imports every ``oracle_*`` symbol and runs one tiny numerical check per op
against the live ``torchsparsegradutils`` implementation. The oracle and the
live package are byte-for-byte the same algorithm today (the oracle is a
frozen copy taken at the moment of the split), so the two must agree exactly
within floating-point tolerance. This is *not* the full parity harness
(that arrives with commit 11) -- it only guards against the extraction
itself having introduced a discrepancy (e.g. a bad import rewrite).
"""

import torch

from tests.oracle import (
    oracle_gather_mm,
    oracle_segment_mm,
    oracle_sparse_bidir_logsumexp,
    oracle_sparse_generic_lstsq,
    oracle_sparse_generic_solve,
    oracle_sparse_logsumexp,
    oracle_sparse_mm,
    oracle_sparse_triangular_solve,
)
from torchsparsegradutils import (
    gather_mm,
    segment_mm,
    sparse_bidir_logsumexp,
    sparse_generic_lstsq,
    sparse_generic_solve,
    sparse_logsumexp,
    sparse_mm,
    sparse_triangular_solve,
)
from torchsparsegradutils.utils import linear_cg


def test_oracle_sparse_mm_matches_live():
    indices = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 2, 1, 3, 0, 2]])
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    A = torch.sparse_coo_tensor(indices, values, (3, 4))
    B = torch.randn(4, 2)

    live = sparse_mm(A, B)
    oracle = oracle_sparse_mm(A, B)
    assert torch.allclose(live, oracle)


def test_oracle_sparse_triangular_solve_matches_live():
    A = torch.sparse_csr_tensor([0, 2, 3, 4], [0, 2, 1, 2], torch.tensor([2.0, 1.0, 3.0, 1.0]), (3, 3))
    B = torch.tensor([[1.0], [2.0], [3.0]])

    live = sparse_triangular_solve(A, B, upper=True)
    oracle = oracle_sparse_triangular_solve(A, B, upper=True)
    assert torch.allclose(live, oracle)


def test_oracle_sparse_generic_solve_matches_live():
    indices = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 0, 1, 2]])
    values = torch.tensor([4.0, -1.0, -1.0, 4.0, 2.0])
    A = torch.sparse_coo_tensor(indices, values, (3, 3))
    B = torch.tensor([1.0, 2.0, 3.0])

    live = sparse_generic_solve(A, B, solve=linear_cg)
    oracle = oracle_sparse_generic_solve(A, B, solve=linear_cg)
    assert torch.allclose(live, oracle, atol=1e-5, rtol=1e-5)


def test_oracle_sparse_generic_lstsq_matches_live():
    indices = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3], [0, 0, 0, 0, 1, 1, 1, 2]])
    values = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0])
    A = torch.sparse_coo_tensor(indices, values, (5, 3)).coalesce()
    B = torch.randn(5)

    live = sparse_generic_lstsq(A, B)
    oracle = oracle_sparse_generic_lstsq(A, B)
    assert torch.allclose(live, oracle, atol=1e-4, rtol=1e-4)


def test_oracle_sparse_logsumexp_matches_live():
    i = torch.tensor([[0, 1, 1], [1, 0, 2]])
    v = torch.tensor([1.0, 2.0, 3.0])
    x = torch.sparse_coo_tensor(i, v, (3, 3))

    live = sparse_logsumexp(x, dim=1)
    oracle = oracle_sparse_logsumexp(x, dim=1)
    assert torch.allclose(live, oracle)


def test_oracle_sparse_bidir_logsumexp_matches_live():
    i = torch.tensor([[0, 1, 1], [1, 0, 2]])
    v = torch.tensor([1.0, 2.0, 3.0])
    x = torch.sparse_coo_tensor(i, v, (3, 3))

    live_col, live_row = sparse_bidir_logsumexp(x)
    oracle_col, oracle_row = oracle_sparse_bidir_logsumexp(x)
    assert torch.allclose(live_col, oracle_col)
    assert torch.allclose(live_row, oracle_row)


def test_oracle_segment_mm_matches_live():
    a = torch.randn(18, 4)
    b = torch.randn(3, 4, 2)
    seglen_a = torch.tensor([10, 5, 3])

    live = segment_mm(a, b, seglen_a)
    oracle = oracle_segment_mm(a, b, seglen_a)
    assert torch.allclose(live, oracle)


def test_oracle_gather_mm_matches_live():
    a = torch.randn(5, 3)
    b = torch.randn(3, 3, 2)
    idx_b = torch.tensor([0, 1, 0, 2, 1])

    live = gather_mm(a, b, idx_b)
    oracle = oracle_gather_mm(a, b, idx_b)
    assert torch.allclose(live, oracle)
