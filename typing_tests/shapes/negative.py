"""Every marked call must fail; an erased shape or permissive fallback fails CI."""

from __future__ import annotations

import torch
from jaxtyping import Shaped
from torch import Tensor

from torchsparsegradutils import (
    SparseCOOTensor,
    SparseCSRTensor,
    is_sparse_csr,
    sparse_generic_lstsq,
    sparse_generic_solve,
    sparse_mm,
    sparse_triangular_solve,
)
from torchsparsegradutils.utils.random_sparse import generate_random_sparse_csr_matrix

csr = generate_random_sparse_csr_matrix((3, 4), 5)
rhs = torch.randn(4, 2)
wrong = torch.randn(5, 2)

sparse_mm(csr, wrong)  # E: No matching overload
sparse_mm(csr.clone(), wrong)  # E: No matching overload
sparse_mm(csr.to_sparse_coo(), wrong)  # E: No matching overload
sparse_mm(csr.to_sparse_csc(), rhs)  # E: No matching overload
sparse_mm(torch.randn(3, 4), rhs)  # E: No matching overload
sparse_mm(csr.T, torch.randn(3, 2))  # E: No matching overload
sparse_triangular_solve(csr, torch.randn(3, 2))  # E: No matching overload
sparse_generic_solve(csr, torch.randn(3, 2))  # E: No matching overload
sparse_generic_lstsq(csr, rhs)  # E: No matching overload
sparse_mm(csr, torch.randn(4))  # E: No matching overload


def guarded(value: Shaped[Tensor, "3 4"]) -> None:
    if is_sparse_csr(value):
        sparse_mm(value, wrong)  # E: No matching overload


def batch_errors(a: Shaped[SparseCOOTensor, "2 3 4"], square: Shaped[SparseCSRTensor, "2 4 4"]) -> None:
    sparse_mm(a, torch.randn(3, 4, 2))  # E: No matching overload
    sparse_mm(a, rhs)  # E: No matching overload
    sparse_mm(a, torch.randn(2, 5, 2))  # E: No matching overload
    sparse_triangular_solve(square, torch.randn(3, 4, 2))  # E: No matching overload
    sparse_generic_solve(square, torch.randn(2, 4, 2))  # E: No matching overload


def rank_errors(a: Shaped[SparseCOOTensor, "1 2 3 4"]) -> None:
    sparse_mm(a, torch.randn(1, 2, 4, 2))  # E: No matching overload
