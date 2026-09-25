"""Calls that Pyrefly must reject as part of the public typing contract."""

import torch

from torchsparsegradutils import (
    require_sparse_csc,
    sparse_generic_lstsq,
    sparse_generic_solve,
    sparse_logsumexp,
    sparse_mm,
    sparse_triangular_solve,
)

dense = torch.eye(3)
csc = require_sparse_csc(dense.to_sparse_csc())

sparse_mm(dense, dense)  # E: Argument `Tensor` is not assignable to parameter `A`
sparse_triangular_solve(dense, dense)  # E: Argument `Tensor` is not assignable to parameter `A`
sparse_generic_solve(dense, dense)  # E: Argument `Tensor` is not assignable to parameter `A`
sparse_generic_lstsq(dense, dense)  # E: Argument `Tensor` is not assignable to parameter `A`
sparse_generic_solve(csc, dense)  # E: Argument `SparseCSCTensor` is not assignable to parameter `A`
sparse_logsumexp(dense, dim=0)  # E: Argument `Tensor` is not assignable to parameter `input`
