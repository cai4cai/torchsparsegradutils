"""Every marked call must fail; an erased shape or permissive fallback fails CI."""

from __future__ import annotations

import torch
from jaxtyping import Shaped
from torch import Tensor

from torchsparsegradutils import (
    SparseCOOTensor,
    SparseCSRTensor,
    is_sparse_csr,
    require_sparse_coo,
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


# No inherited broadcasting method may keep the input's shape and brand.
coo = csr.to_sparse_coo()
broadcast_rhs = torch.ones(2, 3, 4)
coo.mul(torch.ones(5, 4))  # E: Cannot broadcast dimension
sparse_mm(coo.add(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.sub(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.mul(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.div(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.pow(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.remainder(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.eq(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.ne(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.lt(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.le(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.gt(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.ge(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.logical_and(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.logical_or(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.atan2(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.hypot(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.fmod(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.copysign(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.nextafter(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.bitwise_and(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.bitwise_or(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.bitwise_xor(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.bitwise_left_shift(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.bitwise_right_shift(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.maximum(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.minimum(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.fmax(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.fmin(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo**broadcast_rhs, rhs)  # E: No matching overload
sparse_mm(coo.isclose(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.lerp(broadcast_rhs, 0.5), rhs)  # E: No matching overload
sparse_mm(coo.masked_fill(broadcast_rhs, 1.0), rhs)  # E: No matching overload
sparse_mm(coo.masked_scatter(broadcast_rhs, torch.ones(24)), rhs)  # E: No matching overload
sparse_mm(coo.clamp_min(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.clamp_max(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.clamp(broadcast_rhs), rhs)  # E: No matching overload
sparse_mm(coo.clip(broadcast_rhs), rhs)  # E: No matching overload
# Validating the output layout must preserve its new rank and dimensions.
broadcast_coo = require_sparse_coo(coo.mul(broadcast_rhs))
sparse_mm(broadcast_coo, rhs)  # E: No matching overload
sparse_mm(broadcast_coo, torch.ones(2, 5, 2))  # E: No matching overload
