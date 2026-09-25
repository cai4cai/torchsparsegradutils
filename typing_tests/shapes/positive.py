"""Static consumer contracts: both the layout and exact shape must survive."""

from __future__ import annotations

import torch
from jaxtyping import Shaped
from torch import Tensor
from typing_extensions import assert_type

from torchsparsegradutils import (
    SparseCOOTensor,
    SparseCSCTensor,
    SparseCSRTensor,
    gather_mm,
    is_sparse_coo,
    is_sparse_csr,
    require_sparse_coo,
    require_sparse_coo_or_csr,
    require_sparse_csc,
    require_sparse_csr,
    sparse_generic_lstsq,
    sparse_generic_solve,
    sparse_mm,
    sparse_triangular_solve,
)
from torchsparsegradutils.utils import rand_sparse, rand_sparse_tri
from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix,
    generate_random_sparse_csr_matrix,
    generate_random_sparse_strictly_triangular_coo_matrix,
    generate_random_sparse_strictly_triangular_csr_matrix,
    generate_random_sparse_triangular_coo_matrix,
    generate_random_sparse_triangular_csr_matrix,
    make_spd_sparse,
)

coo = generate_random_sparse_coo_matrix((3, 4), 5)
csr = generate_random_sparse_csr_matrix((3, 4), 5)
rhs = torch.randn(4, 2)
assert_type(coo, SparseCOOTensor[[3, 4]])
assert_type(csr, SparseCSRTensor[[3, 4]])
assert_type(sparse_mm(coo, rhs), Tensor[[3, 2]])
assert_type(sparse_mm(csr, rhs), Tensor[[3, 2]])
assert_type(require_sparse_csr(csr), SparseCSRTensor[[3, 4]])
assert_type(SparseCSRTensor(csr), SparseCSRTensor[[3, 4]])
assert_type(require_sparse_coo_or_csr(csr), SparseCOOTensor[[3, 4]] | SparseCSRTensor[[3, 4]])
assert_type(csr.clone(), SparseCSRTensor[[3, 4]])
assert_type(csr.detach(), SparseCSRTensor[[3, 4]])
assert_type(csr.to_dense(), Tensor[[3, 4]])
assert_type(csr.to_sparse_coo(), SparseCOOTensor[[3, 4]])
assert_type(csr.to_sparse_csc(), SparseCSCTensor[[3, 4]])
assert_type(require_sparse_csc(csr.transpose(0, 1)), SparseCSCTensor[[4, 3]])
assert_type(csr.T, Tensor)
assert_type(gather_mm(torch.randn(3, 4), torch.randn(2, 4, 5), torch.tensor([0, 1, 0])), Tensor)

assert_type(rand_sparse((3, 4), 5), SparseCOOTensor[[3, 4]])
assert_type(rand_sparse((3, 4), 5, layout=torch.sparse_csr), SparseCOOTensor[[3, 4]] | SparseCSRTensor[[3, 4]])
assert_type(rand_sparse_tri((4, 4), 5), SparseCOOTensor[[4, 4]])
assert_type(generate_random_sparse_strictly_triangular_coo_matrix((4, 4), 2), SparseCOOTensor[[4, 4]])
assert_type(generate_random_sparse_strictly_triangular_csr_matrix((4, 4), 2), SparseCSRTensor[[4, 4]])
assert_type(generate_random_sparse_triangular_coo_matrix((4, 4), 5), SparseCOOTensor[[4, 4]])
assert_type(generate_random_sparse_triangular_csr_matrix((4, 4), 5), SparseCSRTensor[[4, 4]])
spd, dense_spd = make_spd_sparse(4, torch.sparse_csr, torch.float32, torch.int64, torch.device("cpu"))
assert_type(spd, SparseCOOTensor[[4, 4]] | SparseCSRTensor[[4, 4]])
assert_type(dense_spd, Tensor[[4, 4]])

batch_coo = generate_random_sparse_coo_matrix((2, 3, 4), 5)
batch_csr = generate_random_sparse_csr_matrix((2, 3, 4), 5)
assert_type(sparse_mm(batch_coo, torch.randn(2, 4, 5)), Tensor[[2, 3, 5]])
assert_type(sparse_mm(batch_csr, torch.randn(2, 4, 5)), Tensor[[2, 3, 5]])


def guarded(value: Shaped[Tensor, "3 4"]) -> None:
    if is_sparse_csr(value):
        assert_type(value, SparseCSRTensor[[3, 4]])
        assert_type(sparse_mm(value, rhs), Tensor[[3, 2]])
    if is_sparse_coo(value):
        assert_type(value, SparseCOOTensor[[3, 4]])


def solves(a: Shaped[SparseCSRTensor, "4 4"], batch: Shaped[SparseCOOTensor, "2 4 4"]) -> None:
    assert_type(sparse_triangular_solve(a, torch.randn(4, 2)), Tensor[[4, 2]])
    assert_type(sparse_triangular_solve(batch, torch.randn(2, 4, 2)), Tensor[[2, 4, 2]])
    assert_type(sparse_generic_solve(a, torch.randn(4)), Tensor[[4]])
    assert_type(sparse_generic_solve(a, torch.randn(4, 2)), Tensor[[4, 2]])


def least_squares(a: Shaped[SparseCOOTensor, "5 3"]) -> None:
    assert_type(sparse_generic_lstsq(a, torch.randn(5)), Tensor[[3]])
    assert_type(sparse_generic_lstsq(a, torch.randn(5, 2)), Tensor[[3, 2]])


def symbolic(a: Shaped[SparseCSRTensor, "m n"], b: Shaped[Tensor, "n k"]) -> Shaped[Tensor, "m k"]:
    return sparse_mm(a, b)


assert_type(symbolic(csr, rhs), Tensor[[3, 2]])

# Tensor operands can broadcast. These static assertions do not certify
# PyTorch kernel support for a particular sparse layout or dtype.
broadcast_rhs = torch.ones(2, 3, 4)
assert_type(coo * broadcast_rhs, Tensor[[2, 3, 4]])
assert_type(coo.add(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.sub(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.mul(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.div(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.pow(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.remainder(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.eq(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.ne(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.lt(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.le(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.gt(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.ge(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.logical_and(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.logical_or(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.atan2(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.hypot(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.fmod(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.copysign(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.nextafter(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.bitwise_and(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.bitwise_or(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.bitwise_xor(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.bitwise_left_shift(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.bitwise_right_shift(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.maximum(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.minimum(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.fmax(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.fmin(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo**broadcast_rhs, Tensor[[2, 3, 4]])
assert_type(coo.isclose(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.lerp(broadcast_rhs, 0.5), Tensor[[2, 3, 4]])
assert_type(coo.masked_fill(broadcast_rhs, 1.0), Tensor[[2, 3, 4]])
assert_type(coo.masked_scatter(broadcast_rhs, torch.ones(24)), Tensor[[2, 3, 4]])
assert_type(coo.clamp_min(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.clamp_max(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.clamp(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.clip(broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.clamp(max=broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.clamp(None, broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.clip(max=broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.clip(None, broadcast_rhs), Tensor[[2, 3, 4]])
assert_type(coo.clamp(broadcast_rhs, torch.ones(5, 1, 1, 1)), Tensor[[5, 2, 3, 4]])
assert_type(coo.clip(broadcast_rhs, torch.ones(5, 1, 1, 1)), Tensor[[5, 2, 3, 4]])
assert_type(coo.mul(2), SparseCOOTensor[[3, 4]])
assert_type(coo.div(2), SparseCOOTensor[[3, 4]])
assert_type(coo.pow(2), SparseCOOTensor[[3, 4]])
assert_type(coo.remainder(2), SparseCOOTensor[[3, 4]])
assert_type(coo**2, SparseCOOTensor[[3, 4]])
assert_type(coo.clamp(0, 1), Tensor[[3, 4]])
assert_type(coo.clip(0, 1), Tensor[[3, 4]])
assert_type(coo.clamp_min(0), Tensor[[3, 4]])
assert_type(coo.clamp_max(1), Tensor[[3, 4]])
broadcast_coo = require_sparse_coo(coo.mul(broadcast_rhs))
assert_type(broadcast_coo, SparseCOOTensor[[2, 3, 4]])
assert_type(sparse_mm(broadcast_coo, torch.ones(2, 4, 2)), Tensor[[2, 3, 2]])
