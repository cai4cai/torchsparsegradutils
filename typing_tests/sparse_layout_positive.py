"""Positive consumer contracts for the public sparse-layout types."""

from typing import Union

import torch
from typing_extensions import assert_type

from torchsparsegradutils import (
    SparseCOOTensor,
    SparseCSCTensor,
    SparseCSRTensor,
    is_sparse_coo,
    require_sparse_coo,
    require_sparse_csc,
    sparse_bidir_logsumexp,
    sparse_generic_lstsq,
    sparse_generic_solve,
    sparse_logsumexp,
    sparse_mm,
    sparse_triangular_solve,
)
from torchsparsegradutils.utils import convert_coo_to_csr, sparse_block_diag, sparse_block_diag_split, sparse_eye
from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix,
    generate_random_sparse_csr_matrix,
    rand_sparse,
)

coo = generate_random_sparse_coo_matrix((3, 3), 4)
csr = generate_random_sparse_csr_matrix((3, 3), 4)
csc = require_sparse_csc(torch.eye(3).to_sparse_csc())
dense = torch.eye(3)

assert_type(coo, SparseCOOTensor)
assert_type(csr, SparseCSRTensor)
assert_type(csc, SparseCSCTensor)
assert_type(rand_sparse((3, 3), 4), SparseCOOTensor)
assert_type(
    rand_sparse((3, 3), 4, layout=torch.sparse_csr),
    Union[SparseCOOTensor, SparseCSRTensor],
)

assert_type(convert_coo_to_csr(coo), SparseCSRTensor)
assert_type(sparse_block_diag(coo, coo), SparseCOOTensor)
assert_type(sparse_block_diag(csr, csr), SparseCSRTensor)
assert_type(sparse_block_diag_split(coo, (3, 3)), tuple[SparseCOOTensor, ...])
assert_type(sparse_block_diag_split(csr, (3, 3)), tuple[SparseCSRTensor, ...])
assert_type(sparse_eye((3, 3)), SparseCOOTensor)

external: torch.Tensor = torch.eye(3).to_sparse_coo()
if is_sparse_coo(external):
    assert_type(external, SparseCOOTensor)

validated = require_sparse_coo(torch.eye(3).to_sparse_coo())
assert_type(validated, SparseCOOTensor)
assert_type(sparse_mm(validated, dense), torch.Tensor)
assert_type(sparse_triangular_solve(validated, dense), torch.Tensor)
assert_type(sparse_generic_solve(validated, dense), torch.Tensor)
assert_type(sparse_generic_lstsq(validated, dense), torch.Tensor)
assert_type(sparse_logsumexp(validated, dim=0), torch.Tensor)
assert_type(sparse_logsumexp(csc, dim=0), torch.Tensor)
assert_type(sparse_bidir_logsumexp(validated), tuple[torch.Tensor, torch.Tensor])
assert_type(sparse_bidir_logsumexp(validated, output_layout="padded"), torch.Tensor)
