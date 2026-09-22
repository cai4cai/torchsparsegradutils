"""Nominal types and validators for PyTorch sparse tensor layouts.

The types in this module describe only a tensor's storage layout. They do not
promise a particular rank, shape, dtype, device, coalescence state, or sparse
index invariant. Validation is intentionally centralized here because
``NewType`` itself performs no runtime checking.
"""

from typing import NewType, TypeGuard, Union

import torch

SparseCOOTensor = NewType("SparseCOOTensor", torch.Tensor)
SparseCSRTensor = NewType("SparseCSRTensor", torch.Tensor)
SparseCSCTensor = NewType("SparseCSCTensor", torch.Tensor)
SparseBSRTensor = NewType("SparseBSRTensor", torch.Tensor)
SparseBSCTensor = NewType("SparseBSCTensor", torch.Tensor)

SparseTensor = Union[
    SparseCOOTensor,
    SparseCSRTensor,
    SparseCSCTensor,
    SparseBSRTensor,
    SparseBSCTensor,
]

__all__ = [
    "SparseCOOTensor",
    "SparseCSRTensor",
    "SparseCSCTensor",
    "SparseBSRTensor",
    "SparseBSCTensor",
    "SparseTensor",
    "is_sparse_coo",
    "is_sparse_csr",
    "is_sparse_csc",
    "is_sparse_bsr",
    "is_sparse_bsc",
    "require_sparse_coo",
    "require_sparse_csr",
    "require_sparse_csc",
    "require_sparse_bsr",
    "require_sparse_bsc",
    "require_sparse_coo_or_csr",
    "require_sparse_coo_csr_or_csc",
]


def is_sparse_coo(value: object) -> TypeGuard[SparseCOOTensor]:
    """Return whether ``value`` is a tensor with sparse COO layout."""
    return isinstance(value, torch.Tensor) and value.layout == torch.sparse_coo


def is_sparse_csr(value: object) -> TypeGuard[SparseCSRTensor]:
    """Return whether ``value`` is a tensor with sparse CSR layout."""
    return isinstance(value, torch.Tensor) and value.layout == torch.sparse_csr


def is_sparse_csc(value: object) -> TypeGuard[SparseCSCTensor]:
    """Return whether ``value`` is a tensor with sparse CSC layout."""
    return isinstance(value, torch.Tensor) and value.layout == torch.sparse_csc


def is_sparse_bsr(value: object) -> TypeGuard[SparseBSRTensor]:
    """Return whether ``value`` is a tensor with sparse BSR layout."""
    return isinstance(value, torch.Tensor) and value.layout == torch.sparse_bsr


def is_sparse_bsc(value: object) -> TypeGuard[SparseBSCTensor]:
    """Return whether ``value`` is a tensor with sparse BSC layout."""
    return isinstance(value, torch.Tensor) and value.layout == torch.sparse_bsc


def require_sparse_coo(value: object) -> SparseCOOTensor:
    """Validate and brand a sparse COO tensor."""
    if not is_sparse_coo(value):
        raise TypeError(f"Expected a torch.Tensor with layout torch.sparse_coo, got {_describe(value)}")
    return SparseCOOTensor(value)


def require_sparse_csr(value: object) -> SparseCSRTensor:
    """Validate and brand a sparse CSR tensor."""
    if not is_sparse_csr(value):
        raise TypeError(f"Expected a torch.Tensor with layout torch.sparse_csr, got {_describe(value)}")
    return SparseCSRTensor(value)


def require_sparse_csc(value: object) -> SparseCSCTensor:
    """Validate and brand a sparse CSC tensor."""
    if not is_sparse_csc(value):
        raise TypeError(f"Expected a torch.Tensor with layout torch.sparse_csc, got {_describe(value)}")
    return SparseCSCTensor(value)


def require_sparse_bsr(value: object) -> SparseBSRTensor:
    """Validate and brand a sparse BSR tensor."""
    if not is_sparse_bsr(value):
        raise TypeError(f"Expected a torch.Tensor with layout torch.sparse_bsr, got {_describe(value)}")
    return SparseBSRTensor(value)


def require_sparse_bsc(value: object) -> SparseBSCTensor:
    """Validate and brand a sparse BSC tensor."""
    if not is_sparse_bsc(value):
        raise TypeError(f"Expected a torch.Tensor with layout torch.sparse_bsc, got {_describe(value)}")
    return SparseBSCTensor(value)


def require_sparse_coo_or_csr(value: object) -> Union[SparseCOOTensor, SparseCSRTensor]:
    """Validate and brand a sparse COO or CSR tensor."""
    if is_sparse_coo(value):
        return value
    if is_sparse_csr(value):
        return value
    raise TypeError(f"Expected a torch.Tensor with sparse COO or CSR layout, got {_describe(value)}")


def require_sparse_coo_csr_or_csc(
    value: object,
) -> Union[SparseCOOTensor, SparseCSRTensor, SparseCSCTensor]:
    """Validate and brand a sparse COO, CSR, or CSC tensor."""
    if is_sparse_coo(value):
        return value
    if is_sparse_csr(value):
        return value
    if is_sparse_csc(value):
        return value
    raise TypeError(f"Expected a torch.Tensor with sparse COO, CSR, or CSC layout, got {_describe(value)}")


def _describe(value: object) -> str:
    if isinstance(value, torch.Tensor):
        return str(value.layout)
    return type(value).__name__
