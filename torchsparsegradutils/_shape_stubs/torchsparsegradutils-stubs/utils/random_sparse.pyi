"""Shape inference for the public sparse factories; list sizes remain gradual."""

from typing import TypeVar, overload

import torch
from shape_extensions import Int, IntTuple, IntVar
from torch.types import Device

from ..sparse_types import SparseCOOTensor, SparseCSRTensor

_Shape = TypeVar("_Shape", bound=IntTuple)
_N = IntVar("_N")
SparseSize = torch.Size | list[int] | tuple[int, ...]

@overload
def rand_sparse(
    size: _Shape | list[int],
    nnz: int,
    *,
    indices_dtype: torch.dtype = ...,
    values_dtype: torch.dtype = ...,
    device: Device = ...,
    well_conditioned: bool = ...,
    min_diag_value: float = ...,
) -> SparseCOOTensor[_Shape]: ...
@overload
def rand_sparse(
    size: _Shape | list[int],
    nnz: int,
    layout: torch.layout,
    *,
    indices_dtype: torch.dtype = ...,
    values_dtype: torch.dtype = ...,
    device: Device = ...,
    well_conditioned: bool = ...,
    min_diag_value: float = ...,
) -> SparseCOOTensor[_Shape] | SparseCSRTensor[_Shape]: ...
@overload
def rand_sparse_tri(
    size: _Shape | list[int],
    nnz: int,
    *,
    upper: bool = ...,
    strict: bool = ...,
    indices_dtype: torch.dtype = ...,
    values_dtype: torch.dtype = ...,
    device: Device = ...,
    value_range: tuple[float, float] = ...,
    well_conditioned: bool = ...,
    min_diag_value: float = ...,
) -> SparseCOOTensor[_Shape]: ...
@overload
def rand_sparse_tri(
    size: _Shape | list[int],
    nnz: int,
    layout: torch.layout,
    *,
    upper: bool = ...,
    strict: bool = ...,
    indices_dtype: torch.dtype = ...,
    values_dtype: torch.dtype = ...,
    device: Device = ...,
    value_range: tuple[float, float] = ...,
    well_conditioned: bool = ...,
    min_diag_value: float = ...,
) -> SparseCOOTensor[_Shape] | SparseCSRTensor[_Shape]: ...
def generate_random_sparse_coo_matrix(
    size: _Shape | list[int],
    nnz: int,
    *,
    indices_dtype: torch.dtype = ...,
    values_dtype: torch.dtype = ...,
    device: Device = ...,
    well_conditioned: bool = ...,
    min_diag_value: float = ...,
) -> SparseCOOTensor[_Shape]: ...
def generate_random_sparse_csr_matrix(
    size: _Shape | list[int],
    nnz: int,
    *,
    indices_dtype: torch.dtype = ...,
    values_dtype: torch.dtype = ...,
    device: Device = ...,
    well_conditioned: bool = ...,
    min_diag_value: float = ...,
) -> SparseCSRTensor[_Shape]: ...
def generate_random_sparse_strictly_triangular_coo_matrix(
    size: _Shape | list[int],
    nnz: int,
    *,
    upper: bool = ...,
    indices_dtype: torch.dtype = ...,
    values_dtype: torch.dtype = ...,
    device: Device = ...,
    value_range: tuple[float, float] = ...,
) -> SparseCOOTensor[_Shape]: ...
def generate_random_sparse_strictly_triangular_csr_matrix(
    size: _Shape | list[int],
    nnz: int,
    *,
    upper: bool = ...,
    indices_dtype: torch.dtype = ...,
    values_dtype: torch.dtype = ...,
    device: Device = ...,
    value_range: tuple[float, float] = ...,
) -> SparseCSRTensor[_Shape]: ...
def generate_random_sparse_triangular_coo_matrix(
    size: _Shape | list[int],
    nnz: int,
    *,
    upper: bool = ...,
    indices_dtype: torch.dtype = ...,
    values_dtype: torch.dtype = ...,
    device: Device = ...,
    value_range: tuple[float, float] = ...,
    well_conditioned: bool = ...,
    min_diag_value: float = ...,
) -> SparseCOOTensor[_Shape]: ...
def generate_random_sparse_triangular_csr_matrix(
    size: _Shape | list[int],
    nnz: int,
    *,
    upper: bool = ...,
    indices_dtype: torch.dtype = ...,
    values_dtype: torch.dtype = ...,
    device: Device = ...,
    value_range: tuple[float, float] = ...,
    well_conditioned: bool = ...,
    min_diag_value: float = ...,
) -> SparseCSRTensor[_Shape]: ...
def make_spd_sparse(
    n: Int[_N],
    layout: torch.layout,
    value_dtype: torch.dtype,
    index_dtype: torch.dtype,
    device: Device,
    sparsity_ratio: float = ...,
    nz: int | None = ...,
) -> tuple[SparseCOOTensor[[_N, _N]] | SparseCSRTensor[[_N, _N]], torch.Tensor[[_N, _N]]]: ...
