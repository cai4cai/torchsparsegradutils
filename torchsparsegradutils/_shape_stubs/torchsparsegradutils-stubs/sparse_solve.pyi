from typing import Any, Callable, overload

from jaxtyping import Shaped
from torch import Tensor

from .sparse_types import SparseCOOTensor, SparseCSRTensor

@overload
def sparse_triangular_solve(
    A: Shaped[SparseCOOTensor, "n n"] | Shaped[SparseCSRTensor, "n n"],
    B: Shaped[Tensor, "n k"],
    upper: bool = ...,
    unitriangular: bool = ...,
    transpose: bool = ...,
) -> Shaped[Tensor, "n k"]: ...
@overload
def sparse_triangular_solve(
    A: Shaped[SparseCOOTensor, "b n n"] | Shaped[SparseCSRTensor, "b n n"],
    B: Shaped[Tensor, "b n k"],
    upper: bool = ...,
    unitriangular: bool = ...,
    transpose: bool = ...,
) -> Shaped[Tensor, "b n k"]: ...
@overload
def sparse_generic_solve(
    A: Shaped[SparseCOOTensor, "n n"] | Shaped[SparseCSRTensor, "n n"],
    B: Shaped[Tensor, "n"],  # noqa: F821 - jaxtyping dimension
    solve: Callable[..., Tensor] | None = ...,
    transpose_solve: Callable[..., Tensor] | None = ...,
    **kwargs: Any,
) -> Shaped[Tensor, "n"]: ...  # noqa: F821 - jaxtyping dimension
@overload
def sparse_generic_solve(
    A: Shaped[SparseCOOTensor, "n n"] | Shaped[SparseCSRTensor, "n n"],
    B: Shaped[Tensor, "n k"],
    solve: Callable[..., Tensor] | None = ...,
    transpose_solve: Callable[..., Tensor] | None = ...,
    **kwargs: Any,
) -> Shaped[Tensor, "n k"]: ...
