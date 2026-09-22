from typing import Callable, overload

from jaxtyping import Shaped
from torch import Tensor

from .sparse_types import SparseCOOTensor, SparseCSRTensor

@overload
def sparse_generic_lstsq(
    A: Shaped[SparseCOOTensor, "m n"] | Shaped[SparseCSRTensor, "m n"],
    B: Shaped[Tensor, "m"],  # noqa: F821 - jaxtyping dimension
    lstsq: Callable[[Tensor, Tensor], Tensor] | None = ...,
    transpose_lstsq: Callable[[Tensor, Tensor], Tensor] | None = ...,
) -> Shaped[Tensor, "n"]: ...  # noqa: F821 - jaxtyping dimension
@overload
def sparse_generic_lstsq(
    A: Shaped[SparseCOOTensor, "m n"] | Shaped[SparseCSRTensor, "m n"],
    B: Shaped[Tensor, "m k"],
    lstsq: Callable[[Tensor, Tensor], Tensor] | None = ...,
    transpose_lstsq: Callable[[Tensor, Tensor], Tensor] | None = ...,
) -> Shaped[Tensor, "n k"]: ...
