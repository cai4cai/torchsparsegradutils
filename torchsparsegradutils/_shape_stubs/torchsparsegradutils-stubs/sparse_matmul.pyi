from typing import overload

from jaxtyping import Shaped
from torch import Tensor

from .sparse_types import SparseCOOTensor, SparseCSRTensor

# Wrap each union arm: Pyrefly 1.3.1 drops shapes on Shaped[COO | CSR, ...].
# Separate overloads reflect the implementation's exact 2-D/3-D contract.
@overload
def sparse_mm(
    A: Shaped[SparseCOOTensor, "m n"] | Shaped[SparseCSRTensor, "m n"],
    B: Shaped[Tensor, "n k"],
) -> Shaped[Tensor, "m k"]: ...
@overload
def sparse_mm(
    A: Shaped[SparseCOOTensor, "b m n"] | Shaped[SparseCSRTensor, "b m n"],
    B: Shaped[Tensor, "b n k"],
) -> Shaped[Tensor, "b m k"]: ...
