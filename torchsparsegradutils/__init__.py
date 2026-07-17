from . import _dispatch  # noqa: F401  (import triggers the CUDA backend probe, architecture.md §1)
from .ops import (
    gather_mm,
    segment_mm,
    sparse_bidir_logsumexp,
    sparse_generic_lstsq,
    sparse_generic_solve,
    sparse_logsumexp,
    sparse_mm,
    sparse_triangular_solve,
)

# isort: split
# utils/convert.py defines the tsgu::coo2csr op schema (map.md "Kernel
# routing") -- a bare `import torchsparsegradutils` must register the FULL
# tsgu:: namespace, not just the eight ops/ ops. Pre-existing bug (fixed in
# commit 19): nothing imported utils, so tsgu::coo2csr was only registered
# once something happened to import torchsparsegradutils.utils --
# tests/test_op_schemas.py failed standalone ("operator tsgu::coo2csr does
# not exist"). Deliberately after the .ops import: utils/__init__ pulls in
# solvers/, which reach back into the package internals.
from . import utils  # noqa: F401, E402

__all__ = [
    "sparse_mm",
    "gather_mm",
    "segment_mm",
    "sparse_triangular_solve",
    "sparse_generic_solve",
    "sparse_generic_lstsq",
    "sparse_logsumexp",
    "sparse_bidir_logsumexp",
]
