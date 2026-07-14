# extracted from f19d7b4 — parity Oracle A, never shipped; do not edit
"""Parity Oracle A (testing.md, Pillar 1).

Frozen copies of the pre-rewrite pure-PyTorch core-op implementations
(``main`` @ f19d7b4), exposed here under ``oracle_*`` names for use as a
differential reference against the CUDA-native rewrite. Test-only code,
never shipped in a wheel.

Not wired into the existing test suite yet: this module only re-exports the
frozen implementations. The parity harness that dispatches against these
oracles lands in commit 11 (spec/commit.md).
"""

from .indexed_matmul import gather_mm as oracle_gather_mm
from .indexed_matmul import segment_mm as oracle_segment_mm
from .sparse_logsumexp import sparse_bidir_logsumexp as oracle_sparse_bidir_logsumexp
from .sparse_logsumexp import sparse_logsumexp as oracle_sparse_logsumexp
from .sparse_lstsq import sparse_generic_lstsq as oracle_sparse_generic_lstsq
from .sparse_matmul import sparse_mm as oracle_sparse_mm
from .sparse_solve import sparse_generic_solve as oracle_sparse_generic_solve
from .sparse_solve import sparse_triangular_solve as oracle_sparse_triangular_solve

__all__ = [
    "oracle_sparse_mm",
    "oracle_sparse_triangular_solve",
    "oracle_sparse_generic_solve",
    "oracle_sparse_generic_lstsq",
    "oracle_sparse_logsumexp",
    "oracle_sparse_bidir_logsumexp",
    "oracle_segment_mm",
    "oracle_gather_mm",
]
