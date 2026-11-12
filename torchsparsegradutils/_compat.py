"""Compatibility helpers for PyTorch APIs without feature-equivalent replacements."""

import warnings

import torch
from packaging.version import Version


def linalg_solve_triangular_compat(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    upper: bool,
    unitriangular: bool = False,
    transpose: bool = False,
) -> torch.Tensor:
    """Solve a triangular system with the appropriate dense or sparse backend.

    Dense coefficient matrices use ``torch.linalg.solve_triangular``. Sparse
    matrices continue to use the isolated legacy call because the replacement
    API does not support the sparse layouts required by this package. See
    https://github.com/pytorch/pytorch/issues/87358 for upstream sparse feature
    parity tracking.
    """
    if A.layout == torch.strided:
        if transpose:
            A = A.transpose(-2, -1)
            upper = not upper

        return torch.linalg.solve_triangular(
            A,
            B,
            upper=upper,
            unitriangular=unitriangular,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"torch\.triangular_solve is deprecated.*",
            category=UserWarning,
        )
        return torch.triangular_solve(
            B,
            A,
            upper=upper,
            transpose=transpose,
            unitriangular=unitriangular,
        ).solution


# Official CUDA 11.8/12.1/12.4 builds reproduce this; tested 12.6/13.0/13.2 builds pass.
# CUDA 12.5 is unverified, so conservatively retain protection below 12.6.
_SPARSE_MM_STRIDE_WORKAROUND = torch.version.cuda is not None and Version(torch.version.cuda) < Version("12.6")


def sparse_mm_compat(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Apply the legacy CUDA compressed-sparse single-column stride workaround."""
    # The boundary follows the CUDA build, not the PyTorch release: 2.6/cu124
    # fails while 2.6/cu126 passes. Build version is a proxy for bundled libraries.
    # CSR transposes to CSC in the dense-input backward multiplication.
    if (
        _SPARSE_MM_STRIDE_WORKAROUND
        and A.is_cuda
        and A.layout in (torch.sparse_csr, torch.sparse_csc)
        and B.size(-1) == 1
        and not B.is_contiguous()
    ):
        B = B.contiguous()
    return torch.sparse.mm(A, B)
