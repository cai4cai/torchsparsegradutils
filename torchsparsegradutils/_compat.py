"""Compatibility helpers for PyTorch APIs without feature-equivalent replacements."""

import warnings

import torch


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
