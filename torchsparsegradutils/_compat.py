"""Compatibility helpers for PyTorch APIs without feature-equivalent replacements."""

import warnings

import torch


def sparse_triangular_solve_compat(
    B: torch.Tensor,
    A: torch.Tensor,
    *,
    upper: bool,
    unitriangular: bool,
    transpose: bool,
) -> torch.Tensor:
    """Solve a sparse triangular system using PyTorch's legacy sparse backend.

    ``torch.linalg.solve_triangular`` replaces ``torch.triangular_solve`` for
    dense tensors, but it does not provide the sparse CSR support required by
    this package. Keep the deprecated call isolated here until a documented,
    feature-equivalent sparse API is available in every supported PyTorch
    version.
    """
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
