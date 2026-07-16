"""
Test script for Quick Start Guide examples.

This module tests all code examples from the quickstart documentation to ensure they work correctly.
It is integrated with pytest and runs as part of the CI pipeline.
"""

import warnings

import pytest
import torch

from torchsparsegradutils._dispatch import backend_available

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# spec/commit.md Phase 3 commit 15: sparse_mm now dispatches to tsgu::spmm,
# CUDA-only (architecture.md §4) -- every example below that calls sparse_mm
# (directly, or indirectly via SparseMultivariateNormal.rsample's internal
# spmm/spts chain) now needs a CUDA device and a loaded backend, matching
# test_generic_solve's pre-existing skipif pattern in this same file.
requires_cuda_backend = pytest.mark.skipif(
    not (torch.cuda.is_available() and backend_available()),
    reason="sparse_mm (tsgu::spmm, spec/commit.md Phase 3 commit 15) is CUDA-only; none available here.",
)


@requires_cuda_backend
def test_sparse_mm_example():
    """Test the basic sparse matrix multiplication example from quickstart guide."""
    from torchsparsegradutils import sparse_mm

    # Create a sparse matrix in COO format
    indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.int64)
    values = torch.tensor([3.0, 4.0, 5.0])
    A = torch.sparse_coo_tensor(indices, values, (2, 3)).cuda()
    A.requires_grad_(True)

    # Create a dense matrix
    B = torch.randn(3, 4, requires_grad=True, device="cuda")
    B.requires_grad_(True)

    # Perform sparse matrix multiplication with gradient support
    result = sparse_mm(A, B)
    assert result.shape == (2, 4), f"Expected shape (2, 4), got {result.shape}"

    # The operation preserves sparsity in gradients
    loss = result.sum()
    loss.backward()
    assert A.grad.is_sparse, "A gradient should be sparse"
    assert A.grad._nnz() == 3, f"Expected 3 non-zeros, got {A.grad._nnz()}"


@requires_cuda_backend
def test_batched_operations():
    """Test batched sparse operations from quickstart guide."""
    from torchsparsegradutils import sparse_mm

    # Create batched sparse matrices
    batch_size = 2

    # Method 1: Stack individual sparse matrices
    A1 = torch.sparse_coo_tensor([[0, 1], [0, 1]], [1.0, 2.0], (2, 2)).cuda()
    A2 = torch.sparse_coo_tensor([[0, 1], [1, 0]], [3.0, 4.0], (2, 2)).cuda()
    A_batch = torch.stack([A1, A2])  # Shape: (2, 2, 2)

    # Dense batch
    B = torch.randn(batch_size, 2, 3, device="cuda")

    # Batched multiplication
    result = sparse_mm(A_batch, B)
    assert result.shape == (2, 2, 3), f"Expected shape (2, 2, 3), got {result.shape}"


@requires_cuda_backend
def test_triangular_solve():
    """Test sparse triangular solve example from quickstart guide."""
    from torchsparsegradutils import sparse_mm, sparse_triangular_solve
    from torchsparsegradutils.utils.random_sparse import rand_sparse_tri

    # Create a sparse lower triangular matrix
    L = rand_sparse_tri((3, 3), nnz=5, upper=False, layout=torch.sparse_csr, device="cuda")

    # Right-hand side
    b = torch.randn(3, 2, device="cuda")

    # Solve Lx = b
    x = sparse_triangular_solve(L, b, upper=False)
    assert x.shape == (3, 2), f"Expected shape (3, 2), got {x.shape}"

    # Verify solution (should be close to zero)
    residual = sparse_mm(L, x) - b
    residual_norm = torch.norm(residual)
    assert residual_norm < 1e-5, f"Residual too large: {residual_norm}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_generic_solve():
    """Test generic sparse solve example from quickstart guide."""
    from torchsparsegradutils import sparse_generic_solve
    from torchsparsegradutils.utils import minres
    from torchsparsegradutils.utils.random_sparse import make_spd_sparse

    # Create a sparse symmetric positive definite matrix
    A_sparse, A_dense = make_spd_sparse(10, torch.sparse_coo, torch.float32, torch.int64, "cuda")
    b = torch.randn(10, device="cuda")

    # Solve using MINRES solver
    x_cg = sparse_generic_solve(A_sparse, b, solve=minres)
    assert x_cg.shape == (10,), f"Expected shape (10,), got {x_cg.shape}"

    # Verify solution
    residual = A_sparse @ x_cg - b
    residual_norm = torch.norm(residual)
    assert residual_norm < 1e-4, f"Residual too large: {residual_norm}"


@requires_cuda_backend
def test_sparse_mvn():
    """Test sparse multivariate normal distribution example from quickstart guide."""
    from torchsparsegradutils.distributions import SparseMultivariateNormal
    from torchsparsegradutils.utils.random_sparse import rand_sparse_tri

    # Create parameters
    dim = 10
    loc = torch.zeros(dim, device="cuda")

    # LDL^T parameterization (recommended for stability)
    diagonal = torch.ones(dim, device="cuda") * 0.5
    scale_tril = rand_sparse_tri((dim, dim), nnz=15, upper=False, layout=torch.sparse_coo, strict=False, device="cuda")

    # Create distribution with LDL^T covariance parameterization
    dist = SparseMultivariateNormal(loc=loc, diagonal=diagonal, scale_tril=scale_tril)

    # Sample from the distribution
    samples = dist.rsample((100,))
    assert samples.shape == (100, 10), f"Expected shape (100, 10), got {samples.shape}"


@requires_cuda_backend
def test_mvn_gradients():
    """Test gradient computation with sparse multivariate normal from quickstart guide."""
    from torchsparsegradutils.distributions import SparseMultivariateNormal
    from torchsparsegradutils.utils.random_sparse import rand_sparse_tri

    dim = 10

    # Enable gradients for parameters
    loc = torch.zeros(dim, requires_grad=True, device="cuda")
    diagonal = torch.ones(dim, requires_grad=True, device="cuda")
    scale_tril = rand_sparse_tri((dim, dim), nnz=15, upper=False, layout=torch.sparse_coo, strict=True, device="cuda")
    scale_tril.requires_grad_(True)

    # Create distribution
    dist = SparseMultivariateNormal(loc=loc, diagonal=diagonal, scale_tril=scale_tril)

    # Sample using rsample for gradient flow
    samples = dist.rsample((10,))

    # Compute some loss
    loss = samples.mean()
    loss.backward()

    # Check gradients exist
    assert loc.grad is not None, "Location gradient is None"
    assert diagonal.grad is not None, "Diagonal gradient is None"
    assert scale_tril.grad is not None, "Scale gradient is None"
    assert scale_tril.grad._nnz() == 15, f"Expected 15 non-zeros, got {scale_tril.grad._nnz()}"
