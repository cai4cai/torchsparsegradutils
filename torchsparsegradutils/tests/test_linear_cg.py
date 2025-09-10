import os
import random

import pytest
import torch

from torchsparsegradutils.utils.linear_cg import linear_cg


# Autouse fixture to seed RNG and restore state after each test
@pytest.fixture(autouse=True)
def seed_restore():
    unlock = os.getenv("UNLOCK_SEED")
    if unlock is None or unlock.lower() == "false":
        rng = torch.get_rng_state()
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        random.seed(0)
        yield
        torch.set_rng_state(rng)
    else:
        yield


# Test basic CG solve for vectors and matrices
def test_cg():
    size = 100
    # SPD matrix
    matrix = torch.randn(size, size, dtype=torch.float64)
    matrix = matrix.matmul(matrix.mT)
    matrix.div_(matrix.norm()).add_(torch.eye(size, dtype=torch.float64) * 1e-1)
    # single RHS
    rhs = torch.randn(size, dtype=torch.float64)
    solves = linear_cg(matrix.matmul, rhs=rhs, max_iter=size)
    init = torch.randn(size, dtype=torch.float64)
    solves_init = linear_cg(matrix.matmul, rhs=rhs, max_iter=size, initial_guess=init)
    # reference solve
    chol = torch.linalg.cholesky(matrix)
    actual = torch.cholesky_solve(rhs.unsqueeze(1), chol).squeeze()
    assert torch.allclose(solves, actual, atol=1e-3, rtol=1e-4)
    assert torch.allclose(solves_init, actual, atol=1e-3, rtol=1e-4)
    # multiple RHS
    rhs_mat = torch.randn(size, 50, dtype=torch.float64)
    solves = linear_cg(matrix.matmul, rhs=rhs_mat, max_iter=size)
    init_mat = torch.randn(size, 50, dtype=torch.float64)
    solves_init = linear_cg(matrix.matmul, rhs=rhs_mat, max_iter=size, initial_guess=init_mat)
    actual_mat = torch.cholesky_solve(rhs_mat, chol)
    assert torch.allclose(solves, actual_mat, atol=1e-3, rtol=1e-4)
    assert torch.allclose(solves_init, actual_mat, atol=1e-3, rtol=1e-4)


# Test CG with tridiagonal outputs
def test_cg_with_tridiag():
    size = 10
    matrix = torch.randn(size, size, dtype=torch.float64)
    matrix = matrix.matmul(matrix.mT)
    matrix.div_(matrix.norm()).add_(torch.eye(size, dtype=torch.float64) * 1e-1)
    rhs = torch.randn(size, 50, dtype=torch.float64)
    solves, t_mats = linear_cg(
        matrix.matmul, rhs=rhs, n_tridiag=5, max_tridiag_iter=10, max_iter=size, tolerance=0, eps=1e-15
    )
    chol = torch.linalg.cholesky(matrix)
    actual = torch.cholesky_solve(rhs, chol)
    assert torch.allclose(solves, actual, atol=1e-3, rtol=1e-4)
    eigs = torch.linalg.eigvalsh(matrix)
    for i in range(5):
        approx = torch.linalg.eigvalsh(t_mats[i])
        assert torch.allclose(eigs, approx, atol=1e-3, rtol=1e-4)


# Device parameterized CG tests
@pytest.mark.parametrize("batch", [None, 5])
def test_batch_cg(batch):
    size = 100
    shape = (batch, size, size) if batch else (size, size)
    matrix = torch.randn(*shape, dtype=torch.float64)
    matrix = matrix.matmul(matrix.mT)
    matrix.div_(matrix.norm()).add_(torch.eye(size, dtype=torch.float64) * 1e-1)
    b_shape = (batch, size, 50) if batch else (size, 50)
    rhs = torch.randn(*b_shape, dtype=torch.float64)
    solves = linear_cg(matrix.matmul, rhs=rhs, max_iter=size)
    chol = torch.linalg.cholesky(matrix)
    actual = torch.cholesky_solve(rhs, chol)
    assert torch.allclose(solves, actual, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("batch", [None, 5])
def test_batch_cg_with_tridiag(batch):
    size = 10
    shape = (batch, size, size) if batch else (size, size)
    matrix = torch.randn(*shape, dtype=torch.float64)
    matrix = matrix.matmul(matrix.mT)
    matrix.div_(matrix.norm()).add_(torch.eye(size, dtype=torch.float64) * 1e-1)
    b_shape = (batch, size, 10) if batch else (size, 10)
    rhs = torch.randn(*b_shape, dtype=torch.float64)
    solves, t_mats = linear_cg(
        matrix.matmul, rhs=rhs, n_tridiag=8, max_iter=size, max_tridiag_iter=10, tolerance=0, eps=1e-30
    )
    chol = torch.linalg.cholesky(matrix)
    actual = torch.cholesky_solve(rhs, chol)
    assert torch.allclose(solves, actual, atol=1e-3, rtol=1e-4)
    batch_dim = 5 if batch else 1
    for i in range(batch_dim):
        eigs = torch.linalg.eigvalsh(matrix[i] if batch else matrix)
        for j in range(8):
            approx = torch.linalg.eigvalsh(t_mats[j, i] if batch else t_mats[j])
            assert torch.allclose(eigs, approx, atol=1e-3, rtol=1e-4)


# Test CG initialization reuse
def test_batch_cg_init():
    batch = 5
    size = 100
    matrix = torch.randn(batch, size, size, dtype=torch.float64)
    matrix = matrix.matmul(matrix.mT)
    matrix.div_(matrix.norm()).add_(torch.eye(size, dtype=torch.float64) * 1e-1)
    rhs = torch.randn(batch, size, 50, dtype=torch.float64)
    solves = linear_cg(matrix.matmul, rhs=rhs, max_iter=size, max_tridiag_iter=0)
    solves_init = linear_cg(matrix.matmul, rhs=rhs, max_iter=1, initial_guess=solves, max_tridiag_iter=0)
    chol = torch.linalg.cholesky(matrix)
    actual = torch.cholesky_solve(rhs, chol)
    assert torch.allclose(solves_init, actual, atol=1e-3, rtol=1e-4)
