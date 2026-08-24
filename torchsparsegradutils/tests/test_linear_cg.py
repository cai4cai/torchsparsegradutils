import pytest
import torch

from torchsparsegradutils.utils.linear_cg import linear_cg


# Test basic CG solve for vectors and matrices
def test_cg():
    size = 100
    # SPD matrix
    matrix = torch.randn(size, size, dtype=torch.float64)
    matrix = matrix.matmul(matrix.mT)
    matrix.div_(torch.linalg.vector_norm(matrix)).add_(torch.eye(size, dtype=torch.float64) * 1e-1)
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
    matrix.div_(torch.linalg.vector_norm(matrix)).add_(torch.eye(size, dtype=torch.float64) * 1e-1)
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
    matrix.div_(torch.linalg.vector_norm(matrix)).add_(torch.eye(size, dtype=torch.float64) * 1e-1)
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
    matrix.div_(torch.linalg.vector_norm(matrix)).add_(torch.eye(size, dtype=torch.float64) * 1e-1)
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
    matrix.div_(torch.linalg.vector_norm(matrix)).add_(torch.eye(size, dtype=torch.float64) * 1e-1)
    rhs = torch.randn(batch, size, 50, dtype=torch.float64)
    solves = linear_cg(matrix.matmul, rhs=rhs, max_iter=size, max_tridiag_iter=0)
    solves_init = linear_cg(matrix.matmul, rhs=rhs, max_iter=1, initial_guess=solves, max_tridiag_iter=0)
    chol = torch.linalg.cholesky(matrix)
    actual = torch.cholesky_solve(rhs, chol)
    assert torch.allclose(solves_init, actual, atol=1e-3, rtol=1e-4)


def test_tight_tolerance_does_not_stall_at_historical_absolute_epsilon():
    size = 36
    diagonal = torch.full((size,), 2.0, dtype=torch.float64)
    off_diagonal = torch.full((size - 1,), -0.25, dtype=torch.float64)
    matrix = torch.diag(diagonal) + torch.diag(off_diagonal, 1) + torch.diag(off_diagonal, -1)
    rhs = torch.zeros(size, dtype=torch.float64)
    rhs[size // 2] = 1

    solution, info = linear_cg(matrix, rhs, tolerance=1e-12, max_iter=200, return_info=True)
    expected = torch.linalg.solve(matrix, rhs)

    torch.testing.assert_close(solution, expected, rtol=1e-10, atol=1e-12)
    assert info.reason == "converged"
    assert info.converged.all()
    assert info.true_relative_residual.max() <= 1e-12
    assert info.iterations <= size


def test_all_rhs_convergence_does_not_hide_one_hard_column():
    size = 40
    matrix = torch.diag(torch.linspace(1.0, 100.0, size, dtype=torch.float64))
    rhs = torch.zeros((size, 101), dtype=torch.float64)
    rhs[:, -1] = 1

    with pytest.warns(UserWarning):
        _, mean_info = linear_cg(
            matrix,
            rhs,
            tolerance=1e-4,
            max_iter=size,
            convergence_reduction="mean",
            return_info=True,
        )
    all_solution, all_info = linear_cg(
        matrix,
        rhs,
        tolerance=1e-4,
        max_iter=size,
        convergence_reduction="all",
        return_info=True,
    )

    assert mean_info.reason == "mean_converged"
    assert not mean_info.converged[..., -1].all()
    assert all_info.reason == "converged"
    assert all_info.converged.all()
    assert all_info.true_relative_residual.max() <= 1e-4
    torch.testing.assert_close(all_solution, torch.linalg.solve(matrix, rhs), rtol=2e-4, atol=1e-6)


def test_zero_rhs_returns_zero_even_with_nonzero_initial_guess():
    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    rhs = torch.zeros(3, dtype=torch.float64)
    initial_guess = torch.ones(3, dtype=torch.float64)

    solution, info = linear_cg(matrix, rhs, initial_guess=initial_guess, return_info=True)

    torch.testing.assert_close(solution, torch.zeros_like(rhs), rtol=0, atol=0)
    assert info.iterations == 0
    assert info.converged.all()
    assert info.true_relative_residual.max() == 0


def test_vector_rank_is_preserved_with_vector_initial_guess():
    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    rhs = torch.ones(3, dtype=torch.float64)
    solution = linear_cg(matrix, rhs, initial_guess=torch.zeros_like(rhs))
    assert solution.shape == rhs.shape

    column_guess_solution = linear_cg(matrix, rhs, initial_guess=torch.zeros((3, 1), dtype=torch.float64))
    assert column_guess_solution.shape == rhs.shape

    with pytest.raises(ValueError, match="initial_guess must have shape"):
        linear_cg(matrix, rhs, initial_guess=torch.zeros((3, 2), dtype=torch.float64))


def test_multiple_batch_dimensions_are_supported():
    with pytest.raises(ValueError, match="at least one dimension"):
        linear_cg(torch.ones((1, 1), dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))

    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    rhs = torch.ones((2, 4, 3, 2), dtype=torch.float64)

    solution = linear_cg(matrix, rhs, tolerance=1e-12)
    expected = torch.linalg.solve(matrix, rhs)

    assert solution.shape == rhs.shape
    torch.testing.assert_close(solution, expected, rtol=1e-12, atol=1e-12)
