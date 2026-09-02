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

    solution, info = linear_cg(matrix, rhs, initial_guess=initial_guess, tolerance=0, return_info=True)

    torch.testing.assert_close(solution, torch.zeros_like(rhs), rtol=0, atol=0)
    assert info.iterations == 0
    assert info.converged.all()
    assert info.true_relative_residual.max() == 0
    assert info.tolerance == 0


def test_mixed_zero_and_nonzero_rhs_columns_are_supported():
    matrix = torch.diag(torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64))
    rhs = torch.stack((torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)), dim=-1)

    solution, info = linear_cg(matrix, rhs, tolerance=1e-12, return_info=True)

    expected = torch.linalg.solve(matrix, rhs)
    torch.testing.assert_close(solution, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(solution[:, 0], torch.zeros(3, dtype=torch.float64), rtol=0, atol=0)
    assert info.converged.all()


def test_preconditioner_and_breakdown_validation():
    diagonal = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    matrix = torch.diag(diagonal)
    rhs = torch.ones(3, dtype=torch.float64)

    solution, info = linear_cg(
        matrix,
        rhs,
        preconditioner=lambda value: value / diagonal.unsqueeze(-1),
        return_info=True,
    )
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)
    assert info.reason == "converged"

    indefinite = torch.diag(torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64))
    with pytest.raises(RuntimeError, match=r"p\^T A p"):
        linear_cg(indefinite, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64))

    with pytest.raises(RuntimeError, match="preconditioned residual inner product"):
        linear_cg(matrix, rhs, preconditioner=lambda value: -value)


def test_user_eps_raises_instead_of_freezing_active_column():
    matrix = torch.eye(3, dtype=torch.float64)
    rhs = torch.ones(3, dtype=torch.float64)

    with pytest.raises(RuntimeError, match="at least eps"):
        linear_cg(
            matrix,
            rhs,
            initial_guess=(1 - 1e-6) * rhs,
            tolerance=1e-8,
            eps=1e-10,
            max_tridiag_iter=0,
        )


def test_stopped_initial_updates_warn_and_report_reason():
    matrix = torch.eye(3, dtype=torch.float64)
    rhs = torch.ones(3, dtype=torch.float64)

    with pytest.warns(UserWarning, match="did not converge"):
        _, info = linear_cg(
            matrix,
            rhs,
            initial_guess=0.5 * rhs,
            tolerance=0.1,
            stop_updating_after=0.6,
            return_info=True,
        )

    assert info.iterations == 0
    assert info.reason == "stopped_updating"
    assert info.tolerance == 0.1
    assert not info.converged.any()


def test_recursive_convergence_is_distinguished_from_true_convergence():
    calls = 0

    def matmul_with_final_residual_drift(value):
        nonlocal calls
        calls += 1
        return 1.1 * value if calls == 3 else value

    with pytest.warns(UserWarning, match="did not converge"):
        _, info = linear_cg(
            matmul_with_final_residual_drift,
            torch.ones(3, dtype=torch.float64),
            tolerance=1e-12,
            max_tridiag_iter=0,
            return_info=True,
        )

    assert info.iterations == 1
    assert info.reason == "recursive_converged"
    assert info.recursive_relative_residual.max() == 0
    assert info.true_relative_residual.max() > info.tolerance


def test_tridiagonalization_can_return_info():
    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    rhs = torch.ones(3, dtype=torch.float64)

    solution, tridiagonal, info = linear_cg(
        matrix,
        rhs,
        n_tridiag=1,
        tolerance=0,
        max_iter=3,
        max_tridiag_iter=3,
        return_info=True,
    )

    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)
    assert tridiagonal.shape == (1, 3, 3)
    assert info.iterations == 3
    assert info.matvecs == 5


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"tolerance": float("nan")}, "tolerance"),
        ({"tolerance": float("inf")}, "tolerance"),
        ({"tolerance": -1.0}, "tolerance"),
        ({"eps": float("nan")}, "eps"),
        ({"eps": float("inf")}, "eps"),
        ({"eps": 0.0}, "eps"),
        ({"stop_updating_after": float("nan")}, "stop_updating_after"),
        ({"stop_updating_after": float("inf")}, "stop_updating_after"),
        ({"stop_updating_after": -1.0}, "stop_updating_after"),
        ({"convergence_reduction": "median"}, "convergence_reduction"),
        ({"min_iter": -1}, "min_iter"),
    ],
)
def test_invalid_solver_settings_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        linear_cg(torch.eye(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64), **kwargs)


def test_eps_must_be_representable_in_rhs_dtype():
    with pytest.raises(ValueError, match="representable"):
        linear_cg(torch.eye(2), torch.ones(2), eps=1e-100)


def test_rhs_and_operator_output_validation():
    matrix = torch.eye(2, dtype=torch.float64)
    rhs = torch.ones(2, dtype=torch.float64)

    with pytest.raises(TypeError, match="floating-point"):
        linear_cg(torch.eye(2, dtype=torch.int64), torch.ones(2, dtype=torch.int64))
    with pytest.raises(RuntimeError, match="matmul_closure output"):
        linear_cg(lambda value: value[:-1], rhs)
    with pytest.raises(RuntimeError, match="preconditioner output"):
        linear_cg(matrix, rhs, preconditioner=lambda value: value.to(torch.float32))


def test_vector_rank_is_preserved_with_vector_initial_guess():
    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    rhs = torch.ones(3, dtype=torch.float64)
    solution = linear_cg(matrix, rhs, initial_guess=torch.zeros_like(rhs))
    assert solution.shape == rhs.shape

    column_guess_solution = linear_cg(matrix, rhs, initial_guess=torch.zeros((3, 1), dtype=torch.float64))
    assert column_guess_solution.shape == rhs.shape

    with pytest.raises(ValueError, match="initial_guess must have shape"):
        linear_cg(matrix, rhs, initial_guess=torch.zeros((3, 2), dtype=torch.float64))


def test_min_iter_is_honored_for_converged_initial_guess():
    matrix = torch.eye(3, dtype=torch.float64)
    rhs = torch.ones(3, dtype=torch.float64)

    solution, info = linear_cg(
        matrix,
        rhs,
        initial_guess=rhs,
        min_iter=3,
        max_iter=5,
        max_tridiag_iter=0,
        return_info=True,
    )

    torch.testing.assert_close(solution, rhs, rtol=0, atol=0)
    assert info.iterations == 3
    assert info.matvecs == 5
    assert info.reason == "converged"
    assert info.converged.all()


def test_multiple_batch_dimensions_are_supported():
    with pytest.raises(ValueError, match="at least one dimension"):
        linear_cg(torch.ones((1, 1), dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64))

    matrix = torch.diag(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    rhs = torch.ones((2, 4, 3, 2), dtype=torch.float64)

    solution = linear_cg(matrix, rhs, tolerance=1e-12)
    expected = torch.linalg.solve(matrix, rhs)

    assert solution.shape == rhs.shape
    torch.testing.assert_close(solution, expected, rtol=1e-12, atol=1e-12)
