# MIT-licensed code imported from https://github.com/cornellius-gp/linear_operator
# Minor modifications for torchsparsegradutils to remove dependencies

import pytest
import torch
from test_config import Tolerances

import torchsparsegradutils
from torchsparsegradutils.utils import MINRESInfo, MINRESSettings
from torchsparsegradutils.utils.minres import minres

ATOL, RTOL = Tolerances.iterative(torch.float64)


def _run_minres(rhs_shape, shifts=None, matrix_batch_shape=torch.Size([])):
    # generate random RHS and SPD matrix
    size = rhs_shape[-2] if len(rhs_shape) > 1 else rhs_shape[-1]
    rhs = torch.randn(rhs_shape, dtype=torch.float64)
    matrix = torch.randn(*matrix_batch_shape, size, size, dtype=torch.float64)
    matrix = matrix @ matrix.mT
    matrix = matrix / torch.linalg.vector_norm(matrix)
    matrix = matrix + torch.eye(size, dtype=torch.float64) * 1e-1
    # compute minres
    if shifts is not None:
        shifts = shifts.type_as(rhs)
    settings = MINRESSettings(minres_tolerance=1e-6)
    solves = minres(matrix, rhs=rhs, value=-1, shifts=shifts, settings=settings)
    # adjust matrix dims
    while matrix.dim() < len(rhs_shape):
        matrix = matrix.unsqueeze(0)
    # apply shifts to matrix for exact solve
    if shifts is not None:
        eye = torch.eye(size, dtype=torch.float64)
        matrix = matrix - eye * shifts.view(*shifts.shape, *[1 for _ in matrix.shape])
    # compute direct solve
    actual = torch.linalg.solve(-matrix, rhs.unsqueeze(-1) if rhs.dim() == 1 else rhs)
    if rhs.dim() == 1:
        actual = actual.squeeze(-1)
    # assert closeness
    assert torch.allclose(solves, actual, atol=ATOL, rtol=RTOL)


def test_minres_vec():
    _run_minres(torch.Size([20]))


def test_minres_vec_multiple_shifts():
    shifts = torch.tensor([0.0, 1.0, 2.0])
    _run_minres(torch.Size([5]), shifts=shifts)


def test_minres_mat():
    _run_minres(torch.Size([20, 5]))
    _run_minres(torch.Size([3, 20, 5]))
    _run_minres(torch.Size([3, 20, 5]), matrix_batch_shape=torch.Size([3]))
    _run_minres(torch.Size([20, 5]), matrix_batch_shape=torch.Size([3]))


def test_minres_mat_multiple_shifts():
    shifts = torch.tensor([0.0, 1.0, 2.0])
    _run_minres(torch.Size([20, 5]), shifts=shifts)
    _run_minres(torch.Size([3, 20, 5]), shifts=shifts)
    _run_minres(torch.Size([3, 20, 5]), matrix_batch_shape=torch.Size([3]), shifts=shifts)
    _run_minres(torch.Size([20, 5]), matrix_batch_shape=torch.Size([3]), shifts=shifts)


def _ill_conditioned_spd(size, log10_min_eigenvalue, dtype=torch.float64):
    generator = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(size, size, dtype=dtype, generator=generator))
    eigenvalues = torch.logspace(log10_min_eigenvalue, 0, size, dtype=dtype)
    return q @ torch.diag(eigenvalues) @ q.mT


def _true_relative_residual(matrix, solution, rhs):
    return torch.linalg.vector_norm(rhs - matrix @ solution, dim=-2) / torch.linalg.vector_norm(rhs, dim=-2)


def test_minres_residual_criterion_converges_every_rhs():
    matrix = _ill_conditioned_spd(30, -3)
    rhs = torch.randn(30, 4, dtype=torch.float64)
    # Scale the right-hand sides differently: each one must converge, not their mean
    rhs = rhs * torch.tensor([1.0, 1e3, 1e-3, 1.0], dtype=torch.float64)

    solution, info = minres(matrix, rhs, tolerance=1e-8, return_info=True)

    assert isinstance(info, MINRESInfo)
    assert info.reason in ("converged", "recursive_converged")
    assert info.tolerance == 1e-8
    assert info.matvecs == info.iterations + 2
    assert info.recursive_relative_residual.shape == (4,)
    assert (info.recursive_relative_residual <= 1e-8).all()
    torch.testing.assert_close(info.true_relative_residual, _true_relative_residual(matrix, solution, rhs))
    assert torch.equal(info.converged, info.true_relative_residual <= 1e-8)


def test_minres_max_iter_is_not_capped_by_problem_size():
    size = 30
    matrix = _ill_conditioned_spd(size, -6)
    rhs = torch.randn(size, 2, dtype=torch.float64)

    _, info = minres(matrix, rhs, tolerance=1e-10, max_iter=1000, return_info=True)

    assert info.iterations > size + 1
    assert info.reason in ("converged", "recursive_converged")


def test_minres_explicit_max_iter_is_honored_and_reported():
    matrix = _ill_conditioned_spd(30, -6)
    rhs = torch.randn(30, 2, dtype=torch.float64)

    with pytest.warns(UserWarning, match="MINRES terminated after 3 iterations"):
        _, info = minres(matrix, rhs, tolerance=1e-10, max_iter=3, return_info=True)

    assert info.iterations == 3
    assert info.reason == "max_iter"
    assert not info.converged.any()


def test_minres_tiny_and_zero_rhs():
    matrix = _ill_conditioned_spd(20, -2)
    rhs = torch.randn(20, 3, dtype=torch.float64)
    rhs[:, 1] = 0
    rhs[:, 2] *= 1e-20

    solution, info = minres(matrix, rhs, tolerance=1e-10, return_info=True)

    assert torch.isfinite(solution).all()
    assert torch.equal(solution[:, 1], torch.zeros(20, dtype=torch.float64))
    expected = torch.linalg.solve(matrix, rhs[:, 2])
    torch.testing.assert_close(solution[:, 2], expected, atol=0, rtol=1e-8)
    assert info.recursive_relative_residual[1] == 0
    assert info.true_relative_residual[1] == 0
    assert info.converged[1]


def test_minres_singular_consistent_system_returns_minimum_norm_solution():
    # Graph Laplacian of a connected graph: PSD with a one-dimensional null space
    size = 12
    generator = torch.Generator().manual_seed(0)
    weights = torch.rand(size, size, dtype=torch.float64, generator=generator)
    weights = weights + weights.mT + torch.diag_embed(torch.ones(size - 1, dtype=torch.float64), offset=1)
    weights = (weights + weights.mT).fill_diagonal_(0)
    laplacian = torch.diag(weights.sum(-1)) - weights
    rhs = torch.randn(size, 2, dtype=torch.float64, generator=generator)
    rhs = rhs - rhs.mean(-2, keepdim=True)

    solution, info = minres(laplacian, rhs, tolerance=1e-12, return_info=True)

    assert info.reason == "converged"
    torch.testing.assert_close(solution, torch.linalg.pinv(laplacian) @ rhs, atol=1e-10, rtol=1e-8)


def test_minres_preconditioned_residual_criterion():
    size = 30
    scaling = torch.linspace(1, 100, size, dtype=torch.float64)
    matrix = scaling.sqrt()[:, None] * _ill_conditioned_spd(size, -2) * scaling.sqrt()[None, :]
    rhs = torch.randn(size, 2, dtype=torch.float64)

    solution, info = minres(
        matrix, rhs, tolerance=1e-8, preconditioner=lambda v: v / scaling.unsqueeze(-1), return_info=True
    )

    assert (info.recursive_relative_residual <= 1e-8).all()
    torch.testing.assert_close(info.true_relative_residual, _true_relative_residual(matrix, solution, rhs))
    assert (info.true_relative_residual <= 1e-6).all()


@pytest.mark.parametrize(
    "rhs_shape, shifts, info_shape",
    [
        ((20,), None, (1,)),
        ((20, 5), None, (5,)),
        ((3, 20, 5), None, (3, 5)),
        ((20, 5), torch.tensor([0.0, 2.0, 5.0]), (3, 5)),
        ((3, 20, 5), torch.tensor([0.0, 2.0, 5.0]), (3, 3, 5)),
    ],
)
def test_minres_info_shapes_and_shifted_true_residual(rhs_shape, shifts, info_shape):
    size = 20
    matrix = _ill_conditioned_spd(size, -1)
    rhs = torch.randn(rhs_shape, dtype=torch.float64)
    if shifts is not None:
        shifts = shifts.to(torch.float64)

    solution, info = minres(matrix, rhs, shifts=shifts, value=-1, tolerance=1e-10, return_info=True)

    for tensor in (info.converged, info.recursive_relative_residual, info.true_relative_residual):
        assert tensor.shape == info_shape
    assert info.converged.all()

    rhs_mat = rhs.unsqueeze(-1) if rhs.dim() == 1 else rhs
    solutions = solution.unsqueeze(-1) if rhs.dim() == 1 else solution
    shift_values = torch.zeros(1, dtype=torch.float64) if shifts is None else shifts
    if shifts is None:
        solutions = solutions.unsqueeze(0)
    for shift, shift_solution, shift_info in zip(
        shift_values, solutions, info.true_relative_residual.reshape(len(shift_values), -1)
    ):
        shifted = -matrix + shift * torch.eye(size, dtype=torch.float64)
        expected = _true_relative_residual(shifted, shift_solution, rhs_mat)
        torch.testing.assert_close(shift_info, expected.reshape(-1))


def test_minres_update_criterion_is_still_available():
    matrix = _ill_conditioned_spd(20, -1)
    rhs = torch.randn(20, 3, dtype=torch.float64)
    settings = MINRESSettings(minres_tolerance=1e-8, minres_convergence="update")

    solution, info = minres(matrix, rhs, settings=settings, return_info=True)

    assert info.iterations % 10 == 0
    torch.testing.assert_close(solution, torch.linalg.solve(matrix, rhs), atol=1e-6, rtol=1e-6)


def test_minres_invalid_arguments():
    matrix = _ill_conditioned_spd(5, -1)
    rhs = torch.randn(5, dtype=torch.float64)
    with pytest.raises(ValueError, match="tolerance"):
        minres(matrix, rhs, tolerance=-1.0)
    with pytest.raises(ValueError, match="max_iter"):
        minres(matrix, rhs, max_iter=-1)
    with pytest.raises(ValueError, match="minres_convergence"):
        minres(matrix, rhs, settings=MINRESSettings(minres_convergence="bogus"))
    with pytest.raises(ValueError, match="preconditioner and nonzero shifts"):
        minres(
            matrix,
            rhs,
            shifts=torch.tensor([0.0, 1.0], dtype=torch.float64),
            preconditioner=lambda v: v.clone(),
            return_info=True,
        )
