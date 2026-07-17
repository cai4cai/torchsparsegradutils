# MIT-licensed code imported from https://github.com/cornellius-gp/linear_operator
# Minor modifications for torchsparsegradutils to remove dependencies

import pytest
import torch
from test_config import DEVICES, Tolerances

import torchsparsegradutils
from torchsparsegradutils.solvers.minres import minres
from torchsparsegradutils.utils import MINRESSettings

ATOL, RTOL = Tolerances.iterative(torch.float64)


def _run_minres(rhs_shape, shifts=None, matrix_batch_shape=torch.Size([])):
    # generate random RHS and SPD matrix
    size = rhs_shape[-2] if len(rhs_shape) > 1 else rhs_shape[-1]
    rhs = torch.randn(rhs_shape, dtype=torch.float64)
    matrix = torch.randn(*matrix_batch_shape, size, size, dtype=torch.float64)
    matrix = matrix @ matrix.mT
    matrix = matrix / matrix.norm()
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


def _make_spd(size, batch_size=None, device="cpu"):
    """SPD dense matrix (size, size), or one batched matrix (batch_size, size, size)."""
    shape = (batch_size, size, size) if batch_size else (size, size)
    matrix = torch.randn(*shape, dtype=torch.float64, device=device)
    matrix = matrix @ matrix.mT
    matrix = matrix / matrix.norm()
    return matrix + torch.eye(size, dtype=torch.float64, device=device) * 1e-1


# A right-hand side given as one batched matrix with shape (batch_size, n, n_rhs)
# must solve each batch item exactly as an unbatched (n, n_rhs) solve would.
@pytest.mark.parametrize("device", DEVICES, ids=str)
def test_minres_batched_rhs_matches_loop(device):
    torch.manual_seed(42)
    batch_size, n, n_rhs = 3, 32, 4
    matrix = _make_spd(n, batch_size, device)
    rhs = torch.randn(batch_size, n, n_rhs, dtype=torch.float64, device=device)
    settings = MINRESSettings(minres_tolerance=1e-10)
    solves = minres(matrix, rhs=rhs, settings=settings)
    assert solves.shape == rhs.shape
    for b in range(batch_size):
        single = minres(matrix[b], rhs=rhs[b], settings=settings)
        assert torch.allclose(solves[b], single, atol=1e-8, rtol=1e-8)
    actual = torch.linalg.solve(matrix, rhs)
    assert torch.allclose(solves, actual, atol=ATOL, rtol=RTOL)


# A sparse matrix passed as the operator (matvec routed through the BatchedOperator
# adapter, i.e. tsgu::spmm on CUDA) must match the dense operator's solve.
@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr], ids=["coo", "csr"])
def test_minres_sparse_operator_matches_dense(device, layout):
    torch.manual_seed(42)
    n, n_rhs = 32, 4
    matrix = _make_spd(n, device=device)
    rhs = torch.randn(n, n_rhs, dtype=torch.float64, device=device)
    settings = MINRESSettings(minres_tolerance=1e-10)
    expected = minres(matrix, rhs=rhs, settings=settings)
    matrix_sparse = matrix.to_sparse_coo() if layout == torch.sparse_coo else matrix.to_sparse_csr()
    solves = minres(matrix_sparse, rhs=rhs, settings=settings)
    assert solves.shape == rhs.shape
    assert torch.allclose(solves, expected, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr], ids=["coo", "csr"])
def test_minres_batched_sparse_operator_matches_dense(device, layout):
    torch.manual_seed(42)
    batch_size, n, n_rhs = 3, 32, 4
    matrix = _make_spd(n, batch_size, device)
    rhs = torch.randn(batch_size, n, n_rhs, dtype=torch.float64, device=device)
    settings = MINRESSettings(minres_tolerance=1e-10)
    expected = minres(matrix, rhs=rhs, settings=settings)
    matrix_sparse = matrix.to_sparse_coo() if layout == torch.sparse_coo else matrix.to_sparse_csr()
    solves = minres(matrix_sparse, rhs=rhs, settings=settings)
    assert solves.shape == rhs.shape
    assert torch.allclose(solves, expected, atol=1e-8, rtol=1e-8)
