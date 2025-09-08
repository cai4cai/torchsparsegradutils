# MIT-licensed code imported from https://github.com/cornellius-gp/linear_operator
# Minor modifications for torchsparsegradutils to remove dependencies

import random

import pytest
import torch

import torchsparsegradutils
from torchsparsegradutils.utils import MINRESSettings
from torchsparsegradutils.utils.minres import minres


@pytest.fixture(autouse=True)
def set_seed():
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


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
    assert torch.allclose(solves, actual, atol=1e-3, rtol=1e-4)


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
