import pytest
import torch
from test_config import DEVICES, Tolerances

from torchsparsegradutils.utils import bicgstab

# Get iterative solver tolerances for float64 (used throughout this file)
ATOL, RTOL = Tolerances.iterative(torch.float64)


def _id_device(d):
    return str(d)


@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param


def test_bicgstab(device):
    # setup SPD test problem
    size = 100
    matrix_dense = torch.randn(size, size, dtype=torch.float64, device=device) + 10 * torch.eye(size, device=device)
    matrix_sparse = matrix_dense.to_sparse_csr()
    rhs = torch.randn(size, dtype=torch.float64, device=device)
    # reference solution
    actual = torch.linalg.solve(matrix_dense, rhs)
    # test various bicgstab call signatures
    solves = bicgstab(matrix_dense, rhs=rhs)
    assert torch.allclose(solves, actual, atol=ATOL, rtol=RTOL)
    solves = bicgstab(matrix_dense.matmul, rhs=rhs)
    assert torch.allclose(solves, actual, atol=ATOL, rtol=RTOL)
    solves = bicgstab(matrix_sparse.matmul, rhs=rhs)
    assert torch.allclose(solves, actual, atol=ATOL, rtol=RTOL)
    solves = bicgstab(matrix_sparse, rhs=rhs)
    assert torch.allclose(solves, actual, atol=ATOL, rtol=RTOL)


def test_bicgstab_2d_rhs(device):
    size = 100
    # build SPD test problem
    matrix_dense = torch.randn(size, size, dtype=torch.float64, device=device) + 10 * torch.eye(
        size, dtype=torch.float64, device=device
    )
    matrix_sparse = matrix_dense.to_sparse_csr()

    # multiple RHS columns
    rhs2d = torch.randn(size, 5, dtype=torch.float64, device=device)

    # reference solution
    actual = torch.linalg.solve(matrix_dense, rhs2d)

    # dense-matrix API
    solves = bicgstab(matrix_dense, rhs=rhs2d)
    assert torch.allclose(solves, actual, atol=ATOL, rtol=RTOL)

    # sparse-matrix API
    solves = bicgstab(matrix_sparse, rhs=rhs2d)
    assert torch.allclose(solves, actual, atol=ATOL, rtol=RTOL)
