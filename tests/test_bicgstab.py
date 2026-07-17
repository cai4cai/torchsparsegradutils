import pytest
import torch
from test_config import DEVICES, Tolerances

from torchsparsegradutils.utils import bicgstab

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


# --- Batched-iterate tests (spec/commit.md commit 17) ---


def _small_system(n, dtype, device, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(n, n, dtype=dtype, device=device) + 10 * torch.eye(n, dtype=dtype, device=device)
    return A


def test_bicgstab_multi_rhs_matches_column_solves():
    # A matrix of right-hand sides (n, n_rhs) must match a loop of vector solves.
    n, n_rhs = 32, 4
    device = torch.device("cpu")
    A = _small_system(n, torch.float64, device)
    rhs = torch.randn(n, n_rhs, dtype=torch.float64, device=device)

    solves = bicgstab(A, rhs=rhs)
    for j in range(n_rhs):
        column = bicgstab(A, rhs=rhs[:, j])
        assert torch.allclose(solves[:, j], column, atol=1e-10, rtol=1e-10)


def test_bicgstab_batched_rhs_matches_item_solves():
    # A batched right-hand side (batch_size, n, n_rhs) with a batched dense
    # operator must match a loop of per-item solves.
    batch_size, n, n_rhs = 3, 24, 2
    device = torch.device("cpu")
    torch.manual_seed(1)
    A = torch.randn(batch_size, n, n, dtype=torch.float64, device=device) + 10 * torch.eye(
        n, dtype=torch.float64, device=device
    )
    rhs = torch.randn(batch_size, n, n_rhs, dtype=torch.float64, device=device)

    solves = bicgstab(A, rhs=rhs)
    assert solves.shape == rhs.shape
    for i in range(batch_size):
        item = bicgstab(A[i], rhs=rhs[i])
        assert torch.allclose(solves[i], item, atol=1e-10, rtol=1e-10)


def test_bicgstab_sparse_operator_matches_dense_cpu():
    # Sparse CSR and COO operators (descriptor matvec path) must match the
    # dense-operator result.
    n, n_rhs = 32, 4
    device = torch.device("cpu")
    A = _small_system(n, torch.float64, device, seed=2)
    rhs = torch.randn(n, n_rhs, dtype=torch.float64, device=device)

    expected = bicgstab(A, rhs=rhs)
    for A_sparse in (A.to_sparse_csr(), A.to_sparse_coo()):
        solves = bicgstab(A_sparse, rhs=rhs)
        assert torch.allclose(solves, expected, atol=1e-6, rtol=1e-6)


def test_bicgstab_sparse_operator_cuda():
    from torchsparsegradutils._dispatch import backend_available

    if not (torch.cuda.is_available() and backend_available()):
        pytest.skip("CUDA backend not available")
    n, n_rhs = 32, 2
    device = torch.device("cuda")
    A = _small_system(n, torch.float64, device, seed=3)
    rhs = torch.randn(n, n_rhs, dtype=torch.float64, device=device)

    expected = bicgstab(A, rhs=rhs)
    solves = bicgstab(A.to_sparse_csr(), rhs=rhs)
    assert torch.allclose(solves, expected, atol=1e-6, rtol=1e-6)
