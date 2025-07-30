# for sparse generic solve

import torch
import pytest
from torchsparsegradutils.sparse_solve import sparse_generic_solve
from torchsparsegradutils.utils import convert_coo_to_csr
from torchsparsegradutils.utils import linear_cg, bicgstab, minres

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

TEST_DATA = [
    # name  A_shape, B_shape, A_num_zero
    # ("unbat", (4, 4), (4, 2), 4),
    ("unbat", (12, 12), (12, 6), 32),
    # ("bat", (2, 4, 4), (2, 4, 2), 4),
    # ("bat", (4, 12, 12), (4, 12, 6), 32),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]
SOLVES = [linear_cg, bicgstab, minres]

# TRANSPOSE = [True, False]


# Define Test Names:
def data_id(shapes):
    return shapes[0]


def device_id(device):
    return str(device)


def dtype_id(dtype):
    return str(dtype).split(".")[-1]


def layout_id(layout):
    return "coo" if layout == torch.sparse_coo else "csr"


def solve_id(solve):
    return solve.__name__


# Define Fixtures


@pytest.fixture(params=TEST_DATA, ids=[data_id(d) for d in TEST_DATA])
def shapes(request):
    return request.param


@pytest.fixture(params=VALUE_DTYPES, ids=[dtype_id(d) for d in VALUE_DTYPES])
def value_dtype(request):
    return request.param


@pytest.fixture(params=INDEX_DTYPES, ids=[dtype_id(d) for d in INDEX_DTYPES])
def index_dtype(request):
    return request.param


@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


@pytest.fixture(params=LAYOUTS, ids=[layout_id(d) for d in LAYOUTS])
def layout(request):
    return request.param


@pytest.fixture(params=SOLVES, ids=[solve_id(d) for d in SOLVES])
def solve(request):
    return request.param


def make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=0):
    # generate dense SPD: M M^T + n*I
    M = torch.randn(n, n, dtype=value_dtype, device=device)
    A_dense = M @ M.transpose(-2, -1) + n * torch.eye(n, dtype=value_dtype, device=device)

    # Randomly set 'nz' off-diagonal values to zero before extracting indices
    if nz > 0:
        # Find off-diagonal indices
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=device)
        off_diag_indices = torch.nonzero(off_diag_mask, as_tuple=False)
        num_off_diag = off_diag_indices.size(0)
        if nz < num_off_diag and num_off_diag > 0:
            zero_indices = off_diag_indices[torch.randperm(num_off_diag, device=device)[:nz]]
            A_dense[zero_indices[:, 0], zero_indices[:, 1]] = 0

    idx = A_dense.nonzero(as_tuple=False).t().to(index_dtype)
    vals = A_dense[idx[0], idx[1]].clone()

    if layout == torch.sparse_coo:
        A = torch.sparse_coo_tensor(idx, vals, (n, n), dtype=value_dtype, device=device).coalesce()
    else:
        A_coo = torch.sparse_coo_tensor(idx, vals, (n, n), dtype=value_dtype, device=device).coalesce()
        A = convert_coo_to_csr(A_coo)
    return A, A_dense


@pytest.mark.flaky(reruns=5)
def test_solve_forward_routine(layout, solve, device, value_dtype, index_dtype, shapes):

    _, A_shape, B_shape, num_zero = shapes
    n = A_shape[0]
    A, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=num_zero)
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)

    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=solve)

    # NOTE: bicgstab seems to have tighter tolerances
    # NOTE: increasing num_zero increases the error

    if solve == bicgstab:
        assert torch.allclose(X_test, X_ref, atol=1e-7)
    else:
        assert torch.allclose(X_test, X_ref, atol=1e-2)


@pytest.mark.flaky(reruns=5)
def test_solve_backward_routine(layout, solve, device, value_dtype, index_dtype, shapes):

    _, A_shape, B_shape, num_zero = shapes
    n = A_shape[0]
    As1, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=num_zero)

    Ad2 = A_dense.detach().clone().requires_grad_()
    As1.requires_grad_()
    Bd1 = torch.rand(*B_shape, dtype=value_dtype, device=device).requires_grad_()
    Bd2 = Bd1.clone().detach().requires_grad_()

    res_ref = torch.linalg.solve(Ad2, Bd2)
    res_test = sparse_generic_solve(As1, Bd1, solve=solve)

    grad_output = torch.rand_like(res_test)
    res_ref.backward(grad_output)
    res_test.backward(grad_output)

    nz_mask = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], atol=1e-2)
    assert torch.allclose(Bd1.grad, Bd2.grad, atol=1e-1)  # NOTE: Seems a bit loose
