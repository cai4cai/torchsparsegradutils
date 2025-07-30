import torch
import pytest
import torchsparsegradutils.cupy as tsgucupy
from torchsparsegradutils.utils import convert_coo_to_csr
from torchsparsegradutils.cupy.cupy_sparse_solve import sparse_solve_c4t

# TODO: testing different solvers
# NOTE: I think methods like cg and lsqr are for B being a vector
# from cupyx.scipy.sparse.linalg import cg, lsqr
# from scipy.sparse.linalg import minres, lsqr, cg, bicgstab, gmres

# devices
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))

# test parameters
TEST_DATA = [
    ("unbat", (12, 12), (12, 6), 32),
]
INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]


def data_id(d):
    return d[0]


def dtype_id(d):
    return str(d).split(".")[-1]


def layout_id(layout):
    return "coo" if layout == torch.sparse_coo else "csr"


def device_id(device):
    return str(device)


@pytest.fixture(params=TEST_DATA, ids=lambda d: data_id(d))
def shapes(request):
    return request.param


@pytest.fixture(params=VALUE_DTYPES, ids=lambda d: dtype_id(d))
def value_dtype(request):
    return request.param


@pytest.fixture(params=INDEX_DTYPES, ids=lambda d: dtype_id(d))
def index_dtype(request):
    return request.param


@pytest.fixture(params=DEVICES, ids=lambda d: device_id(d))
def device(request):
    return request.param


@pytest.fixture(params=LAYOUTS, ids=[layout_id(d) for d in LAYOUTS])
def layout(request):
    return request.param


def make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=0):
    M = torch.randn(n, n, dtype=value_dtype, device=device)
    A_dense = M @ M.t() + n * torch.eye(n, dtype=value_dtype, device=device)
    if nz > 0:
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        idxs = torch.nonzero(mask, as_tuple=False)
        sel = idxs[torch.randperm(idxs.size(0), device=device)[:nz]]
        A_dense[sel[:, 0], sel[:, 1]] = 0
    idx = A_dense.nonzero(as_tuple=False).t().to(index_dtype)
    vals = A_dense[idx[0], idx[1]]
    if layout == torch.sparse_coo:
        A = torch.sparse_coo_tensor(idx, vals, (n, n), dtype=value_dtype, device=device).coalesce()
    else:
        A_coo = torch.sparse_coo_tensor(idx, vals, (n, n), dtype=value_dtype, device=device).coalesce()
        A = convert_coo_to_csr(A_coo)
    return A, A_dense


@pytest.mark.flaky(reruns=5)
def test_solve_forward_cupy(layout, device, value_dtype, index_dtype, shapes):
    _, A_shape, B_shape, num_zero = shapes
    n = A_shape[0]
    A_sp, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=num_zero)
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)
    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_solve_c4t(A_sp, B)
    assert torch.allclose(X_test, X_ref, atol=1e-8)


@pytest.mark.flaky(reruns=5)
def test_solve_backward_cupy(layout, device, value_dtype, index_dtype, shapes):
    _, A_shape, B_shape, num_zero = shapes
    n = A_shape[0]
    A_sp, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=num_zero)
    A_sp = A_sp.detach().clone().requires_grad_()
    Ad = A_dense.detach().clone().requires_grad_()
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Bd = B.clone().detach().requires_grad_()
    Bd2 = Bd.clone().detach().requires_grad_()
    # forward
    X1 = sparse_solve_c4t(A_sp, Bd)
    X2 = torch.linalg.solve(Ad, Bd2)
    grad_out = torch.rand_like(X1)
    X1.backward(grad_out)
    X2.backward(grad_out)
    nz = A_sp.grad.to_dense() != 0
    assert torch.allclose(A_sp.grad.to_dense()[nz], Ad.grad[nz], atol=1e-8)
    assert torch.allclose(Bd.grad, Bd2.grad, atol=1e-5)
