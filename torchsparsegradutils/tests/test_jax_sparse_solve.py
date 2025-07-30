import pytest
import torch
import torchsparsegradutils.jax as tsgujax
from torchsparsegradutils.utils import convert_coo_to_csr

pytest.importorskip("jax")
if not tsgujax.have_jax:
    pytest.skip("JAX bindings unavailable, skipping jax tests", allow_module_level=True)
else:
    from jax.scipy.sparse.linalg import cg, bicgstab

# NOTE: JAX seems to use GPU memory preallocation, which causes:
# 90% of GPU:0 to be preallocated by default even for CPU tests
# CPU tests show GPU memory usage (~500MB) on all available GPUs

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
SOLVERS = [None, cg, bicgstab]


def data_id(d):
    return d[0]


def dtype_id(d):
    return str(d).split(".")[-1]


def layout_id(layout):
    return "coo" if layout == torch.sparse_coo else "csr"


def device_id(device):
    return str(device)


def solve_id(s):
    if s:
        return s.__name__
    return "default"


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


@pytest.fixture(params=SOLVERS, ids=[solve_id(s) for s in SOLVERS])
def solver(request):
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
def test_solve_forward_j4t(layout, device, value_dtype, index_dtype, solver, shapes):
    _, A_shape, B_shape, num_zero = shapes
    n = A_shape[0]
    A_sp, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=num_zero)
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)
    X_ref = torch.linalg.solve(A_dense, B)
    X_test = tsgujax.sparse_solve_j4t(A_sp, B, solve=solver)

    assert torch.allclose(X_test, X_ref, atol=1e-2)


@pytest.mark.flaky(reruns=5)
def test_solve_backward_j4t(layout, device, value_dtype, index_dtype, solver, shapes):
    _, A_shape, B_shape, num_zero = shapes
    n = A_shape[0]
    A_sp, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=num_zero)
    A_sp = A_sp.detach().clone().requires_grad_()
    Ad = A_dense.detach().clone().requires_grad_()
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Bd1 = B.clone().detach().requires_grad_()
    Bd2 = B.clone().detach().requires_grad_()
    X1 = tsgujax.sparse_solve_j4t(A_sp, Bd1, solve=solver)
    X2 = torch.linalg.solve(Ad, Bd2)
    grad_out = torch.rand_like(X1)
    X1.backward(grad_out)
    X2.backward(grad_out)
    nz = A_sp.grad.to_dense() != 0
    # same tolerances as generic solve
    assert torch.allclose(A_sp.grad.to_dense()[nz], Ad.grad[nz], atol=1e-2)
    assert torch.allclose(Bd1.grad, Bd2.grad, atol=1e-1)
