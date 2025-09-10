import pytest
import torch

import torchsparsegradutils.jax as tsgujax
from torchsparsegradutils.utils import convert_coo_to_csr
from torchsparsegradutils.utils.random_sparse import make_spd_sparse

pytest.importorskip("jax")
if not tsgujax.have_jax:
    pytest.skip("JAX bindings unavailable, skipping jax tests", allow_module_level=True)
else:
    from jax.scipy.sparse.linalg import bicgstab, cg

# NOTE: JAX seems to use GPU memory preallocation, which causes:
# 90% of GPU:0 to be preallocated by default even for CPU tests
# CPU tests show GPU memory usage (~500MB) on all available GPUs

# devices
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))

# test parameters
TEST_DATA = [
    ("vector_1d", (12, 12), (12,), 32),
    ("vector_2d", (12, 12), (12, 1), 32),
    ("multi_rhs", (12, 12), (12, 6), 32),
]
INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]
SOLVERS = [None, cg, bicgstab]

ATOL = 1e-6


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


@pytest.mark.flaky(reruns=5)
def test_solve_forward_j4t(layout, device, value_dtype, index_dtype, solver, shapes):
    _, A_shape, B_shape, num_zero = shapes
    n = A_shape[0]
    A_sp, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=num_zero)
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)
    X_ref = torch.linalg.solve(A_dense, B)
    X_test = tsgujax.sparse_solve_j4t(A_sp, B, solve=solver)

    assert torch.allclose(X_test, X_ref, atol=ATOL)


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
    assert torch.allclose(A_sp.grad.to_dense()[nz], Ad.grad[nz], atol=ATOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_jax_cg_kwargs(device, value_dtype, layout):
    """Test sparse_solve_j4t with CG solver kwargs."""
    n = 10
    A_sp, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    # Test with custom CG kwargs
    X_ref = torch.linalg.solve(A_dense, B)
    X_test = tsgujax.sparse_solve_j4t(A_sp, B, solve=cg, transpose_solve=cg, tol=1e-6, atol=1e-8, maxiter=500)

    assert torch.allclose(X_test, X_ref, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_jax_bicgstab_kwargs(device, value_dtype, layout):
    """Test sparse_solve_j4t with BiCGSTAB solver kwargs."""
    n = 10
    A_sp, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    # Test with custom BiCGSTAB kwargs
    X_ref = torch.linalg.solve(A_dense, B)
    X_test = tsgujax.sparse_solve_j4t(
        A_sp, B, solve=bicgstab, transpose_solve=bicgstab, tol=1e-6, atol=1e-8, maxiter=1000
    )

    assert torch.allclose(X_test, X_ref, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_jax_kwargs_backward_pass(device, value_dtype, layout):
    """Test that JAX kwargs work correctly during backward pass."""
    n = 10
    A_sp, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)

    # Set up tensors with gradients
    A_sp1 = A_sp.clone().requires_grad_()
    Ad2 = A_dense.detach().clone().requires_grad_()
    Bd1 = torch.rand(n, dtype=value_dtype, device=device).requires_grad_()
    Bd2 = Bd1.clone().detach().requires_grad_()

    # Test with custom kwargs
    res_ref = torch.linalg.solve(Ad2, Bd2)
    res_test = tsgujax.sparse_solve_j4t(A_sp1, Bd1, solve=cg, transpose_solve=cg, tol=1e-6, atol=1e-8, maxiter=500)

    # Backward pass
    grad_output = torch.rand_like(res_test)
    res_ref.backward(grad_output)
    res_test.backward(grad_output)

    # Check gradients
    nz_mask = A_sp1.grad.to_dense() != 0.0
    assert torch.allclose(A_sp1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], atol=ATOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_jax_multiple_kwargs(device, value_dtype):
    """Test sparse_solve_j4t with multiple kwargs (like used in benchmarks)."""
    n = 10
    layout = torch.sparse_csr  # Use CSR for this test
    A_sp, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    # Test with multiple kwargs similar to the benchmark suite
    X_ref = torch.linalg.solve(A_dense, B)
    X_test = tsgujax.sparse_solve_j4t(
        A_sp, B, solve=bicgstab, transpose_solve=bicgstab, tol=1e-5, atol=1e-8, maxiter=1000
    )

    assert torch.allclose(X_test, X_ref, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_jax_kwargs_with_different_solvers():
    """Test that different JAX solvers with their respective kwargs produce similar results."""
    device = torch.device("cpu")
    value_dtype = torch.float64  # Use higher precision for better comparison
    layout = torch.sparse_csr

    n = 10
    A_sp, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    # Reference solution
    X_ref = torch.linalg.solve(A_dense, B)

    # Test CG with kwargs
    X_cg = tsgujax.sparse_solve_j4t(A_sp, B, solve=cg, transpose_solve=cg, tol=1e-8, atol=1e-10, maxiter=1000)

    # Test BiCGSTAB with kwargs
    X_bicgstab = tsgujax.sparse_solve_j4t(
        A_sp, B, solve=bicgstab, transpose_solve=bicgstab, tol=1e-8, atol=1e-10, maxiter=1000
    )

    # All solutions should be close to reference
    assert torch.allclose(X_cg, X_ref, atol=ATOL)
    assert torch.allclose(X_bicgstab, X_ref, atol=ATOL)

    # Solutions should be close to each other
    assert torch.allclose(X_cg, X_bicgstab, atol=ATOL)
