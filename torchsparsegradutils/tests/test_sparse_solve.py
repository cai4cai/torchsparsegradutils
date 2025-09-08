# for sparse generic solve

import pytest
import torch

from torchsparsegradutils.sparse_solve import sparse_generic_solve
from torchsparsegradutils.utils import bicgstab, convert_coo_to_csr, linear_cg, minres
from torchsparsegradutils.utils.random_sparse import make_spd_sparse

DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

TEST_DATA = [
    # name  A_shape, B_shape, A_num_zero
    ("vector_1d", (12, 12), (12,), 32),
    ("vector_2d", (12, 12), (12, 1), 32),
    ("multi_rhs", (12, 12), (12, 6), 32),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]
SOLVES = [None, linear_cg, bicgstab, minres]

ATOL = 1e-6


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
    if solve:
        return solve.__name__
    return "default"


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


@pytest.mark.flaky(reruns=5)
def test_solve_forward_routine(layout, solve, device, value_dtype, index_dtype, shapes):

    _, A_shape, B_shape, num_zero = shapes
    n = A_shape[0]
    A, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=num_zero)
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)

    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=solve, transpose_solve=solve)

    assert torch.allclose(X_test, X_ref, atol=ATOL)


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
    res_test = sparse_generic_solve(As1, Bd1, solve=solve, transpose_solve=solve)

    grad_output = torch.rand_like(res_test)
    res_ref.backward(grad_output)
    res_test.backward(grad_output)

    nz_mask = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], atol=ATOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_linear_cg_kwargs(device, value_dtype, layout):
    """Test sparse_generic_solve with LinearCGSettings kwargs."""
    from torchsparsegradutils.utils.linear_cg import LinearCGSettings

    n = 10
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    # Test with custom LinearCGSettings
    settings = LinearCGSettings(
        cg_tolerance=ATOL, max_cg_iterations=500, terminate_cg_by_size=False, verbose_linalg=False
    )

    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=linear_cg, transpose_solve=linear_cg, settings=settings)

    assert torch.allclose(X_test, X_ref, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_bicgstab_kwargs(device, value_dtype, layout):
    """Test sparse_generic_solve with BICGSTABSettings kwargs."""
    from torchsparsegradutils.utils.bicgstab import BICGSTABSettings

    n = 10
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    # Test with custom BICGSTABSettings
    settings = BICGSTABSettings(reltol=ATOL, abstol=1e-8, matvec_max=1000, precon=None)

    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=bicgstab, transpose_solve=bicgstab, settings=settings)

    assert torch.allclose(X_test, X_ref, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_minres_kwargs(device, value_dtype, layout):
    """Test sparse_generic_solve with MINRESSettings kwargs."""
    from torchsparsegradutils.utils.minres import MINRESSettings

    n = 10
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    # Test with custom MINRESSettings
    settings = MINRESSettings(minres_tolerance=ATOL, max_cg_iterations=500, verbose_linalg=False)

    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=minres, transpose_solve=minres, settings=settings)

    assert torch.allclose(X_test, X_ref, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_kwargs_backward_pass(device, value_dtype, layout):
    """Test that kwargs work correctly during backward pass."""
    from torchsparsegradutils.utils.linear_cg import LinearCGSettings

    n = 10
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)

    # Set up tensors with gradients
    As1 = A.clone().requires_grad_()
    Ad2 = A_dense.detach().clone().requires_grad_()
    Bd1 = torch.rand(n, dtype=value_dtype, device=device).requires_grad_()
    Bd2 = Bd1.clone().detach().requires_grad_()

    # Test with custom settings
    settings = LinearCGSettings(
        cg_tolerance=ATOL, max_cg_iterations=500, terminate_cg_by_size=False, verbose_linalg=False
    )

    res_ref = torch.linalg.solve(Ad2, Bd2)
    res_test = sparse_generic_solve(As1, Bd1, solve=linear_cg, transpose_solve=linear_cg, settings=settings)

    # Backward pass
    grad_output = torch.rand_like(res_test)
    res_ref.backward(grad_output)
    res_test.backward(grad_output)

    # Check gradients
    nz_mask = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], atol=ATOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_multiple_kwargs(device, value_dtype):
    """Test sparse_generic_solve with multiple kwargs (like used in benchmarks)."""
    from torchsparsegradutils.utils.linear_cg import LinearCGSettings

    n = 10
    layout = torch.sparse_csr  # Use CSR for this test
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    # Test with multiple kwargs similar to the benchmark suite
    settings = LinearCGSettings(
        cg_tolerance=1e-5, max_cg_iterations=1000, terminate_cg_by_size=False, verbose_linalg=False
    )

    # Pass both settings and additional kwargs
    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=linear_cg, transpose_solve=linear_cg, settings=settings)

    assert torch.allclose(X_test, X_ref, atol=ATOL)


@pytest.mark.flaky(reruns=5)
def test_kwargs_with_different_solvers_same_matrix():
    """Test that different solvers with their respective kwargs produce similar results."""
    from torchsparsegradutils.utils.bicgstab import BICGSTABSettings
    from torchsparsegradutils.utils.linear_cg import LinearCGSettings
    from torchsparsegradutils.utils.minres import MINRESSettings

    device = torch.device("cpu")
    value_dtype = torch.float64  # Use higher precision for better comparison
    layout = torch.sparse_csr

    n = 10
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    # Reference solution
    X_ref = torch.linalg.solve(A_dense, B)

    # Test linear_cg with settings
    cg_settings = LinearCGSettings(cg_tolerance=1e-8, max_cg_iterations=1000)
    X_cg = sparse_generic_solve(A, B, solve=linear_cg, transpose_solve=linear_cg, settings=cg_settings)

    # Test bicgstab with settings
    bicgstab_settings = BICGSTABSettings(reltol=1e-8, abstol=1e-10, matvec_max=2000)
    X_bicgstab = sparse_generic_solve(A, B, solve=bicgstab, transpose_solve=bicgstab, settings=bicgstab_settings)

    # Test minres with settings
    minres_settings = MINRESSettings(minres_tolerance=1e-8, max_cg_iterations=1000)
    X_minres = sparse_generic_solve(A, B, solve=minres, transpose_solve=minres, settings=minres_settings)

    # All solutions should be close to reference
    assert torch.allclose(X_cg, X_ref, atol=ATOL)
    assert torch.allclose(X_bicgstab, X_ref, atol=ATOL)
    assert torch.allclose(X_minres, X_ref, atol=ATOL)

    # All solutions should be close to each other
    assert torch.allclose(X_cg, X_minres, atol=ATOL)
    assert torch.allclose(X_bicgstab, X_minres, atol=ATOL)
