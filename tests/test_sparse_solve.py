# for sparse generic solve

import pytest
import torch
from test_config import DEVICES, INDEX_DTYPES, VALUE_DTYPES, Tolerances

from torchsparsegradutils.ops.generic_solve import sparse_generic_solve
from torchsparsegradutils.utils import bicgstab, convert_coo_to_csr, linear_cg, minres
from torchsparsegradutils.utils.random_sparse import make_spd_sparse

TEST_DATA = [
    # name  A_shape, B_shape, A_num_zero
    ("vector_1d", (12, 12), (12,), 32),
    ("vector_2d", (12, 12), (12, 1), 32),
    ("multi_rhs", (12, 12), (12, 6), 32),
]

LAYOUTS = [torch.sparse_coo, torch.sparse_csr]
SOLVES = [None, linear_cg, bicgstab, minres]


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


def _make_differentiable_tridiag_spd(theta, layout):
    """Create a small sparse SPD matrix with fixed sparsity and differentiable values."""
    n = theta.numel()
    device = theta.device

    diag = theta.square() + 2.0
    off = -0.1 * torch.sigmoid(theta[:-1])

    rows = torch.cat(
        [
            torch.arange(n, device=device),
            torch.arange(n - 1, device=device),
            torch.arange(1, n, device=device),
        ]
    )
    cols = torch.cat(
        [
            torch.arange(n, device=device),
            torch.arange(1, n, device=device),
            torch.arange(n - 1, device=device),
        ]
    )
    values = torch.cat([diag, off, off])

    A = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n, n)).coalesce()
    if layout == torch.sparse_csr:
        A = A.to_sparse_csr()
    return A, A.to_dense()


def _make_differentiable_nonsymmetric_tridiag(theta, layout):
    """Create a small fixed-sparsity non-symmetric diagonally dominant matrix."""
    n = 6
    device = theta.device
    dtype = theta.dtype

    if theta.numel() != 3 * n - 2:
        raise ValueError(f"theta should have length {3 * n - 2}, got {theta.numel()}")

    diag_theta = theta[:n]
    upper_theta = theta[n : n + n - 1]
    lower_theta = theta[n + n - 1 :]

    diag = diag_theta.square() + 3.0
    upper = 0.05 * torch.tanh(upper_theta)
    lower = -0.08 * torch.sigmoid(lower_theta)

    rows = torch.cat(
        [
            torch.arange(n, device=device),
            torch.arange(n - 1, device=device),
            torch.arange(1, n, device=device),
        ]
    )
    cols = torch.cat(
        [
            torch.arange(n, device=device),
            torch.arange(1, n, device=device),
            torch.arange(n - 1, device=device),
        ]
    )
    values = torch.cat([diag, upper, lower]).to(dtype=dtype)

    A = torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n, n)).coalesce()
    if layout == torch.sparse_csr:
        A = A.to_sparse_csr()
    return A, A.to_dense()


def _bicgstab_transpose(A, B, **kwargs):
    """Solve A.T X = B using BiCGSTAB."""
    if A.layout == torch.sparse_csr:
        # A.T currently triggers aten::as_strided for sparse CSR tensors in PyTorch,
        # so use transpose(...).to_sparse_csr() for the CSR test case.
        # A.transpose(0, 1) for csr will return csc, hence the to_sparse_csr() call.
        return bicgstab(A.transpose(0, 1).to_sparse_csr(), B, **kwargs)
    return bicgstab(A.T, B, **kwargs)


def _bicgstab_higher_order_kwargs(value_dtype):
    from torchsparsegradutils.solvers.bicgstab import BICGSTABSettings

    if value_dtype == torch.float32:
        return {"settings": BICGSTABSettings(reltol=1e-6, abstol=1e-6, matvec_max=1000)}
    return {"settings": BICGSTABSettings(reltol=1e-12, abstol=1e-12, matvec_max=1000)}


def _higher_order_bicgstab_tolerances(value_dtype):
    if value_dtype == torch.float32:
        return {
            "output": (1e-5, 1e-5),
            "grad": (5e-5, 5e-5),
            "hess": (5e-3, 5e-3),
        }
    return {
        "output": (1e-8, 1e-8),
        "grad": (1e-6, 1e-6),
        "hess": (5e-5, 5e-5),
    }


def _settings_for_higher_order_solve(solve, value_dtype):
    if solve is linear_cg:
        from torchsparsegradutils.solvers.cg import LinearCGSettings

        return {"settings": LinearCGSettings(cg_tolerance=1e-5, max_cg_iterations=1000)}
    if solve is minres:
        from torchsparsegradutils.solvers.minres import MINRESSettings

        tolerance = 1e-6 if value_dtype == torch.float32 else 1e-10
        return {"settings": MINRESSettings(minres_tolerance=tolerance, max_cg_iterations=1000)}
    return {}


def _higher_order_spd_tolerances(value_dtype):
    if value_dtype == torch.float32:
        return {
            "grad": (1e-3, 1e-3),
            "hess": (5e-2, 5e-2),
        }
    return {
        "grad": (1e-5, 1e-5),
        "hess": (5e-4, 5e-4),
    }


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


def test_solve_forward_routine(layout, solve, device, value_dtype, index_dtype, shapes):
    _, A_shape, B_shape, num_zero = shapes
    n = A_shape[0]
    A, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=num_zero)
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)

    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=solve, transpose_solve=solve)

    atol, rtol = Tolerances.iterative(value_dtype)
    assert torch.allclose(X_test, X_ref, atol=atol, rtol=rtol)


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

    atol, rtol = Tolerances.iterative(value_dtype)
    nz_mask = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], atol=atol, rtol=rtol)
    assert torch.allclose(Bd1.grad, Bd2.grad, atol=atol, rtol=rtol)


def test_linear_cg_kwargs(device, value_dtype, layout):
    """Test sparse_generic_solve with LinearCGSettings kwargs."""
    from torchsparsegradutils.solvers.cg import LinearCGSettings

    n = 10
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    atol, rtol = Tolerances.iterative(value_dtype)
    # Test with custom LinearCGSettings
    settings = LinearCGSettings(
        cg_tolerance=atol, max_cg_iterations=500, terminate_cg_by_size=False, verbose_linalg=False
    )

    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=linear_cg, transpose_solve=linear_cg, settings=settings)

    assert torch.allclose(X_test, X_ref, atol=atol, rtol=rtol)


def test_bicgstab_kwargs(device, value_dtype, layout):
    """Test sparse_generic_solve with BICGSTABSettings kwargs."""
    from torchsparsegradutils.solvers.bicgstab import BICGSTABSettings

    n = 10
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    atol, rtol = Tolerances.iterative(value_dtype)
    # Test with custom BICGSTABSettings
    settings = BICGSTABSettings(reltol=rtol, abstol=atol / 100, matvec_max=1000, precon=None)

    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=bicgstab, transpose_solve=bicgstab, settings=settings)

    assert torch.allclose(X_test, X_ref, atol=atol, rtol=rtol)


def test_minres_kwargs(device, value_dtype, layout):
    """Test sparse_generic_solve with MINRESSettings kwargs."""
    from torchsparsegradutils.solvers.minres import MINRESSettings

    n = 10
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    atol, rtol = Tolerances.iterative(value_dtype)
    # Test with custom MINRESSettings
    settings = MINRESSettings(minres_tolerance=atol, max_cg_iterations=500, verbose_linalg=False)

    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=minres, transpose_solve=minres, settings=settings)

    assert torch.allclose(X_test, X_ref, atol=atol, rtol=rtol)


def test_kwargs_backward_pass(device, value_dtype, layout):
    """Test that kwargs work correctly during backward pass."""
    from torchsparsegradutils.solvers.cg import LinearCGSettings

    n = 10
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)

    # Set up tensors with gradients
    As1 = A.clone().requires_grad_()
    Ad2 = A_dense.detach().clone().requires_grad_()
    Bd1 = torch.rand(n, dtype=value_dtype, device=device).requires_grad_()
    Bd2 = Bd1.clone().detach().requires_grad_()

    atol, rtol = Tolerances.iterative(value_dtype)
    # Test with custom settings
    settings = LinearCGSettings(
        cg_tolerance=atol, max_cg_iterations=500, terminate_cg_by_size=False, verbose_linalg=False
    )

    res_ref = torch.linalg.solve(Ad2, Bd2)
    res_test = sparse_generic_solve(As1, Bd1, solve=linear_cg, transpose_solve=linear_cg, settings=settings)

    # Backward pass
    grad_output = torch.rand_like(res_test)
    res_ref.backward(grad_output)
    res_test.backward(grad_output)

    # Check gradients
    nz_mask = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], atol=atol, rtol=rtol)
    assert torch.allclose(Bd1.grad, Bd2.grad, atol=atol, rtol=rtol)


def test_multiple_kwargs(device, value_dtype):
    """Test sparse_generic_solve with multiple kwargs (like used in benchmarks)."""
    from torchsparsegradutils.solvers.cg import LinearCGSettings

    n = 10
    layout = torch.sparse_csr  # Use CSR for this test
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    atol, rtol = Tolerances.iterative(value_dtype)
    # Test with multiple kwargs similar to the benchmark suite
    settings = LinearCGSettings(
        cg_tolerance=atol, max_cg_iterations=1000, terminate_cg_by_size=False, verbose_linalg=False
    )

    # Pass both settings and additional kwargs
    X_ref = torch.linalg.solve(A_dense, B)
    X_test = sparse_generic_solve(A, B, solve=linear_cg, transpose_solve=linear_cg, settings=settings)

    assert torch.allclose(X_test, X_ref, atol=atol, rtol=rtol)


def test_kwargs_with_different_solvers_same_matrix():
    """Test that different solvers with their respective kwargs produce similar results."""
    from torchsparsegradutils.solvers.bicgstab import BICGSTABSettings
    from torchsparsegradutils.solvers.cg import LinearCGSettings
    from torchsparsegradutils.solvers.minres import MINRESSettings

    device = torch.device("cpu")
    value_dtype = torch.float64  # Use higher precision for better comparison
    layout = torch.sparse_csr

    n = 10
    A, A_dense = make_spd_sparse(n, layout, value_dtype, torch.int64, device, nz=30)
    B = torch.rand(n, dtype=value_dtype, device=device)

    atol, rtol = Tolerances.iterative(value_dtype)
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
    assert torch.allclose(X_cg, X_ref, atol=atol, rtol=rtol)
    assert torch.allclose(X_bicgstab, X_ref, atol=atol, rtol=rtol)
    assert torch.allclose(X_minres, X_ref, atol=atol, rtol=rtol)

    # All solutions should be close to each other
    assert torch.allclose(X_cg, X_minres, atol=atol, rtol=rtol)
    assert torch.allclose(X_bicgstab, X_minres, atol=atol, rtol=rtol)


@pytest.mark.parametrize("base_solve", [linear_cg, minres], ids=[solve_id(linear_cg), solve_id(minres)])
def test_sparse_generic_solve_higher_order_create_graph_no_out_error(layout, base_solve, device, value_dtype):
    torch.manual_seed(0)

    theta = torch.randn(8, dtype=value_dtype, device=device, requires_grad=True)
    A, _ = _make_differentiable_tridiag_spd(theta, layout)
    B = torch.randn(8, 2, dtype=value_dtype, device=device)

    kwargs = _settings_for_higher_order_solve(base_solve, value_dtype)
    loss = sparse_generic_solve(A, B, solve=base_solve, transpose_solve=base_solve, **kwargs).sum()

    grad_theta = torch.autograd.grad(loss, theta, create_graph=True)[0]

    assert grad_theta.requires_grad
    assert torch.isfinite(grad_theta).all()

    second = torch.autograd.grad(grad_theta.sum(), theta)[0]
    assert torch.isfinite(second).all()


@pytest.mark.parametrize("base_solve", [linear_cg, minres], ids=[solve_id(linear_cg), solve_id(minres)])
def test_sparse_generic_solve_higher_order_matches_dense_reference(layout, base_solve, device, value_dtype):
    torch.manual_seed(1)

    theta_sparse = torch.randn(6, dtype=value_dtype, device=device, requires_grad=True)
    theta_dense = theta_sparse.detach().clone().requires_grad_()

    B = torch.randn(6, 2, dtype=value_dtype, device=device)

    A_sparse, _ = _make_differentiable_tridiag_spd(theta_sparse, layout)
    _, A_dense = _make_differentiable_tridiag_spd(theta_dense, torch.sparse_coo)

    kwargs = _settings_for_higher_order_solve(base_solve, value_dtype)
    tolerances = _higher_order_spd_tolerances(value_dtype)

    out_sparse = sparse_generic_solve(A_sparse, B, solve=base_solve, transpose_solve=base_solve, **kwargs)
    out_dense = torch.linalg.solve(A_dense, B)

    loss_sparse = out_sparse.square().sum()
    loss_dense = out_dense.square().sum()

    grad_sparse = torch.autograd.grad(loss_sparse, theta_sparse, create_graph=True)[0]
    grad_dense = torch.autograd.grad(loss_dense, theta_dense, create_graph=True)[0]

    hess_vec_sparse = torch.autograd.grad(grad_sparse.sum(), theta_sparse)[0]
    hess_vec_dense = torch.autograd.grad(grad_dense.sum(), theta_dense)[0]

    grad_atol, grad_rtol = tolerances["grad"]
    hess_atol, hess_rtol = tolerances["hess"]
    assert torch.allclose(grad_sparse, grad_dense, atol=grad_atol, rtol=grad_rtol)
    assert torch.allclose(hess_vec_sparse, hess_vec_dense, atol=hess_atol, rtol=hess_rtol)


def test_sparse_generic_solve_higher_order_nonsymmetric_bicgstab_matches_dense_reference(layout, device, value_dtype):
    torch.manual_seed(2)

    n = 6
    theta_sparse = torch.randn(3 * n - 2, dtype=value_dtype, device=device, requires_grad=True)
    theta_dense = theta_sparse.detach().clone().requires_grad_()

    B = torch.randn(n, 2, dtype=value_dtype, device=device)

    A_sparse, _ = _make_differentiable_nonsymmetric_tridiag(theta_sparse, layout)
    _, A_dense = _make_differentiable_nonsymmetric_tridiag(theta_dense, torch.sparse_coo)
    assert not torch.allclose(A_dense, A_dense.T)

    kwargs = _bicgstab_higher_order_kwargs(value_dtype)
    tolerances = _higher_order_bicgstab_tolerances(value_dtype)

    out_sparse = sparse_generic_solve(
        A_sparse,
        B,
        solve=bicgstab,
        transpose_solve=_bicgstab_transpose,
        **kwargs,
    )
    out_dense = torch.linalg.solve(A_dense, B)
    output_atol, output_rtol = tolerances["output"]
    assert torch.allclose(A_dense @ out_sparse, B, atol=output_atol, rtol=output_rtol)

    loss_sparse = out_sparse.square().sum()
    loss_dense = out_dense.square().sum()

    grad_sparse = torch.autograd.grad(loss_sparse, theta_sparse, create_graph=True)[0]
    grad_dense = torch.autograd.grad(loss_dense, theta_dense, create_graph=True)[0]

    hess_vec_sparse = torch.autograd.grad(grad_sparse.sum(), theta_sparse)[0]
    hess_vec_dense = torch.autograd.grad(grad_dense.sum(), theta_dense)[0]

    grad_atol, grad_rtol = tolerances["grad"]
    hess_atol, hess_rtol = tolerances["hess"]
    assert torch.allclose(out_sparse, out_dense, atol=output_atol, rtol=output_rtol)
    assert torch.allclose(grad_sparse, grad_dense, atol=grad_atol, rtol=grad_rtol)
    assert torch.allclose(hess_vec_sparse, hess_vec_dense, atol=hess_atol, rtol=hess_rtol)
