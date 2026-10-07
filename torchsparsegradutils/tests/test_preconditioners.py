import pytest
import torch
from test_config import DEVICES

from torchsparsegradutils import sparse_generic_solve, sparse_generic_symmetric_solve
from torchsparsegradutils.utils import (
    BICGSTABSettings,
    GMRESSettings,
    JacobiPreconditioner,
    MatrixPreconditioner,
    Preconditioner,
    bicgstab,
    gmres,
    linear_cg,
    minres,
    sparse_diagonal,
)

LAYOUTS = [torch.sparse_coo, torch.sparse_csr, torch.strided]
LAYOUT_IDS = ["coo", "csr", "dense"]


@pytest.fixture(params=DEVICES, ids=str)
def device(request):
    return request.param


def _to_layout(A, layout):
    if layout == torch.sparse_coo:
        return A.to_sparse_coo()
    if layout == torch.sparse_csr:
        return A.to_sparse_csr()
    return A


def _tridiag(n, diagonal, upper, lower, dtype):
    return (
        diagonal * torch.eye(n, dtype=dtype)
        + torch.diag(torch.full((n - 1,), upper, dtype=dtype), 1)
        + torch.diag(torch.full((n - 1,), lower, dtype=dtype), -1)
    )


def _badly_scaled_spd(n, device, dtype=torch.float64):
    """Tridiagonal SPD matrix D T D whose diagonal scaling D spans four orders of magnitude."""
    T = _tridiag(n, 2.0, -0.5, -0.5, dtype)
    d = torch.logspace(0, 2, n, dtype=dtype)
    return (d[:, None] * T * d[None, :]).to(device)


def _badly_scaled_nonsymmetric(n, device, dtype=torch.float64):
    """Non-symmetric, diagonally dominant matrix with rows scaled over four orders of magnitude."""
    T = _tridiag(n, 3.0, -1.0, -0.5, dtype)
    d = torch.logspace(0, 4, n, dtype=dtype)
    return (d[:, None] * T).to(device)


class _CountingPreconditioner(Preconditioner):
    """Wraps a preconditioner and records how often it is applied."""

    def __init__(self, inner):
        self.inner = inner
        self.shape = inner.shape
        self.calls = 0

    def __call__(self, X):
        self.calls += 1
        return self.inner(X)


# ---------------------------------------------------------------- sparse_diagonal


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
@pytest.mark.parametrize("shape", [(5, 5), (4, 6), (6, 4)])
def test_sparse_diagonal_matches_dense(device, layout, shape):
    A = torch.randn(shape, dtype=torch.float64, device=device)
    A[A.abs() < 0.5] = 0
    A[1, 1] = 0
    torch.testing.assert_close(sparse_diagonal(_to_layout(A, layout)), torch.diagonal(A))


def test_sparse_diagonal_sums_uncoalesced_duplicates(device):
    indices = torch.tensor([[0, 1, 1, 0], [0, 1, 1, 1]], device=device)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device=device)
    A = torch.sparse_coo_tensor(indices, values, (2, 2))
    torch.testing.assert_close(sparse_diagonal(A), torch.tensor([1.0, 5.0], dtype=torch.float64, device=device))


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr], ids=["coo", "csr"])
def test_sparse_diagonal_int32_indices_and_grad(device, layout):
    A_dense = torch.tensor([[2.0, 1.0], [0.0, 3.0]], dtype=torch.float64, device=device)
    A = A_dense.to_sparse_csr() if layout == torch.sparse_csr else A_dense.to_sparse_coo()
    if layout == torch.sparse_csr:
        A = torch.sparse_csr_tensor(
            A.crow_indices().int(), A.col_indices().int(), A.values(), A.shape, requires_grad=True
        )
    else:
        A = torch.sparse_coo_tensor(A.indices(), A.values(), A.shape, requires_grad=True)
    diag = sparse_diagonal(A)
    torch.testing.assert_close(diag.detach(), torch.tensor([2.0, 3.0], dtype=torch.float64, device=device))
    diag.sum().backward()
    torch.testing.assert_close(A.grad.to_dense(), torch.eye(2, dtype=torch.float64, device=device))


def test_sparse_diagonal_rejects_invalid_input():
    with pytest.raises(ValueError, match="2D"):
        sparse_diagonal(torch.ones(3))


# ---------------------------------------------------------------- JacobiPreconditioner


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_jacobi_matches_dense_inverse_diagonal(device, layout):
    A = torch.tensor([[4.0, 1.0, 0.0], [1.0, -2.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64, device=device)
    M_inv = JacobiPreconditioner(_to_layout(A, layout))
    expected = torch.diag(torch.tensor([0.25, -0.5, 1.0], dtype=torch.float64, device=device))

    # Vector, multiple right-hand sides, and batched multiple right-hand sides
    x = torch.randn(3, dtype=torch.float64, device=device)
    X = torch.randn(3, 4, dtype=torch.float64, device=device)
    Xb = torch.randn(2, 3, 4, dtype=torch.float64, device=device)
    torch.testing.assert_close(M_inv(x), expected @ x)
    torch.testing.assert_close(M_inv(X), expected @ X)
    torch.testing.assert_close(M_inv(Xb), expected @ Xb)

    assert M_inv.is_symmetric and M_inv.transpose() is M_inv and M_inv.T is M_inv
    assert not M_inv.is_positive_definite
    assert JacobiPreconditioner(A, absolute=True).is_positive_definite


def test_jacobi_detaches_and_follows_input_dtype(device):
    A = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float64, device=device, requires_grad=True)
    M_inv = JacobiPreconditioner(A)
    assert not M_inv.inv_diag.requires_grad
    y = M_inv(torch.ones(2, dtype=torch.float32, device=device))
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, torch.tensor([0.5, 0.25], device=device))


def test_jacobi_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        JacobiPreconditioner(torch.ones(2, 3))


# ---------------------------------------------------------------- MatrixPreconditioner


@pytest.mark.parametrize("layout", LAYOUTS, ids=LAYOUT_IDS)
def test_matrix_preconditioner_transpose(device, layout):
    M = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [3.0, 0.0, 1.0]], dtype=torch.float64, device=device)
    P = MatrixPreconditioner(_to_layout(M, layout))
    X = torch.randn(3, 2, dtype=torch.float64, device=device)
    torch.testing.assert_close(P(X), M @ X)
    torch.testing.assert_close(P.transpose()(X), M.T @ X)
    assert P.transpose().M_inv.layout == layout
    assert not P.is_symmetric

    S = MatrixPreconditioner(_to_layout(M + M.T, layout), positive_definite=True)
    assert S.is_symmetric and S.transpose() is S


def test_preconditioner_without_transpose_raises():
    class OneSided(Preconditioner):
        shape = (2, 2)

        def __call__(self, X):
            return X

    with pytest.raises(NotImplementedError, match="OneSided does not implement transpose"):
        OneSided().transpose()


# ---------------------------------------------------------------- solvers


def _matvec_counter(A):
    count = [0]

    def op(X):
        count[0] += 1
        return A @ X

    return op, count


def test_jacobi_reduces_cg_and_minres_iterations(device):
    n = 60
    A = _badly_scaled_spd(n, device)
    b = torch.randn(n, 1, dtype=torch.float64, device=device)
    x_ref = torch.linalg.solve(A, b)
    M_inv = JacobiPreconditioner(A.to_sparse_csr())

    for solver in (linear_cg, minres):
        kwargs = dict(max_iter=10 * n, tolerance=1e-10, return_info=True)
        _, info_plain = solver(A.to_sparse_csr(), b, **kwargs)
        x, info_prec = solver(A.to_sparse_csr(), b, preconditioner=M_inv, **kwargs)
        torch.testing.assert_close(x, x_ref, rtol=1e-6, atol=1e-6)
        assert info_prec.iterations < info_plain.iterations / 2, solver.__name__


def test_jacobi_reduces_gmres_and_bicgstab_matvecs(device):
    n = 60
    A = _badly_scaled_nonsymmetric(n, device)
    b = torch.randn(n, dtype=torch.float64, device=device)
    x_ref = torch.linalg.solve(A, b)
    M_inv = JacobiPreconditioner(A)

    gmres_settings = GMRESSettings(rtol=1e-10, restart=20, max_iter=1000)
    bicgstab_settings = BICGSTABSettings(reltol=1e-10, abstol=0.0, matvec_max=20 * n)
    for solve in (
        lambda op, **kw: gmres(op, b, settings=gmres_settings, **kw),
        lambda op, **kw: bicgstab(op, b, settings=bicgstab_settings, **kw),
    ):
        op, plain = _matvec_counter(A)
        solve(op)
        op, prec = _matvec_counter(A)
        x = solve(op, preconditioner=M_inv)
        torch.testing.assert_close(x, x_ref, rtol=1e-6, atol=1e-6)
        assert prec[0] < plain[0] / 2


@pytest.mark.parametrize("solver", [linear_cg, minres, gmres, bicgstab], ids=lambda s: s.__name__)
def test_solvers_accept_tensor_preconditioner(device, solver):
    A = _badly_scaled_spd(8, device)
    b = torch.randn(8, 1, dtype=torch.float64, device=device)
    M_inv = torch.diag(1.0 / torch.diagonal(A))
    x = solver(A, b, preconditioner=M_inv)
    torch.testing.assert_close(x, torch.linalg.solve(A, b), rtol=1e-5, atol=1e-5)
