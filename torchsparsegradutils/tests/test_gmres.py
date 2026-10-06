import sys

import numpy as np
import pytest
import scipy.sparse.linalg as spla
import torch
from test_config import DEVICES, SPARSE_LAYOUTS, VALUE_DTYPES, Tolerances

from torchsparsegradutils import sparse_generic_solve
from torchsparsegradutils.utils import GMRESInfo, GMRESSettings, gmres

ORTHOGONALIZATIONS = ["cgs2", "mgs"]


@pytest.fixture(params=DEVICES, ids=str)
def device(request):
    return request.param


def _nonsymmetric(n, dtype=torch.float64, device="cpu", shift=2.0, seed=0):
    """Well-conditioned dense non-symmetric (and non-normal) test matrix."""
    generator = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, dtype=torch.float64, generator=generator) / n**0.5 + shift * torch.eye(n, dtype=torch.float64)
    A[0, -1] += 3.0
    return A.to(dtype=dtype, device=device)


def _convection_diffusion(n, dtype=torch.float64, device="cpu", diagonal=2.02):
    """Non-symmetric tridiagonal matrix that needs many restarted GMRES iterations."""
    A = diagonal * torch.eye(n, dtype=dtype)
    A -= 1.3 * torch.diag(torch.ones(n - 1, dtype=dtype), -1)
    A -= 0.7 * torch.diag(torch.ones(n - 1, dtype=dtype), 1)
    return A.to(device)


def _relative_residual(A, X, B):
    return torch.linalg.vector_norm(B - A @ X, dim=0) / torch.linalg.vector_norm(B, dim=0)


@pytest.mark.parametrize("orthogonalization", ORTHOGONALIZATIONS)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES, ids=str)
@pytest.mark.parametrize("rhs_shape", [(30,), (30, 1), (30, 4)], ids=["vector", "column", "matrix"])
def test_gmres_matches_direct_solve(device, value_dtype, rhs_shape, orthogonalization):
    A = _nonsymmetric(30, value_dtype, device)
    B = torch.randn(*rhs_shape, dtype=value_dtype, device=device)
    settings = GMRESSettings(rtol=1e-6 if value_dtype == torch.float32 else 1e-12, orthogonalization=orthogonalization)
    X = gmres(A, B, settings=settings)
    assert X.shape == B.shape
    atol, rtol = Tolerances.iterative(value_dtype)
    torch.testing.assert_close(X, torch.linalg.solve(A, B), atol=atol, rtol=rtol)


@pytest.mark.parametrize("layout", SPARSE_LAYOUTS, ids=["coo", "csr"])
@pytest.mark.parametrize("operator_kind", ["tensor", "callable"])
def test_gmres_sparse_operators(device, layout, operator_kind):
    A = _convection_diffusion(60, device=device)
    A_sparse = A.to_sparse_coo() if layout == torch.sparse_coo else A.to_sparse_csr()
    operator = A_sparse if operator_kind == "tensor" else (lambda X: A_sparse @ X)
    B = torch.randn(60, 3, dtype=torch.float64, device=device)
    X, info = gmres(operator, B, settings=GMRESSettings(rtol=1e-10, restart=10), return_info=True)
    assert info.reason == "converged"
    assert info.restarts > 1
    assert _relative_residual(A, X, B).max() <= 1e-10


@pytest.mark.parametrize("restart", [1, 5, 40])
def test_gmres_matches_scipy_mgs(restart):
    """With MGS, the port follows SciPy's iterates up to rounding, restarts and preconditioning included."""
    n = 40
    A = _convection_diffusion(n)
    b = torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    settings = GMRESSettings(rtol=1e-9, restart=restart, orthogonalization="mgs", max_iter=100 * n)
    x = gmres(A, b, settings=settings)
    x_ref, exit_code = spla.gmres(A.numpy(), b.numpy(), rtol=1e-9, restart=restart, maxiter=100 * n)
    assert exit_code == 0
    np.testing.assert_allclose(x.numpy(), x_ref, rtol=1e-8, atol=1e-10)

    inv_diag = 1 / torch.diagonal(A)
    x = gmres(A, b, preconditioner=lambda r: inv_diag.unsqueeze(-1) * r, settings=settings)
    x_ref, exit_code = spla.gmres(
        A.numpy(), b.numpy(), rtol=1e-9, restart=restart, maxiter=100 * n, M=np.diag(inv_diag.numpy())
    )
    assert exit_code == 0
    np.testing.assert_allclose(x.numpy(), x_ref, rtol=1e-8, atol=1e-10)


def test_gmres_per_rhs_convergence():
    """Each column meets its own relative tolerance, whatever the scale of the other columns."""
    n = 80
    A = _convection_diffusion(n)
    B = torch.randn(n, 4, dtype=torch.float64)
    B[:, 1] *= 1e-12
    B[:, 2] *= 1e12
    B[:, 3] = A @ torch.ones(n, dtype=torch.float64)
    X, info = gmres(A, B, settings=GMRESSettings(rtol=1e-8, restart=15), return_info=True)
    assert info.reason == "converged"
    assert info.converged.all()
    torch.testing.assert_close(info.true_relative_residual, _relative_residual(A, X, B), rtol=1e-6, atol=1e-14)
    assert info.true_relative_residual.max() <= 1e-8
    # Every column matches the solution of its own independent solve
    for j in range(4):
        x_j = gmres(A, B[:, j], settings=GMRESSettings(rtol=1e-8, restart=15))
        torch.testing.assert_close(X[:, j], x_j, rtol=1e-6, atol=1e-20)


def test_gmres_zero_rhs_columns():
    A = _nonsymmetric(10)
    B = torch.randn(10, 3, dtype=torch.float64)
    B[:, 1] = 0
    X0 = torch.ones_like(B)
    X, info = gmres(A, B, initial_guess=X0, return_info=True)
    assert torch.equal(X[:, 1], torch.zeros(10, dtype=torch.float64))
    assert info.converged.all()
    assert info.true_relative_residual[1] == 0
    assert torch.equal(gmres(A, torch.zeros(10, dtype=torch.float64)), torch.zeros(10, dtype=torch.float64))


@pytest.mark.parametrize("multiple_rhs", [False, True])
def test_gmres_initial_guess(multiple_rhs):
    A = _nonsymmetric(12)
    B = torch.randn(12, 2, dtype=torch.float64)
    if not multiple_rhs:
        B = B[:, 0]
    expected = torch.linalg.solve(A, B)
    saved = expected.clone()
    X, info = gmres(A, B, initial_guess=expected, return_info=True)
    assert torch.equal(expected, saved)
    assert info.iterations == 0 and info.matvecs == 1 and info.reason == "converged"
    torch.testing.assert_close(X, expected)

    X, info = gmres(A, B, initial_guess=torch.full_like(B, 0.3), settings=GMRESSettings(rtol=1e-12), return_info=True)
    assert info.reason == "converged"
    torch.testing.assert_close(X, expected, rtol=1e-9, atol=1e-11)


def test_gmres_batched_rhs():
    A = torch.stack([_nonsymmetric(15, seed=0), _nonsymmetric(15, seed=1)])
    B = torch.randn(2, 15, 3, dtype=torch.float64)
    X, info = gmres(A, B, settings=GMRESSettings(rtol=1e-12), return_info=True)
    assert X.shape == B.shape
    assert info.converged.shape == (2, 3)
    torch.testing.assert_close(X, torch.linalg.solve(A, B), rtol=1e-9, atol=1e-11)


def test_gmres_max_iter_budget():
    A = _convection_diffusion(100)
    b = torch.randn(100, dtype=torch.float64)
    settings = GMRESSettings(rtol=1e-12, restart=7, max_iter=17)
    calls = []

    def operator(X):
        calls.append(X.shape)
        return A @ X

    x, info = gmres(operator, b, settings=settings, return_info=True)
    assert info.reason == "max_iter"
    assert info.iterations == 17
    assert info.restarts == 3  # cycles of 7, 7 and 3 iterations
    assert info.matvecs == len(calls) == 17 + 3
    assert not info.converged.any()
    torch.testing.assert_close(info.true_relative_residual, _relative_residual(A, x.unsqueeze(-1), b.unsqueeze(-1)))
    with pytest.warns(UserWarning, match="did not reach the tolerance"):
        gmres(A, b, settings=settings)

    x0 = torch.randn(100, dtype=torch.float64)
    x, info = gmres(A, b, initial_guess=x0, settings=settings._replace(max_iter=0), return_info=True)
    assert torch.equal(x, x0)
    assert info.iterations == 0 and info.restarts == 0 and info.matvecs == 1


def test_gmres_restart_capped_at_size():
    A = _nonsymmetric(6)
    b = torch.randn(6, dtype=torch.float64)
    x, info = gmres(A, b, settings=GMRESSettings(rtol=1e-12, restart=50), return_info=True)
    assert info.restarts == 1 and info.iterations <= 6
    torch.testing.assert_close(x, torch.linalg.solve(A, b))


def test_gmres_small_restart_needs_restarts():
    A = _convection_diffusion(50)
    b = torch.randn(50, dtype=torch.float64)
    x, info = gmres(A, b, settings=GMRESSettings(rtol=1e-10, restart=3), return_info=True)
    assert info.reason == "converged"
    assert info.restarts > 2
    assert _relative_residual(A, x.unsqueeze(-1), b.unsqueeze(-1)).item() <= 1e-10


def test_gmres_singular_systems():
    A = torch.diag(torch.tensor([2.0, 1.0, 0.0], dtype=torch.float64))
    A[0, 1] = 1.0
    # Consistent right-hand side: converges
    b = torch.tensor([3.0, 1.0, 0.0], dtype=torch.float64)
    x, info = gmres(A, b, return_info=True)
    assert info.reason == "converged"
    torch.testing.assert_close(A @ x, b)
    # Inconsistent right-hand side: the Krylov space becomes invariant without a solution
    b = torch.tensor([3.0, 1.0, 1.0], dtype=torch.float64)
    B = torch.stack([b, torch.tensor([3.0, 1.0, 0.0], dtype=torch.float64)], dim=1)
    X, info = gmres(A, B, return_info=True)
    assert info.reason == "breakdown"
    assert info.converged.tolist() == [False, True]
    assert torch.isfinite(X).all()
    with pytest.warns(UserWarning, match="breakdown"):
        gmres(A, b)


@pytest.mark.parametrize("kind", ["tensor", "callable"])
def test_gmres_preconditioner(kind):
    A = _convection_diffusion(80, diagonal=4.0)
    A += torch.diag(torch.linspace(0, 100, 80, dtype=torch.float64))
    B = torch.randn(80, 2, dtype=torch.float64)
    inv_diag = 1 / torch.diagonal(A)
    M = torch.diag(inv_diag) if kind == "tensor" else (lambda R: inv_diag.unsqueeze(-1) * R)
    settings = GMRESSettings(rtol=1e-10, restart=10)
    X, info = gmres(A, B, preconditioner=M, settings=settings, return_info=True)
    _, plain = gmres(A, B, settings=settings, return_info=True)
    assert info.reason == "converged"
    assert info.iterations < plain.iterations
    assert _relative_residual(A, X, B).max() <= 1e-10


def test_gmres_preconditioner_breakdown():
    A = _nonsymmetric(5)
    b = torch.randn(5, dtype=torch.float64)
    x, info = gmres(A, b, preconditioner=lambda R: torch.zeros_like(R), return_info=True)
    assert info.reason == "breakdown"
    assert torch.equal(x, torch.zeros_like(b))


@pytest.mark.parametrize("check_every", [1, 4, 100])
def test_gmres_check_every(check_every):
    A = _convection_diffusion(60)
    B = torch.randn(60, 3, dtype=torch.float64)
    X, info = gmres(A, B, settings=GMRESSettings(rtol=1e-10, restart=20, check_every=check_every), return_info=True)
    assert info.reason == "converged"
    assert _relative_residual(A, X, B).max() <= 1e-10


def test_gmres_cgs2_keeps_orthogonality_on_hard_problem():
    """On an ill-conditioned problem, CGS2 is at least as accurate as SciPy's MGS."""
    n = 100
    generator = torch.Generator().manual_seed(0)
    U, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=generator))
    W, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=generator))
    A = U @ torch.diag(torch.logspace(0, -8, n, dtype=torch.float64)) @ W.T
    b = torch.randn(n, dtype=torch.float64, generator=generator)
    results = {}
    for orthogonalization in ORTHOGONALIZATIONS:
        settings = GMRESSettings(rtol=1e-14, restart=n, max_iter=n, orthogonalization=orthogonalization)
        _, info = gmres(A, b, settings=settings, return_info=True)
        results[orthogonalization] = info.true_relative_residual.item()
    assert results["cgs2"] <= 10 * results["mgs"]
    assert results["cgs2"] < 1e-6


def test_gmres_info_fields():
    A = _nonsymmetric(10)
    x, info = gmres(A, torch.randn(10, dtype=torch.float64), return_info=True)
    assert isinstance(info, GMRESInfo)
    assert info.converged.shape == info.true_relative_residual.shape == info.recursive_relative_residual.shape == (1,)
    assert info.rtol == 1e-5 and info.atol == 0.0
    assert info.matvecs == info.iterations + info.restarts


def test_gmres_atol():
    A = _convection_diffusion(50)
    b = torch.randn(50, dtype=torch.float64)
    atol = 1e-3 * torch.linalg.vector_norm(b).item()
    x, info = gmres(A, b, settings=GMRESSettings(rtol=0.0, atol=atol), return_info=True)
    assert info.reason == "converged"
    assert torch.linalg.vector_norm(b - A @ x) <= atol


@pytest.mark.parametrize(
    "settings, message",
    [
        (GMRESSettings(rtol=-1.0), "rtol"),
        (GMRESSettings(atol=float("nan")), "atol"),
        (GMRESSettings(restart=0), "restart"),
        (GMRESSettings(max_iter=-1), "max_iter"),
        (GMRESSettings(orthogonalization="householder"), "orthogonalization"),  # type: ignore[arg-type]
        (GMRESSettings(check_every=0), "check_every"),
    ],
)
def test_gmres_invalid_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        gmres(torch.eye(2), torch.ones(2), settings=settings)


def test_gmres_invalid_arguments():
    with pytest.raises(ValueError, match="float32 and float64"):
        gmres(torch.eye(2, dtype=torch.complex128), torch.ones(2, dtype=torch.complex128))
    for dtype in (torch.int64, torch.bool, torch.float16, torch.bfloat16):
        with pytest.raises(ValueError, match="float32 and float64"):
            gmres(torch.eye(2, dtype=dtype), torch.ones(2, dtype=dtype))
    with pytest.raises(ValueError, match="initial_guess"):
        gmres(torch.eye(2), torch.ones(2), initial_guess=torch.ones(3))
    with pytest.raises(TypeError, match="matmul_closure"):
        gmres(3.0, torch.ones(2))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="preconditioner"):
        gmres(torch.eye(2), torch.ones(2), preconditioner=3.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("layout", SPARSE_LAYOUTS, ids=["coo", "csr"])
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES, ids=str)
def test_sparse_generic_solve_gmres_nonsymmetric_gradients(device, layout, value_dtype):
    """GMRES as the forward and (via A.T) backward solver of sparse_generic_solve on a non-symmetric matrix."""
    n = 20
    A_dense = _convection_diffusion(n, value_dtype, device)
    A_dense[0, 5] = 0.4
    A_sparse = A_dense.to_sparse_coo() if layout == torch.sparse_coo else A_dense.to_sparse_csr()
    A_sparse = A_sparse.detach().requires_grad_()
    A_ref = A_dense.detach().clone().requires_grad_()
    B = torch.randn(n, 3, dtype=value_dtype, device=device, requires_grad=True)
    B_ref = B.detach().clone().requires_grad_()

    rtol = 1e-6 if value_dtype == torch.float32 else 1e-12
    X = sparse_generic_solve(A_sparse, B, solve=gmres, settings=GMRESSettings(rtol=rtol))
    X_ref = torch.linalg.solve(A_ref, B_ref)
    grad = torch.randn_like(X)
    X.backward(grad)
    X_ref.backward(grad)

    atol, tol = Tolerances.iterative(value_dtype)
    torch.testing.assert_close(X, X_ref, atol=atol, rtol=tol)
    torch.testing.assert_close(B.grad, B_ref.grad, atol=atol, rtol=tol)
    grad_A = A_sparse.grad.to_dense()
    mask = A_dense != 0
    torch.testing.assert_close(grad_A[mask], A_ref.grad[mask], atol=atol, rtol=tol)


def test_gmres_restarts_after_rounding_breakdown():
    """In float32, a full Krylov space breaks down just above a tight tolerance: SciPy stops, a restart converges."""
    n = 20
    A = _convection_diffusion(n, torch.float32)
    A[0, 5] = 0.4
    B = torch.randn(n, 8, dtype=torch.float32, generator=torch.Generator().manual_seed(0))
    X, info = gmres(A, B, settings=GMRESSettings(rtol=1e-6), return_info=True)
    assert info.reason == "converged"
    assert info.restarts >= 2
    assert _relative_residual(A.double(), X.double(), B.double()).max() <= 2e-6


@pytest.mark.parametrize("bad_value", [float("inf"), float("nan")])
def test_gmres_non_finite_rhs_is_not_converged(bad_value):
    A = _nonsymmetric(6)
    B = torch.randn(6, 2, dtype=torch.float64)
    B[0, 0] = bad_value
    X, info = gmres(A, B, settings=GMRESSettings(rtol=1e-12), return_info=True)
    assert info.converged.tolist() == [False, True]
    assert info.reason == "breakdown"
    torch.testing.assert_close(X[:, 1], torch.linalg.solve(A, B[:, 1]))
    with pytest.warns(UserWarning, match="breakdown"):
        gmres(A, B[:, 0])


def test_gmres_matvecs_count_residual_updates():
    """max_iter caps Arnoldi iterations only: residual updates and the initial guess residual add matvecs."""
    A = _convection_diffusion(40)
    b = torch.randn(40, dtype=torch.float64)
    settings = GMRESSettings(rtol=1e-12, restart=4, max_iter=10)
    _, info = gmres(A, b, initial_guess=torch.zeros_like(b), settings=settings, return_info=True)
    assert info.iterations == 10 and info.restarts == 3
    assert info.matvecs == info.iterations + info.restarts + 1 > settings.max_iter


@pytest.mark.parametrize("orthogonalization", ORTHOGONALIZATIONS)
def test_gmres_full_basis_projection_matches(monkeypatch, orthogonalization):
    """The fixed-size projection used on MPS gives the same iterates as the default sliced projection."""
    # The package re-exports the function under the module's name: fetch the module itself
    gmres_module = sys.modules["torchsparsegradutils.utils.gmres"]

    A = _convection_diffusion(50)
    B = torch.randn(50, 3, dtype=torch.float64)
    B[:, 1] = 0
    settings = GMRESSettings(rtol=1e-10, restart=8, orthogonalization=orthogonalization)
    X_ref, info_ref = gmres(A, B, settings=settings, return_info=True)
    monkeypatch.setattr(gmres_module, "_project_on_full_basis", lambda device: True)
    X, info = gmres(A, B, settings=settings, return_info=True)
    assert info.iterations == info_ref.iterations and info.restarts == info_ref.restarts
    torch.testing.assert_close(X, X_ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_gmres_mps():
    A = _convection_diffusion(60, torch.float32)
    B = torch.randn(60, 3, dtype=torch.float32)
    for operator in (A.to("mps"), A.to_sparse_coo().to("mps")):
        X, info = gmres(operator, B.to("mps"), settings=GMRESSettings(rtol=1e-5, restart=10), return_info=True)
        assert X.device.type == "mps"
        assert info.reason == "converged"
        assert _relative_residual(A.double(), X.cpu().double(), B.double()).max() <= 2e-5


@pytest.mark.parametrize("with_guess", [False, True])
@pytest.mark.parametrize("with_preconditioner", [False, True])
def test_gmres_recursive_residual_without_iterations(with_guess, with_preconditioner):
    """Without any Arnoldi iteration, the residual estimate is the initial (preconditioned) relative residual."""
    A = _convection_diffusion(30)
    B = torch.randn(30, 2, dtype=torch.float64)
    B[:, 1] = 0
    X0 = torch.randn_like(B) if with_guess else None
    inv_diag = 1 / torch.diagonal(A)
    M = (lambda R: inv_diag.unsqueeze(-1) * R) if with_preconditioner else (lambda R: R)
    _, info = gmres(
        A,
        B,
        initial_guess=X0,
        preconditioner=M if with_preconditioner else None,
        settings=GMRESSettings(max_iter=0),
        return_info=True,
    )
    R0 = B if X0 is None else B - A @ X0
    expected = torch.linalg.vector_norm(M(R0[:, :1])) / torch.linalg.vector_norm(M(B[:, :1]))
    torch.testing.assert_close(info.recursive_relative_residual[0], expected)
    assert info.recursive_relative_residual[1] == 0
    assert info.converged.tolist() == [False, True]


SCALES = [(torch.float32, 1e-25), (torch.float32, 1e20), (torch.float64, 1e-200), (torch.float64, 1e200)]


@pytest.mark.parametrize("dtype, scale", SCALES, ids=[f"{str(d).split('.')[-1]}-{s:g}" for d, s in SCALES])
def test_gmres_extreme_rhs_scale(device, dtype, scale):
    """Norms are scaled: tiny or huge (but representable) right-hand sides do not under/overflow to false results."""
    A = torch.eye(4, device=device, dtype=dtype)
    b = torch.full((4,), scale, device=device, dtype=dtype)
    x, info = gmres(A, b, return_info=True)
    # Scale before taking the reference norm, so that it stays accurate (and a zero solution fails)
    assert torch.linalg.vector_norm((x - b) / scale) / 2 <= 1e-5
    assert info.converged.tolist() == [True] and info.reason == "converged"
    assert torch.isfinite(info.true_relative_residual).all() and torch.isfinite(info.recursive_relative_residual).all()


@pytest.mark.parametrize("dtype, scale", SCALES, ids=[f"{str(d).split('.')[-1]}-{s:g}" for d, s in SCALES])
def test_gmres_extreme_operator_scale(device, dtype, scale):
    A = scale * _convection_diffusion(12, dtype, device, diagonal=4.0)
    b = torch.randn(12, dtype=dtype, device=device)
    x, info = gmres(A, b, settings=GMRESSettings(rtol=1e-5), return_info=True)
    assert info.reason == "converged"
    expected = torch.linalg.solve(A.double() / scale, b.double())
    assert torch.linalg.vector_norm(x.double() * scale - expected) / torch.linalg.vector_norm(expected) <= 1e-4


def test_gmres_mixed_scale_columns():
    """Zero, ordinary, tiny, huge and non-finite columns in one solve: each gets its own outcome."""
    A = _convection_diffusion(16, torch.float32, diagonal=4.0)
    B = torch.randn(16, 5, dtype=torch.float32, generator=torch.Generator().manual_seed(0))
    B[:, 0] = 0
    B[:, 2] *= 1e-25
    B[:, 3] *= 1e20
    B[3, 4] = float("inf")
    X, info = gmres(A, B, settings=GMRESSettings(rtol=1e-5), return_info=True)
    assert info.converged.tolist() == [True, True, True, True, False]
    assert info.reason == "breakdown"
    assert torch.equal(X[:, 0], torch.zeros(16))
    expected = torch.linalg.solve(A.double(), B[:, 1:4].double())
    scale = expected.abs().amax(dim=0)
    error = torch.linalg.vector_norm((X[:, 1:4].double() - expected) / scale, dim=0)
    assert (error / torch.linalg.vector_norm(expected / scale, dim=0)).max() <= 1e-4


@pytest.mark.parametrize("case", ["zero_budget", "exact_guess", "zero_rhs", "short_budget"])
def test_gmres_workspace_not_allocated_without_iterations(monkeypatch, case):
    """No restart-sized Krylov workspace when nothing iterates, and none larger than a short budget."""
    gmres_module = sys.modules["torchsparsegradutils.utils.gmres"]
    allocated = []
    workspace = gmres_module._krylov_workspace

    def recording_workspace(batch, restart, n, dtype, device):
        allocated.append(restart)
        return workspace(batch, restart, n, dtype, device)

    monkeypatch.setattr(gmres_module, "_krylov_workspace", recording_workspace)
    A = _convection_diffusion(40)
    B = torch.randn(40, 3, dtype=torch.float64)
    expected = torch.linalg.solve(A, B)
    rhs = torch.zeros_like(B) if case == "zero_rhs" else B
    guess = expected if case == "exact_guess" else None
    max_iter = {"zero_budget": 0, "short_budget": 3}.get(case)
    settings = GMRESSettings(rtol=1e-8, restart=20, max_iter=max_iter)
    X, info = gmres(A, rhs, initial_guess=guess, settings=settings, return_info=True)

    if case == "short_budget":
        assert allocated == [3] and info.iterations == 3 and info.reason == "max_iter"
        return
    assert allocated == [0] and info.iterations == 0 and info.restarts == 0
    if case == "zero_budget":
        assert info.reason == "max_iter" and torch.equal(X, torch.zeros_like(B)) and info.matvecs == 0
        torch.testing.assert_close(info.true_relative_residual, torch.ones(3, dtype=torch.float64))
    else:
        assert info.reason == "converged"
        torch.testing.assert_close(X, torch.zeros_like(B) if case == "zero_rhs" else expected)
        assert info.matvecs == (1 if case == "exact_guess" else 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gmres_cuda_memory_without_iterations():
    B = torch.ones(20000, 16, device="cuda", dtype=torch.float32)
    gmres(lambda X: X, torch.ones(4, device="cuda"))  # warm up
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    gmres(lambda X: X, B, settings=GMRESSettings(restart=50, max_iter=0), return_info=True)
    torch.cuda.synchronize()
    # Far below the 65 MB of a restart-sized basis: only a few (n, k) temporaries
    assert torch.cuda.max_memory_allocated() - before <= 8 * B.numel() * B.element_size()
