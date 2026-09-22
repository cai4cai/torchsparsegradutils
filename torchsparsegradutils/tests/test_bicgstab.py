import pytest
import torch
from test_config import DEVICES, Tolerances

from torchsparsegradutils.utils import bicgstab
from torchsparsegradutils.utils.bicgstab import BICGSTABSettings

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


@pytest.mark.parametrize("exact_start", [False, True])
@pytest.mark.parametrize("multiple_rhs", [False, True])
@pytest.mark.parametrize("operator_kind", ["tensor", "callable"])
def test_bicgstab_nonzero_initial_guess(device, exact_start, multiple_rhs, operator_kind):
    A = torch.tensor([[4.0, 1.0, 0.0], [0.0, 3.0, -0.5], [0.25, 0.0, 2.0]], dtype=torch.float64, device=device)
    rhs = torch.tensor([[1.0, 3.0], [2.0, -1.0], [-1.0, 2.0]], dtype=A.dtype, device=device)
    if not multiple_rhs:
        rhs = rhs[:, 0]
    expected = torch.linalg.solve(A, rhs)
    initial_guess = expected.clone() if exact_start else torch.full_like(rhs, 0.3)
    saved_guess = initial_guess.clone()
    saved_rhs = rhs.clone()
    matvecs = []

    def operator(x):
        matvecs.append(x.clone())
        return A @ x

    actual = bicgstab(
        A if operator_kind == "tensor" else operator,
        rhs,
        initial_guess=initial_guess,
        settings=BICGSTABSettings(reltol=1e-12, abstol=1e-13, matvec_max=40),
    )

    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(A @ actual, rhs, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(initial_guess, saved_guess)
    torch.testing.assert_close(rhs, saved_rhs)
    if exact_start and operator_kind == "callable":
        # An exact warm start needs only the initial residual check for each column.
        assert len(matvecs) == (2 if multiple_rhs else 1)


def test_bicgstab_initial_residual_counts_toward_matvec_budget(device):
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float64, device=device)
    initial_guess = torch.tensor([0.1, 0.2], dtype=rhs.dtype, device=device)
    calls = []

    def operator(x):
        calls.append(x.clone())
        return 2 * x

    # Computing b - A @ x0 uses the sole matvec, leaving no budget to update the initial guess.
    result = bicgstab(operator, rhs, initial_guess, BICGSTABSettings(matvec_max=1))
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], initial_guess)
    torch.testing.assert_close(result, initial_guess)
    assert result.data_ptr() != initial_guess.data_ptr()


@pytest.mark.parametrize("supplied_guess", [False, True])
@pytest.mark.parametrize("multiple_rhs", [False, True])
def test_bicgstab_zero_matvec_budget(device, supplied_guess, multiple_rhs):
    # No budget means no operator/preconditioner calls, even with multiple RHSs.
    rhs = torch.tensor([[1.0, 3.0], [2.0, -1.0]], dtype=torch.float64, device=device)
    if not multiple_rhs:
        rhs = rhs[:, 0]
    initial_guess = torch.full_like(rhs, 0.3) if supplied_guess else None
    expected = torch.zeros_like(rhs) if initial_guess is None else initial_guess.clone()
    saved_rhs = rhs.clone()

    def unexpected_call(x):
        pytest.fail("A zero budget must not evaluate the operator or preconditioner")

    with pytest.warns(UserWarning, match="matvec_max=0.*without evaluating the operator") as caught:
        result = bicgstab(unexpected_call, rhs, initial_guess, BICGSTABSettings(matvec_max=0, precon=unexpected_call))

    assert len(caught) == 1  # Warn once per solve, not once per RHS column.
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(rhs, saved_rhs)
    assert result.data_ptr() != rhs.data_ptr()
    if initial_guess is not None:
        torch.testing.assert_close(initial_guess, expected)
        assert result.data_ptr() != initial_guess.data_ptr()


@pytest.mark.parametrize("supplied_guess", [False, True])
@pytest.mark.parametrize("multiple_rhs", [False, True])
def test_bicgstab_negative_matvec_budget(device, supplied_guess, multiple_rhs):
    rhs = torch.ones((2, 2) if multiple_rhs else (2,), dtype=torch.float64, device=device)
    initial_guess = torch.full_like(rhs, 0.3) if supplied_guess else None

    def unexpected_call(x):
        pytest.fail("An invalid budget must be rejected before evaluating the operator or preconditioner")

    with pytest.raises(ValueError, match="matvec_max must be nonnegative"):
        bicgstab(unexpected_call, rhs, initial_guess, BICGSTABSettings(matvec_max=-1, precon=unexpected_call))


@pytest.mark.parametrize("matvec_max", [0, 1])
@pytest.mark.parametrize("multiple_rhs", [False, True])
@pytest.mark.parametrize("invalid_argument", ["operator", "preconditioner"])
def test_bicgstab_validates_arguments_before_budget_return(device, matvec_max, multiple_rhs, invalid_argument):
    # A zero budget skips computation, but must still reject invalid argument types.
    rhs = torch.ones((2, 2) if multiple_rhs else (2,), dtype=torch.float64, device=device)

    def unexpected_call(x):
        pytest.fail("Invalid arguments must be rejected before any operator or preconditioner calls")

    operator = object() if invalid_argument == "operator" else unexpected_call
    precon = object() if invalid_argument == "preconditioner" else unexpected_call
    message = (
        "matmul_closure must be a tensor" if invalid_argument == "operator" else "settings.precon must be a tensor"
    )

    with pytest.raises(RuntimeError, match=message):
        bicgstab(operator, rhs, settings=BICGSTABSettings(matvec_max=matvec_max, precon=precon))


def test_bicgstab_accepts_warm_start_within_rhs_tolerance(device):
    # This non-exact guess already meets the default RHS-relative tolerance.
    diagonal = torch.tensor([2.0, 4.0], dtype=torch.float64, device=device)
    rhs = diagonal.clone()
    initial_guess = torch.full_like(rhs, 1.0 - 5e-7)
    calls = []

    def operator(x):
        calls.append(1)
        return diagonal * x

    result = bicgstab(operator, rhs, initial_guess)

    assert len(calls) == 1
    torch.testing.assert_close(result, initial_guess, atol=0, rtol=0)
    assert torch.linalg.vector_norm(rhs - diagonal * result) <= 1e-6 * torch.linalg.vector_norm(rhs)


def test_bicgstab_poor_guess_does_not_relax_rhs_tolerance(device):
    diagonal = torch.tensor([2.0, 4.0], dtype=torch.float64, device=device)
    rhs = torch.ones_like(diagonal)
    initial_guess = torch.full_like(rhs, 100.0)
    settings = BICGSTABSettings(reltol=0.1, abstol=0, matvec_max=20)

    result = bicgstab(lambda x: diagonal * x, rhs, initial_guess, settings)

    assert torch.linalg.vector_norm(rhs - diagonal * result) <= 0.1 * torch.linalg.vector_norm(rhs)


def test_bicgstab_rhs_tolerance_is_per_column(device):
    # A large, already-acceptable column must not loosen the small column's target.
    rhs = torch.tensor([[2e6, 2e-6], [4e6, 4e-6]], dtype=torch.float64, device=device)
    initial_guess = rhs / 2
    initial_guess[:, 0] *= 1.0 - 5e-7
    initial_guess[:, 1] *= 0.5
    calls = []

    def operator(x):
        calls.append(1)
        return 2 * x

    result = bicgstab(operator, rhs, initial_guess, BICGSTABSettings(reltol=1e-6, abstol=0, matvec_max=10))

    torch.testing.assert_close(result[:, 0], initial_guess[:, 0], atol=0, rtol=0)
    residual_norms = torch.linalg.vector_norm(rhs - 2 * result, dim=0)
    assert torch.all(residual_norms <= 1e-6 * torch.linalg.vector_norm(rhs, dim=0))
    assert len(calls) == 3  # One initial check per column, plus one update for the second.


@pytest.mark.parametrize("guess_value", [0.0, 1e-9, 1.0])
def test_bicgstab_zero_rhs_uses_absolute_tolerance(device, guess_value):
    diagonal = torch.tensor([2.0, 4.0], dtype=torch.float64, device=device)
    rhs = torch.zeros_like(diagonal)
    initial_guess = torch.full_like(rhs, guess_value)
    settings = BICGSTABSettings(reltol=0.5, abstol=1e-8, matvec_max=20)

    result = bicgstab(lambda x: diagonal * x, rhs, initial_guess, settings)

    assert torch.linalg.vector_norm(diagonal * result) <= settings.abstol
    if guess_value <= 1e-9:
        torch.testing.assert_close(result, initial_guess, atol=0, rtol=0)
