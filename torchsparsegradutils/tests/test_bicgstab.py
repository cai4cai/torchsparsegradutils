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
