import torch
import pytest
import sys
from torchsparsegradutils import sparse_triangular_solve
from torchsparsegradutils.utils import rand_sparse_tri


# Identify Testing Parameters
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

# Pytest test currently just implemented for unit triangular solve
TEST_DATA = [
    # name  A_shape, B_shape, A_nnz
    ("unbat", (4, 4), (4, 2), 4),
    ("unbat", (12, 12), (12, 6), 32),
    ("bat", (2, 4, 4), (2, 4, 2), 4),
    ("bat", (4, 12, 12), (4, 12, 6), 32),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

UPPER = [True, False]
UNITRIANGULAR = [True, False]  # just unit triangular solve for now
TRANSPOSE = [True, False]

ATOL = 1e-6  # relaxed tolerance to allow for float32
RTOL = 1e-4


# Define Test Names:
def data_id(shapes):
    return shapes[0]


def device_id(device):
    return str(device)


def dtype_id(dtype):
    return str(dtype).split(".")[-1]


def upper_id(upper):
    return "upp" if upper else "low"


def unitriangular_id(unitriangular):
    return "unit" if unitriangular else "nonunit"


def transpose_id(transpose):
    return "t" if transpose else ""


def layout_id(layout):
    return "coo" if layout == torch.sparse_coo else "csr"


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


@pytest.fixture(params=UPPER, ids=[upper_id(d) for d in UPPER])
def upper(request):
    return request.param


@pytest.fixture(params=UNITRIANGULAR, ids=[unitriangular_id(d) for d in UNITRIANGULAR])
def unitriangular(request):
    return request.param


@pytest.fixture(params=TRANSPOSE, ids=[transpose_id(d) for d in TRANSPOSE])
def transpose(request):
    return request.param


@pytest.fixture(params=LAYOUTS, ids=[layout_id(d) for d in LAYOUTS])
def layout(request):
    return request.param


# Define Tests


@pytest.mark.flaky(reruns=5)
def test_tri_solve_forward_routine(layout, device, value_dtype, index_dtype, shapes, upper, unitriangular, transpose):
    if sys.platform == "win32" and device == torch.device("cpu"):
        pytest.skip("Skipping triangular solve CPU tests as solver not implemented for Windows OS")

    _, A_shape, B_shape, A_nnz = shapes
    A = rand_sparse_tri(
        A_shape,
        A_nnz,
        layout,
        upper=upper,
        strict=unitriangular,
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
    )
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Ad = A.to_dense()

    res_ref = torch.triangular_solve(B, Ad, upper=upper, unitriangular=unitriangular, transpose=transpose).solution
    res_test = sparse_triangular_solve(A, B, upper=upper, unitriangular=unitriangular, transpose=transpose)

    assert torch.allclose(res_test, res_ref, atol=ATOL, rtol=RTOL)


@pytest.mark.flaky(reruns=5)
def test_tri_solve_backward_routine(layout, device, value_dtype, index_dtype, shapes, upper, unitriangular, transpose):
    if sys.platform == "win32" and device == torch.device("cpu"):
        pytest.skip("Skipping triangular solve CPU tests as solver not implemented for Windows OS")

    _, A_shape, B_shape, A_nnz = shapes

    As1 = rand_sparse_tri(
        A_shape,
        A_nnz,
        layout,
        upper=upper,
        strict=unitriangular,
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
    )

    Ad2 = As1.to_dense().detach().clone()  # detach and clone to create seperate graph
    Ad3 = Ad2.detach().clone()

    Bd1 = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Bd2 = Bd1.detach().clone()
    Bd3 = Bd1.detach().clone()

    As1.requires_grad_()
    Ad2.requires_grad_()
    Ad3.requires_grad_()
    Bd1.requires_grad_()
    Bd2.requires_grad_()
    Bd3.requires_grad_()

    res_ref = torch.triangular_solve(Bd2, Ad2, upper=upper, unitriangular=unitriangular, transpose=transpose).solution
    res_test = sparse_triangular_solve(As1, Bd1, upper=upper, unitriangular=unitriangular, transpose=transpose)

    # Let's add another test to make sure that the transpose argument is working as epexcted:
    if transpose:
        res_test2 = torch.linalg.solve_triangular(
            Ad3.transpose(-2, -1), Bd3, upper=not upper, unitriangular=unitriangular
        )
    else:
        res_test2 = torch.linalg.solve_triangular(Ad3, Bd3, upper=upper, unitriangular=unitriangular)

    # Generate random gradients for the backward pass
    grad_output = torch.rand_like(res_test, dtype=value_dtype, device=device)

    res_ref.backward(grad_output)
    res_test.backward(grad_output)
    res_test2.backward(grad_output)

    nz_mask = As1.grad.to_dense() != 0.0

    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], atol=ATOL, rtol=RTOL)
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad3.grad[nz_mask], atol=ATOL, rtol=RTOL)

    assert torch.allclose(Bd1.grad, Bd2.grad, atol=ATOL, rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd3.grad, atol=ATOL, rtol=RTOL)


def test_torch_triangular_solve_backward_fail(
    layout, device, value_dtype, index_dtype, shapes, upper, unitriangular, transpose
):
    if sys.platform == "win32" and device == torch.device("cpu"):
        pytest.skip("Skipping backward failure test on Windows CPU")
    # unpack shapes
    _, A_shape, B_shape, A_nnz = shapes
    # create a random sparse triangular A and a dense B
    As1 = rand_sparse_tri(
        A_shape,
        A_nnz,
        layout,
        upper=upper,
        strict=unitriangular,
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
    )
    Bd1 = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Ad = As1.to_dense()
    # torch.triangular_solve does not support backward on general inputs
    with pytest.raises(RuntimeError):
        sol = torch.triangular_solve(Bd1, Ad, upper=upper, unitriangular=unitriangular, transpose=transpose).solution
        sol.sum().backward()


def test_torch_linalg_solve_triangular_backward_fail(
    layout, device, value_dtype, index_dtype, shapes, upper, unitriangular, transpose
):
    if sys.platform == "win32" and device == torch.device("cpu"):
        pytest.skip("Skipping backward failure test on Windows CPU")
    _, A_shape, B_shape, A_nnz = shapes
    As1 = rand_sparse_tri(
        A_shape,
        A_nnz,
        layout,
        upper=upper,
        strict=unitriangular,
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
    )
    Bd1 = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Ad = As1.to_dense()
    with pytest.raises(RuntimeError):
        if transpose:
            sol = torch.linalg.solve_triangular(
                Ad.transpose(-2, -1),
                Bd1,
                upper=not upper,
                unitriangular=unitriangular,
            )
        else:
            sol = torch.linalg.solve_triangular(
                Ad,
                Bd1,
                upper=upper,
                unitriangular=unitriangular,
            )
        sol.sum().backward()
