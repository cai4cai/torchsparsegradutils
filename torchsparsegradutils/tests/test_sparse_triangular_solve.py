import sys
import warnings

import pytest
import torch
from test_config import DEVICES, INDEX_DTYPES, VALUE_DTYPES, Tolerances

from torchsparsegradutils import sparse_triangular_solve
from torchsparsegradutils.utils import rand_sparse_tri

TEST_DATA = [
    # name  A_shape, B_shape, A_nnz
    # ("unbat", (4, 4), (4, 2), 4),
    ("unbat", (12, 12), (12, 6), 32),
    # ("bat", (2, 4, 4), (2, 4, 2), 4),
    ("bat", (4, 12, 12), (4, 12, 6), 32),
]

LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

UPPER = [True, False]
UNITRIANGULAR = [True, False]
TRANSPOSE = [True, False]


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


def dense_triangular_solve(A, B, *, upper, unitriangular, transpose):
    if transpose:
        A = A.transpose(-2, -1)
        upper = not upper

    return torch.linalg.solve_triangular(
        A,
        B,
        upper=upper,
        unitriangular=unitriangular,
    )


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

    res_ref = dense_triangular_solve(Ad, B, upper=upper, unitriangular=unitriangular, transpose=transpose)
    res_test = sparse_triangular_solve(A, B, upper=upper, unitriangular=unitriangular, transpose=transpose)

    atol, rtol = Tolerances.direct(value_dtype)
    assert torch.allclose(res_test, res_ref, atol=atol, rtol=rtol)


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

    Ad_ref = As1.to_dense().detach().clone()

    Bd1 = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Bd_ref = Bd1.detach().clone()

    As1.requires_grad_()
    Ad_ref.requires_grad_()
    Bd1.requires_grad_()
    Bd_ref.requires_grad_()

    res_ref = dense_triangular_solve(Ad_ref, Bd_ref, upper=upper, unitriangular=unitriangular, transpose=transpose)
    res_test = sparse_triangular_solve(As1, Bd1, upper=upper, unitriangular=unitriangular, transpose=transpose)

    # Generate random gradients for the backward pass
    grad_output = torch.rand_like(res_test, dtype=value_dtype, device=device)

    res_ref.backward(grad_output)
    res_test.backward(grad_output)

    nz_mask = As1.grad.to_dense() != 0.0

    atol, rtol = Tolerances.direct(value_dtype)
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad_ref.grad[nz_mask], atol=atol, rtol=rtol)

    assert torch.allclose(Bd1.grad, Bd_ref.grad, atol=atol, rtol=rtol)


def test_sparse_triangular_solve_does_not_emit_upstream_deprecation_warning():
    A = rand_sparse_tri(
        (4, 4),
        6,
        torch.sparse_csr,
        upper=False,
        strict=False,
        indices_dtype=torch.int64,
        values_dtype=torch.float64,
        device=torch.device("cpu"),
    )
    B = torch.rand(4, 2, dtype=torch.float64)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        sparse_triangular_solve(A, B, upper=False)

    deprecation_warnings = [
        warning for warning in caught_warnings if "triangular_solve is deprecated" in str(warning.message)
    ]
    assert not deprecation_warnings


def test_torch_linalg_solve_triangular_rejects_sparse_csr():
    A = torch.eye(4, dtype=torch.float64).to_sparse_csr()
    B = torch.ones(4, 2, dtype=torch.float64)

    with pytest.raises(NotImplementedError, match="SparseCsr"):
        torch.linalg.solve_triangular(A, B, upper=False)


def test_sparse_triangular_solve_optimize_A_multiple_steps(layout, device, value_dtype, index_dtype):
    # small problem
    N, M, NNZ = 30, 10, 50
    A = rand_sparse_tri(
        (N, N),
        NNZ,
        layout,
        upper=True,
        strict=False,
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
    )
    if layout == torch.sparse_coo:
        A = A.coalesce()
    B = torch.randn(N, M, dtype=value_dtype, device=device)

    # make A require gradients on its values
    A.requires_grad_()
    lr = 1e-2

    for step in range(3):
        # forward: solve A X = B
        X = sparse_triangular_solve(A, B, upper=True, unitriangular=False, transpose=False)
        loss = X.sum()

        # backward
        loss.backward()
        assert A.grad is not None
        # B should not get a grad
        assert not hasattr(B, "grad") or B.grad is None

        # grab values and grads
        if layout == torch.sparse_coo:
            vals = A._values()
            gvals = A.grad._values()
        else:
            vals = A.values()
            gvals = A.grad.values()

        old = vals.clone()
        # gradient step on A.values()
        with torch.no_grad():
            vals.sub_(lr * gvals)

        # zero gradients for next iteration
        A.grad = None  # NOTE: only COO CUDA seems to care about this
        # w/o this, CUDA COO: RuntimeError: The size of tensor a (50) must match the size of tensor b (100) at non-singleton dimension 0

        # confirm that the values actually changed
        assert not torch.allclose(old, vals), f"Step {step}: A.values did not update"
