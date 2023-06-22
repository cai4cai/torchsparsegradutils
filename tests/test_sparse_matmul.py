import torch
import pytest

from torchsparsegradutils import sparse_mm  # , sparse_bmm
from torchsparsegradutils.utils import rand_sparse, rand_sparse_tri

# Identify Testing Parameters
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

TEST_DATA = [
    # name  A_shape, B_shape, A_nnz
    ("unbat", (4, 6), (6, 2), 8),  # unbatched
    ("unbat", (8, 16), (16, 10), 32),  # -
    ("unbat", (7, 4), (4, 9), 14),  # -
    ("bat", (1, 4, 6), (1, 6, 2), 8),  # batched
    ("bat", (4, 8, 16), (4, 16, 10), 32),  # -
    ("bat", (11, 7, 4), (11, 4, 9), 14),  # -
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]

ATOL = 1e-6  # relaxed tolerance to allow for float32
RTOL = 1e-4


# Define Test Names:
def data_id(shapes):
    return shapes[0]


def device_id(device):
    return str(device)


def dtype_id(dtype):
    return str(dtype).split(".")[-1]


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


# Define Tests


def forward_routine(op_test, op_ref, layout, device, value_dtype, index_dtype, shapes):
    if index_dtype == torch.int32 and layout is torch.sparse_coo:
        pytest.skip("Skipping test as sparse COO tensors with int32 indices are not supported")

    _, A_shape, B_shape, A_nnz = shapes
    A = rand_sparse(A_shape, A_nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Ad = A.to_dense()

    res_sparse = op_test(A, B)  # both results are dense
    res_dense = op_ref(Ad, B)

    assert torch.allclose(res_sparse, res_dense, atol=ATOL, rtol=RTOL)


def backward_routine(op_test, op_ref, layout, device, value_dtype, index_dtype, shapes, is_backward=False):
    if index_dtype == torch.int32 and layout is torch.sparse_coo:
        pytest.skip("Skipping test as sparse COO tensors with int32 indices are not supported")

    _, A_shape, B_shape, A_nnz = shapes
    As1 = rand_sparse(A_shape, A_nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    Ad2 = As1.detach().clone().to_dense()  # detach and clone to create seperate graph

    Bd1 = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Bd2 = Bd1.detach().clone()

    As1.requires_grad_()
    Ad2.requires_grad_()
    Bd1.requires_grad_()
    Bd2.requires_grad_()

    res1 = op_test(As1, Bd1)  # both results are dense
    res2 = op_ref(Ad2, Bd2)

    # Generate random gradients for the backward pass
    grad_output = torch.rand_like(res1, dtype=value_dtype, device=device)

    res1.backward(grad_output)
    res2.backward(grad_output)

    nz_mask = As1.grad.to_dense() != 0.0

    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], atol=ATOL, rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, atol=ATOL, rtol=RTOL)


def test_sparse_mm_forward_result_coo(device, value_dtype, index_dtype, shapes):
    forward_routine(sparse_mm, torch.matmul, torch.sparse_coo, device, value_dtype, index_dtype, shapes)


def test_sparse_mm_forward_result_csr(device, value_dtype, index_dtype, shapes):
    forward_routine(sparse_mm, torch.matmul, torch.sparse_csr, device, value_dtype, index_dtype, shapes)


def test_sparse_mm_backward_result_coo(device, value_dtype, index_dtype, shapes):
    backward_routine(
        sparse_mm, torch.matmul, torch.sparse_coo, device, value_dtype, index_dtype, shapes, is_backward=True
    )


def test_sparse_mm_backward_result_csr(device, value_dtype, index_dtype, shapes):
    backward_routine(
        sparse_mm, torch.matmul, torch.sparse_csr, device, value_dtype, index_dtype, shapes, is_backward=True
    )


# Additional Testing Parameters
BAD_TEST_DATA = [
    # name, A, B, expected_error, error_msg
    ("bad_tensor", 5, torch.rand(6, 2), ValueError, "Both A and B should be instances of torch.Tensor"),
    (
        "bad_dim_A",
        torch.tensor([0, 1]).to_sparse(),
        torch.rand(6, 2),
        ValueError,
        "Both A and B should be at least 2-dimensional tensors",
    ),
    (
        "bad_dim_B",
        torch.rand(4, 6).to_sparse(),
        torch.rand(6),
        ValueError,
        "Both A and B should be at least 2-dimensional tensors",
    ),
    (
        "bad_dim_mismatch",
        torch.rand(4, 6).to_sparse(),
        torch.rand(1, 6, 2),
        ValueError,
        "Both A and B should have the same number of dimensions",
    ),
    (
        "bad_format",
        torch.rand(4, 6).to_sparse_csc(),
        torch.rand(6, 2),
        ValueError,
        "A should be in either COO or CSR format",
    ),
    (
        "bad_batch",
        torch.stack([torch.rand(4, 6).to_sparse(), torch.rand(4, 6).to_sparse()]),
        torch.rand(1, 6, 2),
        ValueError,
        "If A and B have a leading batch dimension, they should have the same batch size",
    ),
]


# Additional Fixture
@pytest.fixture(params=BAD_TEST_DATA, ids=[data_id(d) for d in BAD_TEST_DATA])
def bad_inputs(request):
    return request.param


# Additional Test
def test_sparse_mm_error(bad_inputs):
    _, A, B, expected_error, error_msg = bad_inputs
    with pytest.raises(expected_error) as e:
        sparse_mm(A, B)
    assert str(e.value) == error_msg
