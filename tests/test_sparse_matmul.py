import torch
import pytest

from torchsparsegradutils import sparse_mm, sparse_bmm
from torchsparsegradutils.utils import rand_sparse, rand_sparse_tri

# Identify Testing Parameters
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

TEST_DATA_UNBATCHED = [
    ((4, 6), (6, 2), 8),
    ((8, 16), (16, 10), 32),
    ((7, 4), (4, 9), 14),
]

TEST_DATA_BATCHED = [
    ((1, 4, 6), (1, 6, 2), 8),
    ((4, 8, 16), (4, 16, 10), 32),
    ((11, 7, 4), (11, 4, 9), 14),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]

ATOL=1e-6  # relaxed tolerance to allow for float32
RTOL=1e-4

# Define Test Names:
TEST_DATA_IDS = list(map(str, range(len(TEST_DATA_BATCHED))))

def device_id(device):
    return str(device)

def dtype_id(dtype):
    return str(dtype).split('.')[-1]

# Define Fixtures
@pytest.fixture(params=TEST_DATA_BATCHED, ids=TEST_DATA_IDS)
def shapes_batched(request):
    return request.param

@pytest.fixture(params=TEST_DATA_UNBATCHED, ids=TEST_DATA_IDS)
def shapes_unbatched(request):
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
        
    A_shape, B_shape, A_nnz = shapes
    A = rand_sparse(A_shape, A_nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Ad = A.to_dense()
    
    res_sparse = op_test(A, B)  # both results are dense
    res_dense = op_ref(Ad, B)
    
    torch.allclose(res_sparse, res_dense, atol=ATOL, rtol=RTOL)

def backward_routine(op_test, op_ref, layout, device, value_dtype, index_dtype, shapes, is_backward=False):
    if index_dtype == torch.int32 and layout is torch.sparse_coo:
        pytest.skip("Skipping test as sparse COO tensors with int32 indices are not supported")
    
    A_shape, B_shape, A_nnz = shapes
    As1 = rand_sparse(A_shape, A_nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device=device)
    Ad2 = As1.detach().clone().to_dense()  # detach and clone to create seperate graph
    
    Bd1 = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Bd2 = Bd1.detach().clone()
    
    As1.requires_grad_()
    Ad2.requires_grad_()
    Bd1.requires_grad_()
    Bd2.requires_grad_()
    
    # As1.retain_grad()  # no-op as leaf node already
    # Bd1.retain_grad()
    # Ad2.retain_grad()
    # Bd2.retain_grad()
    
    res1 = op_test(As1, Bd1)  # both results are dense
    res2 = op_ref(Ad2, Bd2)
    
    res1.sum().backward()
    res2.sum().backward()
    nz_mask = As1.grad.to_dense() != 0.0
    
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], atol=ATOL, rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, atol=ATOL, rtol=RTOL)


def test_sparse_mm_forward_result_coo(device, value_dtype, index_dtype, shapes_unbatched):
    forward_routine(sparse_mm, torch.mm, torch.sparse_coo, device, value_dtype, index_dtype, shapes_unbatched)

def test_sparse_mm_forward_result_csr(device, value_dtype, index_dtype, shapes_unbatched):
    forward_routine(sparse_mm, torch.mm, torch.sparse_csr, device, value_dtype, index_dtype, shapes_unbatched)

def test_sparse_mm_backward_result_coo(device, value_dtype, index_dtype, shapes_unbatched):
    backward_routine(sparse_mm, torch.mm, torch.sparse_coo, device, value_dtype, index_dtype, shapes_unbatched, is_backward=True)

def test_sparse_mm_backward_result_csr(device, value_dtype, index_dtype, shapes_unbatched):
    backward_routine(sparse_mm, torch.mm, torch.sparse_csr, device, value_dtype, index_dtype, shapes_unbatched, is_backward=True)
    
def test_sparse_bmm_forward_result_coo(device, value_dtype, index_dtype, shapes_batched):
    forward_routine(sparse_bmm, torch.bmm, torch.sparse_coo, device, value_dtype, index_dtype, shapes_batched)

def test_sparse_bmm_forward_result_csr(device, value_dtype, index_dtype, shapes_batched):
    forward_routine(sparse_bmm, torch.bmm, torch.sparse_csr, device, value_dtype, index_dtype, shapes_batched)

def test_sparse_bmm_backward_result_coo(device, value_dtype, index_dtype, shapes_batched):
    backward_routine(sparse_bmm, torch.bmm, torch.sparse_coo, device, value_dtype, index_dtype, shapes_batched, is_backward=True)

def test_sparse_bmm_backward_result_csr(device, value_dtype, index_dtype, shapes_batched):
    backward_routine(sparse_bmm, torch.bmm, torch.sparse_csr, device, value_dtype, index_dtype, shapes_batched, is_backward=True)
