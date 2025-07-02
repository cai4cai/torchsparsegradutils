import torch
import pytest
from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix,
    generate_random_sparse_csr_matrix,
    generate_random_sparse_strictly_triangular_coo_matrix,
    generate_random_sparse_strictly_triangular_csr_matrix,
)

# enable sparse invariants checks if available
if hasattr(torch.sparse, 'check_sparse_tensor_invariants'):
    torch.sparse.check_sparse_tensor_invariants.enable()

# Device fixture
DEVICES = [torch.device('cpu')]
if torch.cuda.is_available():
    DEVICES.append(torch.device('cuda:0'))

def _id_device(d):   return str(d)

@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param

# ---------- Tests for generate_random_sparse_coo_matrix ----------

@pytest.mark.parametrize('size, nnz, multiplier', [
    (torch.Size([4, 4]), 12, 1),
    (torch.Size([2, 4, 4]), 12, 2),
    (torch.Size([8, 16]), 32, 1),
    (torch.Size([4, 8, 16]), 32, 4),
])
def test_gen_random_coo_size_nnz(size, nnz, multiplier, device):
    A = generate_random_sparse_coo_matrix(size, nnz, device=device)
    assert A.size() == size
    assert A._nnz() == nnz * multiplier

@pytest.mark.parametrize('indices_dtype', [torch.int8, torch.int16])
def test_gen_random_coo_invalid_indices(indices_dtype, device):
    with pytest.raises(ValueError):
        generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype, device=device)

@pytest.mark.parametrize('values_dtype', [torch.float16, torch.float32, torch.float64])
def test_gen_random_coo_values_dtype(values_dtype, device):
    A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, values_dtype=values_dtype, device=device)
    assert A.values().dtype == values_dtype

def test_gen_random_coo_device(device):
    A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, device=device)
    assert A.device.type == device.type

# ---------- Tests for generate_random_sparse_csr_matrix ----------

@pytest.mark.parametrize('size, nnz', [
    (torch.Size([4, 4]), 12),
    (torch.Size([2, 4, 4]), 12),
    (torch.Size([8, 16]), 32),
    (torch.Size([4, 8, 16]), 32),
])
def test_gen_random_csr_size(device, size, nnz):
    A = generate_random_sparse_csr_matrix(size, nnz, device=device)
    assert A.size() == size

@pytest.mark.parametrize('nnz', [17])
def test_gen_random_csr_too_many_nnz(nnz, device):
    with pytest.raises(ValueError):
        generate_random_sparse_csr_matrix(torch.Size([4, 4]), nnz, device=device)

@pytest.mark.parametrize('indices_dtype', [torch.int8, torch.int16])
def test_gen_random_csr_invalid_indices(indices_dtype, device):
    with pytest.raises(ValueError):
        generate_random_sparse_csr_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype, device=device)

@pytest.mark.parametrize('indices_dtype', [torch.int32, torch.int64])
def test_gen_random_csr_indices_dtype(indices_dtype, device):
    A = generate_random_sparse_csr_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype, device=device)
    assert A.crow_indices().dtype == indices_dtype
    assert A.col_indices().dtype == indices_dtype

@pytest.mark.parametrize('values_dtype', [torch.float16, torch.float32, torch.float64])
def test_gen_random_csr_values_dtype(values_dtype, device):
    A = generate_random_sparse_csr_matrix(torch.Size([4, 4]), 12, values_dtype=values_dtype, device=device)
    assert A.values().dtype == values_dtype

@pytest.mark.parametrize('size, nnz', [
    (torch.Size([4, 4]), 12),
    (torch.Size([2, 4, 4]), 12),
    (torch.Size([8, 16]), 32),
    (torch.Size([4, 8, 16]), 32),
])
def test_gen_random_csr_nnz(size, nnz, device):
    A = generate_random_sparse_csr_matrix(size, nnz, device=device)
    assert A._nnz() == nnz

# ---------- Tests for strictly triangular COO ----------

@pytest.mark.parametrize('size, nnz', [
    (torch.Size([4, 4]), 5),
    (torch.Size([2, 4, 4, 4]), 5),
])
def test_gen_random_strict_tri_coo_invalid_dims(size, nnz, device):
    # too few dims or non-square batches
    if len(size) != 2:
        with pytest.raises(ValueError):
            generate_random_sparse_strictly_triangular_coo_matrix(size, nnz, device=device)

@pytest.mark.parametrize('nnz', [7])
def test_gen_random_strict_tri_coo_too_many_nnz(nnz, device):
    limit = 4 * (4 - 1) // 2
    with pytest.raises(ValueError):
        generate_random_sparse_strictly_triangular_coo_matrix(torch.Size([4, 4]), limit + 1, device=device)

@pytest.mark.parametrize('indices_dtype', [torch.int8, torch.int16])
def test_gen_random_strict_tri_coo_invalid_indices(indices_dtype, device):
    with pytest.raises(ValueError):
        generate_random_sparse_strictly_triangular_coo_matrix(torch.Size([4, 4]), 5, indices_dtype=indices_dtype, device=device)

@pytest.mark.parametrize('values_dtype', [torch.float16, torch.float32, torch.float64])
def test_gen_random_strict_tri_coo_values_dtype(values_dtype, device):
    A = generate_random_sparse_strictly_triangular_coo_matrix(torch.Size([4, 4]), 5, values_dtype=values_dtype, device=device)
    assert A.values().dtype == values_dtype

@pytest.mark.parametrize('size, upper, multiplier', [
    (torch.Size([4, 4]), True, 1),
    (torch.Size([2, 4, 4]), False, 2),
    (torch.Size([8, 8]), True, 1),
])
def test_gen_random_strict_tri_coo_properties(size, upper, multiplier, device):
    nnz = 5
    A = generate_random_sparse_strictly_triangular_coo_matrix(size, nnz, upper=upper, device=device)
    assert A.size() == size
    assert A._nnz() == nnz * multiplier
    Ad = A.to_dense()
    if upper:
        assert torch.equal(Ad, Ad.triu(1))
    else:
        assert torch.equal(Ad, Ad.tril(-1))

# ---------- Tests for strictly triangular CSR ----------

@pytest.mark.parametrize('size, nnz', [
    (torch.Size([4, 4]), 5),
])
def test_gen_random_strict_tri_csr_invalid_dims(size, nnz, device):
    # only square supported
    with pytest.raises(ValueError):
        generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4, 8]), nnz, device=device)

@pytest.mark.parametrize('nnz', [7])
def test_gen_random_strict_tri_csr_too_many_nnz(nnz, device):
    limit = 4 * (4 - 1) // 2
    with pytest.raises(ValueError):
        generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4]), limit + 1, device=device)

@pytest.mark.parametrize('indices_dtype', [torch.int8, torch.int16])
def test_gen_random_strict_tri_csr_invalid_indices(indices_dtype, device):
    with pytest.raises(ValueError):
        generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4]), 5, indices_dtype=indices_dtype, device=device)

@pytest.mark.parametrize('values_dtype', [torch.float16, torch.float32, torch.float64])
def test_gen_random_strict_tri_csr_values_dtype(values_dtype, device):
    A = generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4]), 5, values_dtype=values_dtype, device=device)
    assert A.values().dtype == values_dtype

@pytest.mark.parametrize('upper', [True, False])
def test_gen_random_strict_tri_csr_properties(upper, device):
    nnz = 5
    size = torch.Size([4, 4])
    A = generate_random_sparse_strictly_triangular_csr_matrix(size, nnz, upper=upper, device=device)
    assert A.size() == size
    assert A._nnz() == nnz
    Ad = A.to_dense()
    if upper:
        assert torch.equal(Ad, Ad.triu(1))
    else:
        assert torch.equal(Ad, Ad.tril(-1))
