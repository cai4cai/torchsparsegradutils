from unittest.mock import Mock

import pytest
import torch

from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix,
    generate_random_sparse_csr_matrix,
)
from torchsparsegradutils.utils.utils import (
    _compress_row_indices,
    _demcompress_crow_indices,
    _sort_coo_indices,
    convert_coo_to_csr,
    sparse_block_diag,
    sparse_block_diag_split,
    sparse_eye,
    stack_csr,
)

# Device fixture
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))


def _id_device(d):
    return str(d)


@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param
    return request.param


# Test stack_csr
@pytest.mark.parametrize(
    "size, nnz, dim",
    [
        ((4, 4), 12, 0),
        ((8, 16), 32, 0),
        ((4, 4), 12, -1),
        ((8, 16), 32, -1),
    ],
)
def test_stack_csr(device, size, nnz, dim):
    csr_list = [generate_random_sparse_csr_matrix(size, nnz, device=device) for _ in range(3)]
    dense_list = [csr.to_dense() for csr in csr_list]
    csr_stacked = stack_csr(csr_list)
    dense_stacked = torch.stack(dense_list)
    assert torch.equal(csr_stacked.to_dense(), dense_stacked)


# Test _sort_coo_indices


def test_unbatched_sort(device):
    nr, nc = 4, 4
    indices = torch.randperm(nr * nc, device=device)
    indices = torch.stack([indices // nc, indices % nc])
    values = torch.arange(nr * nc, device=device)
    coalesced = torch.sparse_coo_tensor(indices, values).coalesce()
    sorted_indices_coalesced = coalesced.indices()
    coalesce_permutation = coalesced.values()
    sorted_indices, permutation = _sort_coo_indices(indices)
    assert torch.equal(sorted_indices_coalesced, sorted_indices)
    assert torch.equal(coalesce_permutation, permutation)


def test_batched_sort(device):
    nr, nc = 4, 4
    batch = 3
    indices = torch.randperm(nr * nc, device=device)
    indices = torch.stack([indices // nc, indices % nc])
    sparse_indices = torch.cat([indices] * batch, dim=-1)
    batch_indices = torch.arange(batch, device=device).repeat(nr * nc).unsqueeze(0)
    batched_sparse_indices = torch.cat([batch_indices, sparse_indices])
    values = torch.arange(nr * nc * batch, device=device)
    coalesced = torch.sparse_coo_tensor(batched_sparse_indices, values).coalesce()
    sorted_indices_coalesced = coalesced.indices()
    coalesce_permutation = coalesced.values()
    sorted_indices, permutation = _sort_coo_indices(batched_sparse_indices)
    assert torch.equal(sorted_indices_coalesced, sorted_indices)
    assert torch.equal(coalesce_permutation, permutation)


# Test COO to CSR conversion and values
@pytest.mark.parametrize(
    "size, nnz",
    [
        ((4, 4), 12),
        ((8, 16), 32),
        ((4, 4, 4), 12),
        ((6, 8, 14), 32),
    ],
)
def test_convert_coo_to_csr(device, size, nnz):
    A_coo = generate_random_sparse_coo_matrix(size, nnz, device=device)
    A_csr = convert_coo_to_csr(A_coo)
    if len(size) == 2:
        A_csr_2 = A_coo.to_sparse_csr()
    elif len(size) == 3:
        sizes = A_coo.size()
        A_crow, A_row, A_val = [], [], []
        for a in A_coo:
            csr = a.detach().to_sparse_csr()
            A_crow.append(csr.crow_indices())
            A_row.append(csr.col_indices())
            A_val.append(csr.values())
        A_csr_2 = torch.sparse_csr_tensor(
            crow_indices=torch.stack(A_crow),
            col_indices=torch.stack(A_row),
            values=torch.stack(A_val),
            size=sizes,
        )
    else:
        pytest.skip("Unsupported size for convert_coo_to_csr test")
    # Compare indices and values
    assert torch.equal(A_csr.crow_indices(), A_csr_2.crow_indices())
    assert torch.equal(A_csr.col_indices(), A_csr_2.col_indices())
    assert torch.equal(A_csr.values(), A_csr_2.values())


# Test CSR to COO conversion row indices decompression
@pytest.mark.parametrize(
    "size, nnz",
    [
        ((4, 4), 12),
        ((8, 16), 32),
    ],
)
def test_demcompress_crow_indices(device, size, nnz):
    A_csr = generate_random_sparse_csr_matrix(size, nnz, device=device)
    A_coo = A_csr.to_sparse_coo()
    row_indices = A_coo.indices()[0]
    decompressed = _demcompress_crow_indices(A_csr.crow_indices(), A_coo.size()[0])
    assert torch.equal(row_indices, decompressed)


# Test sparse_block_diag forward for COO and CSR
@pytest.mark.parametrize(
    "layout, size, nnz",
    [
        ("coo", (1, 4, 4), 12),
        ("coo", (4, 4, 4), 12),
        ("coo", (6, 8, 14), 32),
        ("csr", (1, 4, 4), 12),
        ("csr", (4, 4, 4), 12),
        ("csr", (6, 8, 14), 32),
    ],
)
def test_sparse_block_diag_forward(device, layout, size, nnz):
    if layout == "coo":
        A = generate_random_sparse_coo_matrix(size, nnz, device=device)
    else:
        A = generate_random_sparse_csr_matrix(size, nnz, device=device)
    A_d = A.to_dense()
    B = sparse_block_diag(*A)
    D = torch.block_diag(*A_d)
    assert torch.equal(B.to_dense(), D)


# Test sparse_block_diag backward for COO
# TODO: what about CSR?
@pytest.mark.parametrize(
    "size, nnz",
    [
        ((1, 4, 4), 12),
        ((4, 4, 4), 12),
        ((6, 8, 14), 32),
    ],
)
def test_sparse_block_diag_coo_backward(device, size, nnz):
    A = generate_random_sparse_coo_matrix(size, nnz, device=device)
    A_d = A.detach().clone().to_dense()
    A.requires_grad_(True)
    A_d.requires_grad_(True)
    B = sparse_block_diag(*A)
    D = torch.block_diag(*A_d)
    torch.sparse.sum(B).backward()
    D.sum().backward()
    mask = A.grad.to_dense() != 0
    assert torch.allclose(A.grad.to_dense()[mask], A_d.grad[mask])


# Test sparse_block_diag error cases
def test_sparse_block_diag_errors():
    with pytest.raises(ValueError):
        sparse_block_diag()
    coo_tensor = Mock(spec=torch.Tensor)
    coo_tensor.layout = torch.sparse_coo
    csr_tensor = Mock(spec=torch.Tensor)
    csr_tensor.layout = torch.sparse_csr
    with pytest.raises(ValueError):
        sparse_block_diag(coo_tensor, csr_tensor)
    coo = Mock(spec=torch.Tensor)
    coo.layout = torch.sparse_coo
    coo.sparse_dim.return_value = 1
    with pytest.raises(ValueError):
        sparse_block_diag(coo)
    coo.dense_dim.return_value = 1
    with pytest.raises(ValueError):
        sparse_block_diag(coo)
    with pytest.raises(TypeError):
        sparse_block_diag("not a list or tuple")
    # generate a small sparse COO without specifying device (defaults to CPU)
    tensor1 = generate_random_sparse_coo_matrix((5, 5), 5)
    with pytest.raises(TypeError):
        sparse_block_diag(tensor1, "bad")


# Test sparse_block_diag_split
@pytest.mark.parametrize(
    "layout, shape, nnz",
    [
        ("coo", (1, 4, 4), 12),
        ("coo", (4, 4, 4), 12),
        ("coo", (6, 8, 14), 32),
        ("csr", (1, 4, 4), 12),
        ("csr", (4, 4, 4), 12),
        ("csr", (6, 8, 14), 32),
    ],
)
def test_sparse_block_diag_split(device, layout, shape, nnz):
    if layout == "coo":
        A = generate_random_sparse_coo_matrix(shape, nnz, device=device)
    else:
        A = generate_random_sparse_csr_matrix(shape, nnz, device=device)
    B = sparse_block_diag(*A)
    # build a tuple of original block shapes: (rows, cols) repeated batch times
    block_shapes = tuple((shape[-2], shape[-1]) for _ in range(shape[0]))
    parts = sparse_block_diag_split(B, *block_shapes)
    for orig, part in zip(A, parts):
        assert torch.equal(orig.to_dense(), part.to_dense())


# Test sparse_eye
TEST_EYE = [
    ("unbat", (4, 4)),
    ("unbat", (8, 8)),
    ("bat", (2, 4, 4)),
    ("bat", (4, 8, 8)),
]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]


@pytest.fixture(params=TEST_EYE, ids=lambda x: x[0])
def shapes(request):
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64], ids=lambda x: str(x).split(".")[-1])
def values_dtype(request):
    return request.param


@pytest.fixture(params=[torch.int32, torch.int64], ids=lambda x: str(x).split(".")[-1])
def indices_dtype(request):
    return request.param


@pytest.fixture(params=LAYOUTS, ids=lambda x: str(x).split(".")[-1].upper())
def layout(request):
    return request.param


def test_sparse_eye(shapes, layout, values_dtype, indices_dtype, device):
    name, size = shapes
    if layout is torch.sparse_coo and indices_dtype == torch.int32:
        pytest.skip("sparse COO does not return int32 indices")
    Id = sparse_eye(
        size,
        layout=layout,
        values_dtype=values_dtype,
        indices_dtype=indices_dtype,
        device=device,
    )
    assert Id.shape == size
    assert Id.layout == layout
    assert Id.device == device
    assert Id.values().dtype == values_dtype
    if layout is torch.sparse_coo:
        assert Id.indices().dtype == indices_dtype
    else:
        assert Id.crow_indices().dtype == indices_dtype
        assert Id.col_indices().dtype == indices_dtype
    if len(size) == 2:
        expected = torch.eye(size[-1], dtype=values_dtype, device=device)
        assert torch.equal(Id.to_dense(), expected)
