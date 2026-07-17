from unittest.mock import Mock

import pytest
import torch
from test_config import DEVICES, devices_match

from torchsparsegradutils.utils.convert import (
    _compress_row_indices,
    _demcompress_crow_indices,
    _sort_coo_indices,
    convert_coo_to_csr,
    convert_coo_to_csr_indices_values,
    sparse_eye,
    stack_csr,
)
from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix,
    generate_random_sparse_csr_matrix,
)


def _id_device(d):
    return str(d)


@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
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


# Test convert_coo_to_csr_indices_values directly (commit 19 regressions:
# the batched path used to derive num_batches via torch.unique(batch_indices),
# silently DROPPING batch items with zero entries — wrong num_batches,
# misaligned (num_batches, -1) reshapes. It now derives num_batches from
# max(batch)+1 and raises a clear ValueError on ragged nse per item instead
# of silently mis-reshaping.)


def test_convert_coo_to_csr_indices_values_unbatched(device):
    A_coo = generate_random_sparse_coo_matrix((6, 5), 10, device=device)
    crow, col, vals = convert_coo_to_csr_indices_values(A_coo.indices(), 6, A_coo.values())
    A_csr = A_coo.to_sparse_csr()
    assert torch.equal(crow, A_csr.crow_indices())
    assert torch.equal(col, A_csr.col_indices())
    assert torch.equal(vals, A_csr.values())


def test_convert_coo_to_csr_indices_values_unbatched_permutation(device):
    A_coo = generate_random_sparse_coo_matrix((6, 5), 10, device=device)
    crow, col, perm = convert_coo_to_csr_indices_values(A_coo.indices(), 6)
    # values is None -> third output is the sort permutation: applying it to
    # the input-order values must reproduce the values-aligned conversion.
    _, _, vals = convert_coo_to_csr_indices_values(A_coo.indices(), 6, A_coo.values())
    assert torch.equal(A_coo.values()[perm.long()], vals)


def test_convert_coo_to_csr_indices_values_batched_equal_nse(device):
    A_coo = generate_random_sparse_coo_matrix((3, 4, 5), 6, device=device)  # equal nse per item by construction
    crow, col, vals = convert_coo_to_csr_indices_values(A_coo.indices(), 4, A_coo.values())
    assert crow.shape == (3, 5)
    assert col.shape == (3, 6)
    assert vals.shape == (3, 6)
    for b in range(3):
        item_csr = A_coo[b].detach().to_sparse_csr()
        assert torch.equal(crow[b], item_csr.crow_indices())
        assert torch.equal(col[b], item_csr.col_indices())
        assert torch.equal(vals[b], item_csr.values())


def test_convert_coo_to_csr_indices_values_raises_on_empty_batch_item(device):
    # Batch item 1 has zero entries. An empty item among non-empty ones is
    # ragged nse — pre-fix this silently produced wrong-shaped output
    # (torch.unique dropped item 1 entirely); now it must raise clearly.
    indices = torch.tensor([[0, 0, 2, 2], [0, 1, 0, 1], [1, 0, 1, 0]], device=device)
    values = torch.arange(4.0, device=device)
    with pytest.raises(ValueError, match="equal nse per batch item"):
        convert_coo_to_csr_indices_values(indices, 2, values)


def test_convert_coo_to_csr_indices_values_raises_on_ragged_nse(device):
    # nse per item [1, 2]: previously silently mis-reshaped; now raises.
    indices = torch.tensor([[0, 1, 1], [0, 0, 1], [1, 0, 1]], device=device)
    values = torch.arange(3.0, device=device)
    with pytest.raises(ValueError, match="equal nse per batch item"):
        convert_coo_to_csr_indices_values(indices, 2, values)


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
    assert devices_match(Id.device, device)
    assert Id.values().dtype == values_dtype
    if layout is torch.sparse_coo:
        assert Id.indices().dtype == indices_dtype
    else:
        assert Id.crow_indices().dtype == indices_dtype
        assert Id.col_indices().dtype == indices_dtype
    if len(size) == 2:
        expected = torch.eye(size[-1], dtype=values_dtype, device=device)
        assert torch.equal(Id.to_dense(), expected)
