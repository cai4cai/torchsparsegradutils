import pytest
import torch
from test_config import DEVICES, SPARSE_LAYOUTS, VALUE_DTYPES, Tolerances

from torchsparsegradutils import sparse_logsumexp

# dim arguments to exercise: per-row, per-column, and full reduction.
DIMS = [0, 1, [0, 1]]


def dtype_id(dtype):
    return str(dtype).split(".")[-1]


def device_id(device):
    return str(device)


def layout_id(layout):
    return "coo" if layout == torch.sparse_coo else "csr"


def dim_id(dim):
    return f"dim{dim}"


def _to_layout(dense, layout):
    if layout == torch.sparse_coo:
        return dense.to_sparse_coo()
    if layout == torch.sparse_csr:
        return dense.to_sparse_csr()
    if layout == torch.sparse_csc:
        return dense.to_sparse_csc()
    raise ValueError(layout)


def _dense_reference(dense, dim, include_zeros):
    """Dense torch.logsumexp reference honouring the include_zeros semantics."""
    if not include_zeros:
        dense = dense.masked_fill(dense == 0, float("-inf"))
    return torch.logsumexp(dense, dim=tuple(dim) if isinstance(dim, list) else dim)


def _assert_close(actual, expected, dtype):
    """allclose that also treats matching -inf entries as equal."""
    atol, rtol = Tolerances.direct(dtype)
    inf_mask = torch.isinf(expected)
    assert torch.equal(torch.isinf(actual), inf_mask), f"inf pattern mismatch:\n{actual}\n{expected}"
    assert torch.allclose(actual[~inf_mask], expected[~inf_mask], atol=atol, rtol=rtol), f"{actual}\n!=\n{expected}"


@pytest.fixture(params=SPARSE_LAYOUTS, ids=[layout_id(x) for x in SPARSE_LAYOUTS])
def layout(request):
    return request.param


@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


@pytest.fixture(params=VALUE_DTYPES, ids=[dtype_id(d) for d in VALUE_DTYPES])
def value_dtype(request):
    return request.param


@pytest.fixture(params=DIMS, ids=[dim_id(d) for d in DIMS])
def dim(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["incl_zeros", "excl_zeros"])
def include_zeros(request):
    return request.param


def _make_dense(device, value_dtype, seed):
    """A 5x4 tensor with a deterministic mix of zeros, incl. an all-zero row/col."""
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(5, 4, generator=g, dtype=torch.float64)
    dense = dense.masked_fill(torch.rand(5, 4, generator=g) < 0.5, 0.0)
    dense[2] = 0.0  # all-zero row -> empty segment when reducing dim=1
    dense[:, 3] = 0.0  # all-zero column -> empty segment when reducing dim=0
    return dense.to(device=device, dtype=value_dtype)


def test_matches_dense_reference(layout, device, value_dtype, dim, include_zeros):
    dense = _make_dense(device, value_dtype, seed=0)
    sp = _to_layout(dense, layout)
    out = sparse_logsumexp(sp, dim=dim, include_zeros=include_zeros)
    ref = _dense_reference(dense, dim, include_zeros)
    _assert_close(out, ref, value_dtype)


def test_keepdim_shapes(layout, device):
    dense = _make_dense(device, torch.float64, seed=1)
    sp = _to_layout(dense, layout)
    assert sparse_logsumexp(sp, dim=1, keepdim=True).shape == (5, 1)
    assert sparse_logsumexp(sp, dim=0, keepdim=True).shape == (1, 4)
    assert sparse_logsumexp(sp, dim=[0, 1], keepdim=True).shape == (1, 1)


def test_csc_layout(device):
    """CSC is supported even though it is not in the standard test matrix."""
    dense = _make_dense(device, torch.float64, seed=2)
    out = sparse_logsumexp(dense.to_sparse_csc(), dim=1, include_zeros=True)
    _assert_close(out, torch.logsumexp(dense, dim=1), torch.float64)


def _make_batched_dense(device, value_dtype, seed):
    """A (3, 5, 4) batched tensor with zeros, incl. an empty row inside one slice."""
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(3, 5, 4, generator=g, dtype=torch.float64)
    dense = dense.masked_fill(torch.rand(3, 5, 4, generator=g) < 0.5, 0.0)
    dense[1, 2] = 0.0  # empty row within the second slice
    return dense.to(device=device, dtype=value_dtype)


@pytest.fixture(params=[1, 2, [1, 2]], ids=["dim1", "dim2", "dim12"])
def batched_dim(request):
    return request.param


def test_batched_matches_dense(device, value_dtype, batched_dim, include_zeros):
    dense = _make_batched_dense(device, value_dtype, seed=4)
    out = sparse_logsumexp(dense.to_sparse_coo(), dim=batched_dim, include_zeros=include_zeros)
    ref = _dense_reference(dense, batched_dim, include_zeros)
    _assert_close(out, ref, value_dtype)


def test_batched_keepdim_shapes(device):
    dense = _make_batched_dense(device, torch.float64, seed=5)
    sp = dense.to_sparse_coo()
    assert sparse_logsumexp(sp, dim=1, keepdim=True).shape == (3, 1, 4)
    assert sparse_logsumexp(sp, dim=2, keepdim=True).shape == (3, 5, 1)
    assert sparse_logsumexp(sp, dim=[1, 2], keepdim=True).shape == (3, 1, 1)


def test_batched_cannot_reduce_batch_dim(device):
    sp = _make_batched_dense(device, torch.float64, seed=6).to_sparse_coo()
    with pytest.raises(NotImplementedError):
        sparse_logsumexp(sp, dim=0)


def test_gradient(layout, device):
    """Gradients through the explicit values match autograd on the dense op."""
    dense = _make_dense(device, torch.float64, seed=3)
    sp = _to_layout(dense, layout).coalesce() if layout == torch.sparse_coo else _to_layout(dense, layout)
    vals = sp.values().clone().requires_grad_(True)

    # Rebuild a sparse tensor that carries grad on its values.
    if layout == torch.sparse_coo:
        sp_grad = torch.sparse_coo_tensor(sp.indices(), vals, sp.shape)
    else:
        sp_grad = torch.sparse_csr_tensor(sp.crow_indices(), sp.col_indices(), vals, sp.shape)

    sparse_logsumexp(sp_grad, dim=1, include_zeros=True).sum().backward()

    # Reference: same reduction on the dense tensor, gradient read at the nnz.
    dense_leaf = dense.clone().requires_grad_(True)
    torch.logsumexp(dense_leaf, dim=1).sum().backward()
    expected = dense_leaf.grad[dense != 0]
    _assert_close(vals.grad, expected, torch.float64)


def test_unsupported_rank_raises():
    x = torch.randn(2, 3, 4, 5).to_sparse_coo()  # 4-D: neither 2-D nor batched 3-D
    with pytest.raises(NotImplementedError):
        sparse_logsumexp(x, dim=1)


def test_dense_layout_raises():
    with pytest.raises(NotImplementedError):
        sparse_logsumexp(torch.randn(3, 3), dim=1)
