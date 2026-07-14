import pytest
import torch
from packaging.version import parse as parse_version
from test_config import DEVICES, INDEX_DTYPES, SPARSE_LAYOUTS, VALUE_DTYPES

# Reuse the sibling module's helpers/fixtures so the two suites stay in lock-step.
from test_sparse_logsumexp import (
    FWD_LAYOUTS,
    _assert_close,
    _dense_reference,
    _make_batched_dense,
    _make_batched_dense_equal_nnz,
    _make_dense,
    _to_layout,
    device_id,
    dtype_id,
    layout_id,
)

from torchsparsegradutils import sparse_bidir_logsumexp, sparse_logsumexp
from torchsparsegradutils.ops.logsumexp import _bidir_2d, _bidir_batched

_NESTED_OK = parse_version(torch.__version__) >= parse_version("2.4")


@pytest.fixture(params=SPARSE_LAYOUTS, ids=[layout_id(x) for x in SPARSE_LAYOUTS])
def layout(request):
    return request.param


@pytest.fixture(params=FWD_LAYOUTS, ids=[layout_id(x) for x in FWD_LAYOUTS])
def fwd_layout(request):
    return request.param


@pytest.fixture(params=FWD_LAYOUTS, ids=[layout_id(x) for x in FWD_LAYOUTS])
def batched_layout(request):
    return request.param


@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


@pytest.fixture(params=VALUE_DTYPES, ids=[dtype_id(d) for d in VALUE_DTYPES])
def value_dtype(request):
    return request.param


@pytest.fixture(params=INDEX_DTYPES, ids=[dtype_id(d) for d in INDEX_DTYPES])
def index_dtype(request):
    return request.param


@pytest.fixture(params=[True, False], ids=["incl_zeros", "excl_zeros"])
def include_zeros(request):
    return request.param


def test_matches_two_call_and_dense(fwd_layout, device, value_dtype, index_dtype, include_zeros):
    """(col_lse, row_lse) equals two sparse_logsumexp calls and the dense reference."""
    dense = _make_dense(device, value_dtype, seed=0)  # 5x4: has an all-zero row and column
    sp = _to_layout(dense, fwd_layout, index_dtype)
    col_lse, row_lse = sparse_bidir_logsumexp(sp, include_zeros=include_zeros)
    _assert_close(col_lse, sparse_logsumexp(sp, dim=0, include_zeros=include_zeros), value_dtype)
    _assert_close(row_lse, sparse_logsumexp(sp, dim=1, include_zeros=include_zeros), value_dtype)
    _assert_close(col_lse, _dense_reference(dense, 0, include_zeros), value_dtype)
    _assert_close(row_lse, _dense_reference(dense, 1, include_zeros), value_dtype)


def test_output_layouts_agree(fwd_layout, device, include_zeros):
    """padded and nested carry the same values as tuple; padded pads to -inf."""
    dense = _make_dense(device, torch.float64, seed=1)  # 5x4 -> G=5, col_lse padded by one
    sp = _to_layout(dense, fwd_layout)
    nrows, ncols = dense.shape
    G = max(nrows, ncols)
    col_lse, row_lse = sparse_bidir_logsumexp(sp, include_zeros=include_zeros)

    padded = sparse_bidir_logsumexp(sp, include_zeros=include_zeros, output_layout="padded")
    assert padded.shape == (2, G)
    _assert_close(padded[0, :ncols], col_lse, torch.float64)
    _assert_close(padded[1, :nrows], row_lse, torch.float64)
    assert torch.isneginf(padded[0, ncols:]).all()  # col_lse tail padded with -inf

    if _NESTED_OK:
        nested = sparse_bidir_logsumexp(sp, include_zeros=include_zeros, output_layout="nested")
        parts = nested.unbind()
        _assert_close(parts[0], col_lse, torch.float64)
        _assert_close(parts[1], row_lse, torch.float64)


def test_wide_matrix_row_padding(fwd_layout, device):
    """ncols > nrows exercises the row-side -inf padding; the tall (5x4) fixtures only
    ever hit the column-side pad, so transpose to a wide (4x5) matrix."""
    dense = _make_dense(device, torch.float64, seed=1).t().contiguous()  # (4, 5), wide
    sp = _to_layout(dense, fwd_layout)
    nrows, ncols = dense.shape  # 4, 5 -> G=5
    G = max(nrows, ncols)
    col_lse, row_lse = sparse_bidir_logsumexp(sp)
    _assert_close(col_lse, _dense_reference(dense, 0, True), torch.float64)
    _assert_close(row_lse, _dense_reference(dense, 1, True), torch.float64)
    padded = sparse_bidir_logsumexp(sp, output_layout="padded")
    assert padded.shape == (2, G)
    _assert_close(padded[1, :nrows], row_lse, torch.float64)
    assert torch.isneginf(padded[1, nrows:]).all()  # row_lse tail padded with -inf


def test_keepdim_shapes(fwd_layout, device):
    dense = _make_dense(device, torch.float64, seed=2)  # (5, 4)
    sp = _to_layout(dense, fwd_layout)
    col_lse, row_lse = sparse_bidir_logsumexp(sp, keepdim=True)
    assert col_lse.shape == (1, 4)  # dim=0 reduction kept
    assert row_lse.shape == (5, 1)  # dim=1 reduction kept


def test_keepdim_rejected_for_non_tuple_layouts(device):
    sp = _make_dense(device, torch.float64, seed=3).to_sparse_coo()
    for output_layout in ("padded", "nested"):
        with pytest.raises(ValueError, match="keepdim is only supported"):
            sparse_bidir_logsumexp(sp, keepdim=True, output_layout=output_layout)


def test_unknown_output_layout_raises(device):
    sp = _make_dense(device, torch.float64, seed=3).to_sparse_coo()
    with pytest.raises(ValueError, match="unknown output_layout"):
        sparse_bidir_logsumexp(sp, output_layout="bogus")


def test_all_negative_values_stability_both_axes(fwd_layout, device, include_zeros):
    """Very-negative values on both axes: structural zeros must enter the shift so
    neither axis overflows to +inf or produces nan (the #87 blocking case, verified
    for the column reduction too)."""
    dense = torch.tensor(
        [[-1000.0, -5.0, 0.0], [-3.0, -900.0, -2.0], [0.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float64,
    )
    col_lse, row_lse = sparse_bidir_logsumexp(_to_layout(dense, fwd_layout), include_zeros=include_zeros)
    _assert_close(col_lse, _dense_reference(dense, 0, include_zeros), torch.float64)
    _assert_close(row_lse, _dense_reference(dense, 1, include_zeros), torch.float64)
    assert not torch.isnan(col_lse).any() and not torch.isnan(row_lse).any()


def test_positive_inf_value_both_axes(fwd_layout, device, include_zeros):
    """An explicit +inf must yield +inf on both axes, not nan from inf - inf."""
    dense = torch.tensor([[float("inf"), 0.0, 1.0], [-2.0, 3.0, 0.0]], device=device, dtype=torch.float64)
    col_lse, row_lse = sparse_bidir_logsumexp(_to_layout(dense, fwd_layout), include_zeros=include_zeros)
    _assert_close(col_lse, _dense_reference(dense, 0, include_zeros), torch.float64)
    _assert_close(row_lse, _dense_reference(dense, 1, include_zeros), torch.float64)
    assert not torch.isnan(col_lse).any() and not torch.isnan(row_lse).any()


def test_batched_matches_two_call(device, value_dtype, include_zeros):
    dense = _make_batched_dense(device, value_dtype, seed=4)  # (3, 5, 4)
    sp = dense.to_sparse_coo()
    col_lse, row_lse = sparse_bidir_logsumexp(sp, include_zeros=include_zeros)
    _assert_close(col_lse, sparse_logsumexp(sp, dim=1, include_zeros=include_zeros), value_dtype)
    _assert_close(row_lse, sparse_logsumexp(sp, dim=2, include_zeros=include_zeros), value_dtype)


def test_batched_output_shapes(device):
    dense = _make_batched_dense(device, torch.float64, seed=5)  # (3, 5, 4)
    sp = dense.to_sparse_coo()
    b, nrows, ncols = dense.shape
    G = max(nrows, ncols)
    col_lse, row_lse = sparse_bidir_logsumexp(sp)
    assert col_lse.shape == (b, ncols) and row_lse.shape == (b, nrows)
    assert sparse_bidir_logsumexp(sp, output_layout="padded").shape == (2, b, G)
    ck, rk = sparse_bidir_logsumexp(sp, keepdim=True)
    assert ck.shape == (b, 1, ncols) and rk.shape == (b, nrows, 1)


def test_batched_output_layouts_agree(device, include_zeros):
    """Batched padded/nested carry the same values as tuple (not just the right shape),
    so a col/row plane swap in the padded assembly cannot pass unnoticed."""
    dense = _make_batched_dense(device, torch.float64, seed=5)  # (3, 5, 4)
    sp = dense.to_sparse_coo()
    b, nrows, ncols = dense.shape
    G = max(nrows, ncols)
    col_lse, row_lse = sparse_bidir_logsumexp(sp, include_zeros=include_zeros)

    padded = sparse_bidir_logsumexp(sp, include_zeros=include_zeros, output_layout="padded")
    assert padded.shape == (2, b, G)
    _assert_close(padded[0, :, :ncols], col_lse, torch.float64)
    _assert_close(padded[1, :, :nrows], row_lse, torch.float64)
    assert torch.isneginf(padded[0, :, ncols:]).all()  # col-side -inf pad

    if _NESTED_OK:
        parts = sparse_bidir_logsumexp(sp, include_zeros=include_zeros, output_layout="nested").unbind()
        _assert_close(parts[0], col_lse, torch.float64)  # (b, ncols)
        _assert_close(parts[1], row_lse, torch.float64)  # (b, nrows)


def test_batched_all_layouts_match_two_call(batched_layout, device, value_dtype, include_zeros):
    """Batched COO, CSR and CSC agree with the two-call baseline. The compressed layouts
    take a different route in (converted, since batched CSR/CSC expose no per-slice index
    accessors), so COO coverage alone would not exercise them."""
    dense = _make_batched_dense_equal_nnz(device, value_dtype, seed=4)
    sp = _to_layout(dense, batched_layout)
    col_lse, row_lse = sparse_bidir_logsumexp(sp, include_zeros=include_zeros)
    _assert_close(col_lse, sparse_logsumexp(sp, dim=1, include_zeros=include_zeros), value_dtype)
    _assert_close(row_lse, sparse_logsumexp(sp, dim=2, include_zeros=include_zeros), value_dtype)
    _assert_close(col_lse, _dense_reference(dense, 1, include_zeros), value_dtype)
    _assert_close(row_lse, _dense_reference(dense, 2, include_zeros), value_dtype)


def test_batched_all_layouts_output_shapes(batched_layout, device):
    """The (2, b, G) padded buffer and the tuple/keepdim shapes hold for every layout."""
    dense = _make_batched_dense_equal_nnz(device, torch.float64, seed=5)
    sp = _to_layout(dense, batched_layout)
    b, nrows, ncols = dense.shape
    G = max(nrows, ncols)
    col_lse, row_lse = sparse_bidir_logsumexp(sp)
    assert col_lse.shape == (b, ncols) and row_lse.shape == (b, nrows)

    padded = sparse_bidir_logsumexp(sp, output_layout="padded")
    assert padded.shape == (2, b, G)
    _assert_close(padded[0, :, :ncols], col_lse, torch.float64)
    _assert_close(padded[1, :, :nrows], row_lse, torch.float64)


def test_bidir_returns_share_one_allocation(device):
    """Both _bidir_* helpers return the scatter's native buffer plus two views into it,
    and every output_layout is served from that one allocation — no layout copies it.
    The direction leads in both ranks, so no transpose is needed to reach 'padded'."""
    same_storage = lambda a, b: a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

    dense = _make_dense(device, torch.float64, seed=7)  # (5, 4)
    nrows, ncols = dense.shape
    col, row, padded = _bidir_2d(dense.to_sparse_coo(), True)
    assert same_storage(col, padded) and same_storage(row, padded)
    assert padded.shape == (2, max(nrows, ncols))

    bdense = _make_batched_dense(device, torch.float64, seed=7)  # (3, 5, 4)
    b, nrows, ncols = bdense.shape
    col, row, padded = _bidir_batched(bdense.to_sparse_coo(), True)
    assert same_storage(col, padded) and same_storage(row, padded)
    assert padded.shape == (2, b, max(nrows, ncols))  # native (direction, batch, group) order


def test_gradient_parity_with_two_call(layout, device):
    """The values.expand(2, nnz) backward must sum each nonzero's row + column
    gradient — identical to summing the two separate reductions' gradients."""
    dense = _make_dense(device, torch.float64, seed=6)

    def _grad(bidir):
        leaf = dense.clone().requires_grad_(True)
        sp = _to_layout(leaf, layout) if layout != torch.sparse_coo else leaf.to_sparse_coo()
        if bidir:
            col_lse, row_lse = sparse_bidir_logsumexp(sp)
            (col_lse.sum() + row_lse.sum()).backward()
        else:
            (sparse_logsumexp(sp, dim=0).sum() + sparse_logsumexp(sp, dim=1).sum()).backward()
        return leaf.grad

    _assert_close(_grad(bidir=True), _grad(bidir=False), torch.float64)


def test_batched_gradient_parity_with_two_call(device):
    """_bidir_batched is a distinct backward path; its gradient must still match the
    sum of the two separate batched reductions' gradients."""
    dense = _make_batched_dense(device, torch.float64, seed=6)  # (3, 5, 4)

    def _grad(bidir):
        leaf = dense.clone().requires_grad_(True)
        sp = leaf.to_sparse_coo()
        if bidir:
            col_lse, row_lse = sparse_bidir_logsumexp(sp)
            (col_lse.sum() + row_lse.sum()).backward()
        else:
            (sparse_logsumexp(sp, dim=1).sum() + sparse_logsumexp(sp, dim=2).sum()).backward()
        return leaf.grad

    _assert_close(_grad(bidir=True), _grad(bidir=False), torch.float64)


def test_duplicate_coordinates_are_coalesced(device, include_zeros):
    """An uncoalesced COO with repeated coordinates must sum the duplicates before
    exp (guards against a refactor to _values()/_indices() that would double-count)."""
    indices = torch.tensor([[0, 0, 1], [1, 1, 2]], device=device)  # (0,1) appears twice
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=device)
    sp = torch.sparse_coo_tensor(indices, values, (2, 3), device=device)  # NOT coalesced
    assert not sp.is_coalesced()
    dense = sp.to_dense()  # (0,1) -> 3.0 after summing
    col_lse, row_lse = sparse_bidir_logsumexp(sp, include_zeros=include_zeros)
    _assert_close(col_lse, _dense_reference(dense, 0, include_zeros), torch.float64)
    _assert_close(row_lse, _dense_reference(dense, 1, include_zeros), torch.float64)


def test_unsupported_rank_raises(device):
    x = torch.randn(2, 3, 4, 5, device=device).to_sparse_coo()
    with pytest.raises(NotImplementedError):
        sparse_bidir_logsumexp(x)


def test_dense_layout_raises(device):
    with pytest.raises(NotImplementedError):
        sparse_bidir_logsumexp(torch.randn(3, 3, device=device))
