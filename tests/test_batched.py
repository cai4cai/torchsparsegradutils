"""Tests for the BatchedCSR / BatchedCSC descriptors (torchsparsegradutils/_batched.py).

Covers: every from_torch() input form + to_torch(like=...) round-trip, folded-row
math (row_global = b * n_rows + r), ragged batches (incl. an empty batch item),
B=1 unbatched, the int32 eligibility rule, and naming.md §1 error-message
conformance. Property-based round-trip exactness over random shapes/nse/dtypes
via hypothesis (plain @given — profiles arrive commit 11).

Note: torch's ``sparse_coo_tensor``/``.coalesce()`` always coerce indices to
int64 regardless of the dtype passed in (map.md invariant 6's "upstream COO
int64 coercion excepted"). So genuine int32-index fixtures below are built
directly as CSR (``torch.sparse_csr_tensor``), never via a COO detour.
"""

import itertools

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from test_config import INDEX_DTYPES, VALUE_DTYPES

from torchsparsegradutils._batched import BatchedCSC, BatchedCSR, _resolve_index_dtype
from torchsparsegradutils.utils.convert import stack_csr

# --------------------------------------------------------------------------- #
# Fixture builders — independent of _batched.py, so round-trip tests aren't
# circularly validated against the code under test.
# --------------------------------------------------------------------------- #


def _make_2d_coo(coords, n_rows, n_cols, value_dtype):
    """A 2D COO tensor. Indices are always int64 (upstream coercion)."""
    if coords:
        idx = torch.tensor(coords, dtype=torch.int64).t().contiguous()
        vals = torch.arange(1, len(coords) + 1, dtype=value_dtype)
    else:
        idx = torch.empty((2, 0), dtype=torch.int64)
        vals = torch.empty((0,), dtype=value_dtype)
    return torch.sparse_coo_tensor(idx, vals, (n_rows, n_cols)).coalesce()


def _make_batched_coo(coords, batch_size, n_rows, n_cols, value_dtype):
    """A 3D (batched) COO tensor. Indices are always int64 (upstream coercion)."""
    if coords:
        idx = torch.tensor(coords, dtype=torch.int64).t().contiguous()
        vals = torch.arange(1, len(coords) + 1, dtype=value_dtype)
    else:
        idx = torch.empty((3, 0), dtype=torch.int64)
        vals = torch.empty((0,), dtype=value_dtype)
    return torch.sparse_coo_tensor(idx, vals, (batch_size, n_rows, n_cols)).coalesce()


def _make_2d_csr(coords, n_rows, n_cols, index_dtype, value_dtype):
    """A 2D CSR tensor built directly from (row, col) coords, with a genuinely
    controlled index dtype (int32 survives — no COO detour)."""
    coords = sorted(set(coords))
    crow = [0] * (n_rows + 1)
    for r, _c in coords:
        crow[r + 1] += 1
    for i in range(1, n_rows + 1):
        crow[i] += crow[i - 1]
    crow_t = torch.tensor(crow, dtype=index_dtype)
    col_t = torch.tensor([c for _r, c in coords], dtype=index_dtype)
    vals = torch.arange(1, len(coords) + 1, dtype=value_dtype)
    return torch.sparse_csr_tensor(crow_t, col_t, vals, (n_rows, n_cols))


def _all_coords_2d(n_rows, n_cols):
    return list(itertools.product(range(n_rows), range(n_cols)))


def _all_coords_batched(batch_size, n_rows, n_cols):
    return list(itertools.product(range(batch_size), range(n_rows), range(n_cols)))


def _equal_nse_csr_batch(batch_size, n_rows, n_cols, nse, index_dtype, value_dtype):
    """A list of 2D CSR tensors, each with exactly `nse` entries (equal nse per
    item), independent of _batched.py."""
    all_coords = _all_coords_2d(n_rows, n_cols)
    items = []
    for b in range(batch_size):
        # Deterministic-but-distinct pattern per item, still exactly `nse` entries.
        chosen = [all_coords[(i + b) % len(all_coords)] for i in range(nse)]
        items.append(_make_2d_csr(chosen, n_rows, n_cols, index_dtype, value_dtype))
    return items


# --------------------------------------------------------------------------- #
# Unit tests — one per accepted from_torch() input form + round-trip.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_from_torch_csr_2d_is_zero_copy(value_dtype, index_dtype):
    csr = _make_2d_csr([(0, 1), (1, 0), (1, 2)], 2, 3, index_dtype, value_dtype)

    descriptor = BatchedCSR.from_torch(csr)

    assert descriptor.shape == (1, 2, 3)
    assert descriptor.values.data_ptr() == csr.values().data_ptr()
    assert descriptor.rowptr.data_ptr() == csr.crow_indices().data_ptr()
    assert descriptor.col.data_ptr() == csr.col_indices().data_ptr()
    assert descriptor.index_dtype == index_dtype

    back = descriptor.to_torch(like=csr)
    assert torch.equal(back.crow_indices(), csr.crow_indices())
    assert torch.equal(back.col_indices(), csr.col_indices())
    assert torch.equal(back.values(), csr.values())
    assert back.crow_indices().dtype == index_dtype


@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_from_torch_coo_2d_round_trip(value_dtype):
    coo = _make_2d_coo([(0, 1), (1, 0), (1, 2), (2, 2)], 3, 3, value_dtype)

    descriptor = BatchedCSR.from_torch(coo)
    assert descriptor.shape == (1, 3, 3)
    assert descriptor.index_dtype == torch.int64
    assert descriptor.nse_total == 4

    back = descriptor.to_torch(like=coo)
    assert back.layout == torch.sparse_coo
    assert torch.equal(back.indices(), coo.indices())
    assert torch.equal(back.values(), coo.values())


def test_from_torch_coo_batched_ragged_with_empty_item():
    # 3 batch items, batch 1 is entirely empty (ragged nse, first-class per naming.md §2).
    coords = [(0, 0, 1), (0, 1, 0), (2, 0, 0), (2, 1, 1)]
    coo = _make_batched_coo(coords, batch_size=3, n_rows=2, n_cols=2, value_dtype=torch.float64)

    descriptor = BatchedCSR.from_torch(coo)
    assert descriptor.shape == (3, 2, 2)
    assert descriptor.nse_total == 4

    # Batch item 1 (folded rows 2,3) must show zero specified entries.
    assert descriptor.rowptr[2] == descriptor.rowptr[4]

    back = descriptor.to_torch(like=coo)
    assert torch.equal(back.indices(), coo.indices())
    assert torch.equal(back.values(), coo.values())


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_from_torch_csr_batched_equal_nse(index_dtype):
    items = _equal_nse_csr_batch(3, 2, 2, nse=2, index_dtype=index_dtype, value_dtype=torch.float32)
    stacked = stack_csr(items)
    assert stacked.layout == torch.sparse_csr and stacked.ndim == 3

    descriptor = BatchedCSR.from_torch(stacked)
    assert descriptor.shape == (3, 2, 2)
    assert descriptor.nse_total == 6
    assert descriptor.index_dtype == index_dtype

    back = descriptor.to_torch(like=stacked)
    assert torch.equal(back.crow_indices(), stacked.crow_indices())
    assert torch.equal(back.col_indices(), stacked.col_indices())
    assert torch.equal(back.values(), stacked.values())


@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_from_torch_csr_list_ragged_with_empty_item(index_dtype):
    empty = _make_2d_csr([], 2, 3, index_dtype, torch.float64)
    one = _make_2d_csr([(0, 1)], 2, 3, index_dtype, torch.float64)
    two = _make_2d_csr([(0, 0), (1, 2)], 2, 3, index_dtype, torch.float64)
    items = [empty, one, two]

    descriptor = BatchedCSR.from_torch(items)
    assert descriptor.shape == (3, 2, 3)
    assert descriptor.nse_total == 3
    assert descriptor.index_dtype == index_dtype

    back = descriptor.to_torch(like=items)
    assert isinstance(back, list) and len(back) == 3
    for reconstructed, original in zip(back, items):
        assert torch.equal(reconstructed.crow_indices(), original.crow_indices())
        assert torch.equal(reconstructed.col_indices(), original.col_indices())
        assert torch.equal(reconstructed.values(), original.values())


# --------------------------------------------------------------------------- #
# Folded-row math, B=1, transposed.
# --------------------------------------------------------------------------- #


def test_folded_row_math_row_global_equals_batch_times_n_plus_r():
    # batch_size=2, n_rows=3: folded rows are 0..5. Put one entry in each
    # folded row so row_indices should read back exactly [0, 1, 2, 3, 4, 5].
    coords = [(b, r, 0) for b in range(2) for r in range(3)]
    coo = _make_batched_coo(coords, batch_size=2, n_rows=3, n_cols=1, value_dtype=torch.float64)

    descriptor = BatchedCSR.from_torch(coo)
    expected = torch.arange(6, dtype=torch.int64)
    assert torch.equal(descriptor.row_indices, expected)

    # And explicitly: row_global // n_rows == batch, row_global % n_rows == row.
    for entry, (b, r, _c) in zip(descriptor.row_indices.tolist(), sorted(coords)):
        assert entry // 3 == b
        assert entry % 3 == r


def test_b1_encodes_unbatched():
    coo = _make_2d_coo([(0, 0)], 1, 1, torch.float64)
    descriptor = BatchedCSR.from_torch(coo)
    assert descriptor.shape[0] == 1
    assert descriptor.rowptr.shape == (2,)  # B * n_rows + 1 == 1 * 1 + 1
    assert torch.equal(descriptor.row_indices, torch.zeros(1, dtype=torch.int64))


def test_transposed_round_trip_identity():
    coords = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    coo = _make_batched_coo(coords, batch_size=2, n_rows=2, n_cols=2, value_dtype=torch.float64)
    descriptor = BatchedCSR.from_torch(coo)

    transposed = descriptor.transposed
    assert isinstance(transposed, BatchedCSC)
    assert transposed.shape == descriptor.shape

    back = transposed.transposed
    assert isinstance(back, BatchedCSR)
    assert torch.equal(back.rowptr, descriptor.rowptr)
    assert torch.equal(back.col, descriptor.col)
    assert torch.equal(back.values, descriptor.values)


def test_transposed_matches_dense_transpose():
    coords = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]
    coo = _make_batched_coo(coords, batch_size=2, n_rows=2, n_cols=2, value_dtype=torch.float64)
    descriptor = BatchedCSR.from_torch(coo)
    dense = coo.to_dense()

    csc = descriptor.transposed
    col_global = csc.col_indices
    batch = col_global // 2
    col_local = col_global % 2
    dense_t = torch.zeros_like(dense)
    for b, c, r, v in zip(batch.tolist(), col_local.tolist(), csc.row.tolist(), csc.values.tolist()):
        dense_t[b, r, c] = v
    assert torch.equal(dense_t, dense)


def test_plans_placeholder_is_empty_dict():
    coo = _make_2d_coo([(0, 0)], 1, 1, torch.float64)
    descriptor = BatchedCSR.from_torch(coo)
    assert descriptor.plans == {}
    assert descriptor.transposed.plans == {}


def test_index_dtype_and_nse_total_properties():
    csr = _make_2d_csr([(0, 0), (0, 1)], 2, 2, torch.int32, torch.float32)
    descriptor = BatchedCSR.from_torch(csr)
    assert descriptor.index_dtype == torch.int32
    assert descriptor.nse_total == 2


# --------------------------------------------------------------------------- #
# int32 eligibility rule.
# --------------------------------------------------------------------------- #


def test_int32_source_kept_when_eligible():
    csr = _make_2d_csr([(0, 0), (1, 1)], 4, 4, torch.int32, torch.float32)
    descriptor = BatchedCSR.from_torch(csr)
    assert descriptor.index_dtype == torch.int32


def test_int64_source_never_downcast_even_when_tiny():
    csr = _make_2d_csr([(0, 0)], 2, 2, torch.int64, torch.float32)
    descriptor = BatchedCSR.from_torch(csr)
    assert descriptor.index_dtype == torch.int64


def test_coo_source_always_int64_upstream_coercion():
    # map.md invariant 6: "upstream COO int64 coercion excepted" — COO can
    # never actually supply int32 indices, regardless of eligibility.
    coo = _make_2d_coo([(0, 0)], 2, 2, torch.float32)
    descriptor = BatchedCSR.from_torch(coo)
    assert descriptor.index_dtype == torch.int64


@pytest.mark.parametrize(
    "nse_total,rowptr_len,n_cols,expected",
    [
        (10, 10, 10, torch.int32),  # comfortably eligible
        (2**31 - 1, 10, 10, torch.int32),  # just under the bound
        (2**31, 10, 10, torch.int64),  # nse_total crosses the bound
        (10, 2**31, 10, torch.int64),  # rowptr length crosses the bound
        (10, 10, 2**31, torch.int64),  # n_cols crosses the bound
    ],
)
def test_resolve_index_dtype_boundary(nse_total, rowptr_len, n_cols, expected):
    # Exercised directly (no giant tensor allocation) since the boundary is 2**31.
    assert _resolve_index_dtype(torch.int32, nse_total=nse_total, rowptr_len=rowptr_len, n_cols=n_cols) == expected


def test_resolve_index_dtype_int64_source_always_int64():
    assert _resolve_index_dtype(torch.int64, nse_total=1, rowptr_len=1, n_cols=1) == torch.int64


def test_resolve_index_dtype_rejects_unsupported_dtype():
    with pytest.raises(TypeError):
        _resolve_index_dtype(torch.float32, nse_total=1, rowptr_len=1, n_cols=1)


# --------------------------------------------------------------------------- #
# Error-message conformance (naming.md §1 template: accepted forms + received).
# --------------------------------------------------------------------------- #


def test_from_torch_rejects_dense_tensor():
    dense = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="accepts a 2D sparse matrix"):
        BatchedCSR.from_torch(dense)


def test_from_torch_rejects_wrong_ndim_coo():
    idx = torch.zeros((4, 0), dtype=torch.int64)
    vals = torch.zeros((0,))
    bad = torch.sparse_coo_tensor(idx, vals, (2, 2, 2, 2))
    with pytest.raises(ValueError, match="got a COO tensor of shape"):
        BatchedCSR.from_torch(bad)


def test_from_torch_rejects_empty_list():
    with pytest.raises(ValueError, match="empty list"):
        BatchedCSR.from_torch([])


def test_from_torch_rejects_mismatched_list_shapes():
    a = _make_2d_csr([(0, 0)], 2, 2, torch.int64, torch.float64)
    b = _make_2d_csr([(0, 0)], 3, 3, torch.int64, torch.float64)
    with pytest.raises(ValueError, match="mismatched shapes"):
        BatchedCSR.from_torch([a, b])


def test_from_torch_rejects_mismatched_list_index_dtypes():
    a = _make_2d_csr([(0, 0)], 2, 2, torch.int64, torch.float64)
    b = _make_2d_csr([(0, 0)], 2, 2, torch.int32, torch.float64)
    with pytest.raises(ValueError, match="mismatched index dtypes"):
        BatchedCSR.from_torch([a, b])


def test_to_torch_rejects_ragged_for_batched_csr_target():
    empty = _make_2d_csr([], 2, 2, torch.int64, torch.float64)
    one = _make_2d_csr([(0, 0)], 2, 2, torch.int64, torch.float64)
    descriptor = BatchedCSR.from_torch([empty, one])
    like = stack_csr(_equal_nse_csr_batch(2, 2, 2, nse=1, index_dtype=torch.int64, value_dtype=torch.float64))
    with pytest.raises(ValueError, match="equal nse per batch item"):
        descriptor.to_torch(like=like)


def test_to_torch_rejects_batch_size_mismatch_for_list_target():
    items = _equal_nse_csr_batch(2, 2, 2, nse=1, index_dtype=torch.int64, value_dtype=torch.float64)
    descriptor = BatchedCSR.from_torch(items)
    with pytest.raises(ValueError, match="does not match"):
        descriptor.to_torch(like=[items[0]])  # only 1 item, descriptor has batch_size=2


def test_to_torch_rejects_batched_descriptor_for_2d_forms():
    items = _equal_nse_csr_batch(2, 2, 2, nse=1, index_dtype=torch.int64, value_dtype=torch.float64)
    descriptor = BatchedCSR.from_torch(items)
    coo_2d_like = _make_2d_coo([(0, 0)], 2, 2, torch.float64)
    with pytest.raises(ValueError, match="requires batch_size == 1"):
        descriptor.to_torch(like=coo_2d_like)


def test_post_init_rejects_mismatched_rowptr_length():
    with pytest.raises(ValueError, match="rowptr must be absolute over folded rows"):
        BatchedCSR(
            values=torch.zeros(1),
            rowptr=torch.tensor([0, 1]),  # wrong length for shape (1, 2, 2)
            col=torch.zeros(1, dtype=torch.int64),
            shape=(1, 2, 2),
        )


def test_post_init_rejects_dtype_mismatch_between_rowptr_and_col():
    with pytest.raises(ValueError, match="must share one index dtype"):
        BatchedCSR(
            values=torch.zeros(1),
            rowptr=torch.tensor([0, 1, 1], dtype=torch.int64),
            col=torch.zeros(1, dtype=torch.int32),
            shape=(1, 2, 2),
        )


def test_post_init_rejects_unsupported_index_dtype():
    with pytest.raises(ValueError, match="must be torch.int32 or torch.int64"):
        BatchedCSR(
            values=torch.zeros(1),
            rowptr=torch.tensor([0, 1, 1], dtype=torch.float32),
            col=torch.zeros(1, dtype=torch.float32),
            shape=(1, 2, 2),
        )


# --------------------------------------------------------------------------- #
# tsgu::coo2csr internal switch (commit 19): from_torch's COO paths route
# through the kernel when it is available; the descriptor must come out
# IDENTICAL to the pure-torch compress path.
# --------------------------------------------------------------------------- #


def _coo2csr_cuda_kernel_ready() -> bool:
    import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
    from torchsparsegradutils._dispatch import backend_available

    return (
        torch.cuda.is_available()
        and backend_available()
        and torch._C._dispatch_has_kernel_for_dispatch_key("tsgu::coo2csr", "CUDA")
    )


@pytest.mark.skipif(
    not _coo2csr_cuda_kernel_ready(),
    reason="tsgu::coo2csr has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs a CUDA device, a loaded backend, and the coo2csr kernel registered (commit 19 kernel lane)",
)
def test_from_coo_op_path_matches_pure_torch_path(monkeypatch):
    import torchsparsegradutils.utils.convert as convert_mod

    gen = torch.Generator().manual_seed(7)
    batch_size, n_rows, n_cols = 4, 5, 6
    # Random ragged batched COO, batch item 2 entirely empty (ragged nse is
    # first-class for COO batching, naming.md §2).
    coords = []
    for b in range(batch_size):
        if b == 2:
            continue
        n_item = int(torch.randint(1, n_rows * n_cols, (1,), generator=gen))
        flat = torch.randperm(n_rows * n_cols, generator=gen)[:n_item]
        coords.extend((b, int(f) // n_cols, int(f) % n_cols) for f in flat)
    coo = _make_batched_coo(coords, batch_size, n_rows, n_cols, torch.float64).cuda().coalesce()

    descriptor_op = BatchedCSR.from_torch(coo)

    monkeypatch.setattr(convert_mod, "_coo2csr_backend_ready", lambda _t: False)
    descriptor_torch = BatchedCSR.from_torch(coo)

    assert descriptor_op.shape == descriptor_torch.shape
    assert descriptor_op.index_dtype == descriptor_torch.index_dtype
    assert torch.equal(descriptor_op.rowptr, descriptor_torch.rowptr)
    assert torch.equal(descriptor_op.col, descriptor_torch.col)
    assert torch.equal(descriptor_op.values, descriptor_torch.values)

    # 2D (unbatched) path too.
    coo_2d = _make_2d_coo([(2, 1), (0, 2), (1, 0), (0, 0)], 3, 3, torch.float32).cuda().coalesce()
    monkeypatch.undo()
    descriptor_op_2d = BatchedCSR.from_torch(coo_2d)
    monkeypatch.setattr(convert_mod, "_coo2csr_backend_ready", lambda _t: False)
    descriptor_torch_2d = BatchedCSR.from_torch(coo_2d)
    assert torch.equal(descriptor_op_2d.rowptr, descriptor_torch_2d.rowptr)
    assert torch.equal(descriptor_op_2d.col, descriptor_torch_2d.col)
    assert torch.equal(descriptor_op_2d.values, descriptor_torch_2d.values)


# --------------------------------------------------------------------------- #
# Hypothesis property tests — round-trip exactness over random shapes/nse/dtypes.
# --------------------------------------------------------------------------- #


@st.composite
def _random_2d(draw, min_dim=1, max_dim=5):
    n_rows = draw(st.integers(min_dim, max_dim))
    n_cols = draw(st.integers(min_dim, max_dim))
    all_coords = _all_coords_2d(n_rows, n_cols)
    coords = draw(st.lists(st.sampled_from(all_coords), unique=True, max_size=len(all_coords)))
    value_dtype = draw(st.sampled_from(VALUE_DTYPES))
    return n_rows, n_cols, coords, value_dtype


@st.composite
def _random_batched(draw, min_batch=1, max_batch=3, min_dim=1, max_dim=4):
    batch_size = draw(st.integers(min_batch, max_batch))
    n_rows = draw(st.integers(min_dim, max_dim))
    n_cols = draw(st.integers(min_dim, max_dim))
    all_coords = _all_coords_batched(batch_size, n_rows, n_cols)
    coords = draw(st.lists(st.sampled_from(all_coords), unique=True, max_size=len(all_coords)))
    value_dtype = draw(st.sampled_from(VALUE_DTYPES))
    return batch_size, n_rows, n_cols, coords, value_dtype


@given(_random_2d())
@settings(deadline=None, max_examples=50)
def test_hypothesis_round_trip_coo_2d(data):
    n_rows, n_cols, coords, value_dtype = data
    coo = _make_2d_coo(coords, n_rows, n_cols, value_dtype)

    descriptor = BatchedCSR.from_torch(coo)
    back = descriptor.to_torch(like=coo)

    assert back.indices().dtype == torch.int64
    assert torch.equal(back.indices(), coo.indices())
    assert torch.equal(back.values(), coo.values())


@given(_random_batched())
@settings(deadline=None, max_examples=50)
def test_hypothesis_round_trip_coo_batched(data):
    batch_size, n_rows, n_cols, coords, value_dtype = data
    coo = _make_batched_coo(coords, batch_size, n_rows, n_cols, value_dtype)

    descriptor = BatchedCSR.from_torch(coo)
    assert descriptor.shape == (batch_size, n_rows, n_cols)

    back = descriptor.to_torch(like=coo)
    assert back.indices().dtype == torch.int64
    assert torch.equal(back.indices(), coo.indices())
    assert torch.equal(back.values(), coo.values())


@given(_random_batched())
@settings(deadline=None, max_examples=50)
def test_hypothesis_transpose_of_transpose_is_identity(data):
    batch_size, n_rows, n_cols, coords, value_dtype = data
    coo = _make_batched_coo(coords, batch_size, n_rows, n_cols, value_dtype)
    descriptor = BatchedCSR.from_torch(coo)

    back = descriptor.transposed.transposed

    assert torch.equal(back.rowptr, descriptor.rowptr)
    assert torch.equal(back.col, descriptor.col)
    assert torch.equal(back.values, descriptor.values)


@given(
    st.integers(1, 3),
    st.integers(1, 4),
    st.integers(1, 4),
    st.sampled_from(VALUE_DTYPES),
    st.sampled_from(INDEX_DTYPES),
    st.data(),
)
@settings(deadline=None, max_examples=50)
def test_hypothesis_round_trip_csr_list_ragged(batch_size, n_rows, n_cols, value_dtype, index_dtype, data):
    all_coords = _all_coords_2d(n_rows, n_cols)
    items = []
    for _ in range(batch_size):
        coords = data.draw(st.lists(st.sampled_from(all_coords), unique=True, max_size=len(all_coords)))
        items.append(_make_2d_csr(coords, n_rows, n_cols, index_dtype, value_dtype))

    descriptor = BatchedCSR.from_torch(items)
    assert descriptor.shape == (batch_size, n_rows, n_cols)
    assert descriptor.index_dtype == index_dtype

    back = descriptor.to_torch(like=items)
    for reconstructed, original in zip(back, items):
        assert torch.equal(reconstructed.crow_indices(), original.crow_indices())
        assert torch.equal(reconstructed.col_indices(), original.col_indices())
        assert torch.equal(reconstructed.values(), original.values())
