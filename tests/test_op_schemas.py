"""Schema / fake-kernel / value-independence tests for the nine ``tsgu::``
ops (spec/commit.md Phase 1 #9, spec/map.md "Kernel routing").

No CUDA/CPU implementation is registered for any of these ops yet — each op's
Python body raises ``NotImplementedError`` if actually executed. That means
``torch.library.opcheck`` (which *executes* the op to cross-check schema,
fake-kernel, and autograd-registration consistency, spec/testing.md gate 2)
is not runnable here: it would immediately hit the ``NotImplementedError``.
Full opcheck is deferred to each op's own kernel commit (spec/commit.md
Phase 3), per the commit-9 spec ("Full opcheck is NOT runnable yet ... defer
to kernel commits").

This module instead exercises the parts of legitimacy that *are* checkable
against schema + fake kernel alone:

(a) the op is registered under ``torch.ops.tsgu`` with a schema that parses;
(b) the registered fake (meta) kernel produces correct output shapes/dtypes
    under ``FakeTensorMode``, for both unbatched (``B = 1``) and batched
    (``B = 3``) shapes, crossed with int32/int64 index dtypes
    (spec/testing.md's ``INDEX_DTYPES`` fixture intent);
(c) fake outputs are value-independent (architecture.md §2): two fake calls
    with identical shapes but independently-constructed (and, outside
    ``FakeTensorMode``, differently-valued) inputs produce identical output
    shapes/dtypes — guarding against a fake kernel that accidentally reads
    tensor contents rather than deriving shape from index lengths and shape
    ints alone.
"""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import torchsparsegradutils  # noqa: F401  (import side-effect: registers all tsgu:: ops)

INDEX_DTYPES = (torch.int32, torch.int64)
VALUE_DTYPES = (torch.float32, torch.float64)

ALL_TSGU_OPS = (
    "spmm",
    "sddmm",
    "spsm",
    "seglse",
    "seglse_bwd",
    "seglse_bidir",
    "seglse_bidir_bwd",
    "coo2csr",
    "grouped_gemm",
)


def test_all_nine_ops_registered():
    """(a) every op from naming.md §2's list exists under torch.ops.tsgu with
    a schema that parses (querying ._schema itself exercises schema
    parsing)."""
    for name in ALL_TSGU_OPS:
        assert hasattr(torch.ops.tsgu, name), f"torch.ops.tsgu.{name} is not registered"
        op = getattr(torch.ops.tsgu, name)
        schema = op.default._schema
        assert schema is not None
        assert schema.name == f"tsgu::{name}"


def _toy_pattern(B: int, n: int, m: int, index_dtype: torch.dtype, value_dtype: torch.dtype):
    """A tiny, valid folded-CSR pattern (naming.md §2): row 0 of every batch
    item gets ``min(2, m)`` entries (local columns ``0, 1, ...``); every
    other row of every batch item is empty. Returns
    ``(vals, rowptr, col, nse_total)``.
    """
    nse_per_row0 = min(2, m)
    row0_counts = [nse_per_row0] + [0] * (n - 1)
    rowptr_list = [0]
    for _b in range(B):
        for c in row0_counts:
            rowptr_list.append(rowptr_list[-1] + c)
    nse_total = rowptr_list[-1]
    rowptr = torch.tensor(rowptr_list, dtype=index_dtype)
    col = torch.tensor(list(range(nse_per_row0)) * B, dtype=index_dtype)
    vals = torch.arange(1, nse_total + 1, dtype=value_dtype)
    return vals, rowptr, col, nse_total


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_spmm_fake(B, index_dtype, value_dtype):
    n, m, p = 3, 4, 5
    with FakeTensorMode():
        vals, rowptr, col, _ = _toy_pattern(B, n, m, index_dtype, value_dtype)
        dense = torch.randn(B, m, p, dtype=value_dtype)
        out = torch.ops.tsgu.spmm(vals, rowptr, col, dense, B, n, m)
        assert out.shape == (B, n, p)
        assert out.dtype == value_dtype

        # (c) value-independence: a second, independently-built fake input of
        # the same shapes gives the same output shape/dtype.
        vals2, rowptr2, col2, _ = _toy_pattern(B, n, m, index_dtype, value_dtype)
        dense2 = torch.randn(B, m, p, dtype=value_dtype) * 37 + 5
        out2 = torch.ops.tsgu.spmm(vals2, rowptr2, col2, dense2, B, n, m)
        assert out2.shape == out.shape
        assert out2.dtype == out.dtype


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("negate", [False, True])
def test_sddmm_fake(B, index_dtype, value_dtype, negate):
    n, m, p = 3, 4, 5
    with FakeTensorMode():
        _vals, rowptr, col, nse_total = _toy_pattern(B, n, m, index_dtype, value_dtype)
        g = torch.randn(B, n, p, dtype=value_dtype)
        mat = torch.randn(B, m, p, dtype=value_dtype)
        out = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B, n, m, negate)
        assert out.shape == (nse_total,)
        assert out.dtype == value_dtype

        g2 = torch.randn(B, n, p, dtype=value_dtype) * 11 - 3
        mat2 = torch.randn(B, m, p, dtype=value_dtype) * 11 - 3
        out2 = torch.ops.tsgu.sddmm(rowptr, col, g2, mat2, B, n, m, negate)
        assert out2.shape == out.shape
        assert out2.dtype == out.dtype


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("upper,unitriangular,transpose", [(True, False, False), (False, True, True)])
def test_spsm_fake(B, index_dtype, value_dtype, upper, unitriangular, transpose):
    n, p = 3, 5
    with FakeTensorMode():
        vals, rowptr, col, _ = _toy_pattern(B, n, n, index_dtype, value_dtype)
        rhs = torch.randn(B, n, p, dtype=value_dtype)
        out = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n, upper, unitriangular, transpose)
        assert out.shape == (B, n, p)
        assert out.dtype == value_dtype

        rhs2 = torch.randn(B, n, p, dtype=value_dtype) * 4 + 1
        out2 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs2, B, n, upper, unitriangular, transpose)
        assert out2.shape == out.shape
        assert out2.dtype == out.dtype


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("include_zeros", [False, True])
def test_seglse_fake(B, index_dtype, value_dtype, include_zeros):
    n, m = 3, 4
    with FakeTensorMode():
        vals, rowptr, _col, _ = _toy_pattern(B, n, m, index_dtype, value_dtype)
        out = torch.ops.tsgu.seglse(vals, rowptr, B, n, m, include_zeros)
        assert out.shape == (B, n)
        assert out.dtype == value_dtype

        vals2, rowptr2, _col2, _ = _toy_pattern(B, n, m, index_dtype, value_dtype)
        out2 = torch.ops.tsgu.seglse(vals2 * 9 + 2, rowptr2, B, n, m, include_zeros)
        assert out2.shape == out.shape
        assert out2.dtype == out.dtype


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_seglse_bwd_fake(B, index_dtype, value_dtype):
    n, m = 3, 4
    with FakeTensorMode():
        vals, rowptr, _col, nse_total = _toy_pattern(B, n, m, index_dtype, value_dtype)
        lse = torch.randn(B, n, dtype=value_dtype)
        gout = torch.randn(B, n, dtype=value_dtype)
        out = torch.ops.tsgu.seglse_bwd(vals, rowptr, lse, gout, B, n)
        assert out.shape == (nse_total,)
        assert out.dtype == value_dtype

        lse2 = torch.randn(B, n, dtype=value_dtype) * 5
        gout2 = torch.randn(B, n, dtype=value_dtype) * 5
        out2 = torch.ops.tsgu.seglse_bwd(vals, rowptr, lse2, gout2, B, n)
        assert out2.shape == out.shape
        assert out2.dtype == out.dtype


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("include_zeros", [False, True])
def test_seglse_bidir_fake(B, index_dtype, value_dtype, include_zeros):
    n, m = 3, 4
    G = max(n, m)
    with FakeTensorMode():
        vals, rowptr, col, _ = _toy_pattern(B, n, m, index_dtype, value_dtype)
        out = torch.ops.tsgu.seglse_bidir(vals, rowptr, col, B, n, m, include_zeros)
        assert out.shape == (2, B, G)
        assert out.dtype == value_dtype

        vals2, rowptr2, col2, _ = _toy_pattern(B, n, m, index_dtype, value_dtype)
        out2 = torch.ops.tsgu.seglse_bidir(vals2 * 3 - 1, rowptr2, col2, B, n, m, include_zeros)
        assert out2.shape == out.shape
        assert out2.dtype == out.dtype


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_seglse_bidir_bwd_fake(B, index_dtype, value_dtype):
    n, m = 3, 4
    G = max(n, m)
    with FakeTensorMode():
        vals, rowptr, col, nse_total = _toy_pattern(B, n, m, index_dtype, value_dtype)
        padded = torch.randn(2, B, G, dtype=value_dtype)
        gout = torch.randn(2, B, G, dtype=value_dtype)
        out = torch.ops.tsgu.seglse_bidir_bwd(vals, rowptr, col, padded, gout, B, n, m)
        assert out.shape == (nse_total,)
        assert out.dtype == value_dtype

        padded2 = torch.randn(2, B, G, dtype=value_dtype) * 6
        gout2 = torch.randn(2, B, G, dtype=value_dtype) * 6
        out2 = torch.ops.tsgu.seglse_bidir_bwd(vals, rowptr, col, padded2, gout2, B, n, m)
        assert out2.shape == out.shape
        assert out2.dtype == out.dtype


@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_coo2csr_fake(B, index_dtype):
    n = 3
    with FakeTensorMode():
        nse_total = 2 * B
        batch = torch.arange(B, dtype=index_dtype).repeat_interleave(2)
        row = torch.zeros(nse_total, dtype=index_dtype)
        col = torch.tensor([0, 1] * B, dtype=index_dtype)
        rowptr, col_sorted, perm = torch.ops.tsgu.coo2csr(batch, row, col, B, n)
        assert rowptr.shape == (B * n + 1,)
        assert col_sorted.shape == (nse_total,)
        assert perm.shape == (nse_total,)
        assert rowptr.dtype == index_dtype
        assert col_sorted.dtype == index_dtype
        assert perm.dtype == index_dtype

        batch2 = torch.arange(B, dtype=index_dtype).repeat_interleave(2)
        row2 = torch.ones(nse_total, dtype=index_dtype) if n > 1 else row
        col2 = torch.tensor([1, 0] * B, dtype=index_dtype)
        rowptr2, col_sorted2, perm2 = torch.ops.tsgu.coo2csr(batch2, row2, col2, B, n)
        assert rowptr2.shape == rowptr.shape
        assert col_sorted2.shape == col_sorted.shape
        assert perm2.shape == perm.shape


@pytest.mark.parametrize("num_groups", [1, 3])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("reduce", [False, True])
def test_grouped_gemm_fake(num_groups, index_dtype, value_dtype, reduce):
    N, D1, D2 = 7, 4, 6
    with FakeTensorMode():
        a = torch.randn(N, D1, dtype=value_dtype)
        idx = (torch.arange(N) % num_groups).to(index_dtype)
        if reduce:
            b = torch.randn(N, D2, dtype=value_dtype)
        else:
            b = torch.randn(num_groups, D1, D2, dtype=value_dtype)
        out = torch.ops.tsgu.grouped_gemm(a, b, idx, num_groups, reduce)
        expected_shape = (num_groups, D1, D2) if reduce else (N, D2)
        assert out.shape == expected_shape
        assert out.dtype == value_dtype

        a2 = torch.randn(N, D1, dtype=value_dtype) * 8 + 3
        b2 = torch.randn(*b.shape, dtype=value_dtype) * 8 + 3
        out2 = torch.ops.tsgu.grouped_gemm(a2, b2, idx, num_groups, reduce)
        assert out2.shape == out.shape
        assert out2.dtype == out.dtype
