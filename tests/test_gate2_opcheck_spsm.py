"""Gate 2 — opcheck for tsgu::spsm (spec/testing.md "Gates & ordering":
"opcheck (all ops x representative inputs)"; spec/commit.md Phase 3
commit 16).

Mirrors tests/test_gate2_opcheck_spmm.py's structure: tsgu::spsm both gets a
real CUDA implementation in this commit AND is wired into a public wrapper
(``sparse_triangular_solve``) with a real ``register_autograd`` (landed in
commit 9) -- opcheck exercises schema, fake-kernel, and autograd-
registration consistency, including ``requires_grad=True`` cases.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate2

_SKIP_REASON = (
    "tsgu::spsm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)

INDEX_DTYPES = (torch.int32, torch.int64)
VALUE_DTYPES = (torch.float32, torch.float64)


def _make_pattern(index_dtype, value_dtype, *, unitriangular=False):
    # B=1, n=3, upper triangular: row0 -> {0,1,2}, row1 -> {1,2}, row2 -> {2}
    # (unitriangular=True drops the diagonal entries: row0->{1,2}, row1->{2}, row2->{})
    if unitriangular:
        rowptr = torch.tensor([0, 2, 3, 3], dtype=index_dtype, device="cuda")
        col = torch.tensor([1, 2, 2], dtype=index_dtype, device="cuda")
    else:
        rowptr = torch.tensor([0, 3, 5, 6], dtype=index_dtype, device="cuda")
        col = torch.tensor([0, 1, 2, 1, 2, 2], dtype=index_dtype, device="cuda")
    vals = torch.rand(col.numel(), dtype=value_dtype, device="cuda") + 1.0  # keep diagonal-ish entries well away from 0
    return rowptr, col, vals


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_opcheck_spsm_contiguous(index_dtype, value_dtype, requires_grad):
    rowptr, col, vals = _make_pattern(index_dtype, value_dtype)
    vals = vals.detach().requires_grad_(requires_grad)
    rhs = torch.randn(1, 3, 4, dtype=value_dtype, device="cuda", requires_grad=requires_grad)
    torch.library.opcheck(torch.ops.tsgu.spsm.default, (vals, rowptr, col, rhs, 1, 3, True, False, False))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_opcheck_spsm_noncontiguous_rhs(index_dtype):
    """testing.md Pillar 2: "strided and non-contiguous dense operands"."""
    rowptr, col, vals = _make_pattern(index_dtype, torch.float32)
    rhs = torch.randn(1, 4, 3, device="cuda").transpose(1, 2)  # (1, 3, 4), non-contiguous
    assert not rhs.is_contiguous()
    torch.library.opcheck(torch.ops.tsgu.spsm.default, (vals, rowptr, col, rhs, 1, 3, True, False, False))


@requires_cuda_backend
@pytest.mark.parametrize("requires_grad", [False, True])
def test_opcheck_spsm_batched(requires_grad):
    rowptr = torch.cat(
        [
            torch.tensor([0, 3, 5, 6], dtype=torch.int64, device="cuda"),
            torch.tensor([9, 11, 12], dtype=torch.int64, device="cuda"),
        ]
    )
    col = torch.tensor([0, 1, 2, 1, 2, 2, 0, 1, 2, 1, 2, 2], dtype=torch.int64, device="cuda")
    vals = (torch.rand(12, dtype=torch.float64, device="cuda") + 1.0).detach().requires_grad_(requires_grad)
    rhs = torch.randn(2, 3, 5, dtype=torch.float64, device="cuda", requires_grad=requires_grad)
    torch.library.opcheck(torch.ops.tsgu.spsm.default, (vals, rowptr, col, rhs, 2, 3, True, False, False))


@requires_cuda_backend
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_opcheck_spsm_upper_lower_transpose(upper, transpose):
    rowptr, col, vals = _make_pattern(torch.int64, torch.float64) if upper else _lower_pattern()
    rhs = torch.randn(1, 3, 2, dtype=torch.float64, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.spsm.default, (vals, rowptr, col, rhs, 1, 3, upper, False, transpose))


def _lower_pattern():
    rowptr = torch.tensor([0, 1, 3, 6], dtype=torch.int64, device="cuda")
    col = torch.tensor([0, 0, 1, 0, 1, 2], dtype=torch.int64, device="cuda")
    vals = torch.rand(6, dtype=torch.float64, device="cuda") + 1.0
    return rowptr, col, vals


@requires_cuda_backend
def test_opcheck_spsm_unitriangular():
    rowptr, col, vals = _make_pattern(torch.int32, torch.float32, unitriangular=True)
    rhs = torch.randn(1, 3, 3, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.spsm.default, (vals, rowptr, col, rhs, 1, 3, True, True, False))


@requires_cuda_backend
def test_opcheck_spsm_p_equals_one():
    rowptr, col, vals = _make_pattern(torch.int32, torch.float32)
    rhs = torch.randn(1, 3, 1, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.spsm.default, (vals, rowptr, col, rhs, 1, 3, True, False, False))


@requires_cuda_backend
def test_opcheck_spsm_empty_pattern_unitriangular():
    # unitriangular=True with an entirely empty pattern is valid (all implicit 1s).
    rowptr = torch.zeros(4, dtype=torch.int64, device="cuda")
    col = torch.zeros(0, dtype=torch.int64, device="cuda")
    vals = torch.zeros(0, device="cuda")
    rhs = torch.randn(1, 3, 2, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.spsm.default, (vals, rowptr, col, rhs, 1, 3, True, True, False))
