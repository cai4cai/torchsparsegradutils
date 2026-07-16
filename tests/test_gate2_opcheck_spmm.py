"""Gate 2 — opcheck for tsgu::spmm (spec/testing.md "Gates & ordering":
"opcheck (all ops x representative inputs)"; spec/commit.md Phase 3 commit
15).

Unlike tsgu::sddmm (commit 14, no wrapper yet), tsgu::spmm both gets a real
CUDA implementation in this commit AND is wired into a public wrapper
(``sparse_mm``) with a real ``register_autograd`` (torchsparsegradutils/
ops/matmul.py, landed already in commit 9) -- so opcheck here exercises
schema, fake-kernel, AND autograd-registration consistency for real,
including ``requires_grad=True`` cases (testing.md Pillar 2: "requires_grad
on/off").
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate2

_SKIP_REASON = (
    "tsgu::spmm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)

INDEX_DTYPES = (torch.int32, torch.int64)
VALUE_DTYPES = (torch.float32, torch.float64)


def _make_pattern(index_dtype, value_dtype):
    # B=1, n=2, m=3: row 0 has 2 entries (local cols 0, 2); row 1 has 1 (col 1).
    rowptr = torch.tensor([0, 2, 3], dtype=index_dtype, device="cuda")
    col = torch.tensor([0, 2, 1], dtype=index_dtype, device="cuda")
    vals = torch.randn(3, dtype=value_dtype, device="cuda")
    return rowptr, col, vals


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_opcheck_spmm_contiguous(index_dtype, value_dtype, requires_grad):
    rowptr, col, vals = _make_pattern(index_dtype, value_dtype)
    vals = vals.requires_grad_(requires_grad)
    dense = torch.randn(1, 3, 4, dtype=value_dtype, device="cuda", requires_grad=requires_grad)
    torch.library.opcheck(torch.ops.tsgu.spmm.default, (vals, rowptr, col, dense, 1, 2, 3))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_opcheck_spmm_noncontiguous_dense_operand(index_dtype):
    """testing.md Pillar 2: "strided and non-contiguous dense operands" --
    dense built via a transpose so it is not contiguous; the launcher's
    host-side ``.contiguous()`` (spmm.cu) must make this a real, correct
    execution for opcheck's real-vs-fake comparison to pass."""
    rowptr, col, vals = _make_pattern(index_dtype, torch.float32)
    dense = torch.randn(1, 4, 3, device="cuda").transpose(1, 2)  # (1, 3, 4), non-contiguous
    assert not dense.is_contiguous()
    torch.library.opcheck(torch.ops.tsgu.spmm.default, (vals, rowptr, col, dense, 1, 2, 3))


@requires_cuda_backend
@pytest.mark.parametrize("requires_grad", [False, True])
def test_opcheck_spmm_batched(requires_grad):
    # B=2, n=2: batch 0 row 0 has 1 entry, row 1 has 0; batch 1 row 0 has 0, row 1 has 1.
    rowptr = torch.tensor([0, 1, 1, 1, 2], dtype=torch.int64, device="cuda")
    col = torch.tensor([1, 0], dtype=torch.int64, device="cuda")
    vals = torch.randn(2, dtype=torch.float64, device="cuda", requires_grad=requires_grad)
    dense = torch.randn(2, 3, 5, dtype=torch.float64, device="cuda", requires_grad=requires_grad)
    torch.library.opcheck(torch.ops.tsgu.spmm.default, (vals, rowptr, col, dense, 2, 2, 3))


@requires_cuda_backend
def test_opcheck_spmm_p_equals_one():
    """SpMV shape (p=1, kernels.md: "spmv = spmm with p = 1, no separate op")."""
    rowptr, col, vals = _make_pattern(torch.int32, torch.float32)
    dense = torch.randn(1, 3, 1, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.spmm.default, (vals, rowptr, col, dense, 1, 2, 3))


@requires_cuda_backend
def test_opcheck_spmm_p_greater_than_32():
    """Wide-p path (spmm.cu spmm_wide_kernel, p >= kWarpSize)."""
    rowptr, col, vals = _make_pattern(torch.int64, torch.float64)
    dense = torch.randn(1, 3, 65, dtype=torch.float64, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.spmm.default, (vals, rowptr, col, dense, 1, 2, 3))


@requires_cuda_backend
def test_opcheck_spmm_empty_pattern():
    rowptr = torch.zeros(3, dtype=torch.int64, device="cuda")  # B=1, n=2, no specified entries at all
    col = torch.zeros(0, dtype=torch.int64, device="cuda")
    vals = torch.zeros(0, device="cuda")
    dense = torch.randn(1, 3, 4, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.spmm.default, (vals, rowptr, col, dense, 1, 2, 3))
