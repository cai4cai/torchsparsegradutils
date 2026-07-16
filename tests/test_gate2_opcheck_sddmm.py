"""Gate 2 — opcheck for tsgu::sddmm (spec/testing.md "Gates & ordering":
"opcheck (all ops x representative inputs)"; spec/commit.md Phase 3 commit
14). No wrapper switches to this op in this commit ("nothing routes to it
alone" per the commit's own map.md entry), so it is exercised directly via
``torch.ops.tsgu.sddmm``.

torch.library.opcheck exercises schema, fake-kernel, and autograd-
registration consistency against a real (CUDA) execution (testing.md Pillar
2). ``requires_grad`` is left off throughout: ops/matmul.py's commit-9
docstring notes ``tsgu::sddmm`` gets no ``register_autograd`` in this
commit -- it is only ever invoked as a backward primitive by other ops
(routed in later commits), never differentiated through directly, so
opcheck's autograd-consistency subtests have nothing to check here (and
"requires_grad off is fine" per this commit's own instructions).
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate2

_SKIP_REASON = (
    "tsgu::sddmm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)

INDEX_DTYPES = (torch.int32, torch.int64)
VALUE_DTYPES = (torch.float32, torch.float64)


def _make_pattern(index_dtype):
    # B=1, n=2, m=3: row 0 has 2 entries (local cols 0, 2); row 1 has 1 (col 1).
    rowptr = torch.tensor([0, 2, 3], dtype=index_dtype, device="cuda")
    col = torch.tensor([0, 2, 1], dtype=index_dtype, device="cuda")
    return rowptr, col


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("negate", [False, True])
def test_opcheck_sddmm_contiguous(index_dtype, value_dtype, negate):
    rowptr, col = _make_pattern(index_dtype)
    g = torch.randn(1, 2, 4, dtype=value_dtype, device="cuda")
    mat = torch.randn(1, 3, 4, dtype=value_dtype, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.sddmm.default, (rowptr, col, g, mat, 1, 2, 3, negate))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_opcheck_sddmm_noncontiguous_dense_operands(index_dtype):
    """testing.md Pillar 2: "strided and non-contiguous dense operands" --
    g/mat built via a transpose so neither is contiguous; the launcher's
    host-side ``.contiguous()`` (sddmm.cu) must make this a real, correct
    execution, not a raise, for opcheck's real-vs-fake comparison to pass."""
    rowptr, col = _make_pattern(index_dtype)
    g = torch.randn(1, 4, 2, device="cuda").transpose(1, 2)  # (1, 2, 4), non-contiguous
    mat = torch.randn(1, 4, 3, device="cuda").transpose(1, 2)  # (1, 3, 4), non-contiguous
    assert not g.is_contiguous()
    assert not mat.is_contiguous()
    torch.library.opcheck(torch.ops.tsgu.sddmm.default, (rowptr, col, g, mat, 1, 2, 3, False))


@requires_cuda_backend
@pytest.mark.parametrize("negate", [False, True])
def test_opcheck_sddmm_batched(negate):
    # B=2, n=2: batch 0 row 0 has 1 entry, row 1 has 0; batch 1 row 0 has 0, row 1 has 1.
    rowptr = torch.tensor([0, 1, 1, 1, 2], dtype=torch.int64, device="cuda")
    col = torch.tensor([1, 0], dtype=torch.int64, device="cuda")
    g = torch.randn(2, 2, 5, device="cuda")
    mat = torch.randn(2, 3, 5, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.sddmm.default, (rowptr, col, g, mat, 2, 2, 3, negate))


@requires_cuda_backend
def test_opcheck_sddmm_p_equals_one():
    rowptr, col = _make_pattern(torch.int32)
    g = torch.randn(1, 2, 1, device="cuda")
    mat = torch.randn(1, 3, 1, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.sddmm.default, (rowptr, col, g, mat, 1, 2, 3, False))


@requires_cuda_backend
def test_opcheck_sddmm_empty_pattern():
    rowptr = torch.zeros(3, dtype=torch.int64, device="cuda")  # B=1, n=2, no specified entries at all
    col = torch.zeros(0, dtype=torch.int64, device="cuda")
    g = torch.randn(1, 2, 4, device="cuda")
    mat = torch.randn(1, 3, 4, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.sddmm.default, (rowptr, col, g, mat, 1, 2, 3, False))
