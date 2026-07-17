"""Gate 2 — opcheck for tsgu::seglse_bidir / tsgu::seglse_bidir_bwd (spec/testing.md
"Gates & ordering": "opcheck (all ops x representative inputs)";
spec/commit.md Phase 3 commit 13). Mirrors test_gate2_opcheck_seglse.py's
structure for the fused bidirectional op pair.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate2

_SKIP_REASON = (
    "tsgu::seglse_bidir has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)


def _make_inputs(index_dtype, value_dtype):
    # 2 rows, 3 specified entries total: row 0 has 2 (cols 0, 1), row 1 has 1 (col 1).
    # n_cols = 2.
    vals = torch.tensor([1.0, -2.0, 0.5], dtype=value_dtype, device="cuda", requires_grad=True)
    rowptr = torch.tensor([0, 2, 3], dtype=index_dtype, device="cuda")
    col = torch.tensor([0, 1, 1], dtype=index_dtype, device="cuda")
    return vals, rowptr, col


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("include_zeros", [True, False])
def test_opcheck_seglse_bidir(index_dtype, value_dtype, include_zeros):
    vals, rowptr, col = _make_inputs(index_dtype, value_dtype)
    torch.library.opcheck(torch.ops.tsgu.seglse_bidir.default, (vals, rowptr, col, 1, 2, 2, include_zeros))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
def test_opcheck_seglse_bidir_bwd(index_dtype, value_dtype):
    vals, rowptr, col = _make_inputs(index_dtype, value_dtype)
    padded = torch.ops.tsgu.seglse_bidir(vals, rowptr, col, 1, 2, 2, True).detach()
    gout = torch.randn_like(padded)
    torch.library.opcheck(torch.ops.tsgu.seglse_bidir_bwd.default, (vals.detach(), rowptr, col, padded, gout, 1, 2, 2))


@requires_cuda_backend
def test_seglse_bidir_batched_smoke():
    """B > 1: two batch items folded into one (B * n + 1,) rowptr."""
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device="cuda", requires_grad=True)
    rowptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cuda")  # B=2, n=2, one entry per segment
    col = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device="cuda")
    out = torch.ops.tsgu.seglse_bidir(vals, rowptr, col, 2, 2, 2, True)
    assert out.shape == (2, 2, 2)
    out.sum().backward()
    assert vals.grad is not None and vals.grad.shape == vals.shape
