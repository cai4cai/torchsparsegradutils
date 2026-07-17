"""Gate 2 — opcheck for tsgu::seglse / tsgu::seglse_bwd (spec/testing.md
"Gates & ordering": "opcheck (all ops x representative inputs)";
spec/commit.md Phase 3 commit 12).

test_op_schemas.py's docstring notes opcheck was NOT runnable before a
kernel commit gave these ops a real implementation -- this module is that
follow-up for seglse/seglse_bwd specifically. torch.library.opcheck exercises
schema, fake-kernel, and autograd-registration consistency against a real
(CUDA) execution, per testing.md Pillar 2.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate2

_SKIP_REASON = (
    "tsgu::seglse has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)


def _make_inputs(index_dtype, value_dtype, include_zeros):
    # 2 segments, 3 specified entries total: seg 0 has 2, seg 1 has 1.
    vals = torch.tensor([1.0, -2.0, 0.5], dtype=value_dtype, device="cuda", requires_grad=True)
    rowptr = torch.tensor([0, 2, 3], dtype=index_dtype, device="cuda")
    return vals, rowptr


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("include_zeros", [True, False])
def test_opcheck_seglse(index_dtype, value_dtype, include_zeros):
    vals, rowptr = _make_inputs(index_dtype, value_dtype, include_zeros)
    torch.library.opcheck(torch.ops.tsgu.seglse.default, (vals, rowptr, 1, 2, 4, include_zeros))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
def test_opcheck_seglse_bwd(index_dtype, value_dtype):
    vals, rowptr = _make_inputs(index_dtype, value_dtype, True)
    lse = torch.ops.tsgu.seglse(vals, rowptr, 1, 2, 4, True).detach()
    gout = torch.randn(1, 2, dtype=value_dtype, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.seglse_bwd.default, (vals.detach(), rowptr, lse, gout, 1, 2))


@requires_cuda_backend
def test_seglse_batched_smoke():
    """B > 1: two batch items folded into one (B * n + 1,) rowptr."""
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, device="cuda", requires_grad=True)
    rowptr = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device="cuda")  # B=2, n=2, one entry per segment
    out = torch.ops.tsgu.seglse(vals, rowptr, 2, 2, 3, True)
    assert out.shape == (2, 2)
    out.sum().backward()
    assert vals.grad is not None and vals.grad.shape == vals.shape
