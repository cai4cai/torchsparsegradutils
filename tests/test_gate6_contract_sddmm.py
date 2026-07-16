"""Gate 6 — contract conformance for tsgu::sddmm (spec/testing.md "Gates &
ordering": "contract conformance + workspace assertion"; map.md invariant 7:
"raise, never silently accept"; spec/commit.md Phase 3 commit 14).

Every check below is a defensive ``STD_TORCH_CHECK`` in the launcher
(cuda/csrc/kernels/sddmm/sddmm.cu) -- this module proves each one actually
raises (never silently produces a wrong-shaped/garbage result) and that the
message states the accepted logical shape/dtype alongside the received one
(naming.md §1's error template).
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate6

_SKIP_REASON = (
    "tsgu::sddmm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)


def _valid_inputs():
    rowptr = torch.tensor([0, 2, 3], dtype=torch.int64, device="cuda")  # B=1, n=2
    col = torch.tensor([0, 1, 1], dtype=torch.int64, device="cuda")
    g = torch.randn(1, 2, 4, device="cuda")
    mat = torch.randn(1, 2, 4, device="cuda")
    return rowptr, col, g, mat


@requires_cuda_backend
def test_sanity_valid_call_does_not_raise():
    rowptr, col, g, mat = _valid_inputs()
    torch.ops.tsgu.sddmm(rowptr, col, g, mat, 1, 2, 2, False)


@requires_cuda_backend
def test_rejects_cpu_tensors():
    rowptr, col, g, mat = _valid_inputs()
    rowptr_cpu = rowptr.cpu()
    with pytest.raises(RuntimeError, match=r"CUDA tensors"):
        torch.ops.tsgu.sddmm(rowptr_cpu, col, g, mat, 1, 2, 2, False)


@requires_cuda_backend
def test_rejects_wrong_rowptr_length():
    rowptr, col, g, mat = _valid_inputs()
    bad_rowptr = torch.cat([rowptr, rowptr[-1:]])  # length 4, expected 3
    with pytest.raises(RuntimeError, match=r"rowptr of shape \(B \* n \+ 1,\)"):
        torch.ops.tsgu.sddmm(bad_rowptr, col, g, mat, 1, 2, 2, False)


@requires_cuda_backend
def test_rejects_col_wrong_ndim():
    rowptr, col, g, mat = _valid_inputs()
    bad_col = col.unsqueeze(0)
    with pytest.raises(RuntimeError, match=r"col of shape \(nse_total,\)"):
        torch.ops.tsgu.sddmm(rowptr, bad_col, g, mat, 1, 2, 2, False)


@requires_cuda_backend
def test_rejects_mismatched_index_dtypes():
    rowptr, col, g, mat = _valid_inputs()
    bad_col = col.to(torch.int32)
    with pytest.raises(RuntimeError, match=r"rowptr and col to share one index dtype"):
        torch.ops.tsgu.sddmm(rowptr, bad_col, g, mat, 1, 2, 2, False)


@requires_cuda_backend
def test_rejects_g_wrong_batch():
    rowptr, col, g, mat = _valid_inputs()
    bad_g = torch.randn(2, 2, 4, device="cuda")
    with pytest.raises(RuntimeError, match=r"g of shape \(B, n, p\)"):
        torch.ops.tsgu.sddmm(rowptr, col, bad_g, mat, 1, 2, 2, False)


@requires_cuda_backend
def test_rejects_g_wrong_rows():
    rowptr, col, g, mat = _valid_inputs()
    bad_g = torch.randn(1, 3, 4, device="cuda")
    with pytest.raises(RuntimeError, match=r"g of shape \(B, n, p\)"):
        torch.ops.tsgu.sddmm(rowptr, col, bad_g, mat, 1, 2, 2, False)


@requires_cuda_backend
def test_rejects_mat_wrong_cols():
    rowptr, col, g, mat = _valid_inputs()
    bad_mat = torch.randn(1, 3, 4, device="cuda")
    with pytest.raises(RuntimeError, match=r"mat of shape \(B, m, p\)"):
        torch.ops.tsgu.sddmm(rowptr, col, g, bad_mat, 1, 2, 2, False)


@requires_cuda_backend
def test_rejects_p_mismatch():
    rowptr, col, g, mat = _valid_inputs()
    bad_mat = torch.randn(1, 2, 5, device="cuda")
    with pytest.raises(RuntimeError, match=r"share their trailing dimension p"):
        torch.ops.tsgu.sddmm(rowptr, col, g, bad_mat, 1, 2, 2, False)


@requires_cuda_backend
def test_rejects_value_dtype_mismatch():
    rowptr, col, g, mat = _valid_inputs()
    bad_mat = mat.to(torch.float64)
    with pytest.raises(RuntimeError, match=r"g and mat to share one value dtype"):
        torch.ops.tsgu.sddmm(rowptr, col, g, bad_mat, 1, 2, 2, False)


@requires_cuda_backend
def test_error_message_states_accepted_and_received_shape():
    """naming.md §1 template: "state the accepted logical forms and the
    received shape" -- spot-check one message contains both."""
    rowptr, col, g, mat = _valid_inputs()
    bad_g = torch.randn(1, 3, 4, device="cuda")
    with pytest.raises(RuntimeError) as exc_info:
        torch.ops.tsgu.sddmm(rowptr, col, bad_g, mat, 1, 2, 2, False)
    message = str(exc_info.value)
    assert "(B, n, p)" in message  # accepted logical form
    assert "(1, 3, *)" in message  # received shape
