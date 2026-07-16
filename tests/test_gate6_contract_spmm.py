"""Gate 6 — contract conformance + workspace assertion for tsgu::spmm
(spec/testing.md "Gates & ordering": "contract conformance + workspace
assertion"; map.md invariant 7: "raise, never silently accept";
spec/commit.md Phase 3 commit 15).

Every check below is a defensive ``STD_TORCH_CHECK`` in the launcher
(cuda/csrc/kernels/spmm/spmm.cu) -- this module proves each one actually
raises (never silently produces a wrong-shaped/garbage result) and that the
message states the accepted logical shape/dtype alongside the received one
(naming.md §1's error template). The final test asserts the workspace bound
(testing.md/benchmarks.md: "peak memory minus tensors <= O(nse) bound").
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate6

_SKIP_REASON = (
    "tsgu::spmm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)


def _valid_inputs():
    rowptr = torch.tensor([0, 2, 3], dtype=torch.int64, device="cuda")  # B=1, n=2
    col = torch.tensor([0, 1, 1], dtype=torch.int64, device="cuda")
    vals = torch.randn(3, device="cuda")
    dense = torch.randn(1, 2, 4, device="cuda")
    return vals, rowptr, col, dense


@requires_cuda_backend
def test_sanity_valid_call_does_not_raise():
    vals, rowptr, col, dense = _valid_inputs()
    torch.ops.tsgu.spmm(vals, rowptr, col, dense, 1, 2, 2)


@requires_cuda_backend
def test_rejects_cpu_tensors():
    vals, rowptr, col, dense = _valid_inputs()
    with pytest.raises(RuntimeError, match=r"CUDA tensors"):
        torch.ops.tsgu.spmm(vals, rowptr.cpu(), col, dense, 1, 2, 2)


@requires_cuda_backend
def test_rejects_wrong_rowptr_length():
    vals, rowptr, col, dense = _valid_inputs()
    bad_rowptr = torch.cat([rowptr, rowptr[-1:]])  # length 4, expected 3
    with pytest.raises(RuntimeError, match=r"rowptr of shape \(B \* n \+ 1,\)"):
        torch.ops.tsgu.spmm(vals, bad_rowptr, col, dense, 1, 2, 2)


@requires_cuda_backend
def test_rejects_col_wrong_ndim():
    vals, rowptr, col, dense = _valid_inputs()
    bad_col = col.unsqueeze(0)
    with pytest.raises(RuntimeError, match=r"col of shape \(nse_total,\)"):
        torch.ops.tsgu.spmm(vals, rowptr, bad_col, dense, 1, 2, 2)


@requires_cuda_backend
def test_rejects_vals_length_mismatch():
    vals, rowptr, col, dense = _valid_inputs()
    bad_vals = vals[:-1]
    with pytest.raises(RuntimeError, match=r"vals of shape \(nse_total,\)"):
        torch.ops.tsgu.spmm(bad_vals, rowptr, col, dense, 1, 2, 2)


@requires_cuda_backend
def test_rejects_mismatched_index_dtypes():
    vals, rowptr, col, dense = _valid_inputs()
    bad_col = col.to(torch.int32)
    with pytest.raises(RuntimeError, match=r"rowptr and col to share one index dtype"):
        torch.ops.tsgu.spmm(vals, rowptr, bad_col, dense, 1, 2, 2)


@requires_cuda_backend
def test_rejects_dense_wrong_batch():
    vals, rowptr, col, dense = _valid_inputs()
    bad_dense = torch.randn(2, 2, 4, device="cuda")
    with pytest.raises(RuntimeError, match=r"dense of shape \(B, m, p\)"):
        torch.ops.tsgu.spmm(vals, rowptr, col, bad_dense, 1, 2, 2)


@requires_cuda_backend
def test_rejects_dense_wrong_cols():
    vals, rowptr, col, dense = _valid_inputs()
    bad_dense = torch.randn(1, 3, 4, device="cuda")
    with pytest.raises(RuntimeError, match=r"dense of shape \(B, m, p\)"):
        torch.ops.tsgu.spmm(vals, rowptr, col, bad_dense, 1, 2, 2)


@requires_cuda_backend
def test_rejects_value_dtype_mismatch():
    vals, rowptr, col, dense = _valid_inputs()
    bad_dense = dense.to(torch.float64)
    with pytest.raises(RuntimeError, match=r"vals and dense to share one value dtype"):
        torch.ops.tsgu.spmm(vals, rowptr, col, bad_dense, 1, 2, 2)


@requires_cuda_backend
def test_error_message_states_accepted_and_received_shape():
    """naming.md §1 template: "state the accepted logical forms and the
    received shape" -- spot-check one message contains both."""
    vals, rowptr, col, dense = _valid_inputs()
    bad_dense = torch.randn(2, 2, 4, device="cuda")
    with pytest.raises(RuntimeError) as exc_info:
        torch.ops.tsgu.spmm(vals, rowptr, col, bad_dense, 1, 2, 2)
    message = str(exc_info.value)
    assert "(B, m, p)" in message  # accepted logical form
    assert "(2, 2, *)" in message  # received shape


@requires_cuda_backend
def test_workspace_bound_o_nse():
    """benchmarks.md/testing.md workspace assertion: peak memory minus the
    tensor inputs/outputs stays O(nse_total) / O(n_rows) -- no hidden dense
    (n * m) materialisation anywhere in the launcher (kernels.md shared
    rule: "no dense materialisation, ever")."""
    torch.manual_seed(0)
    n, m, p = 4000, 4000, 32
    nnz = 8000
    row = torch.randint(0, n, (nnz,), device="cuda")
    col_idx = torch.randint(0, m, (nnz,), device="cuda")
    order = torch.argsort(row)
    row = row[order]
    col_idx = col_idx[order]
    counts = torch.bincount(row, minlength=n)
    rowptr = torch.zeros(n + 1, dtype=torch.int64, device="cuda")
    rowptr[1:] = torch.cumsum(counts, dim=0)
    vals = torch.randn(nnz, device="cuda")
    dense = torch.randn(1, m, p, device="cuda")

    io_bytes = (
        vals.numel() * vals.element_size()
        + rowptr.numel() * rowptr.element_size()
        + col_idx.numel() * col_idx.element_size()
        + dense.numel() * dense.element_size()
        + n * p * dense.element_size()  # output
    )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = torch.ops.tsgu.spmm(vals, rowptr, col_idx, dense, 1, n, m)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    workspace = max(0, peak - io_bytes)
    # O(nse) bound with a generous floor: the point of this assertion is
    # "no hidden dense (n * m) materialisation" (kernels.md shared rule), not
    # a tight perf bound -- CUDA's caching allocator rounds every allocation
    # up to its own block granularity (~2MB-class chunks), so even a
    # correctly O(nse) launcher legitimately shows several MB of "workspace"
    # here that has nothing to do with nse. A 16MB floor absorbs that
    # allocator overhead while staying far below the dense counterfactual
    # this matrix would materialise if the launcher were wrong (n * m *
    # element_size = 4000 * 4000 * 4 bytes = 64MB here) -- a real dense
    # materialisation bug still fails this assertion loudly.
    nse_bound = 8 * (nnz * (vals.element_size() + col_idx.element_size()) + rowptr.numel() * rowptr.element_size())
    bound = max(16 * 1024 * 1024, nse_bound)
    dense_counterfactual = n * m * vals.element_size()
    assert bound < dense_counterfactual, "test bug: bound should stay well below the dense (n*m) counterfactual"
    assert workspace <= bound, f"workspace {workspace} bytes exceeds O(nse) bound {bound} bytes"
    assert out.shape == (1, n, p)
