"""Gate 6 — contract conformance + workspace assertion + plan-cache tests
for tsgu::spsm (spec/testing.md "Gates & ordering": "contract conformance +
workspace assertion"; map.md invariant 7: "raise, never silently accept";
spec/commit.md Phase 3 commit 16).

Every shape/dtype check below is a defensive ``STD_TORCH_CHECK`` in the
launcher (cuda/csrc/kernels/spsm/spsm.cu) or the analysis
(cuda/csrc/kernels/spsm/plan.cpp) -- this module proves each one actually
raises and that the message states the accepted logical shape/dtype
alongside the received one (naming.md §1's error template). Plan-cache
tests use ``tsgu::_spsm_plan_cache_stats`` (torchsparsegradutils/ops/
triangular_solve.py) to assert "same BatchedCSR solved twice -> analysis
computed once" directly (this commit's own T5 instruction), not via timing.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate6

_SKIP_REASON = (
    "tsgu::spsm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)


def _valid_inputs():
    # B=1, n=2, upper: row0={0,1}, row1={1}.
    rowptr = torch.tensor([0, 2, 3], dtype=torch.int64, device="cuda")
    col = torch.tensor([0, 1, 1], dtype=torch.int64, device="cuda")
    vals = torch.tensor([2.0, 0.5, 3.0], device="cuda")
    rhs = torch.randn(1, 2, 4, device="cuda")
    return vals, rowptr, col, rhs


@requires_cuda_backend
def test_sanity_valid_call_does_not_raise():
    vals, rowptr, col, rhs = _valid_inputs()
    torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_rejects_cpu_tensors():
    vals, rowptr, col, rhs = _valid_inputs()
    with pytest.raises(RuntimeError, match=r"CUDA tensors"):
        torch.ops.tsgu.spsm(vals, rowptr.cpu(), col, rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_rejects_wrong_rowptr_length():
    vals, rowptr, col, rhs = _valid_inputs()
    bad_rowptr = torch.cat([rowptr, rowptr[-1:]])  # length 4, expected 3
    with pytest.raises(RuntimeError, match=r"rowptr of shape \(B \* n \+ 1,\)"):
        torch.ops.tsgu.spsm(vals, bad_rowptr, col, rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_rejects_col_wrong_ndim():
    vals, rowptr, col, rhs = _valid_inputs()
    bad_col = col.unsqueeze(0)
    with pytest.raises(RuntimeError, match=r"col of shape \(nse_total,\)"):
        torch.ops.tsgu.spsm(vals, rowptr, bad_col, rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_rejects_vals_length_mismatch():
    vals, rowptr, col, rhs = _valid_inputs()
    bad_vals = vals[:-1]
    with pytest.raises(RuntimeError, match=r"vals of shape \(nse_total,\)"):
        torch.ops.tsgu.spsm(bad_vals, rowptr, col, rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_rejects_mismatched_index_dtypes():
    vals, rowptr, col, rhs = _valid_inputs()
    bad_col = col.to(torch.int32)
    with pytest.raises(RuntimeError, match=r"rowptr and col to share one index dtype"):
        torch.ops.tsgu.spsm(vals, rowptr, bad_col, rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_rejects_rhs_wrong_batch():
    vals, rowptr, col, rhs = _valid_inputs()
    bad_rhs = torch.randn(2, 2, 4, device="cuda")
    with pytest.raises(RuntimeError, match=r"rhs of shape \(B, n, p\)"):
        torch.ops.tsgu.spsm(vals, rowptr, col, bad_rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_rejects_rhs_wrong_rows():
    vals, rowptr, col, rhs = _valid_inputs()
    bad_rhs = torch.randn(1, 3, 4, device="cuda")
    with pytest.raises(RuntimeError, match=r"rhs of shape \(B, n, p\)"):
        torch.ops.tsgu.spsm(vals, rowptr, col, bad_rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_rejects_value_dtype_mismatch():
    vals, rowptr, col, rhs = _valid_inputs()
    bad_rhs = rhs.to(torch.float64)
    with pytest.raises(RuntimeError, match=r"vals and rhs to share one value dtype"):
        torch.ops.tsgu.spsm(vals, rowptr, col, bad_rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_error_message_states_accepted_and_received_shape():
    """naming.md §1 template: "state the accepted logical forms and the
    received shape" -- spot-check one message contains both."""
    vals, rowptr, col, rhs = _valid_inputs()
    bad_rhs = torch.randn(2, 2, 4, device="cuda")
    with pytest.raises(RuntimeError) as exc_info:
        torch.ops.tsgu.spsm(vals, rowptr, col, bad_rhs, 1, 2, True, False, False)
    message = str(exc_info.value)
    assert "(B, n, p)" in message  # accepted logical form
    assert "(2, 2, *)" in message  # received shape


@requires_cuda_backend
def test_rejects_missing_diagonal_on_nonunit_solve():
    """map.md invariant 7 / this commit's own contract: "for non-unit
    diagonal the diagonal entry must exist in the pattern -- if missing,
    that's singular input: raise"."""
    rowptr = torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda")  # row0->{1}, row1->{} (no diagonals at all)
    col = torch.tensor([1, 0], dtype=torch.int64, device="cuda")
    vals = torch.tensor([1.0, 1.0], device="cuda")
    rhs = torch.randn(1, 2, 2, device="cuda")
    with pytest.raises(RuntimeError, match=r"explicit diagonal entry in every row"):
        torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_rejects_explicit_diagonal_when_unitriangular():
    """The inverse contract: unitriangular=True requires NO explicit
    diagonal entries (map.md docstring: "the stored matrix must be strictly
    triangular")."""
    rowptr = torch.tensor([0, 2, 3], dtype=torch.int64, device="cuda")  # row0->{0,1} (has diagonal!), row1->{1}
    col = torch.tensor([0, 1, 1], dtype=torch.int64, device="cuda")
    vals = torch.tensor([2.0, 0.5, 3.0], device="cuda")
    rhs = torch.randn(1, 2, 2, device="cuda")
    with pytest.raises(RuntimeError, match=r"unitriangular=True"):
        torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, 2, True, True, False)


@requires_cuda_backend
def test_rejects_non_triangular_pattern():
    """A pattern with an off-diagonal entry on the WRONG side of the
    diagonal for the claimed `upper` flag -- this commit's own analysis-time
    contract check (plan.cpp): raise rather than silently reading an
    unwritten level."""
    # Claim upper=True but store a genuinely lower-triangular entry (1, 0).
    rowptr = torch.tensor([0, 1, 3], dtype=torch.int64, device="cuda")  # row0->{0}, row1->{0,1}
    col = torch.tensor([0, 0, 1], dtype=torch.int64, device="cuda")
    vals = torch.tensor([1.0, 1.0, 1.0], device="cuda")
    rhs = torch.randn(1, 2, 2, device="cuda")
    with pytest.raises(RuntimeError, match=r"genuinely upper-triangular"):
        torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, 2, True, False, False)


@requires_cuda_backend
def test_workspace_bound_o_nse():
    """benchmarks.md/testing.md workspace assertion: peak memory minus the
    tensor inputs/outputs stays O(nse_total)/O(n_rows) -- no hidden dense
    (n * n) materialisation (kernels.md shared rule)."""
    torch.manual_seed(0)
    n, p, bw = 4000, 8, 6
    rows, cols = [], []
    for r in range(n):
        lo = max(0, r - bw)
        for c in range(lo, r + 1):
            rows.append(r)
            cols.append(c)
    row_t = torch.tensor(rows, dtype=torch.int64)
    col_t = torch.tensor(cols, dtype=torch.int64)
    counts = torch.bincount(row_t, minlength=n)
    rowptr = torch.zeros(n + 1, dtype=torch.int64)
    rowptr[1:] = torch.cumsum(counts, dim=0)
    vals = torch.rand(len(cols)) * 0.1
    is_diag = row_t == col_t
    vals[is_diag] = vals[is_diag].abs() + 4.0
    rowptr, col_t, vals = rowptr.cuda(), col_t.cuda(), vals.cuda()
    rhs = torch.randn(1, n, p, device="cuda")

    io_bytes = (
        vals.numel() * vals.element_size()
        + rowptr.numel() * rowptr.element_size()
        + col_t.numel() * col_t.element_size()
        + rhs.numel() * rhs.element_size()
        + n * p * rhs.element_size()  # output
    )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = torch.ops.tsgu.spsm(vals, rowptr, col_t, rhs, 1, n, False, False, False)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    workspace = max(0, peak - io_bytes)
    # Same generous-floor reasoning as test_gate6_contract_spmm.py's
    # identical assertion: the point is "no hidden dense (n*n)
    # materialisation", not a tight perf bound; the plan itself (eff_ptr/
    # eff_dep/eff_val_idx/diag_val_idx/row_order, all O(nse)/O(n)) is real,
    # legitimate workspace that must be counted, not exempted.
    nse_bound = 8 * (
        len(cols) * (vals.element_size() + col_t.element_size() + 8 * 3)  # plan's own int64 arrays, generously counted
        + rowptr.numel() * rowptr.element_size()
    )
    bound = max(16 * 1024 * 1024, nse_bound)
    dense_counterfactual = n * n * vals.element_size()
    assert bound < dense_counterfactual, "test bug: bound should stay well below the dense (n*n) counterfactual"
    assert workspace <= bound, f"workspace {workspace} bytes exceeds O(nse) bound {bound} bytes"
    assert out.shape == (1, n, p)


# ---------------------------------------------------------------------------
# Plan-cache tests (this commit's own T5 instruction).
# ---------------------------------------------------------------------------


def _plan_cache_stats():
    stats = torch.ops.tsgu._spsm_plan_cache_stats(torch.zeros(1, device="cuda"))
    return int(stats[0].item()), int(stats[1].item())


@requires_cuda_backend
def test_plan_cache_same_batchedcsr_builds_once():
    """spec/commit.md Phase 3 commit 16 T5: "same BatchedCSR solved twice ->
    analysis computed once (assert via the lazy member's identity/a
    counter)". Uses fresh, uniquely-shaped tensors (n=997, an unusual prime
    size) so this test's own cache entries can't collide with any other
    test's leftover pattern in the bounded LRU."""
    n = 997
    rowptr = torch.arange(n + 1, dtype=torch.int64, device="cuda")  # diagonal-only, unitriangular=False
    col = torch.arange(n, dtype=torch.int64, device="cuda")
    vals = torch.rand(n, device="cuda") + 1.0
    rhs = torch.randn(1, n, 3, device="cuda")

    builds_before, _ = _plan_cache_stats()
    torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, n, True, False, False)
    builds_after_first, hits_after_first = _plan_cache_stats()
    torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, n, True, False, False)
    builds_after_second, hits_after_second = _plan_cache_stats()

    assert builds_after_first == builds_before + 1, "first call on a fresh pattern must build exactly one plan"
    assert builds_after_second == builds_after_first, "second call on the SAME tensors must not rebuild"
    assert hits_after_second == hits_after_first + 1, "second call must register as a cache hit"


@requires_cuda_backend
def test_plan_cache_different_rhs_still_reuses_plan():
    """ "different rhs reuse" (this commit's own T5 instruction): the plan
    is keyed on the PATTERN (rowptr/col), not on rhs -- a different rhs on
    the same pattern must still hit the cache."""
    n = 991  # a different unusual prime -- avoid colliding with the test above
    rowptr = torch.arange(n + 1, dtype=torch.int64, device="cuda")
    col = torch.arange(n, dtype=torch.int64, device="cuda")
    vals = torch.rand(n, device="cuda") + 1.0

    torch.ops.tsgu.spsm(vals, rowptr, col, torch.randn(1, n, 2, device="cuda"), 1, n, True, False, False)
    builds_before, _ = _plan_cache_stats()
    torch.ops.tsgu.spsm(vals, rowptr, col, torch.randn(1, n, 9, device="cuda"), 1, n, True, False, False)
    builds_after, hits_after = _plan_cache_stats()

    assert builds_after == builds_before, "a new rhs on the same pattern must not trigger a rebuild"


@requires_cuda_backend
def test_plan_cache_transposed_plan_cached_separately():
    """ "transposed plan cached separately" (this commit's own T5
    instruction): transpose=True and transpose=False on the SAME rowptr/col
    are two distinct cache entries (plan.h: transpose is part of the key)."""
    n = 983  # another unusual prime
    rowptr = torch.arange(n + 1, dtype=torch.int64, device="cuda")
    col = torch.arange(n, dtype=torch.int64, device="cuda")
    vals = torch.rand(n, device="cuda") + 1.0
    rhs = torch.randn(1, n, 2, device="cuda")

    builds_before, _ = _plan_cache_stats()
    torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, n, True, False, False)
    builds_after_direct, _ = _plan_cache_stats()
    torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, n, True, False, True)
    builds_after_transposed, _ = _plan_cache_stats()
    torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, n, True, False, False)
    builds_after_direct_again, hits_after = _plan_cache_stats()

    assert builds_after_direct == builds_before + 1
    assert builds_after_transposed == builds_after_direct + 1, "transpose=True must build its OWN plan"
    assert builds_after_direct_again == builds_after_transposed, (
        "repeating transpose=False must hit its own cached plan"
    )
