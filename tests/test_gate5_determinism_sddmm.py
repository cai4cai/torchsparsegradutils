"""Gate 5 — determinism + adversarial structure for tsgu::sddmm
(spec/testing.md "Gates & ordering": "property + adversarial + determinism";
spec/commit.md Phase 3 commit 14).

Determinism: sddmm.cu's module comment states the kernel has no atomics --
each warp owns exactly one output element, so there is nothing to race.
kernels.md's open Q1 resolution requires two things for a kernel making that
claim: a run-twice bitwise-equality test, and confirmation it still runs
(and matches) under ``torch.use_deterministic_algorithms(True)`` (no
separate deterministic path needed since the only path already qualifies).

Adversarial (testing.md, seglse-family wording generalised to this op's own
structure): zero-row segments (folded rows with no specified entries --
already exercised implicitly by every ragged/empty-batch-item pattern, but
covered explicitly here) and single-entry rows.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from tests._sddmm_helpers import dense_reference, make_batched_pattern, random_masks
from tests._tolerances import assert_close
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate5

_SKIP_REASON = (
    "tsgu::sddmm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)


def _random_inputs(index_dtype, value_dtype, *, seed=42, B=3, n=6, m=5, p=9, density=0.4):
    gen = torch.Generator().manual_seed(seed)
    masks = random_masks(B, n, m, density=density, generator=gen)
    rowptr, col, B_, n_, m_ = make_batched_pattern(masks, index_dtype, device="cuda")
    g = torch.randn(B_, n_, p, dtype=value_dtype, device="cuda")
    mat = torch.randn(B_, m_, p, dtype=value_dtype, device="cuda")
    return rowptr, col, g, mat, B_, n_, m_


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
def test_determinism_run_twice_bitwise_equal(index_dtype, value_dtype):
    rowptr, col, g, mat, B, n, m = _random_inputs(index_dtype, value_dtype)
    out1 = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B, n, m, False)
    out2 = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B, n, m, False)
    assert torch.equal(out1, out2), "tsgu::sddmm has no atomics (sddmm.cu) -- repeat calls must be bitwise identical"


@requires_cuda_backend
def test_determinism_under_use_deterministic_algorithms():
    rowptr, col, g, mat, B, n, m = _random_inputs(torch.int64, torch.float64, seed=43)
    torch.use_deterministic_algorithms(True)
    try:
        out1 = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B, n, m, False)
        out2 = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B, n, m, False)
        ref = dense_reference(rowptr, col, g, mat, B, n, m, False)
    finally:
        torch.use_deterministic_algorithms(False)
    assert torch.equal(out1, out2)
    assert_close(out1, ref, reduction_length=g.shape[-1])


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
def test_adversarial_zero_row_segments(index_dtype, value_dtype):
    """Some folded rows have zero specified entries -- rowptr[k] ==
    rowptr[k+1] for those rows; find_row (sddmm.cu) must skip over them
    correctly for every later entry's binary search."""
    n, m, p = 5, 4, 6
    # Row 0: empty. Row 1: empty. Row 2: 2 entries. Row 3: empty. Row 4: 1 entry.
    mask = torch.zeros(n, m)
    mask[2, 0] = 1.0
    mask[2, 3] = 1.0
    mask[4, 1] = 1.0
    rowptr, col, B, n_, m_ = make_batched_pattern([mask], index_dtype, device="cuda")

    g = torch.randn(B, n_, p, dtype=value_dtype, device="cuda")
    mat = torch.randn(B, m_, p, dtype=value_dtype, device="cuda")

    ours = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B, n_, m_, False)
    ref = dense_reference(rowptr, col, g, mat, B, n_, m_, False)
    assert_close(ours, ref, reduction_length=p)
    assert ours.shape == (3,)  # exactly the 3 specified entries above


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
def test_adversarial_single_entry_rows(index_dtype, value_dtype):
    """Every row has exactly one specified entry -- the minimal non-empty
    row shape, stress-testing find_row's boundary condition
    (rowptr[row] + 1 == rowptr[row + 1]) for every row."""
    n, m, p = 6, 6, 4
    mask = torch.eye(n, m)  # exactly one entry per row, on the diagonal
    rowptr, col, B, n_, m_ = make_batched_pattern([mask], index_dtype, device="cuda")

    g = torch.randn(B, n_, p, dtype=value_dtype, device="cuda")
    mat = torch.randn(B, m_, p, dtype=value_dtype, device="cuda")

    ours = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B, n_, m_, False)
    ref = dense_reference(rowptr, col, g, mat, B, n_, m_, False)
    assert_close(ours, ref, reduction_length=p)
    assert ours.shape == (n_,)


@requires_cuda_backend
def test_adversarial_all_empty_pattern():
    """Every row of every batch item is empty -- nse_total == 0, the
    kernel's own early-return path (sddmm.cu launcher)."""
    n, m, p = 3, 3, 4
    masks = [torch.zeros(n, m), torch.zeros(n, m)]
    rowptr, col, B_, n_, m_ = make_batched_pattern(masks, torch.int64, device="cuda")
    g = torch.randn(B_, n_, p, device="cuda")
    mat = torch.randn(B_, m_, p, device="cuda")

    out = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B_, n_, m_, False)
    assert out.shape == (0,)
