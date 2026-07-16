"""Gate 5 — determinism + adversarial structure for tsgu::spmm
(spec/testing.md "Gates & ordering": "property + adversarial + determinism";
spec/commit.md Phase 3 commit 15).

Determinism: spmm.cu's module comment states neither kernel path
(spmm_wide_kernel, spmm_narrow_kernel) uses atomics -- every folded row is
owned by exactly one warp, so there is nothing to race. kernels.md's open
Q1 resolution requires two things for a kernel making that claim: a
run-twice bitwise-equality test, and confirmation it still runs (and
matches) under ``torch.use_deterministic_algorithms(True)`` (no separate
deterministic path needed since the only path already qualifies).

Adversarial (testing.md): empty rows (folded rows with zero specified
entries -- must still write zero, not skip, per the "out is dense" module
comment in spmm.cu), an empty batch item, and p=0 (zero-size p).
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from tests._spmm_helpers import dense_reference, make_batched_csr, random_masks
from tests._tolerances import assert_close
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate5

_SKIP_REASON = (
    "tsgu::spmm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)


def _random_inputs(index_dtype, value_dtype, *, seed=42, B=3, n=6, m=5, p=9, density=0.4):
    gen = torch.Generator().manual_seed(seed)
    masks = random_masks(B, n, m, density=density, generator=gen)
    rowptr, col, vals, B_, n_, m_ = make_batched_csr(masks, index_dtype, value_dtype, device="cuda", generator=gen)
    dense = torch.randn(B_, m_, p, dtype=value_dtype, device="cuda")
    return rowptr, col, vals, dense, B_, n_, m_


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p", [1, 16, 64])
def test_determinism_run_twice_bitwise_equal(index_dtype, value_dtype, p):
    rowptr, col, vals, dense, B, n, m = _random_inputs(index_dtype, value_dtype, p=p)
    out1 = torch.ops.tsgu.spmm(vals, rowptr, col, dense, B, n, m)
    out2 = torch.ops.tsgu.spmm(vals, rowptr, col, dense, B, n, m)
    assert torch.equal(out1, out2), "tsgu::spmm has no atomics (spmm.cu) -- repeat calls must be bitwise identical"


@requires_cuda_backend
def test_determinism_under_use_deterministic_algorithms():
    rowptr, col, vals, dense, B, n, m = _random_inputs(torch.int64, torch.float64, seed=43, p=17)
    torch.use_deterministic_algorithms(True)
    try:
        out1 = torch.ops.tsgu.spmm(vals, rowptr, col, dense, B, n, m)
        out2 = torch.ops.tsgu.spmm(vals, rowptr, col, dense, B, n, m)
        ref = dense_reference(rowptr, col, vals, dense, B, n, m)
    finally:
        torch.use_deterministic_algorithms(False)
    assert torch.equal(out1, out2)
    assert_close(out1, ref, reduction_length=m)


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p", [1, 16])
def test_adversarial_empty_rows(index_dtype, value_dtype, p):
    """Some folded rows have zero specified entries -- out[row, :] must be
    written as exactly zero (spmm.cu: "out is dense -- every row must be
    initialized"), never skipped."""
    n, m = 5, 4
    mask = torch.zeros(n, m)
    mask[2, 0] = 1.0
    mask[2, 3] = 1.0
    mask[4, 1] = 1.0
    rowptr, col, vals, B, n_, m_ = make_batched_csr([mask], index_dtype, value_dtype, device="cuda")

    dense = torch.randn(B, m_, p, dtype=value_dtype, device="cuda")
    out = torch.ops.tsgu.spmm(vals, rowptr, col, dense, B, n_, m_)
    ref = dense_reference(rowptr, col, vals, dense, B, n_, m_)
    assert_close(out, ref, reduction_length=m_)

    # Rows 0, 1, 3 are empty -> exactly zero.
    for empty_row in (0, 1, 3):
        assert torch.equal(out[0, empty_row], torch.zeros(p, dtype=value_dtype, device="cuda"))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_adversarial_empty_batch_item(index_dtype):
    n, m, p = 4, 4, 6
    masks = [torch.zeros(n, m), (torch.rand(n, m, generator=torch.Generator().manual_seed(5)) < 0.5).float()]
    rowptr, col, vals, B, n_, m_ = make_batched_csr(masks, index_dtype, torch.float32, device="cuda")

    dense = torch.randn(B, m_, p, device="cuda")
    out = torch.ops.tsgu.spmm(vals, rowptr, col, dense, B, n_, m_)
    ref = dense_reference(rowptr, col, vals, dense, B, n_, m_)
    assert_close(out, ref, reduction_length=m_)
    assert torch.equal(out[0], torch.zeros_like(out[0]))


@requires_cuda_backend
def test_adversarial_all_empty_pattern():
    """Every row of every batch item is empty -- nse_total == 0 (spmm.cu
    launcher writes a well-formed all-zero output, never skips the whole
    kernel launch's semantics)."""
    n, m, p = 3, 3, 4
    masks = [torch.zeros(n, m), torch.zeros(n, m)]
    rowptr, col, vals, B, n_, m_ = make_batched_csr(masks, torch.int64, torch.float32, device="cuda")
    assert vals.numel() == 0

    dense = torch.randn(B, m_, p, device="cuda")
    out = torch.ops.tsgu.spmm(vals, rowptr, col, dense, B, n_, m_)
    assert out.shape == (B, n_, p)
    assert torch.equal(out, torch.zeros_like(out))


@requires_cuda_backend
def test_adversarial_p_zero():
    """Zero-size p (testing.md Pillar 2 adversarial cases): a degenerate but
    well-formed dense operand with no columns at all."""
    n, m = 4, 3
    mask = (torch.rand(n, m, generator=torch.Generator().manual_seed(6)) < 0.5).float()
    rowptr, col, vals, B, n_, m_ = make_batched_csr([mask], torch.int64, torch.float32, device="cuda")

    dense = torch.randn(B, m_, 0, device="cuda")
    out = torch.ops.tsgu.spmm(vals, rowptr, col, dense, B, n_, m_)
    assert out.shape == (B, n_, 0)
