"""Gate 5 — determinism + adversarial structure for tsgu::spsm
(spec/testing.md "Gates & ordering": "property + adversarial + determinism";
spec/commit.md Phase 3 commit 16).

Determinism: spsm.cu's module comment states the solve kernel has no
atomics -- every folded row/owner is written by exactly one warp, in
exactly one level's launch, and level order is fixed by the (cached, or
freshly rebuilt identically) analysis. Run-twice bitwise equality, plus
confirmation under torch.use_deterministic_algorithms(True) (kernels.md
open Q1's resolution -- same as every other kernel in this codebase).

Adversarial (testing.md + this commit's brief): ill-conditioned diagonal
(tiny but nonzero -- must not raise, only a numerically poor answer),
single-row, dense-triangular row (every column below/above the diagonal
specified), deep dependency chain (bidiagonal -- worst case: n levels, one
row each), wide level (diagonal-only pattern -- one level, n rows).
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from tests._spsm_helpers import bidiagonal_chain_csr, dense_reference_solve, make_batched_triangular_csr
from tests._tolerances import assert_close
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate5

_SKIP_REASON = (
    "tsgu::spsm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)


def _random_inputs(index_dtype, value_dtype, *, seed=42, n=12, p=5, density=0.35):
    gen = torch.Generator().manual_seed(seed)
    rowptr, col, vals, B, n_ = make_batched_triangular_csr(
        n, 1, index_dtype, value_dtype, "cuda", upper=True, unitriangular=False, density=density, generator=gen
    )
    rhs = torch.randn(B, n_, p, dtype=value_dtype, device="cuda")
    return rowptr, col, vals, rhs, B, n_


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("p", [1, 8, 33])
def test_determinism_run_twice_bitwise_equal(index_dtype, value_dtype, p):
    rowptr, col, vals, _, B, n = _random_inputs(index_dtype, value_dtype, p=p)
    rhs = torch.randn(B, n, p, dtype=value_dtype, device="cuda")
    out1 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n, True, False, False)
    out2 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n, True, False, False)
    assert torch.equal(out1, out2), "tsgu::spsm has no atomics (spsm.cu) -- repeat calls must be bitwise identical"


@requires_cuda_backend
def test_determinism_under_use_deterministic_algorithms():
    rowptr, col, vals, rhs, B, n = _random_inputs(torch.int64, torch.float64, seed=43, p=7)
    torch.use_deterministic_algorithms(True)
    try:
        out1 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n, True, False, False)
        out2 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n, True, False, False)
        ref = dense_reference_solve(rowptr, col, vals, rhs, B, n, upper=True, unitriangular=False, transpose=False)
    finally:
        torch.use_deterministic_algorithms(False)
    assert torch.equal(out1, out2)
    assert_close(out1, ref, reduction_length=n)


@requires_cuda_backend
def test_plan_cache_separate_across_calls_still_deterministic():
    """Same BatchedCSR called with DIFFERENT flags (transpose True/False) --
    separate cached plans (plan.h: cache key includes transpose), each
    individually deterministic."""
    rowptr, col, vals, rhs, B, n = _random_inputs(torch.int64, torch.float64, seed=44)
    for transpose in (False, True, False, True):
        out1 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n, True, False, transpose)
        out2 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n, True, False, transpose)
        assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Adversarial.
# ---------------------------------------------------------------------------


@requires_cuda_backend
def test_adversarial_ill_conditioned_tiny_diagonal():
    """Tiny-but-nonzero diagonal -- must not raise (the contract only
    requires the diagonal entry to EXIST, map.md invariant 7), only produce
    a numerically large (but finite, non-crashing) answer."""
    n = 5
    rowptr = torch.tensor([0, 1, 3, 6, 10, 15], dtype=torch.int64, device="cuda")
    col = torch.tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4], dtype=torch.int64, device="cuda")
    vals = torch.rand(15, dtype=torch.float64, device="cuda") * 0.1 + 0.5
    diag_positions = [0, 2, 5, 9, 14]
    vals = vals.clone()
    vals[diag_positions] = torch.tensor([1e-6, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64, device="cuda")
    rhs = torch.randn(1, n, 2, dtype=torch.float64, device="cuda")
    out = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, n, False, False, False)
    assert torch.isfinite(out).all()


@requires_cuda_backend
def test_adversarial_single_row():
    rowptr = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    col = torch.tensor([0], dtype=torch.int64, device="cuda")
    vals = torch.tensor([2.0], dtype=torch.float64, device="cuda")
    rhs = torch.tensor([[[3.0, 4.0]]], dtype=torch.float64, device="cuda")
    out = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, 1, True, False, False)
    assert_close(out, torch.tensor([[[1.5, 2.0]]], dtype=torch.float64, device="cuda"), reduction_length=1)


@requires_cuda_backend
def test_adversarial_dense_triangular_row():
    """Every column below/above the diagonal specified (a fully dense
    triangular row, not just banded/sparse) -- the deepest per-row
    dependency chain the solve kernel's inner loop can see."""
    n = 16
    rowptr, col, vals, B, n_ = make_batched_triangular_csr(
        n, 1, torch.int64, torch.float64, "cuda", upper=True, unitriangular=False, density=1.0
    )
    rhs = torch.randn(B, n_, 3, dtype=torch.float64, device="cuda")
    out = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n_, True, False, False)
    ref = dense_reference_solve(rowptr, col, vals, rhs, B, n_, upper=True, unitriangular=False, transpose=False)
    assert_close(out, ref, reduction_length=n_)


@requires_cuda_backend
def test_adversarial_deep_dependency_chain_bidiagonal():
    """Worst case for the level schedule: n levels, one row each
    (spsm.cu/this commit's brief)."""
    n = 64
    rowptr, col, vals, B, n_ = bidiagonal_chain_csr(n, torch.int64, torch.float64, "cuda", upper=False)
    rhs = torch.randn(B, n_, 4, dtype=torch.float64, device="cuda")
    out1 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n_, False, False, False)
    out2 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n_, False, False, False)
    assert torch.equal(out1, out2)
    ref = dense_reference_solve(rowptr, col, vals, rhs, B, n_, upper=False, unitriangular=False, transpose=False)
    assert_close(out1, ref, reduction_length=n_)


@requires_cuda_backend
def test_adversarial_wide_level_diagonal_only():
    """Diagonal-only pattern -- exactly one level, n rows, all independent
    (the opposite adversarial extreme from the bidiagonal chain)."""
    n = 500
    idx = torch.arange(n, dtype=torch.int64, device="cuda")
    rowptr = idx.clone()
    rowptr = torch.arange(n + 1, dtype=torch.int64, device="cuda")
    col = idx
    vals = torch.rand(n, dtype=torch.float64, device="cuda") + 1.0
    rhs = torch.randn(1, n, 6, dtype=torch.float64, device="cuda")
    out1 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, n, True, False, False)
    out2 = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, 1, n, True, False, False)
    assert torch.equal(out1, out2)
    ref = rhs[0] / vals.unsqueeze(-1)
    assert_close(out1[0], ref, reduction_length=1)


@requires_cuda_backend
def test_adversarial_p_zero():
    rowptr, col, vals, _, B, n = _random_inputs(torch.int64, torch.float32)
    rhs = torch.randn(B, n, 0, device="cuda")
    out = torch.ops.tsgu.spsm(vals, rowptr, col, rhs, B, n, True, False, False)
    assert out.shape == (B, n, 0)
