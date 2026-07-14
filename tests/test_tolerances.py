"""Unit tests for tests/_tolerances.py (the ONE tolerance policy table,
spec/testing.md). CPU-only, no GPU needed.
"""

from __future__ import annotations

import pytest
import torch

from tests._tolerances import (
    assert_close,
    half_ulp_tolerance,
    reduction_allowance,
    rtol_atol,
)


def test_f64_tighter_than_f32():
    rtol64, atol64 = rtol_atol(torch.float64)
    rtol32, atol32 = rtol_atol(torch.float32)
    assert rtol64 < rtol32
    assert atol64 < atol32


def test_reduction_length_loosens_f32_but_not_below_one():
    assert reduction_allowance(1) == 1.0
    assert reduction_allowance(0) == 1.0  # degenerate input still gets the floor, not a smaller factor
    r_small, _ = rtol_atol(torch.float32, reduction_length=1)
    r_big, _ = rtol_atol(torch.float32, reduction_length=10_000)
    assert r_big > r_small


def test_rtol_atol_rejects_half_precision():
    with pytest.raises(ValueError):
        rtol_atol(torch.float16)
    with pytest.raises(ValueError):
        rtol_atol(torch.bfloat16)


def test_half_ulp_tolerance_positive_and_dtype_gated():
    assert half_ulp_tolerance(torch.float16) > 0
    assert half_ulp_tolerance(torch.bfloat16) > 0
    with pytest.raises(ValueError):
        half_ulp_tolerance(torch.float32)


def test_assert_close_f32_f64_pass_and_fail():
    ref = torch.randn(8, dtype=torch.float64)
    assert_close(ref.to(torch.float32), ref)  # within f32 policy
    with pytest.raises(AssertionError):
        assert_close(ref.to(torch.float32) + 1.0, ref)  # far outside any policy


def test_assert_close_half_precision_vs_fp32_accum_reference():
    fp32_ref = torch.randn(8, dtype=torch.float32)
    half = fp32_ref.to(torch.bfloat16)
    assert_close(half, fp32_ref)  # round-trip through bf16 should be within a few ULPs
