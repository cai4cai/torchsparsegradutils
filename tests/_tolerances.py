"""tests/_tolerances.py — the ONE tolerance policy table.

spec/testing.md, Pillar 1: "Tolerance is a named, per-dtype policy — never
ad-hoc ``atol`` per test: f64 → tight rel (1e-12); f32 → rel 1e-5 with
accumulation-order allowance scaled by reduction length; bf16/fp16 rows →
ULP bounds vs the fp32-accum reference. The tolerance table lives in one
fixture module; a test that needs a looser bound documents why inline or
fails review."

Every parity/gradcheck test in this suite gets its numerical bound from this
module — nowhere else defines a fresh ``atol``/``rtol`` pair. This is
distinct from ``tests/test_config.py``'s pre-existing ``Tolerances`` class,
which is scoped to the iterative-solver suites (CG/BiCGSTAB/LSMR/MINRES —
convergence tolerances, not dtype-accuracy tolerances) and predates this
policy; it is left as-is (one concern per commit, spec/commit.md).
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# f64 / f32 — relative-tolerance policy (testing.md).
# ---------------------------------------------------------------------------

# f64: "tight rel (1e-12)".
F64_RTOL = 1e-12
# A tiny absolute floor so comparisons against exact zeros don't divide by
# zero in relative terms; still far tighter than any f32 bound below.
F64_ATOL = 1e-12

# f32: "rel 1e-5 with accumulation-order allowance scaled by reduction
# length" — the base bound before any reduction-length scaling.
F32_RTOL_BASE = 1e-5
F32_ATOL_BASE = 1e-6

# Dtypes this policy has an opinion on. Anything else is a review-time
# decision (testing.md: "a test that needs a looser bound documents why
# inline"), not a silent default here.
_FLOAT_POLICY_DTYPES = (torch.float32, torch.float64)
_HALF_DTYPES = (torch.float16, torch.bfloat16)


def reduction_allowance(reduction_length: int) -> float:
    """Accumulation-order allowance factor for a sum/reduction over
    ``reduction_length`` terms (e.g. nse per row for spmm/sddmm/seglse-style
    reductions; 1 for elementwise ops).

    Naive sequential/pairwise floating-point summation accumulates rounding
    error that grows roughly with ``sqrt(reduction_length)`` for
    uncorrelated rounding directions (a standard, conservative rule of
    thumb — not a tight bound, but the right order of magnitude for
    "different but valid summation order" discrepancies between our kernel
    and a reference that doesn't sum in the same order). ``reduction_length
    <= 1`` gets no allowance (factor 1.0).
    """
    if reduction_length <= 1:
        return 1.0
    return math.sqrt(float(reduction_length))


def rtol_atol(dtype: torch.dtype, *, reduction_length: int = 1) -> tuple[float, float]:
    """Return ``(rtol, atol)`` for `dtype`, scaled by `reduction_length`.

    Raises for half precision (bf16/fp16) and any dtype outside the
    f32/f64 policy — those go through :func:`assert_close` /
    :func:`half_ulp_bound` instead, since they're not a plain rel-tolerance
    comparison (testing.md: ULP bounds vs the fp32-accum reference).
    """
    if dtype not in _FLOAT_POLICY_DTYPES:
        raise ValueError(
            f"rtol_atol() has no rel-tolerance policy for {dtype} — half "
            "precision (fp16/bf16) uses half_ulp_bound()/assert_close() "
            "instead (testing.md: ULP bounds vs the fp32-accum reference); "
            "any other dtype needs a reviewed, documented addition to this "
            "module, never an ad-hoc atol at the call site."
        )
    factor = reduction_allowance(reduction_length)
    if dtype == torch.float64:
        return F64_RTOL * factor, F64_ATOL * factor
    return F32_RTOL_BASE * factor, F32_ATOL_BASE * factor


# ---------------------------------------------------------------------------
# bf16 / fp16 — ULP bounds vs the fp32-accum reference (testing.md).
# ---------------------------------------------------------------------------

# Default ULP budget for a half-precision result computed by (conceptually)
# accumulating in fp32 and rounding once to the storage dtype at the end —
# kernels.md's fp32-accumulation policy for the half-precision (v2) rows.
# A handful of ULPs covers round-to-nearest-even differences in the final
# cast plus a small amount of epilogue reordering; a test that needs more
# documents why inline (testing.md).
DEFAULT_ULPS = 4


def half_ulp_tolerance(dtype: torch.dtype, ulps: int = DEFAULT_ULPS) -> float:
    """The (rtol == atol, deliberately) bound for `ulps` units-in-last-place
    of `dtype`, expressed as ``ulps * eps(dtype)``. This is an
    epsilon-relative approximation to a true bit-level ULP count — adequate
    for the "is this within a handful of rounding steps of the fp32-accum
    reference" check this policy exists for, not a substitute for exact
    integer ULP distance."""
    if dtype not in _HALF_DTYPES:
        raise ValueError(f"half_ulp_tolerance() is for {_HALF_DTYPES}, got {dtype}")
    return ulps * torch.finfo(dtype).eps


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    reduction_length: int = 1,
    ulps: int = DEFAULT_ULPS,
) -> None:
    """One-stop comparison using this module's policy, dispatched on
    ``actual.dtype``:

    - f32/f64: :func:`rtol_atol`, scaled by ``reduction_length``.
    - bf16/fp16: ULP bound vs ``expected`` cast down to ``actual``'s dtype
      (the "fp32-accum reference" — ``expected`` is expected to already be
      the higher-precision reference computation).

    ``expected`` is cast to ``actual``'s dtype before comparison in the half
    case (mirrors testing.md: "ULP bounds vs the fp32-accum reference").
    """
    dtype = actual.dtype
    if dtype in _HALF_DTYPES:
        tol = half_ulp_tolerance(dtype, ulps)
        torch.testing.assert_close(actual, expected.to(dtype), rtol=tol, atol=tol)
        return
    rtol, atol = rtol_atol(dtype, reduction_length=reduction_length)
    torch.testing.assert_close(actual, expected.to(dtype), rtol=rtol, atol=atol)
