"""Gate 5 — determinism for tsgu::grouped_gemm through the public
``segment_mm``/``gather_mm`` wrappers (spec/testing.md "Gates & ordering":
"property + adversarial + determinism"; spec/commit.md Phase 3 commit 18).

The kernel is deterministic by construction: gather mode owns each output
row exclusively, and scatter-reduce mode requires idx sorted non-decreasing
precisely so each group accumulates its rows in a fixed order with no
atomics (the op's documented contract). kernels.md's open-Q1 resolution
(testing.md) requires two things of a kernel making that claim: a run-twice
bitwise-equality test for the forward AND both gradients, and confirmation
it still runs (and matches) under
``torch.use_deterministic_algorithms(True)`` -- no separate deterministic
path needed since the only path already qualifies. The gradB path
additionally goes through a stable ``argsort``
(``_grouped_gemm_backward``), itself deterministic (ties broken by input
position), so the whole chain is covered.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils import gather_mm, segment_mm
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate5

_SKIP_REASON = (
    "tsgu::grouped_gemm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)


def _grouped_gemm_cuda_ready() -> bool:
    # Identical in effect to the other gate files' machine-level condition
    # once commit 18's kernel is registered; the extra dispatch-key check
    # keeps this file skipping (not erroring) when the front-package wiring
    # is present but the cuda/ tree has not yet been rebuilt with the
    # grouped_gemm kernel (commit 18 is developed in two concurrent lanes).
    return (
        torch.cuda.is_available()
        and backend_available()
        and torch._C._dispatch_has_kernel_for_dispatch_key("tsgu::grouped_gemm", "CUDA")
    )


requires_cuda_backend = pytest.mark.skipif(not _grouped_gemm_cuda_ready(), reason=_SKIP_REASON)

INDEX_DTYPES = (torch.int32, torch.int64)
VALUE_DTYPES = (torch.float32, torch.float64)


def _segment_inputs(index_dtype, value_dtype, *, seed=42):
    torch.manual_seed(seed)
    seglens = [7, 0, 12, 5]  # ragged incl. a zero-length segment
    N, R, D1, D2 = sum(seglens), len(seglens), 8, 6
    a = torch.randn(N, D1, dtype=value_dtype, device="cuda")
    b = torch.randn(R, D1, D2, dtype=value_dtype, device="cuda")
    seglen_a = torch.tensor(seglens, dtype=index_dtype, device="cuda")
    return a, b, seglen_a


def _gather_inputs(index_dtype, value_dtype, *, seed=43):
    torch.manual_seed(seed)
    N, R, D1, D2 = 24, 5, 8, 6
    a = torch.randn(N, D1, dtype=value_dtype, device="cuda")
    b = torch.randn(R, D1, D2, dtype=value_dtype, device="cuda")
    idx_b = torch.randint(0, R, (N,), dtype=index_dtype, device="cuda")
    return a, b, idx_b


def _fwd_and_grads(fn, a, b, third):
    a = a.detach().clone().requires_grad_(True)
    b = b.detach().clone().requires_grad_(True)
    out = fn(a, b, third)
    gout = torch.ones_like(out)  # fixed upstream gradient, identical across runs
    grad_a, grad_b = torch.autograd.grad(out, (a, b), gout)
    return out, grad_a, grad_b


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_determinism_segment_mm_run_twice_bitwise_equal(index_dtype, value_dtype):
    a, b, seglen_a = _segment_inputs(index_dtype, value_dtype)
    out1, ga1, gb1 = _fwd_and_grads(segment_mm, a, b, seglen_a)
    out2, ga2, gb2 = _fwd_and_grads(segment_mm, a, b, seglen_a)
    assert torch.equal(out1, out2), "grouped_gemm gather mode has no atomics -- repeat forwards must be bitwise equal"
    assert torch.equal(ga1, ga2), "gradA (gather mode, transposed groups) must be bitwise repeatable"
    assert torch.equal(gb1, gb2), "gradB (scatter mode, ordered accumulation) must be bitwise repeatable"


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_determinism_gather_mm_run_twice_bitwise_equal(index_dtype, value_dtype):
    a, b, idx_b = _gather_inputs(index_dtype, value_dtype)
    out1, ga1, gb1 = _fwd_and_grads(gather_mm, a, b, idx_b)
    out2, ga2, gb2 = _fwd_and_grads(gather_mm, a, b, idx_b)
    assert torch.equal(out1, out2)
    assert torch.equal(ga1, ga2)
    # gradB goes through stable argsort + ordered scatter accumulation --
    # the arbitrary-idx case is exactly where a hidden atomic would show up.
    assert torch.equal(gb1, gb2)


@requires_cuda_backend
@pytest.mark.parametrize("op_name", ["segment_mm", "gather_mm"])
def test_determinism_under_use_deterministic_algorithms(op_name):
    """The kernel must run (not raise) and stay bitwise repeatable under
    torch's global determinism switch -- deterministic by construction, no
    fallback path to engage (testing.md / kernels.md open Q1)."""
    if op_name == "segment_mm":
        a, b, third = _segment_inputs(torch.int64, torch.float64, seed=7)
        fn = segment_mm
    else:
        a, b, third = _gather_inputs(torch.int64, torch.float64, seed=7)
        fn = gather_mm

    torch.use_deterministic_algorithms(True)
    try:
        out1, ga1, gb1 = _fwd_and_grads(fn, a, b, third)
        out2, ga2, gb2 = _fwd_and_grads(fn, a, b, third)
    finally:
        torch.use_deterministic_algorithms(False)

    assert torch.equal(out1, out2)
    assert torch.equal(ga1, ga2)
    assert torch.equal(gb1, gb2)
