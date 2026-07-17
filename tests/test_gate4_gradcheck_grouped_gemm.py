"""Gate 4 — gradcheck for ``segment_mm``/``gather_mm`` (the public wrappers,
spec/testing.md "Gates & ordering": "gradcheck / gradgradcheck (f64)";
spec/commit.md Phase 3 commit 18).

``tsgu::grouped_gemm`` has a real ``register_autograd``
(torchsparsegradutils/ops/indexed_matmul.py, commit 9): gradA is gather mode
on transposed groups, gradB is the scatter-reduce mode behind a stable
``argsort`` (the kernel's sorted-idx contract for reduce=True) -- so
``torch.autograd.gradcheck`` here numerically proves both adjoint formulas,
including the sort-then-scatter gradB path for gather_mm's arbitrary idx.

No gradgradcheck: reduce=True is a deliberate gradient leaf (its backward
raises NotImplementedError -- map.md routing never differentiates through a
gradB computation), matching the spmm gate-4 file, which carries no
gradgradcheck either (testing.md scopes gradgradcheck to the solve ops).

All f64, tiny sizes (N <= 32).
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils import gather_mm, segment_mm
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate4

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

SEGLENS = {
    "uniform": [3, 3, 3],
    "ragged-with-zero": [4, 0, 2, 3],
}


@requires_cuda_backend
@pytest.mark.parametrize("seglen_name", list(SEGLENS))
def test_gradcheck_segment_mm_both_operands(seglen_name):
    seglens = SEGLENS[seglen_name]
    N, R, D1, D2 = sum(seglens), len(seglens), 4, 3
    a = torch.randn(N, D1, dtype=torch.float64, device="cuda", requires_grad=True)
    b = torch.randn(R, D1, D2, dtype=torch.float64, device="cuda", requires_grad=True)
    seglen_a = torch.tensor(seglens, dtype=torch.int64, device="cuda")

    assert torch.autograd.gradcheck(lambda a_, b_: segment_mm(a_, b_, seglen_a), (a, b))


@requires_cuda_backend
@pytest.mark.parametrize("operand", ["a", "b"])
def test_gradcheck_segment_mm_single_operand(operand):
    seglens = [2, 5, 1]
    N, R, D1, D2 = sum(seglens), len(seglens), 3, 4
    a = torch.randn(N, D1, dtype=torch.float64, device="cuda", requires_grad=(operand == "a"))
    b = torch.randn(R, D1, D2, dtype=torch.float64, device="cuda", requires_grad=(operand == "b"))
    seglen_a = torch.tensor(seglens, dtype=torch.int64, device="cuda")

    inputs = (a,) if operand == "a" else (b,)
    if operand == "a":
        assert torch.autograd.gradcheck(lambda a_: segment_mm(a_, b, seglen_a), inputs)
    else:
        assert torch.autograd.gradcheck(lambda b_: segment_mm(a, b_, seglen_a), inputs)


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_gradcheck_gather_mm_both_operands_arbitrary_idx(index_dtype):
    """Arbitrary (unsorted, repeating) idx -- numerically proves the
    argsort-then-scatter gradB path against gradcheck's finite differences."""
    N, R, D1, D2 = 9, 3, 4, 3
    a = torch.randn(N, D1, dtype=torch.float64, device="cuda", requires_grad=True)
    b = torch.randn(R, D1, D2, dtype=torch.float64, device="cuda", requires_grad=True)
    idx_b = torch.tensor([2, 0, 1, 2, 2, 0, 1, 0, 2], dtype=index_dtype, device="cuda")

    assert torch.autograd.gradcheck(lambda a_, b_: gather_mm(a_, b_, idx_b), (a, b))


@requires_cuda_backend
@pytest.mark.parametrize("operand", ["a", "b"])
def test_gradcheck_gather_mm_single_operand(operand):
    N, R, D1, D2 = 7, 3, 3, 5
    a = torch.randn(N, D1, dtype=torch.float64, device="cuda", requires_grad=(operand == "a"))
    b = torch.randn(R, D1, D2, dtype=torch.float64, device="cuda", requires_grad=(operand == "b"))
    idx_b = torch.tensor([1, 1, 0, 2, 0, 2, 1], dtype=torch.int64, device="cuda")

    if operand == "a":
        assert torch.autograd.gradcheck(lambda a_: gather_mm(a_, b, idx_b), (a,))
    else:
        assert torch.autograd.gradcheck(lambda b_: gather_mm(a, b_, idx_b), (b,))


@requires_cuda_backend
def test_gradcheck_gather_mm_unreferenced_group():
    """A group never selected by idx_b must get an exactly-zero gradient
    slice -- gradcheck's finite differences confirm the zero columns of the
    Jacobian too."""
    N, R, D1, D2 = 6, 3, 3, 3
    a = torch.randn(N, D1, dtype=torch.float64, device="cuda", requires_grad=True)
    b = torch.randn(R, D1, D2, dtype=torch.float64, device="cuda", requires_grad=True)
    idx_b = torch.tensor([0, 2, 0, 2, 0, 2], dtype=torch.int64, device="cuda")  # group 1 unused

    assert torch.autograd.gradcheck(lambda a_, b_: gather_mm(a_, b_, idx_b), (a, b))
