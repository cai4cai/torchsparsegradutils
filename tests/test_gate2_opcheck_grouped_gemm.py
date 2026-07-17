"""Gate 2 — opcheck for tsgu::grouped_gemm (spec/testing.md "Gates &
ordering": "opcheck (all ops x representative inputs)"; spec/commit.md Phase
3 commit 18).

tsgu::grouped_gemm gets its real CUDA implementation in this commit AND is
wired into the public ``segment_mm``/``gather_mm`` wrappers with a real
``register_autograd`` (torchsparsegradutils/ops/indexed_matmul.py, landed in
commit 9) -- so opcheck here exercises schema, fake-kernel, AND
autograd-registration consistency for real, including ``requires_grad=True``
cases (testing.md Pillar 2: "requires_grad on/off").

Mode-specific notes:

- Gather mode (``reduce=False``) accepts arbitrary ``idx``; samples below
  include repeated, out-of-order indices.
- Scatter-reduce mode (``reduce=True``) requires ``idx`` sorted
  non-decreasing (deterministic ordered accumulation, no atomics -- the
  op's documented contract), so every scatter sample uses sorted ``idx``.
  Scatter samples run with ``requires_grad=False`` only: reduce=True is a
  gradient leaf by design (its registered backward raises
  NotImplementedError, see ``_grouped_gemm_backward``), and opcheck's
  ``test_aot_dispatch_dynamic`` differentiates any input that requires
  grad, which would exercise exactly that deliberate raise rather than a
  real adjoint.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate2

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

# Small representative sizes (hard resource guard: tiny tensors only).
_N, _D1, _D2, _R = 6, 4, 3, 3


def _gather_inputs(index_dtype, value_dtype, *, requires_grad=False):
    a = torch.randn(_N, _D1, dtype=value_dtype, device="cuda", requires_grad=requires_grad)
    b = torch.randn(_R, _D1, _D2, dtype=value_dtype, device="cuda", requires_grad=requires_grad)
    # Arbitrary order with repeats -- gather mode's contract allows it.
    idx = torch.tensor([2, 0, 2, 1, 0, 0], dtype=index_dtype, device="cuda")
    return a, b, idx


def _scatter_inputs(index_dtype, value_dtype):
    a = torch.randn(_N, _D1, dtype=value_dtype, device="cuda")
    b = torch.randn(_N, _D2, dtype=value_dtype, device="cuda")
    # Sorted non-decreasing -- scatter mode's hard contract (op docstring).
    idx = torch.tensor([0, 0, 1, 1, 1, 2], dtype=index_dtype, device="cuda")
    return a, b, idx


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("requires_grad", [False, True])
def test_opcheck_gather_mode(index_dtype, value_dtype, requires_grad):
    a, b, idx = _gather_inputs(index_dtype, value_dtype, requires_grad=requires_grad)
    torch.library.opcheck(torch.ops.tsgu.grouped_gemm.default, (a, b, idx, _R, False))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_opcheck_scatter_mode_sorted_idx(index_dtype, value_dtype):
    a, b, idx = _scatter_inputs(index_dtype, value_dtype)
    torch.library.opcheck(torch.ops.tsgu.grouped_gemm.default, (a, b, idx, _R, True))


@requires_cuda_backend
@pytest.mark.parametrize("reduce", [False, True])
def test_opcheck_noncontiguous_a(reduce):
    """testing.md Pillar 2: "strided and non-contiguous dense operands" --
    ``a`` built via a transpose so it is not contiguous."""
    a = torch.randn(_D1, _N, device="cuda").t()  # (N, D1), non-contiguous
    assert not a.is_contiguous()
    if reduce:
        b = torch.randn(_N, _D2, device="cuda")
        idx = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64, device="cuda")
    else:
        b = torch.randn(_R, _D1, _D2, device="cuda")
        idx = torch.tensor([2, 0, 2, 1, 0, 0], dtype=torch.int64, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.grouped_gemm.default, (a, b, idx, _R, reduce))


@requires_cuda_backend
@pytest.mark.parametrize("reduce", [False, True])
def test_opcheck_n_zero(reduce):
    """N=0 edge: no rows at all, every group empty (still a well-formed
    call; scatter must produce an all-zero (R, D1, D2) output)."""
    a = torch.randn(0, _D1, device="cuda")
    idx = torch.zeros(0, dtype=torch.int64, device="cuda")
    b = torch.randn(0, _D2, device="cuda") if reduce else torch.randn(_R, _D1, _D2, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.grouped_gemm.default, (a, b, idx, _R, reduce))


@requires_cuda_backend
def test_opcheck_single_group():
    """R=1: every row maps to group 0 (idx trivially sorted)."""
    a = torch.randn(_N, _D1, device="cuda", requires_grad=True)
    b = torch.randn(1, _D1, _D2, device="cuda", requires_grad=True)
    idx = torch.zeros(_N, dtype=torch.int32, device="cuda")
    torch.library.opcheck(torch.ops.tsgu.grouped_gemm.default, (a, b, idx, 1, False))
