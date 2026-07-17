"""Gate 3 — parity for ``segment_mm``/``gather_mm`` (the public wrappers,
spec/commit.md Phase 3 commit 18) against BOTH oracles (spec/testing.md
"Gates & ordering": "parity (Oracle A + B, full matrix)").

Oracle A (tests/oracle/indexed_matmul.py, the pre-rewrite nested-tensor
implementation): forward parity on the shapes it supports. One frozen quirk
documented inline: Oracle A's ``gather_mm`` allocates its output with
``torch.empty((N, D2), device=...)`` -- default dtype (f32) regardless of the
input dtype -- so its f64 output is silently cast down to f32. The oracle is
frozen ("do not edit"), so f64 gather_mm parity vs Oracle A is compared in
the oracle's own output dtype at f32 tolerance, and true f64 parity is
carried by Oracle B.

Oracle B (fp64 dense per-segment/per-row ``@`` loops, local to this module):
independent dense fp64 reference for forward AND both gradients (manual
adjoint formulas -- gradA is gather mode on transposed groups, gradB is the
scatter-reduce mode, so the gradient rows here are the only parity coverage
of ``reduce=True``), plus the edges Oracle A can't reach (N=0).

Variant matrix per this commit's T5: uniform and ragged segment lengths
incl. zero-length segments, arbitrary idx for gather_mm, f32/f64 x i32/i64,
N=0 edge, single group. Tolerances come from tests/_tolerances.py (the one
policy module), reduction_length = D1 (each output element is a length-D1
inner product).
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from tests._tolerances import assert_close
from tests.oracle.indexed_matmul import gather_mm as oracle_gather_mm
from tests.oracle.indexed_matmul import segment_mm as oracle_segment_mm
from torchsparsegradutils import gather_mm, segment_mm
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate3

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

# Segment-length variants (R segments; sums differ, N follows the sum).
SEGLENS = {
    "uniform": [8, 8, 8, 8],
    "ragged": [3, 11, 1, 6],
    "zero-length": [5, 0, 9, 0],
    "single-group": [16],
}


# ---------------------------------------------------------------------------
# Oracle B reference formulas -- fp64 dense loops, independent of the oracle
# AND of tsgu::grouped_gemm.
# ---------------------------------------------------------------------------


def _idx_from_seglens(seglens, index_dtype, device):
    return torch.repeat_interleave(torch.arange(len(seglens), device=device), torch.tensor(seglens, device=device)).to(
        index_dtype
    )


def _dense_ref_forward(a, b, idx):
    """out[i] = a[i] @ b[idx[i]] in fp64."""
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    out = torch.zeros(a.shape[0], b.shape[-1], dtype=torch.float64, device=a.device)
    for i in range(a.shape[0]):
        out[i] = a64[i] @ b64[idx[i]]
    return out


def _dense_ref_grads(a, b, idx, gout):
    """Manual fp64 adjoints of the gather-mode forward:
    grad_a[i] = gout[i] @ b[idx[i]].T; grad_b[k] = sum_{i: idx[i]==k}
    outer(a[i], gout[i])."""
    a64, b64, g64 = a.to(torch.float64), b.to(torch.float64), gout.to(torch.float64)
    grad_a = torch.zeros_like(a64)
    grad_b = torch.zeros_like(b64)
    for i in range(a.shape[0]):
        k = int(idx[i])
        grad_a[i] = g64[i] @ b64[k].T
        grad_b[k] += torch.outer(a64[i], g64[i])
    return grad_a, grad_b


# ---------------------------------------------------------------------------
# Oracle A -- forward parity.
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("seglen_name", list(SEGLENS))
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_a_segment_mm(seglen_name, index_dtype, value_dtype):
    seglens = SEGLENS[seglen_name]
    N, R, D1, D2 = sum(seglens), len(seglens), 5, 4
    a = torch.randn(N, D1, dtype=value_dtype, device="cuda")
    b = torch.randn(R, D1, D2, dtype=value_dtype, device="cuda")
    seglen_a = torch.tensor(seglens, dtype=index_dtype, device="cuda")

    ours = segment_mm(a, b, seglen_a)
    ref = oracle_segment_mm(a, b, seglen_a)
    assert ours.shape == (N, D2)
    assert_close(ours, ref, reduction_length=D1)


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_a_gather_mm_arbitrary_idx(index_dtype, value_dtype):
    N, R, D1, D2 = 20, 4, 5, 4
    a = torch.randn(N, D1, dtype=value_dtype, device="cuda")
    b = torch.randn(R, D1, D2, dtype=value_dtype, device="cuda")
    idx_b = torch.randint(0, R, (N,), dtype=index_dtype, device="cuda")

    ours = gather_mm(a, b, idx_b)
    ref = oracle_gather_mm(a, b, idx_b)
    # Frozen Oracle A quirk (module docstring): its output is always f32
    # (torch.empty default dtype), so compare in the oracle's output dtype --
    # f64-tight parity for this op comes from Oracle B below.
    assert_close(ours.to(ref.dtype), ref, reduction_length=D1)


@requires_cuda_backend
def test_oracle_a_gather_mm_group_never_referenced():
    """Some groups of b are never selected by idx_b -- DGL-exact semantics
    just ignore them (and their gradB is zero, covered by Oracle B)."""
    N, R, D1, D2 = 10, 5, 4, 3
    a = torch.randn(N, D1, device="cuda")
    b = torch.randn(R, D1, D2, device="cuda")
    idx_b = torch.tensor([0, 0, 3, 3, 3, 0, 3, 0, 0, 3], device="cuda")

    ours = gather_mm(a, b, idx_b)
    ref = oracle_gather_mm(a, b, idx_b)
    assert_close(ours, ref, reduction_length=D1)


# ---------------------------------------------------------------------------
# Oracle B -- forward + both gradients (manual fp64 adjoints; the gradB rows
# are the parity coverage of reduce=True / scatter mode).
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("seglen_name", list(SEGLENS))
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_b_segment_mm_forward_and_grads(seglen_name, index_dtype, value_dtype):
    seglens = SEGLENS[seglen_name]
    N, R, D1, D2 = sum(seglens), len(seglens), 5, 4
    a = torch.randn(N, D1, dtype=value_dtype, device="cuda", requires_grad=True)
    b = torch.randn(R, D1, D2, dtype=value_dtype, device="cuda", requires_grad=True)
    seglen_a = torch.tensor(seglens, dtype=index_dtype, device="cuda")
    idx = _idx_from_seglens(seglens, index_dtype, torch.device("cuda"))

    out = segment_mm(a, b, seglen_a)
    ref = _dense_ref_forward(a.detach(), b.detach(), idx)
    assert_close(out, ref, reduction_length=D1)

    gout = torch.randn_like(out)
    out.backward(gout)
    grad_a_ref, grad_b_ref = _dense_ref_grads(a.detach(), b.detach(), idx, gout)
    assert_close(a.grad, grad_a_ref, reduction_length=D2)
    assert_close(b.grad, grad_b_ref, reduction_length=max(1, max(seglens)))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_b_gather_mm_forward_and_grads(index_dtype, value_dtype):
    N, R, D1, D2 = 24, 5, 4, 6
    a = torch.randn(N, D1, dtype=value_dtype, device="cuda", requires_grad=True)
    b = torch.randn(R, D1, D2, dtype=value_dtype, device="cuda", requires_grad=True)
    idx_b = torch.randint(0, R, (N,), dtype=index_dtype, device="cuda")

    out = gather_mm(a, b, idx_b)
    ref = _dense_ref_forward(a.detach(), b.detach(), idx_b)
    assert_close(out, ref, reduction_length=D1)

    gout = torch.randn_like(out)
    out.backward(gout)
    grad_a_ref, grad_b_ref = _dense_ref_grads(a.detach(), b.detach(), idx_b, gout)
    assert_close(a.grad, grad_a_ref, reduction_length=D2)
    assert_close(b.grad, grad_b_ref, reduction_length=N)


@requires_cuda_backend
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_b_segment_mm_n_zero(value_dtype):
    """N=0 edge: every segment empty; forward is a well-formed (0, D2)
    output and gradB is exactly zero (Oracle A's nested-tensor path can't
    represent this, so Oracle B carries it alone)."""
    R, D1, D2 = 3, 4, 5
    a = torch.zeros(0, D1, dtype=value_dtype, device="cuda", requires_grad=True)
    b = torch.randn(R, D1, D2, dtype=value_dtype, device="cuda", requires_grad=True)
    seglen_a = torch.zeros(R, dtype=torch.int64, device="cuda")

    out = segment_mm(a, b, seglen_a)
    assert out.shape == (0, D2)

    out.sum().backward()
    assert a.grad.shape == (0, D1)
    assert torch.equal(b.grad, torch.zeros_like(b))


@requires_cuda_backend
def test_oracle_b_gather_mm_n_zero():
    R, D1, D2 = 3, 4, 5
    a = torch.zeros(0, D1, device="cuda", requires_grad=True)
    b = torch.randn(R, D1, D2, device="cuda", requires_grad=True)
    idx_b = torch.zeros(0, dtype=torch.int64, device="cuda")

    out = gather_mm(a, b, idx_b)
    assert out.shape == (0, D2)

    out.sum().backward()
    assert a.grad.shape == (0, D1)
    assert torch.equal(b.grad, torch.zeros_like(b))


@requires_cuda_backend
def test_oracle_b_gather_mm_single_group():
    """R=1 gather is exactly a @ b[0] -- checked against the plain dense
    matmul in fp64."""
    N, D1, D2 = 16, 5, 4
    a = torch.randn(N, D1, device="cuda")
    b = torch.randn(1, D1, D2, device="cuda")
    idx_b = torch.zeros(N, dtype=torch.int32, device="cuda")

    ours = gather_mm(a, b, idx_b)
    ref = (a.to(torch.float64) @ b[0].to(torch.float64)).to(ours.dtype)
    assert_close(ours, ref, reduction_length=D1)
