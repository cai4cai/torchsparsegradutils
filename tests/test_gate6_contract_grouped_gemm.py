"""Gate 6 — contract conformance + workspace assertion for
``segment_mm``/``gather_mm``/tsgu::grouped_gemm (spec/testing.md "Gates &
ordering": "contract conformance + workspace assertion"; map.md invariant 7:
"raise, never silently accept"; spec/commit.md Phase 3 commit 18).

Two sections:

- **Wrapper validation (device-independent).** segment_mm/gather_mm's shape
  and dtype checks (torchsparsegradutils/ops/indexed_matmul.py) raise
  BEFORE any op dispatch, so these tests run on CPU tensors everywhere --
  no CUDA, no backend needed. (The spmm/spsm gate-6 files have no
  wrapper-level section because those wrappers' validation predates the
  rewrite and is covered by their frozen public-API suites; grouped_gemm's
  wrapper validation is new in this commit, so it is proven here.) Messages
  must state the accepted logical shape alongside the received one
  (naming.md §1's error template).
- **Kernel-side contract + workspace (CUDA-gated).** A sanity valid call
  and the workspace assertion (testing.md/benchmarks.md: "peak memory minus
  tensors <= O(nse) bound" -- for this dense op: no hidden per-row
  materialisation of the gathered ``b[idx]`` stack).
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils import gather_mm, segment_mm
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate6

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


# ---------------------------------------------------------------------------
# Wrapper validation -- device-independent (CPU tensors; raises before any
# op dispatch, so no backend is needed).
# ---------------------------------------------------------------------------


def _valid_segment_inputs():
    a = torch.randn(10, 4)
    b = torch.randn(3, 4, 2)
    seglen_a = torch.tensor([5, 2, 3])
    return a, b, seglen_a


def _valid_gather_inputs():
    a = torch.randn(5, 4)
    b = torch.randn(3, 4, 2)
    idx_b = torch.tensor([0, 2, 1, 0, 2])
    return a, b, idx_b


def test_segment_mm_rejects_non_tensor():
    a, b, seglen_a = _valid_segment_inputs()
    with pytest.raises(ValueError, match="instances of torch.Tensor"):
        segment_mm(a, b, [5, 2, 3])


def test_segment_mm_rejects_a_wrong_rank():
    a, b, seglen_a = _valid_segment_inputs()
    with pytest.raises(ValueError, match=r"a must be a matrix with shape \(N, D1\); got shape \(2, 5, 4\)"):
        segment_mm(a.reshape(2, 5, 4), b, seglen_a)


def test_segment_mm_rejects_b_wrong_rank():
    a, b, seglen_a = _valid_segment_inputs()
    with pytest.raises(ValueError, match=r"b must be a batched matrix with shape \(R, D1, D2\); got shape \(12, 2\)"):
        segment_mm(a, b.reshape(12, 2), seglen_a)


def test_segment_mm_rejects_seglen_wrong_rank():
    a, b, seglen_a = _valid_segment_inputs()
    with pytest.raises(ValueError, match=r"seglen_a must be a vector with shape \(R,\); got shape \(1, 3\)"):
        segment_mm(a, b, seglen_a.unsqueeze(0))


def test_segment_mm_rejects_non_integer_seglen():
    a, b, seglen_a = _valid_segment_inputs()
    with pytest.raises(ValueError, match=r"seglen_a must be a vector with shape \(R,\) of integer dtype"):
        segment_mm(a, b, seglen_a.to(torch.float32))


def test_segment_mm_rejects_mismatched_d1():
    a, b, seglen_a = _valid_segment_inputs()
    bad_b = torch.randn(3, 5, 2)  # D1=5, a has D1=4
    with pytest.raises(ValueError, match=r"a and b must share the inner dimension D1"):
        segment_mm(a, bad_b, seglen_a)


def test_segment_mm_rejects_seglen_length_mismatch():
    a, b, seglen_a = _valid_segment_inputs()
    bad_seglen = torch.tensor([5, 5])  # R=2, b has R=3
    with pytest.raises(ValueError, match=r"seglen_a must be a vector with shape \(R,\) with R = b.shape\[0\] = 3"):
        segment_mm(a, b, bad_seglen)


def test_segment_mm_rejects_seglen_sum_mismatch():
    """DGL-exact semantics (map.md contract): seglen_a must sum to N."""
    a, b, seglen_a = _valid_segment_inputs()
    bad_seglen = torch.tensor([5, 2, 2])  # sums to 9, N=10
    with pytest.raises(ValueError, match=r"seglen_a must sum to N = a.shape\[0\] = 10; got sum 9"):
        segment_mm(a, b, bad_seglen)


def test_gather_mm_rejects_non_tensor():
    a, b, idx_b = _valid_gather_inputs()
    with pytest.raises(ValueError, match="instances of torch.Tensor"):
        gather_mm(a, b, [0, 2, 1, 0, 2])


def test_gather_mm_rejects_a_wrong_rank():
    a, b, idx_b = _valid_gather_inputs()
    with pytest.raises(ValueError, match=r"a must be a matrix with shape \(N, D1\); got shape \(20,\)"):
        gather_mm(a.reshape(20), b, idx_b)


def test_gather_mm_rejects_b_wrong_rank():
    a, b, idx_b = _valid_gather_inputs()
    with pytest.raises(ValueError, match=r"b must be a batched matrix with shape \(R, D1, D2\); got shape \(3, 8\)"):
        gather_mm(a, b.reshape(3, 8), idx_b)


def test_gather_mm_rejects_idx_wrong_rank():
    a, b, idx_b = _valid_gather_inputs()
    with pytest.raises(ValueError, match=r"idx_b must be a vector with shape \(N,\); got shape \(1, 5\)"):
        gather_mm(a, b, idx_b.unsqueeze(0))


def test_gather_mm_rejects_non_integer_idx():
    a, b, idx_b = _valid_gather_inputs()
    with pytest.raises(ValueError, match=r"idx_b must be a vector with shape \(N,\) of integer dtype"):
        gather_mm(a, b, idx_b.to(torch.float32))


def test_gather_mm_rejects_idx_length_mismatch():
    a, b, idx_b = _valid_gather_inputs()
    bad_idx = torch.tensor([0, 1, 2])  # length 3, N=5
    with pytest.raises(ValueError, match=r"idx_b must be a vector with shape \(N,\) with N = a.shape\[0\] = 5"):
        gather_mm(a, b, bad_idx)


def test_gather_mm_rejects_mismatched_d1():
    a, b, idx_b = _valid_gather_inputs()
    bad_b = torch.randn(3, 5, 2)  # D1=5, a has D1=4
    with pytest.raises(ValueError, match=r"a and b must share the inner dimension D1"):
        gather_mm(a, bad_b, idx_b)


def test_error_message_states_accepted_and_received_shape():
    """naming.md §1 template: "state the accepted logical forms and the
    received shape" -- spot-check one message contains both."""
    a, b, seglen_a = _valid_segment_inputs()
    with pytest.raises(ValueError) as exc_info:
        segment_mm(a.reshape(2, 5, 4), b, seglen_a)
    message = str(exc_info.value)
    assert "(N, D1)" in message  # accepted logical form
    assert "(2, 5, 4)" in message  # received shape


# ---------------------------------------------------------------------------
# Kernel-side contract + workspace -- CUDA-gated.
# ---------------------------------------------------------------------------


@requires_cuda_backend
def test_sanity_valid_calls_do_not_raise():
    a = torch.randn(6, 4, device="cuda")
    b = torch.randn(3, 4, 2, device="cuda")
    segment_mm(a, b, torch.tensor([2, 3, 1], device="cuda"))
    gather_mm(a, b, torch.tensor([2, 0, 1, 1, 0, 2], device="cuda"))


@requires_cuda_backend
def test_workspace_bound_gather_mode():
    """Workspace assertion (testing.md/benchmarks.md): the gather-mode
    kernel fuses the gather into the GEMM (map.md: "gather fused into
    grouped GEMM") -- it must NOT materialise the gathered ``b[idx]`` stack
    (N x D1 x D2 elements). Peak minus tensor I/O stays under an
    allocator-granularity floor far below that counterfactual."""
    torch.manual_seed(0)
    N, R, D1, D2 = 4096, 8, 64, 64
    a = torch.randn(N, D1, device="cuda")
    b = torch.randn(R, D1, D2, device="cuda")
    idx = torch.randint(0, R, (N,), dtype=torch.int64, device="cuda")

    io_bytes = (
        a.numel() * a.element_size()
        + b.numel() * b.element_size()
        + idx.numel() * idx.element_size()
        + N * D2 * a.element_size()  # output
    )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = torch.ops.tsgu.grouped_gemm(a, b, idx, R, False)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    workspace = max(0, peak - io_bytes)
    # 16MB floor absorbs CUDA caching-allocator block granularity (same
    # rationale as test_gate6_contract_spmm.py's workspace test); the
    # counterfactual a materialised gather would allocate is
    # N * D1 * D2 * 4 bytes = 64MB here, so a real materialisation bug
    # still fails loudly.
    bound = 16 * 1024 * 1024
    counterfactual = N * D1 * D2 * a.element_size()
    assert bound < counterfactual, "test bug: bound should stay well below the gathered-stack counterfactual"
    assert workspace <= bound, f"workspace {workspace} bytes exceeds bound {bound} bytes"
    assert out.shape == (N, D2)


@requires_cuda_backend
def test_workspace_bound_scatter_mode():
    """Scatter-reduce mode: ordered per-group accumulation (sorted idx, no
    atomics) must not materialise the N per-row outer products
    (N x D1 x D2 elements) before reducing."""
    torch.manual_seed(1)
    N, R, D1, D2 = 4096, 8, 64, 64
    a = torch.randn(N, D1, device="cuda")
    b = torch.randn(N, D2, device="cuda")
    idx, _ = torch.sort(torch.randint(0, R, (N,), dtype=torch.int64, device="cuda"))

    io_bytes = (
        a.numel() * a.element_size()
        + b.numel() * b.element_size()
        + idx.numel() * idx.element_size()
        + R * D1 * D2 * a.element_size()  # output
    )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    out = torch.ops.tsgu.grouped_gemm(a, b, idx, R, True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    workspace = max(0, peak - io_bytes)
    bound = 16 * 1024 * 1024
    counterfactual = N * D1 * D2 * a.element_size()
    assert bound < counterfactual, "test bug: bound should stay well below the outer-product counterfactual"
    assert workspace <= bound, f"workspace {workspace} bytes exceeds bound {bound} bytes"
    assert out.shape == (R, D1, D2)
