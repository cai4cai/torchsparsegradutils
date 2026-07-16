"""Gate 3 — parity for ``sparse_triangular_solve`` (the public wrapper,
spec/commit.md Phase 3 commit 16) against BOTH oracles (spec/testing.md
"Gates & ordering": "parity (Oracle A + B, full matrix)").

Oracle A (tests/oracle/sparse_solve.py, the pre-rewrite pure-PyTorch
implementation via ``torch.triangular_solve``): forward + both gradients,
unbatched and batched.

Oracle B (tests/_spsm_helpers.dense_reference_solve): independent fp64
``torch.linalg.solve_triangular`` reference, covering the p sweep, ragged/
empty-item batching, and both upper/lower x unitriangular combinations.

Full matrix (per this commit's T5 instructions): COO+CSR x batched(incl.
ragged, empty item)/unbatched x f32/f64 x i32/i64 x p sweep x upper/lower x
unitriangular on/off; fwd + both grads.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from tests._spsm_helpers import (
    bidiagonal_chain_csr,
    csr_to_sparse_tensor,
    dense_reference_solve,
    make_batched_triangular_csr,
)
from tests._tolerances import assert_close
from torchsparsegradutils import sparse_triangular_solve
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate3

_SKIP_REASON = (
    "tsgu::spsm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)

INDEX_DTYPES = (torch.int32, torch.int64)
VALUE_DTYPES = (torch.float32, torch.float64)
LAYOUTS = (torch.sparse_coo, torch.sparse_csr)
P_SWEEP = (1, 4, 16, 33)


# ---------------------------------------------------------------------------
# Oracle A -- forward + both gradients, unbatched and batched.
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("unitriangular", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_oracle_a_unbatched(layout, index_dtype, value_dtype, upper, unitriangular, transpose):
    from tests.oracle.sparse_solve import sparse_triangular_solve as oracle_solve

    n, p = 10, 4
    rowptr, col, vals, B_, n_ = make_batched_triangular_csr(
        n, 1, index_dtype, value_dtype, "cuda", upper=upper, unitriangular=unitriangular, density=0.4
    )
    A_ref = csr_to_sparse_tensor(rowptr, col, vals, B_, n_, layout=layout).requires_grad_(True)
    B_ref = torch.randn(n, p, dtype=value_dtype, device="cuda")
    A_ours = A_ref.detach().clone().requires_grad_(True)
    B_ours = B_ref.detach().clone().requires_grad_(True)
    A_ref = A_ref.detach().clone().requires_grad_(True)
    B_ref = B_ref.detach().clone().requires_grad_(True)

    out_ours = sparse_triangular_solve(A_ours, B_ours, upper=upper, unitriangular=unitriangular, transpose=transpose)
    out_ref = oracle_solve(A_ref, B_ref, upper=upper, unitriangular=unitriangular, transpose=transpose)
    assert_close(out_ours, out_ref, reduction_length=max(1, n_))

    gout = torch.randn_like(out_ours)
    out_ours.backward(gout)
    out_ref.backward(gout)

    assert A_ours.grad.layout == layout
    assert_close(A_ours.grad.to_dense(), A_ref.grad.to_dense(), reduction_length=max(1, p))
    assert_close(B_ours.grad, B_ref.grad, reduction_length=max(1, n_))


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_a_batched(layout, index_dtype, value_dtype):
    from tests.oracle.sparse_solve import sparse_triangular_solve as oracle_solve

    B, n, p = 3, 8, 5
    rowptr, col, vals, B_, n_ = make_batched_triangular_csr(
        n, B, index_dtype, value_dtype, "cuda", upper=True, unitriangular=False, density=0.35
    )
    A_ref = csr_to_sparse_tensor(rowptr, col, vals, B_, n_, layout=layout).requires_grad_(True)
    B_ref = torch.randn(B_, n_, p, dtype=value_dtype, device="cuda")
    A_ours = A_ref.detach().clone().requires_grad_(True)
    B_ours = B_ref.detach().clone().requires_grad_(True)
    A_ref = A_ref.detach().clone().requires_grad_(True)
    B_ref = B_ref.detach().clone().requires_grad_(True)

    out_ours = sparse_triangular_solve(A_ours, B_ours, upper=True, unitriangular=False, transpose=False)
    out_ref = oracle_solve(A_ref, B_ref, upper=True, unitriangular=False, transpose=False)
    assert_close(out_ours, out_ref, reduction_length=max(1, n_))

    gout = torch.randn_like(out_ours)
    out_ours.backward(gout)
    out_ref.backward(gout)

    assert A_ours.grad.layout == layout
    assert_close(A_ours.grad.to_dense(), A_ref.grad.to_dense(), reduction_length=max(1, p))
    assert_close(B_ours.grad, B_ref.grad, reduction_length=max(1, n_))


# ---------------------------------------------------------------------------
# Oracle B -- p sweep, both layouts, upper/lower x unitriangular, transpose.
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("p", P_SWEEP)
def test_oracle_b_unbatched_p_sweep(layout, index_dtype, value_dtype, p):
    n = 9
    rowptr, col, vals, B_, n_ = make_batched_triangular_csr(
        n, 1, index_dtype, value_dtype, "cuda", upper=True, unitriangular=False, density=0.4
    )
    A = csr_to_sparse_tensor(rowptr, col, vals, B_, n_, layout=layout).requires_grad_(True)
    dense = torch.randn(n_, p, dtype=value_dtype, device="cuda", requires_grad=True)

    out = sparse_triangular_solve(A, dense, upper=True, unitriangular=False, transpose=False)
    ref = dense_reference_solve(
        rowptr, col, vals, dense.detach().unsqueeze(0), B_, n_, upper=True, unitriangular=False, transpose=False
    ).squeeze(0)
    assert_close(out, ref, reduction_length=max(1, n_))


@requires_cuda_backend
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("unitriangular", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_oracle_b_upper_lower_unitriangular_transpose(upper, unitriangular, transpose):
    n, p = 11, 3
    rowptr, col, vals, B_, n_ = make_batched_triangular_csr(
        n, 1, torch.int64, torch.float64, "cuda", upper=upper, unitriangular=unitriangular, density=0.35
    )
    A = csr_to_sparse_tensor(rowptr, col, vals, B_, n_, layout=torch.sparse_csr).requires_grad_(True)
    dense = torch.randn(n_, p, dtype=torch.float64, device="cuda", requires_grad=True)

    out = sparse_triangular_solve(A, dense, upper=upper, unitriangular=unitriangular, transpose=transpose)
    ref = dense_reference_solve(
        rowptr,
        col,
        vals,
        dense.detach().unsqueeze(0),
        B_,
        n_,
        upper=upper,
        unitriangular=unitriangular,
        transpose=transpose,
    ).squeeze(0)
    assert_close(out, ref, reduction_length=max(1, n_))

    gout = torch.randn_like(out)
    out.backward(gout)
    assert A.grad._nnz() == vals.numel()


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_b_batched_ragged(index_dtype, value_dtype):
    """Ragged nse per batch item -- COO only (batched CSR requires equal nse
    per item, naming.md §1). Built by concatenating two independently-sized
    triangular CSR patterns."""
    n, p = 6, 3
    gen = torch.Generator().manual_seed(3)
    rowptr0, col0, vals0, _, _ = make_batched_triangular_csr(
        n, 1, index_dtype, value_dtype, "cuda", upper=True, unitriangular=False, density=0.25, generator=gen
    )
    rowptr1, col1, vals1, _, _ = make_batched_triangular_csr(
        n, 1, index_dtype, value_dtype, "cuda", upper=True, unitriangular=False, density=0.6, generator=gen
    )
    nse0 = int(rowptr0[-1].item())
    rowptr1_off = rowptr1 + nse0
    rowptr = torch.cat([rowptr0, rowptr1_off[1:]])
    col = torch.cat([col0, col1])
    vals = torch.cat([vals0, vals1])
    B_, n_ = 2, n

    A = csr_to_sparse_tensor(rowptr, col, vals, B_, n_, layout=torch.sparse_coo).requires_grad_(True)
    dense = torch.randn(B_, n_, p, dtype=value_dtype, device="cuda", requires_grad=True)

    out = sparse_triangular_solve(A, dense, upper=True, unitriangular=False, transpose=False)
    ref = dense_reference_solve(
        rowptr, col, vals, dense.detach(), B_, n_, upper=True, unitriangular=False, transpose=False
    )
    assert_close(out, ref, reduction_length=max(1, n_))

    gout = torch.randn_like(out)
    out.backward(gout)
    assert A.grad._nnz() == vals.numel()


@requires_cuda_backend
def test_deep_dependency_chain_bidiagonal():
    """Adversarial (testing.md/this commit's brief): a bidiagonal pattern is
    the worst case for the level schedule (n levels, one row each) -- still
    must be numerically correct, just exercised via many small kernel
    launches (spsm.cu's own module comment)."""
    n = 40
    rowptr, col, vals, B_, n_ = bidiagonal_chain_csr(n, torch.int64, torch.float64, "cuda", upper=False)
    A = csr_to_sparse_tensor(rowptr, col, vals, B_, n_, layout=torch.sparse_csr)
    rhs = torch.randn(n_, 2, dtype=torch.float64, device="cuda")
    out = sparse_triangular_solve(A, rhs, upper=False, unitriangular=False, transpose=False)
    ref = dense_reference_solve(
        rowptr, col, vals, rhs.unsqueeze(0), B_, n_, upper=False, unitriangular=False, transpose=False
    ).squeeze(0)
    assert_close(out, ref, reduction_length=max(1, n_))
