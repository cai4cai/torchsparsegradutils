"""Gate 3 — parity for ``sparse_mm`` (the public wrapper, spec/commit.md
Phase 3 commit 15) against BOTH oracles (spec/testing.md "Gates & ordering":
"parity (Oracle A + B, full matrix)").

Oracle A (tests/oracle/sparse_matmul.py, the pre-rewrite pure-PyTorch
implementation): forward + both gradients, unbatched and batched (equal nse
per item, since Oracle A's own batching goes through ``sparse_block_diag``,
which does not require equal nse but ``rand_sparse`` always generates equal
nse per item).

Oracle B (tests/_spmm_helpers.dense_reference): independent dense fp64
reference, covering the parts of the variant matrix Oracle A's own
``rand_sparse``-based construction cannot reach -- ragged nse per batch item
and an empty batch item (naming.md §1) -- plus a p sweep including p=1.

``tests/test_sparse_matmul.py`` already covers the public API surface (error
messages, conditional-gradient cases, memory advantage, optimizer-step
loop, double-backward) and stays green through this commit's wrapper switch
-- this module adds only what that suite doesn't: the full oracle variant
matrix (both oracles, both layouts, both index dtypes, both value dtypes,
ragged nse, empty batch items, p sweep incl. p=1).
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from tests._spmm_helpers import csr_to_sparse_tensor, dense_reference, make_batched_csr, random_masks
from tests._tolerances import assert_close
from torchsparsegradutils import sparse_mm
from torchsparsegradutils._dispatch import backend_available
from torchsparsegradutils.utils import rand_sparse

pytestmark = pytest.mark.gate3

_SKIP_REASON = (
    "tsgu::spmm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)

INDEX_DTYPES = (torch.int32, torch.int64)
VALUE_DTYPES = (torch.float32, torch.float64)
LAYOUTS = (torch.sparse_coo, torch.sparse_csr)
P_SWEEP = (1, 4, 16, 33)


# ---------------------------------------------------------------------------
# Oracle A -- forward + both gradients, unbatched and batched (equal nse).
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_a_unbatched(layout, index_dtype, value_dtype):
    from tests.oracle.sparse_matmul import sparse_mm as oracle_sparse_mm

    n, m, p, nnz = 8, 6, 5, 12
    A_ref = rand_sparse((n, m), nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device="cuda")
    B_ref = torch.randn(m, p, dtype=value_dtype, device="cuda")
    A_ours = A_ref.detach().clone().requires_grad_(True)
    B_ours = B_ref.detach().clone().requires_grad_(True)
    A_ref = A_ref.detach().clone().requires_grad_(True)
    B_ref = B_ref.detach().clone().requires_grad_(True)

    out_ours = sparse_mm(A_ours, B_ours)
    out_ref = oracle_sparse_mm(A_ref, B_ref)
    assert_close(out_ours, out_ref, reduction_length=max(1, A_ours.shape[-1]))

    gout = torch.randn_like(out_ours)
    out_ours.backward(gout)
    out_ref.backward(gout)

    assert A_ours.grad.layout == layout
    assert_close(A_ours.grad.to_dense(), A_ref.grad.to_dense(), reduction_length=max(1, p))
    assert_close(B_ours.grad, B_ref.grad, reduction_length=max(1, n))


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_a_batched(layout, index_dtype, value_dtype):
    from tests.oracle.sparse_matmul import sparse_mm as oracle_sparse_mm

    B, n, m, p, nnz = 4, 7, 5, 6, 10
    A_ref = rand_sparse((B, n, m), nnz, layout, indices_dtype=index_dtype, values_dtype=value_dtype, device="cuda")
    B_ref = torch.randn(B, m, p, dtype=value_dtype, device="cuda")
    A_ours = A_ref.detach().clone().requires_grad_(True)
    B_ours = B_ref.detach().clone().requires_grad_(True)
    A_ref = A_ref.detach().clone().requires_grad_(True)
    B_ref = B_ref.detach().clone().requires_grad_(True)

    out_ours = sparse_mm(A_ours, B_ours)
    out_ref = oracle_sparse_mm(A_ref, B_ref)
    assert_close(out_ours, out_ref, reduction_length=max(1, m))

    gout = torch.randn_like(out_ours)
    out_ours.backward(gout)
    out_ref.backward(gout)

    assert A_ours.grad.layout == layout
    assert_close(A_ours.grad.to_dense(), A_ref.grad.to_dense(), reduction_length=max(1, p))
    assert_close(B_ours.grad, B_ref.grad, reduction_length=max(1, n))


# ---------------------------------------------------------------------------
# Oracle B -- p sweep (incl. p=1), unbatched, both layouts. Both forward and
# both gradients, all checked against manually-derived fp64 reference
# formulas (no second autograd pass on a shared leaf -- calling .backward()
# twice on the same `dense` leaf would accumulate, not compare).
# ---------------------------------------------------------------------------


def _manual_grad_refs(rowptr, col, vals, dense, gout, B, n, m, *, dtype=torch.float64):
    """gradA (dense, at A's own pattern) and gradB analytic adjoints, computed
    directly in fp64 -- Oracle B's own reference formulas, independent of
    autograd and of tsgu::spmm/tsgu::sddmm."""
    row_g = torch.repeat_interleave(
        torch.arange(B * n, device=rowptr.device, dtype=torch.int64), (rowptr[1:] - rowptr[:-1]).long()
    )
    batch = row_g // n
    local_row = row_g % n
    dense64 = dense.to(dtype)
    gout64 = gout.to(dtype)

    grad_a_dense = torch.zeros(B, n, m, dtype=dtype, device=rowptr.device)
    if vals.numel() > 0:
        full = torch.einsum("bnp,bmp->bnm", gout64, dense64)
        grad_a_dense[batch, local_row, col.long()] = full[batch, local_row, col.long()]

    dense_pattern = torch.zeros(B, n, m, dtype=dtype, device=rowptr.device)
    if vals.numel() > 0:
        dense_pattern[batch, local_row, col.long()] = vals.to(dtype)
    grad_b = torch.einsum("bnm,bnp->bmp", dense_pattern, gout64)
    return grad_a_dense, grad_b


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("p", P_SWEEP)
def test_oracle_b_unbatched_p_sweep(layout, index_dtype, value_dtype, p):
    gen = torch.Generator().manual_seed(0)
    n, m = 9, 7
    masks = random_masks(1, n, m, density=0.4, generator=gen)
    rowptr, col, vals, B_, n_, m_ = make_batched_csr(masks, index_dtype, value_dtype, device="cuda", generator=gen)
    A = csr_to_sparse_tensor(rowptr, col, vals, B_, n_, m_, layout=layout).requires_grad_(True)
    dense = torch.randn(m_, p, dtype=value_dtype, device="cuda", requires_grad=True)

    out = sparse_mm(A, dense)
    ref = dense_reference(rowptr, col, vals, dense.detach().unsqueeze(0), B_, n_, m_).squeeze(0)
    assert_close(out, ref, reduction_length=max(1, m_))

    gout = torch.randn_like(out)
    out.backward(gout)
    grad_a_ref, grad_b_ref = _manual_grad_refs(
        rowptr, col, vals, dense.detach().unsqueeze(0), gout.unsqueeze(0), B_, n_, m_
    )
    assert_close(A.grad.to_dense(), grad_a_ref.squeeze(0), reduction_length=max(1, p))
    assert_close(dense.grad, grad_b_ref.squeeze(0), reduction_length=max(1, n_))


# ---------------------------------------------------------------------------
# Oracle B -- batched, ragged nse per item + an empty batch item (COO only:
# batched CSR requires equal nse per item, naming.md §1).
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_b_batched_ragged_empty_item(index_dtype, value_dtype):
    gen = torch.Generator().manual_seed(1)
    B, n, m, p = 4, 5, 6, 8
    masks = random_masks(B, n, m, density=0.5, generator=gen, empty_item=True)
    masks[1] = (torch.rand(n, m, generator=gen) < 0.85).to(torch.float32)  # force genuine raggedness
    rowptr, col, vals, B_, n_, m_ = make_batched_csr(masks, index_dtype, value_dtype, device="cuda", generator=gen)
    A = csr_to_sparse_tensor(rowptr, col, vals, B_, n_, m_, layout=torch.sparse_coo).requires_grad_(True)
    dense = torch.randn(B_, m_, p, dtype=value_dtype, device="cuda", requires_grad=True)

    out = sparse_mm(A, dense)
    ref = dense_reference(rowptr, col, vals, dense.detach(), B_, n_, m_)
    assert_close(out, ref, reduction_length=max(1, m_))

    # Batch item 0 is entirely empty -- its output rows must be exactly zero.
    item0_nse = int(rowptr[n_].item()) - int(rowptr[0].item())
    assert item0_nse == 0
    assert torch.equal(out[0], torch.zeros_like(out[0]))

    gout = torch.randn_like(out)
    out.backward(gout)
    assert A.grad._nnz() == vals.numel()
    grad_a_ref, grad_b_ref = _manual_grad_refs(rowptr, col, vals, dense.detach(), gout, B_, n_, m_)
    assert_close(A.grad.to_dense(), grad_a_ref, reduction_length=max(1, p))
    assert_close(dense.grad, grad_b_ref, reduction_length=max(1, n_))
