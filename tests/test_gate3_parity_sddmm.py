"""Gate 3 — parity for tsgu::sddmm against BOTH oracles (spec/testing.md
"Gates & ordering": "parity (Oracle A + B, full matrix)"; spec/commit.md
Phase 3 commit 14).

Oracle A (tests/oracle/sparse_matmul.py's ``SparseMatMul.backward``): the
pre-rewrite pure-PyTorch ``sparse_mm`` backward computes exactly this
sampled product for gradA (``dotprod(grad[i,:], b[j,:])`` at A's own
sparsity pattern) -- cross-checked here unbatched (CSR layout; the oracle's
own batching goes through ``sparse_block_diag``/``stack_csr``, out of scope
for a *pattern-alignment* check like this one, which cares about exact
per-entry order matching our own ``(rowptr, col)`` -- the unbatched case
already exercises the same per-entry math the batched kernel path reuses).

Oracle B (tests/_sddmm_helpers.dense_reference): independent dense fp64
reference, run across the **full** variant matrix -- unbatched/batched
(including ragged nse per item and an empty batch item), f32/f64, i32/i64,
a p sweep including p=1, negate on/off (this commit's own T5 instructions).
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from tests._sddmm_helpers import dense_reference, make_batched_pattern, random_masks
from tests._tolerances import assert_close
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate3

_SKIP_REASON = (
    "tsgu::sddmm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)

INDEX_DTYPES = (torch.int32, torch.int64)
VALUE_DTYPES = (torch.float32, torch.float64)
P_SWEEP = (1, 4, 16)


# ---------------------------------------------------------------------------
# Oracle A -- exact pattern-aligned cross-check, unbatched CSR.
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("p", P_SWEEP)
@pytest.mark.parametrize("negate", [False, True])
def test_oracle_a_sddmm_unbatched(index_dtype, value_dtype, p, negate):
    from tests.oracle.sparse_matmul import SparseMatMul

    torch.manual_seed(0)
    n, m = 6, 5
    dense_pattern = (torch.rand(n, m) < 0.4).to(torch.float64)
    if dense_pattern.sum() == 0:
        dense_pattern[0, 0] = 1.0
    csr = dense_pattern.to_sparse_csr()
    rowptr_cpu = csr.crow_indices()
    col_cpu = csr.col_indices()
    nse = col_cpu.numel()

    # Oracle A: values are irrelevant to the backward's gradA (it only reads
    # A's indices), so a placeholder values array is fine.
    A = torch.sparse_csr_tensor(rowptr_cpu, col_cpu, torch.zeros(nse, dtype=torch.float64), (n, m)).requires_grad_(True)
    B_dense = torch.randn(m, p, dtype=torch.float64)
    C = SparseMatMul.apply(A, B_dense)
    g_dense = torch.randn(n, p, dtype=torch.float64)
    (gradA_sparse,) = torch.autograd.grad(C, A, grad_outputs=g_dense)
    oracle_a = gradA_sparse.values()
    if negate:
        oracle_a = -oracle_a

    rowptr = rowptr_cpu.to(index_dtype).cuda()
    col = col_cpu.to(index_dtype).cuda()
    g = g_dense.to(value_dtype).unsqueeze(0).cuda()
    mat = B_dense.to(value_dtype).unsqueeze(0).cuda()
    ours = torch.ops.tsgu.sddmm(rowptr, col, g, mat, 1, n, m, negate).cpu().to(torch.float64)

    reduction_length = max(1, p)  # dot product over p terms
    assert_close(ours.to(value_dtype), oracle_a.to(value_dtype), reduction_length=reduction_length)


# ---------------------------------------------------------------------------
# Oracle B -- full variant matrix, dense fp64 reference.
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("p", P_SWEEP)
@pytest.mark.parametrize("negate", [False, True])
def test_oracle_b_sddmm_unbatched(index_dtype, value_dtype, p, negate):
    torch.manual_seed(1)
    gen = torch.Generator().manual_seed(1)
    n, m = 7, 6
    masks = random_masks(1, n, m, density=0.4, generator=gen)
    rowptr, col, B, n_, m_ = make_batched_pattern(masks, index_dtype, device="cuda")

    g = torch.randn(B, n_, p, dtype=value_dtype, device="cuda")
    mat = torch.randn(B, m_, p, dtype=value_dtype, device="cuda")

    ours = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B, n_, m_, negate)
    ref = dense_reference(rowptr, col, g, mat, B, n_, m_, negate)

    assert_close(ours, ref, reduction_length=max(1, p))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
@pytest.mark.parametrize("p", P_SWEEP)
@pytest.mark.parametrize("negate", [False, True])
def test_oracle_b_sddmm_batched_ragged(index_dtype, value_dtype, p, negate):
    """Batched, ragged nse per item (naming.md §1: "items may have unequal
    nse (even zero)")."""
    gen = torch.Generator().manual_seed(2)
    B, n, m = 4, 5, 6
    masks = random_masks(B, n, m, density=0.5, generator=gen)
    # Force genuine raggedness: item 0 sparser than item 1.
    masks[0] = (torch.rand(n, m, generator=gen) < 0.15).to(torch.float32)
    masks[1] = (torch.rand(n, m, generator=gen) < 0.85).to(torch.float32)
    rowptr, col, B_, n_, m_ = make_batched_pattern(masks, index_dtype, device="cuda")

    g = torch.randn(B_, n_, p, dtype=value_dtype, device="cuda")
    mat = torch.randn(B_, m_, p, dtype=value_dtype, device="cuda")

    ours = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B_, n_, m_, negate)
    ref = dense_reference(rowptr, col, g, mat, B_, n_, m_, negate)

    assert_close(ours, ref, reduction_length=max(1, p))


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("value_dtype", VALUE_DTYPES)
def test_oracle_b_sddmm_empty_batch_item(index_dtype, value_dtype):
    """testing.md's "empty batch items": one full batch item has zero
    specified entries (all-zero mask), the rest are populated."""
    gen = torch.Generator().manual_seed(3)
    B, n, m, p = 3, 4, 5, 8
    masks = random_masks(B, n, m, density=0.4, generator=gen, empty_item=True)
    rowptr, col, B_, n_, m_ = make_batched_pattern(masks, index_dtype, device="cuda")

    g = torch.randn(B_, n_, p, dtype=value_dtype, device="cuda")
    mat = torch.randn(B_, m_, p, dtype=value_dtype, device="cuda")

    for negate in (False, True):
        ours = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B_, n_, m_, negate)
        ref = dense_reference(rowptr, col, g, mat, B_, n_, m_, negate)
        assert_close(ours, ref, reduction_length=max(1, p))

    # The empty batch item (0) must contribute exactly zero entries to nse_total.
    item0_nse = int(rowptr[n_].item()) - int(rowptr[0].item())
    assert item0_nse == 0
