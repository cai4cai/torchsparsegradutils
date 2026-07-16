"""Gate 4 — gradcheck for tsgu::sddmm via a probe op (spec/testing.md "Gates
& ordering": "gradcheck / gradgradcheck (f64)"; spec/commit.md Phase 3
commit 14).

``tsgu::sddmm`` itself gets no ``register_autograd`` in this commit
(ops/matmul.py's commit-9 note: it is only ever used as a backward
primitive by other ops, never differentiated through directly -- its own
higher-order-gradient support is "decided at its own kernel commit", i.e.
here). ``torch.autograd.gradcheck`` needs *some* backward to check against
a numerical Jacobian, so this module wraps the real kernel call in a small,
test-local ``torch.autograd.Function`` (:class:`_SddmmProbe`) whose
``backward`` is a hand-derived, dense-torch-ops implementation of the
sampled dot product's own adjoint (test-only code, never shipped) --
*not* another call into the kernel. gradcheck perturbs ``g``/``mat`` and
re-invokes the probe's ``forward`` (the real CUDA kernel) each time, so a
passing gradcheck proves the **forward** kernel's output is numerically
consistent with the analytic adjoint anyone differentiating through this
sampling pattern would write -- exactly what future commits (15-17) need
when they call ``tsgu::sddmm`` from inside their own ``register_autograd``.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from tests._sddmm_helpers import make_batched_pattern, random_masks
from torchsparsegradutils._dispatch import backend_available

pytestmark = pytest.mark.gate4

_SKIP_REASON = (
    "tsgu::sddmm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)


class _SddmmProbe(torch.autograd.Function):
    """Test-only autograd wrapper around ``torch.ops.tsgu.sddmm``. The
    backward below is a from-scratch, dense-torch-ops derivation of
    ``out[k] = sign * dot(g[row(k), :], mat[col(k), :])``'s adjoint w.r.t.
    ``g``/``mat`` -- it does not call back into the kernel or reuse any of
    its logic, so a gradcheck pass is not circular."""

    @staticmethod
    def forward(ctx, rowptr, col, g, mat, B, n, m, negate):
        ctx.save_for_backward(rowptr, col, g, mat)
        ctx.B, ctx.n, ctx.m, ctx.negate = B, n, m, negate
        return torch.ops.tsgu.sddmm(rowptr, col, g, mat, B, n, m, negate)

    @staticmethod
    def backward(ctx, grad_output):
        rowptr, col, g, mat = ctx.saved_tensors
        B, n, m, negate = ctx.B, ctx.n, ctx.m, ctx.negate
        sign = -1.0 if negate else 1.0

        row_g = torch.repeat_interleave(
            torch.arange(B * n, device=rowptr.device, dtype=torch.int64), (rowptr[1:] - rowptr[:-1]).long()
        )
        batch = row_g // n
        local_row = row_g % n
        col_long = col.long()

        grad_g = None
        if ctx.needs_input_grad[2]:
            contrib = sign * grad_output.unsqueeze(-1) * mat[batch, col_long, :]
            idx = batch * n + local_row
            grad_g_flat = torch.zeros(B * n, g.shape[-1], dtype=g.dtype, device=g.device)
            grad_g_flat.index_add_(0, idx, contrib)
            grad_g = grad_g_flat.reshape(B, n, -1)

        grad_mat = None
        if ctx.needs_input_grad[3]:
            contrib2 = sign * grad_output.unsqueeze(-1) * g[batch, local_row, :]
            idx2 = batch * m + col_long
            grad_mat_flat = torch.zeros(B * m, mat.shape[-1], dtype=mat.dtype, device=mat.device)
            grad_mat_flat.index_add_(0, idx2, contrib2)
            grad_mat = grad_mat_flat.reshape(B, m, -1)

        return None, None, grad_g, grad_mat, None, None, None, None


@requires_cuda_backend
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("p", [1, 4, 16])
@pytest.mark.parametrize("negate", [False, True])
def test_gradcheck_sddmm_unbatched(index_dtype, p, negate):
    gen = torch.Generator().manual_seed(7)
    n, m = 5, 4
    masks = random_masks(1, n, m, density=0.5, generator=gen)
    rowptr, col, B, n_, m_ = make_batched_pattern(masks, index_dtype, device="cuda")

    g = torch.randn(B, n_, p, dtype=torch.float64, device="cuda", requires_grad=True)
    mat = torch.randn(B, m_, p, dtype=torch.float64, device="cuda", requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda g_, mat_: _SddmmProbe.apply(rowptr, col, g_, mat_, B, n_, m_, negate),
        (g, mat),
    )


@requires_cuda_backend
@pytest.mark.parametrize("negate", [False, True])
def test_gradcheck_sddmm_batched_ragged(negate):
    gen = torch.Generator().manual_seed(8)
    B, n, m, p = 3, 4, 5, 6
    masks = random_masks(B, n, m, density=0.5, generator=gen)
    masks[0] = (torch.rand(n, m, generator=gen) < 0.2).to(torch.float32)
    rowptr, col, B_, n_, m_ = make_batched_pattern(masks, torch.int64, device="cuda")

    g = torch.randn(B_, n_, p, dtype=torch.float64, device="cuda", requires_grad=True)
    mat = torch.randn(B_, m_, p, dtype=torch.float64, device="cuda", requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda g_, mat_: _SddmmProbe.apply(rowptr, col, g_, mat_, B_, n_, m_, negate),
        (g, mat),
    )


@requires_cuda_backend
def test_gradcheck_sddmm_only_g_requires_grad():
    gen = torch.Generator().manual_seed(9)
    n, m, p = 4, 4, 3
    masks = random_masks(1, n, m, density=0.5, generator=gen)
    rowptr, col, B, n_, m_ = make_batched_pattern(masks, torch.int64, device="cuda")

    g = torch.randn(B, n_, p, dtype=torch.float64, device="cuda", requires_grad=True)
    mat = torch.randn(B, m_, p, dtype=torch.float64, device="cuda", requires_grad=False)

    assert torch.autograd.gradcheck(
        lambda g_: _SddmmProbe.apply(rowptr, col, g_, mat, B, n_, m_, False),
        (g,),
    )
