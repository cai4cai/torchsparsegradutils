"""Gate 4 — gradcheck + gradgradcheck for ``sparse_triangular_solve`` (the
public wrapper, spec/testing.md "Gates & ordering": "gradcheck /
gradgradcheck (f64)"; spec/commit.md Phase 3 commit 16).

Gradcheck: ``tsgu::spsm`` already has a real ``register_autograd``
(torchsparsegradutils/ops/triangular_solve.py, commit 9; transpose-indexing
resolved commit 16), composed with PyTorch's own sparse-tensor ``values()``
adjoint at the wrapper boundary (same pattern as ``sparse_mm``'s
``test_gate4_gradcheck_spmm.py``) -- ``torch.autograd.gradcheck`` runs
directly on ``sparse_triangular_solve``.

Gradgradcheck (PR #85: "solve ops support higher-order" -- verified here,
not assumed):

- **w.r.t. B (dense) alone: SUPPORTED and verified below.** ``gradB`` is
  computed by calling ``torch.ops.tsgu.spsm`` again (module docstring:
  "the same op with `transpose` flipped") -- itself a ``register_autograd``
  op, so this recursive call stays fully in the autograd graph to any
  order.
- **w.r.t. A's sparse values: NOT supported, same as the pre-existing
  ``sparse_mm`` limitation (commit 15) -- documented, not silently
  papered over.** ``gradA`` is computed via ``torch.ops.tsgu.sddmm``
  (negate epilogue), and ``tsgu::sddmm`` has **no ``register_autograd``**
  (torchsparsegradutils/ops/matmul.py, commit 9's own note: "in this
  commit it is only ever used as a backward primitive... its own
  higher-order-gradient support is decided at its own kernel commit" --
  commit 14 shipped ``tsgu::sddmm`` without one, and no later commit
  (15, 16) has added it; that is Family-1's own decision to make, not
  this commit's). The practical effect, verified directly below via
  ``create_graph=True`` (``torch.autograd.gradgradcheck`` itself cannot
  even run this comparison -- PyTorch's gradcheck machinery raises
  "Sparse output is not supported at gradcheck yet" on the sparse gradA
  output before ever reaching the second-order check): the first-order
  ``gradA`` tensor comes back with ``requires_grad=False`` (no
  ``grad_fn``), so ``torch.autograd.grad`` on any expression combining it
  cleanly raises ``"...does not require grad and does not have a
  grad_fn"`` -- a clear, catchable failure, never a silently wrong zero
  second derivative.

  The **old (Oracle A) implementation had the identical limitation for a
  different reason**: ``SparseTriangularSolve.forward`` calls
  ``A.detach()``/``B.detach()`` before the solve and only ``save_for_backward``s
  the *detached* ``A``/``x`` -- so any use of those saved tensors inside
  ``backward`` (index_select, elementwise ops) has no graph connection back
  to the original ``A``/``B`` either, and double-backward through the
  legacy path was never possible. This commit does not regress anything
  that used to work; it inherits a documented, pre-existing gap.
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils import sparse_triangular_solve
from torchsparsegradutils._dispatch import backend_available
from torchsparsegradutils.utils import rand_sparse_tri

pytestmark = pytest.mark.gate4

_SKIP_REASON = (
    "tsgu::spsm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)

LAYOUTS = (torch.sparse_coo, torch.sparse_csr)


# ---------------------------------------------------------------------------
# gradcheck (first order) -- full variant sweep.
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("unitriangular", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_gradcheck_sparse_triangular_solve_unbatched(layout, upper, unitriangular, transpose):
    n, p, nnz = 7, 3, 12
    A = rand_sparse_tri(
        (n, n), nnz, layout, upper=upper, strict=unitriangular, values_dtype=torch.float64, device="cuda"
    ).requires_grad_(True)
    B = torch.randn(n, p, dtype=torch.float64, device="cuda", requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda a, b: sparse_triangular_solve(a, b, upper=upper, unitriangular=unitriangular, transpose=transpose),
        (A, B),
        masked=True,
        check_undefined_grad=False,
    )


@requires_cuda_backend
@pytest.mark.parametrize(
    "layout",
    [
        torch.sparse_coo,
        pytest.param(
            torch.sparse_csr,
            marks=pytest.mark.skip(
                reason=(
                    "torch.autograd.gradcheck._iter_tensor's batched-(3D)-CSR bug -- "
                    "identical to tests/test_gate4_gradcheck_spmm.py's documented skip, "
                    "unrelated to tsgu::spsm. Batched-CSR backward correctness is still "
                    "verified analytically in tests/test_gate3_parity_spsm.py."
                )
            ),
        ),
    ],
)
def test_gradcheck_sparse_triangular_solve_batched(layout):
    from tests._spsm_helpers import csr_to_sparse_tensor, make_batched_triangular_csr

    B_, n, p = 3, 5, 2
    rowptr, col, vals, B_, n_ = make_batched_triangular_csr(
        n, B_, torch.int64, torch.float64, "cuda", upper=True, unitriangular=False, density=0.4
    )
    A = csr_to_sparse_tensor(rowptr, col, vals, B_, n_, layout=layout).requires_grad_(True)
    Bd = torch.randn(B_, n_, p, dtype=torch.float64, device="cuda", requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda a, b: sparse_triangular_solve(a, b, upper=True, unitriangular=False, transpose=False),
        (A, Bd),
        masked=True,
        check_undefined_grad=False,
    )


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
def test_gradcheck_sparse_triangular_solve_only_dense_requires_grad(layout):
    n, p, nnz = 6, 2, 10
    A = rand_sparse_tri((n, n), nnz, layout, upper=True, strict=False, values_dtype=torch.float64, device="cuda")
    B = torch.randn(n, p, dtype=torch.float64, device="cuda", requires_grad=True)

    assert torch.autograd.gradcheck(lambda b: sparse_triangular_solve(A, b, upper=True), (B,))


# ---------------------------------------------------------------------------
# gradgradcheck (second order) -- w.r.t. B: supported, verified.
# ---------------------------------------------------------------------------


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("transpose", [True, False])
def test_gradgradcheck_sparse_triangular_solve_wrt_B_only(layout, upper, transpose):
    n, p, nnz = 6, 2, 10
    A = rand_sparse_tri((n, n), nnz, layout, upper=upper, strict=False, values_dtype=torch.float64, device="cuda")
    B = torch.randn(n, p, dtype=torch.float64, device="cuda", requires_grad=True)

    assert torch.autograd.gradgradcheck(lambda b: sparse_triangular_solve(A, b, upper=upper, transpose=transpose), (B,))


# ---------------------------------------------------------------------------
# gradgradcheck w.r.t. A's sparse values: documented NOT supported (module
# docstring) -- this test proves the failure is the EXPECTED, clean one
# (no grad_fn on gradA), not a silently wrong second derivative.
# ---------------------------------------------------------------------------


@requires_cuda_backend
def test_higher_order_wrt_A_values_cleanly_unsupported():
    n, p, nnz = 6, 2, 10
    A = rand_sparse_tri(
        (n, n), nnz, torch.sparse_coo, upper=True, strict=False, values_dtype=torch.float64, device="cuda"
    )
    coalesced = A.coalesce()
    theta = coalesced.values().clone().requires_grad_(True)
    indices = coalesced.indices()

    def make_A(t):
        return torch.sparse_coo_tensor(indices, t, A.shape).coalesce()

    B = torch.randn(n, p, dtype=torch.float64, device="cuda", requires_grad=True)
    x = sparse_triangular_solve(make_A(theta), B, upper=True)
    loss = x.square().sum()
    grad_theta, grad_B = torch.autograd.grad(loss, (theta, B), create_graph=True)

    # gradA (via tsgu::sddmm, no register_autograd -- module docstring) has
    # no graph connection back to theta; gradB (via the recursive tsgu::spsm
    # call, which DOES have register_autograd) does.
    assert grad_theta.requires_grad is False
    assert grad_B.requires_grad is True

    with pytest.raises(RuntimeError, match=r"does not require grad"):
        torch.autograd.grad(grad_theta.sum(), theta)

    # The B-only second derivative is unaffected by A's limitation.
    second_B = torch.autograd.grad(grad_B.sum(), B)[0]
    assert torch.isfinite(second_B).all()
