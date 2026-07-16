"""Gate 4 — gradcheck for ``sparse_mm`` (the public wrapper, spec/testing.md
"Gates & ordering": "gradcheck / gradgradcheck (f64)"; spec/commit.md Phase
3 commit 15).

Unlike tsgu::sddmm (commit 14, tested via a test-local probe Function since
it had no ``register_autograd`` of its own yet), ``tsgu::spmm`` already has a
real ``register_autograd`` (torchsparsegradutils/ops/matmul.py, commit 9),
and the public ``sparse_mm`` wrapper composes it with PyTorch's own
sparse-tensor ``values()`` adjoint (no custom ``torch.autograd.Function`` at
the wrapper level at all -- see ``_tsgu_sparse_mm``'s docstring). So
``torch.autograd.gradcheck`` runs directly on ``sparse_mm`` itself, perturbing
the sparse tensor's ``.values()`` (via ``torch.autograd.gradcheck``'s own
sparse-tensor support) and the dense operand.

Both layouts (COO, CSR), unbatched and batched, per this commit's own T5
instructions.

``check_undefined_grad=False`` on every A-requires-grad call: gradcheck's
undefined-grad subtest calls ``.eq(0).all()`` directly on the returned
gradient to confirm an "undefined" incoming grad_output produces an
all-zero/None gradient. ``torch.library.custom_op``'s Python-level
``register_autograd`` backward (unlike an ATen-native C++ derivative
formula) never sees a true "undefined" grad_output -- it always receives a
materialized tensor -- so it legitimately computes a real (all-zero-valued)
*sparse* gradient for A, and ``aten::eq.Scalar`` has no SparseCUDA kernel,
crashing that subtest with a NotImplementedError unrelated to the adjoint
formula gradcheck exists to verify (confirmed by direct comparison: the same
call with ``check_undefined_grad=False`` passes cleanly, and the default
Jacobian check -- the actual adjoint-correctness proof -- is untouched by
this flag).
"""

from __future__ import annotations

import pytest
import torch

import torchsparsegradutils  # noqa: F401  (import side-effect: registers tsgu:: ops)
from torchsparsegradutils import sparse_mm
from torchsparsegradutils._dispatch import backend_available
from torchsparsegradutils.utils import rand_sparse

pytestmark = pytest.mark.gate4

_SKIP_REASON = (
    "tsgu::spmm has no CPU implementation (CUDA-only, architecture.md §4) -- "
    "needs both a CUDA device and a loaded, version-matched backend."
)
requires_cuda_backend = pytest.mark.skipif(not (torch.cuda.is_available() and backend_available()), reason=_SKIP_REASON)

LAYOUTS = (torch.sparse_coo, torch.sparse_csr)


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
def test_gradcheck_sparse_mm_unbatched(layout):
    n, m, p, nnz = 5, 4, 3, 8
    A = rand_sparse((n, m), nnz, layout, values_dtype=torch.float64, device="cuda").requires_grad_(True)
    B = torch.randn(m, p, dtype=torch.float64, device="cuda", requires_grad=True)

    assert torch.autograd.gradcheck(lambda a, b: sparse_mm(a, b), (A, B), masked=True, check_undefined_grad=False)


@requires_cuda_backend
@pytest.mark.parametrize(
    "layout",
    [
        torch.sparse_coo,
        pytest.param(
            torch.sparse_csr,
            marks=pytest.mark.skip(
                reason=(
                    "torch.autograd.gradcheck._iter_tensor has its own bug for batched "
                    "(3D) sparse CSR tensors, unrelated to tsgu::spmm: it sets "
                    "x_nnz = x_tensor._nnz() (naming.md/PyTorch semantics: nnz PER BATCH "
                    "ITEM for CSR) but then indexes x_tensor.values()[i] for i in "
                    "range(x_nnz) -- values() for a batched CSR tensor has shape "
                    "(B, nnz_per_item), so that loop indexes along the BATCH axis and "
                    "raises IndexError once nnz_per_item > B. Reproduced with gradcheck "
                    "on `lambda a: a.to_dense()` for a bare batched-CSR leaf -- no "
                    "sparse_mm, no tsgu:: op involved at all -- confirming this is a "
                    "PyTorch limitation, not a bug in this kernel. Batched-CSR backward "
                    "correctness is still verified analytically in "
                    "tests/test_gate3_parity_spmm.py's test_oracle_a_batched and "
                    "test_oracle_b_batched_ragged_empty_item (both layouts); the batched "
                    "COO case above exercises gradcheck's numeric Jacobian for the batched "
                    "path directly."
                )
            ),
        ),
    ],
)
def test_gradcheck_sparse_mm_batched(layout):
    B_, n, m, p, nnz = 3, 4, 5, 3, 6
    A = rand_sparse((B_, n, m), nnz, layout, values_dtype=torch.float64, device="cuda").requires_grad_(True)
    Bd = torch.randn(B_, m, p, dtype=torch.float64, device="cuda", requires_grad=True)

    assert torch.autograd.gradcheck(lambda a, b: sparse_mm(a, b), (A, Bd), masked=True, check_undefined_grad=False)


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
def test_gradcheck_sparse_mm_only_dense_requires_grad(layout):
    n, m, p, nnz = 5, 4, 3, 8
    A = rand_sparse((n, m), nnz, layout, values_dtype=torch.float64, device="cuda")
    B = torch.randn(m, p, dtype=torch.float64, device="cuda", requires_grad=True)

    assert torch.autograd.gradcheck(lambda b: sparse_mm(A, b), (B,))


@requires_cuda_backend
@pytest.mark.parametrize("layout", LAYOUTS)
def test_gradcheck_sparse_mm_p_equals_one(layout):
    """SpMV shape through the public wrapper (kernels.md: "spmv = spmm with
    p = 1, no separate op")."""
    n, m, nnz = 6, 5, 8
    A = rand_sparse((n, m), nnz, layout, values_dtype=torch.float64, device="cuda").requires_grad_(True)
    B = torch.randn(m, 1, dtype=torch.float64, device="cuda", requires_grad=True)

    assert torch.autograd.gradcheck(lambda a, b: sparse_mm(a, b), (A, B), masked=True, check_undefined_grad=False)
