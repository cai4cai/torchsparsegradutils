from typing import cast

import torch
from torch import Tensor

from torchsparsegradutils.utils import convert_coo_to_csr, sparse_block_diag, sparse_block_diag_split, stack_csr

# ---------------------------------------------------------------------------
# tsgu::spsm — op schema, fake kernel, autograd registration.
# spec/commit.md Phase 1 #9; routing verbatim from spec/map.md "Kernel
# routing". Schema takes plain dense tensors only (architecture.md §2). The
# `sparse_triangular_solve` wrapper below still calls
# `_legacy_sparse_triangular_solve`; nothing wires this op in until commit 16
# (spec/commit.md Phase 3), which also adds the real SpSM analysis-plan
# cache (architecture.md §3).
#
# No CUDA/CPU implementation is registered in this commit — the op exists
# only as schema + fake (meta) kernel and raises NotImplementedError if
# actually invoked. Its `register_autograd` backward below therefore also
# references torch.ops.tsgu.spsm / torch.ops.tsgu.sddmm with no real
# implementation yet; nothing exercises those code paths until later.
# ---------------------------------------------------------------------------


@torch.library.custom_op("tsgu::spsm", mutates_args=())
def spsm(
    vals: Tensor,
    rowptr: Tensor,
    col: Tensor,
    rhs: Tensor,
    B: int,
    n: int,
    upper: bool,
    unitriangular: bool,
    transpose: bool,
) -> Tensor:
    r"""Batched sparse triangular solve (map.md: ``sparse_triangular_solve``
    forward; also serves its own ``gradB`` via the "transposed plan" —
    calling this same op again with ``transpose`` flipped, map.md routing;
    the real analysis-plan cache reuse, architecture.md §3, lands with the
    kernel in commit 16).

    Solves, per batch item ``b``, :math:`A[b]\,x[b] = \mathrm{rhs}[b]` (or
    :math:`A[b]^\top\,x[b] = \mathrm{rhs}[b]` when ``transpose=True``) for
    triangular ``A[b]``.

    Parameters
    ----------
    vals : Tensor, shape ``(nse_total,)``
        Stored values of the triangular matrix (naming.md §2 ``vals``).
    rowptr : Tensor, shape ``(B * n + 1,)``
        Absolute CSR pointer over folded rows (naming.md §2 ``rowptr``).
    col : Tensor, shape ``(nse_total,)``
        Local column indices in ``[0, n)`` (naming.md §2 ``col``) — ``A`` is
        square, ``n_rows == n_cols == n``.
    rhs : Tensor, shape ``(B, n, p)``
        Right-hand side.
    B, n : int
        ``batch_size``, ``n_rows`` (== ``n_cols``).
    upper, unitriangular, transpose : bool
        Kept verbatim from the public ``sparse_triangular_solve`` contract
        (map.md invariant 1).

    Returns
    -------
    Tensor, shape ``(B, n, p)``
        Solution ``x``, same shape as ``rhs``.
    """
    raise NotImplementedError(
        "tsgu::spsm has no implementation registered yet — lands in spec/commit.md Phase 3 (commit 16)."
    )


@spsm.register_fake
def _spsm_fake(
    vals: Tensor,
    rowptr: Tensor,
    col: Tensor,
    rhs: Tensor,
    B: int,
    n: int,
    upper: bool,
    unitriangular: bool,
    transpose: bool,
) -> Tensor:
    # Value-independent: output shape/dtype exactly mirror rhs.
    return rhs.new_empty(rhs.shape)


def _spsm_setup_context(ctx, inputs, output):
    vals, rowptr, col, rhs, B, n, upper, unitriangular, transpose = inputs
    ctx.save_for_backward(vals, rowptr, col, output)
    ctx.B, ctx.n = B, n
    ctx.upper, ctx.unitriangular, ctx.transpose = upper, unitriangular, transpose
    ctx.vals_requires_grad = vals.requires_grad
    ctx.rhs_requires_grad = rhs.requires_grad


def _spsm_backward(ctx, grad_output):
    vals, rowptr, col, x = ctx.saved_tensors
    B, n = ctx.B, ctx.n

    need_gradB = ctx.rhs_requires_grad or ctx.vals_requires_grad
    gradB = None
    if need_gradB:
        # gradB (map.md "transposed plan"): the same op with `transpose`
        # flipped. Unlike tsgu::spmm's gradB, spsm carries its own
        # `transpose` flag, so no separately-transposed (CSC) pattern is
        # needed — `vals`/`rowptr`/`col` are reused as-is.
        gradB = torch.ops.tsgu.spsm(
            vals, rowptr, col, grad_output, B, n, ctx.upper, ctx.unitriangular, not ctx.transpose
        )

    grad_vals = None
    if ctx.vals_requires_grad:
        # gradA (map.md): sampled outer product -gradB @ x^T at A's own
        # pattern, via tsgu::sddmm's negate epilogue. NOTE: this uses the
        # non-transposed (g=gradB, mat=x) index convention; the exact
        # row/col role swap the reference implementation applies when
        # transpose=True (see the legacy SparseTriangularSolve.backward) is
        # a kernel-commit-16 detail, not a schema-level concern here.
        grad_vals = torch.ops.tsgu.sddmm(rowptr, col, gradB, x, B, n, n, True)

    grad_rhs = gradB if ctx.rhs_requires_grad else None

    return grad_vals, None, None, grad_rhs, None, None, None, None, None


spsm.register_autograd(_spsm_backward, setup_context=_spsm_setup_context)


def sparse_triangular_solve(
    A: torch.Tensor,
    B: torch.Tensor,
    upper: bool = True,
    unitriangular: bool = False,
    transpose: bool = False,
) -> torch.Tensor:
    r"""Sparse triangular solve with memory-efficient sparse gradients.

     Solves the triangular system :math:`\mathbf{A}\,\mathbf{x} = \mathbf{B}` (or
     :math:`\mathbf{A}^{\top}\,\mathbf{x} = \mathbf{B}` if ``transpose=True``), where
     :math:`\mathbf{A} \in \mathbb{R}^{m\times m}` is sparse triangular (COO/CSR) and
     :math:`\mathbf{B} \in \mathbb{R}^{m\times p}` is dense. Gradients preserve the sparsity
     pattern of :math:`\mathbf{A}` by evaluating only at its nonzero entries. Supports
     unbatched 2D and batched 3D inputs; COO inputs are converted to CSR internally for the
     factor solve.

     Let the upstream gradient be :math:`\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}}`
     for a scalar objective :math:`\mathcal{L}` and solution :math:`\mathbf{x}`. The dense-form
     gradients are

     Gradient with respect to B (dense):

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{B}} \;=\; \mathbf{A}^{-\top} \, \mathbf{G},

     and for ``transpose=True`` replace :math:`\mathbf{A}` by :math:`\mathbf{A}^{\top}` so that
     :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{B}} = \left(\mathbf{A}^{\top}\right)^{-\top} \mathbf{G} = \mathbf{A}^{-1} \mathbf{G}`.

     Gradient with respect to A (sparse):

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{A}} \;=\; -\big(\mathbf{A}^{-\top} \, \mathbf{G}\big)\, \mathbf{x}^{\top},

     and only entries at the nonzeros of :math:`\mathbf{A}` are evaluated. Equivalently,
     for a nonzero :math:`\mathbf{A}_{ij}` the contribution is

     .. math::
         \bigg[\frac{\partial \mathcal{L}}{\partial \mathbf{A}}\bigg]_{ij}
         \;=\; -\, \big(\mathbf{A}^{-\top} \, \mathbf{G}\big)_{i,:} \,\cdot\, \mathbf{x}_{j,:},

     where the dot denotes a row-wise inner product across the :math:`p` right-hand sides.

    Parameters
    ----------
    A : torch.Tensor, sparse COO or CSR, shape ``(m, m)`` or ``(b, m, m)``
        Sparse triangular coefficient matrix. Must be square per batch. All tensors must
        be on the same device.
    B : torch.Tensor, dense (strided), shape ``(m, p)`` or ``(b, m, p)``
        Right-hand side. ``B.shape[-2]`` must equal ``A.shape[-2]`` (``m``).
    upper : bool, optional
        If ``True`` (default), treat ``A`` as upper-triangular; else lower-triangular.
    unitriangular : bool, optional
        If ``True``, assume unit diagonal (implicit ones). The stored matrix must be
        strictly triangular (no explicit diagonal entries). Default: ``False``.
    transpose : bool, optional
        If ``True``, solves :math:`A^\top x = B`; otherwise :math:`A x = B`. Default: ``False``.

    Returns
    -------
    torch.Tensor
        Solution with the same shape as ``B``: ``(m, p)`` or ``(b, m, p)``.

    Raises
    ------
    ValueError
        If inputs are not tensors; ranks are < 2 or not both 2D/3D; layouts are
        incompatible (``A`` not COO/CSR or ``B`` not dense); shapes are incompatible;
        batch sizes differ; or if ``unitriangular=True`` but explicit diagonal
        entries are present.
    RuntimeError
        If the underlying triangular solve fails.

    Notes
    -----
    Backprop computes gradients only at nonzero entries of :math:`\mathbf{A}`, keeping the
    gradient sparse and reducing memory. COO inputs are converted to CSR since PyTorch's
    triangular solver requires CSR [1e]_. For autograd implementation details, see [2e]_.

    See Also
    --------
    torch.sparse.mm : Sparse ``@`` dense multiply.
    torch.linalg.solve_triangular : Dense triangular solver (modern API).

    References
    ----------
    .. [1e] PyTorch issue on sparse triangular solve:
           https://github.com/pytorch/pytorch/issues/87358
    .. [2e] PyTorch issue on autograd/triangular solve:
           https://github.com/pytorch/pytorch/issues/88890

    Examples
    --------
    Upper-triangular::

        >>> import torch
        >>> from torchsparsegradutils import sparse_triangular_solve
        >>> A = torch.sparse_csr_tensor([0, 2, 3, 4], [0, 2, 1, 2],
        ...                             torch.tensor([2.0, 1.0, 3.0, 1.0]), (3, 3))
        >>> B = torch.tensor([[1.0], [2.0], [3.0]])
        >>> x = sparse_triangular_solve(A, B, upper=True)
        >>> x.shape
        torch.Size([3, 1])

    Lower-triangular::

        >>> A_low = torch.sparse_csr_tensor([0, 1, 3, 5], [0, 0, 1, 0, 2],
        ...                                 torch.tensor([2.0, 1.0, 3.0, 0.5, 1.0]), (3, 3))
        >>> x = sparse_triangular_solve(A_low, B, upper=False)

    Batched::

        >>> # Convert to COO for batching (since torch.stack doesn't work with CSR)
        >>> A_coo = A.to_sparse_coo()
        >>> A_b = torch.stack([A_coo, A_coo])   # (2, 3, 3)
        >>> B_b = torch.stack([B, B])   # (2, 3, 1)
        >>> x_b = sparse_triangular_solve(A_b, B_b)
        >>> x_b.shape
        torch.Size([2, 3, 1])
    """
    # --- minimal validations to match the docstring expectations ---
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise ValueError("Both A and B should be instances of torch.Tensor")
    if A.dim() < 2 or B.dim() < 2:
        raise ValueError("Both A and B should be at least 2-dimensional tensors")
    if A.dim() != B.dim() or A.dim() not in (2, 3):
        raise ValueError("A and B must both be 2D or both be 3D tensors")
    if A.layout not in {torch.sparse_coo, torch.sparse_csr}:
        raise ValueError("A should be in either COO or CSR sparse format")
    if B.layout != torch.strided:
        raise ValueError("B must be a dense (strided) tensor")
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("A must be square on its last two dimensions")
    if A.size(-2) != B.size(-2):
        raise ValueError(f"Incompatible inner dimensions: A[..., {A.size(-2)}] vs B[..., {B.size(-2)}]")
    if A.dim() == 3 and A.size(0) != B.size(0):
        raise ValueError("If batched, A and B must have the same batch size")

    return _legacy_sparse_triangular_solve(A, B, upper, unitriangular, transpose)


# deleted by its kernel commit (spec/commit.md Phase 3)
def _legacy_sparse_triangular_solve(
    A: torch.Tensor, B: torch.Tensor, upper: bool, unitriangular: bool, transpose: bool
) -> torch.Tensor:
    return cast(torch.Tensor, SparseTriangularSolve.apply(A, B, upper, unitriangular, transpose))


class SparseTriangularSolve(torch.autograd.Function):
    r"""
    Autograd function for memory-efficient sparse triangular system solving.

    See Also
    --------
    sparse_triangular_solve : User-facing function that calls this autograd function.
    torch.triangular_solve : PyTorch's native triangular solver.
    """

    @staticmethod
    def forward(ctx, A, B, upper, unitriangular, transpose):
        ctx.batch_size = B.size()[0] if B.dim() == 3 else None
        ctx.A_shape = A.size()  # (b), m, m
        ctx.B_shape = B.size()  # (b), m, p
        ctx.csr = True
        ctx.upper = upper
        ctx.unitriangular = unitriangular
        ctx.transpose = transpose

        grad_flag = A.requires_grad or B.requires_grad

        if ctx.batch_size is not None:
            A = sparse_block_diag(*A)
            B = B.reshape(-1, B.size(-1))

        if A.layout == torch.sparse_coo:
            A = convert_coo_to_csr(A)  # NOTE: triangular solve doesn't work with sparse coo
            ctx.csr = False

        # NOTE: DEPRECATED: Check if a workaround for https://github.com/pytorch/pytorch/issues/88890 is needed

        x = torch.triangular_solve(
            B.detach(), A.detach(), upper=upper, unitriangular=unitriangular, transpose=transpose
        ).solution

        x.requires_grad = grad_flag
        ctx.save_for_backward(A, x.detach())

        if ctx.batch_size is not None:
            x = x.view(ctx.batch_size, ctx.A_shape[-2], ctx.B_shape[-1])

        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        if ctx.batch_size is not None:
            grad = grad.reshape(-1, grad.size(-1))

        A, x = ctx.saved_tensors

        # Backprop rule: gradB = A^{-T} grad
        # NOTE: DEPRECATED: Check if a workaround for https://github.com/pytorch/pytorch/issues/88890 is needed

        gradB = torch.triangular_solve(
            grad, A, upper=ctx.upper, transpose=not ctx.transpose, unitriangular=ctx.unitriangular
        ).solution

        # The gradient with respect to the matrix A seen as a dense matrix would
        # lead to a backprop rule as follows
        # gradA = -(A^{-T} grad)(A^{-1} B) = - gradB @ x.T
        # but we are only interested in the gradient with respect to
        # the (non-zero) values of A. To save memory, instead of computing the full
        # dense matrix gradB @ x.T and then subsampling at the nnz locations in a,
        # we can directly only compute the required values:
        # gradA[i,j] = - dotprod(gradB[i,:], x[j,:])

        # We start by getting the i and j indices:
        A_col_idx = A.col_indices()
        A_crow_idx = A.crow_indices()
        # Uncompress row indices:
        A_row_idx = torch.repeat_interleave(
            torch.arange(A.size()[0], device=A.device), A_crow_idx[1:] - A_crow_idx[:-1]
        )

        if ctx.transpose:
            mgradbselect = -gradB.index_select(0, A_col_idx)  # -gradB[j, :]
            xselect = x.index_select(0, A_row_idx)  # x[i, :]
        else:
            mgradbselect = -gradB.index_select(0, A_row_idx)  # -gradB[i, :]
            xselect = x.index_select(0, A_col_idx)  # x[j, :]

        if ctx.unitriangular is True and torch.any(A_row_idx == A_col_idx):
            raise ValueError("First input should be strictly triangular (i.e. unit diagonals is implicit)")

        # Dot product:
        mgbx = mgradbselect * xselect
        gradA = torch.sum(mgbx, dim=1)

        if ctx.csr is False:
            gradA = torch.sparse_coo_tensor(torch.stack([A_row_idx, A_col_idx]), gradA, A.shape)
        else:
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)

        if ctx.batch_size is not None:
            shapes = ctx.A_shape[0] * (ctx.A_shape[-2:],)
            gradA = sparse_block_diag_split(gradA, *shapes)
            if not ctx.csr:
                gradA = torch.stack([*gradA])
            else:
                gradA = stack_csr([*gradA])

            gradB = gradB.view(ctx.B_shape)

        return gradA, gradB, None, None, None
