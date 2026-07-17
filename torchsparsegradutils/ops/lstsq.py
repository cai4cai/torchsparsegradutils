from typing import Callable, Optional, cast

import torch

from torchsparsegradutils._batched import BatchedCSR


def sparse_generic_lstsq(
    A: torch.Tensor,
    B: torch.Tensor,
    lstsq: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    transpose_lstsq: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    r"""Sparse linear least squares with sparse-aware gradients.

     Solves the overdetermined problem :math:`\min_x \|\mathbf{A}x - \mathbf{B}\|_2^2` where
     :math:`\mathbf{A} \in \mathbb{R}^{m\times n}` is sparse and tall (:math:`m>n`) and
     :math:`\mathbf{B} \in \mathbb{R}^{m\times p}` is dense. Backprop preserves the
     sparsity pattern by returning sparse gradients for :math:`\mathbf{A}` at its nonzero
     entries only.

     We assume :math:`\mathbf{A}` has full column rank so that :math:`\mathbf{A}^{+}\mathbf{A}=\mathbf{I}`
     (with :math:`\,\cdot^{+}` the Moore–Penrose pseudoinverse). Let
     :math:`\mathbf{x} \in \mathbb{R}^{n\times p}` denote the solution and let the upstream
     gradient be :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{x}} \in \mathbb{R}^{n\times p}` for some
     scalar objective :math:`\mathcal{L}`.
     Using Golub & Pereyra (1973) [1f]_, the gradients are:

    Gradient with respect to B (dense):

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{B}} \;=\; (\mathbf{A}^{\top})^{+} \, \frac{\partial \mathcal{L}}{\partial \mathbf{x}} \;\equiv\; \mathbf{G}_B.

    Gradient with respect to A (sparse): The dense form is

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{A}} \;=\; -\, \mathbf{G}_B\, \mathbf{x}^{\top}
         \; -\; (\mathbf{A}\,\mathbf{x} - \mathbf{B})\; \big(\mathbf{A}^{+}\, \mathbf{G}_B\big)^{\top},

     and we evaluate only the entries corresponding to nonzeros of :math:`\mathbf{A}` to keep
     the gradient sparse. Equivalently, for a nonzero entry :math:`\mathbf{A}_{ij}` with residuals
     :math:`\mathbf{r}=\mathbf{A}\,\mathbf{x}-\mathbf{B}` and :math:`\mathbf{H}=\mathbf{A}^{+}\,\mathbf{G}_B`,
     the contribution is

     .. math::
         \bigg[\frac{\partial \mathcal{L}}{\partial \mathbf{A}}\bigg]_{ij}
         \;=\; -\, (\mathbf{G}_B)_{i,:}\,\cdot\, \mathbf{x}_{j,:}
         \; -\; \mathbf{r}_{i,:}\,\cdot\, \mathbf{H}_{j,:},

     where dots denote row-wise inner products over the :math:`p` right-hand sides.

    Parameters
    ----------
    A : torch.Tensor
        Sparse COO/CSR tensor of shape ``(m, n)`` with ``m>n`` and full column rank.
    B : torch.Tensor
        Dense RHS of shape ``(m,)`` or ``(m, k)`` with ``B.shape[0] == A.shape[0]``.
    lstsq : callable, optional
        Solver ``lstsq(A,B)->X`` (``(n,)`` or ``(n,k)``). Default: LSMR
        (:func:`torchsparsegradutils.utils.lsmr`).
    transpose_lstsq : callable, optional
        Solver for transpose system in backward (``(A^T) Y = G``). Default: LSMR on ``A^T``.

    Returns
    -------
    torch.Tensor
        Solution ``X`` minimizing :math:`\|AX - B\|_2^2` with shape ``(n,)`` or ``(n,k)``.

    Raises
    ------
    TypeError
        If ``A`` is not sparse COO/CSR.
    ValueError
        If dimension mismatch or if backward encounters non-tall ``A``.
    RuntimeError
        If a provided solver fails or returns unexpected shape.

    See Also
    --------
    sparse_generic_solve : Square-system counterpart with sparse-aware gradients.

    References
    ----------
    .. [1f] Gene H. Golub and Victor Pereyra. The Differentiation of
        Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables Separate.
        SIAM Journal on Numerical Analysis, 10(2):413-432, 1973.

    Examples
    --------
    The backward routes gradA through the CUDA-only ``tsgu::sddmm`` /
    ``tsgu::spmm`` ops (architecture.md §4) -- these examples run on CUDA
    tensors.

    >>> # Simple sparse least squares example:
    >>> import torch
    >>> from torchsparsegradutils import sparse_generic_lstsq
    >>> indices = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3],
    ...                         [0, 0, 0, 0, 1, 1, 1, 2]])
    >>> values = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0])
    >>> A = torch.sparse_coo_tensor(indices, values, (5, 3)).coalesce().cuda()
    >>> B = torch.randn(5).cuda()
    >>> x = sparse_generic_lstsq(A, B)
    >>> x.shape
    torch.Size([3])

    >>> # Multiple RHS:
    >>> Bm = torch.randn(5, 4).cuda()
    >>> Xm = sparse_generic_lstsq(A, Bm)
    >>> Xm.shape
    torch.Size([3, 4])

    >>> # Custom solver:
    >>> from torchsparsegradutils.utils import lsmr
    >>> def my_lstsq(A_, B_):
    ...     return lsmr(A_, B_, atol=1e-10, btol=1e-10)[0]
    >>> _ = sparse_generic_lstsq(A, B, lstsq=my_lstsq)

    >>> # Gradients:
    >>> A.requires_grad_(True)  # doctest: +ELLIPSIS
    tensor(...)
    >>> B.requires_grad_(True)  # doctest: +ELLIPSIS
    tensor(...)
    >>> x = sparse_generic_lstsq(A, B)
    >>> loss = x.sum()  # Simple loss to preserve sparsity
    >>> loss.backward()
    >>> A.grad.is_sparse
    True
    """
    # Input validation (map.md invariant 7: invalid input raises, never
    # silently accepted; messages state the accepted logical forms).
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise ValueError("Both A and B should be instances of torch.Tensor")
    if A.layout not in (torch.sparse_coo, torch.sparse_csr):
        raise TypeError(f"Unsupported sparse layout: {A.layout}. Only COO and CSR are supported.")
    if A.dim() != 2:
        raise ValueError(f"A must be a sparse matrix with shape (n_rows, n_cols); got shape {tuple(A.shape)}")
    if B.layout != torch.strided:
        raise TypeError("B must be a dense (strided) tensor")
    if B.dim() not in (1, 2):
        raise ValueError(
            f"B must be a right-hand side with shape (n_rows,) or (n_rows, n_rhs); got shape {tuple(B.shape)}"
        )
    if B.shape[0] != A.shape[0]:
        raise ValueError(f"Incompatible dimensions: A has shape {tuple(A.shape)}, B has shape {tuple(B.shape)}")

    if lstsq is None or transpose_lstsq is None:
        from ..utils import lsmr

        if lstsq is None:
            # lsmr iterates batched (batch_size, n, n_rhs) right-hand sides
            # natively (spec/commit.md commit 17) — no per-column loop.
            def lstsq_default(AA, BB):
                return lsmr(AA, BB)[0]

            lstsq = lstsq_default
        if transpose_lstsq is None:

            def transpose_lstsq_default(AA, BB):
                # torch.adjoint of CSR yields CSC — lsmr's operator adapter
                # (solvers/_matvec.py) reads CSC as the O(1) transpose view,
                # so this still routes through tsgu::spmm on CUDA.
                return lsmr(torch.adjoint(AA), BB, AA)[0]

            transpose_lstsq = transpose_lstsq_default

    return _tsgu_sparse_generic_lstsq(A, B, lstsq, transpose_lstsq)


def _tsgu_sparse_generic_lstsq(A, B, lstsq, transpose_lstsq):
    """Host-loop forward + ``tsgu::sddmm`` backward (spec/commit.md Phase 3
    commit 17; map.md routing: ``sparse_generic_lstsq`` forward = host loop →
    ``tsgu::spmm`` on A and its cached CSC — inside lsmr, via
    ``solvers/_matvec.py`` — backward = host ``transpose_lstsq`` → gradA
    ``tsgu::sddmm``; replaces ``_legacy_sparse_generic_lstsq`` /
    ``SparseGenericLstsq``, deleted in this commit).

    Same construction as ``_tsgu_sparse_generic_solve``: the pluggable
    ``lstsq``/``transpose_lstsq`` callables cannot cross a ``torch.library``
    schema, so the Golub–Pereyra adjoint is wired with a thin private
    ``autograd.Function``; the descriptor's stored values thread through it
    so the sampled gradient rewraps at ``A``'s own pattern in ``A``'s own
    layout automatically (map.md invariant 3).
    """
    descriptor = BatchedCSR.from_torch(A)
    return cast(
        torch.Tensor,
        _GenericLstsqIFT.apply(descriptor.values, B, descriptor.rowptr, descriptor.col, A, lstsq, transpose_lstsq),
    )


class _GenericLstsqIFT(torch.autograd.Function):
    """Golub–Pereyra (1973, eq. 4.12) adjoint for the pluggable least-squares
    solve, assuming tall full-column-rank A (so :math:`A^+ A = I`).

    gradA at A's pattern is two ``tsgu::sddmm`` calls (map.md routing):
    ``-dot(gradB[i, :], x[j, :])`` (negate epilogue fused) plus
    ``dot((B - A x)[i, :], (A^+ gradB)[j, :])``; the residual's ``A x`` term
    is one ``tsgu::spmm`` at the same descriptor arrays.
    """

    @staticmethod
    def forward(ctx, values, B, rowptr, col, A, lstsq, transpose_lstsq):
        x = lstsq(A.detach(), B.detach())

        if B.dim() == 1:
            if x.dim() == 2:
                x = x.squeeze()
        else:
            if x.dim() == 1:
                x = x.unsqueeze(1)

        ctx.save_for_backward(values, rowptr, col, A, B, x)
        ctx.lstsq = lstsq
        ctx.transpose_lstsq = transpose_lstsq
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        values, rowptr, col, A, B, x = ctx.saved_tensors
        n_rows, n_cols = A.shape[-2], A.shape[-1]

        # We assume A is tall and full rank to get A^+ A = Id and simplify the
        # derivation; we don't try to compute the rank of A for computational
        # reasons, but at least check that A is a tall matrix.
        if n_cols > n_rows:
            raise ValueError(f"A should be a tall full-rank matrix. Got A.shape={tuple(A.shape)}")

        B_matrix = B.unsqueeze(1) if B.ndim == 1 else B
        x_matrix = x.unsqueeze(1) if x.ndim == 1 else x

        # Backprop rule: gradB = (A^T)^{+} grad
        gradB = ctx.transpose_lstsq(A, grad)
        if gradB.ndim == 1:
            gradB = gradB.unsqueeze(1)

        grad_values = None
        if ctx.needs_input_grad[0]:
            # gradA = -gradB @ x^T - (A x - B) @ (A^+ gradB)^T evaluated only
            # at A's specified entries: two sddmm calls at A's pattern.
            # First term, negate epilogue fused in the kernel:
            grad_values = torch.ops.tsgu.sddmm(
                rowptr, col, gradB.unsqueeze(0), x_matrix.unsqueeze(0), 1, n_rows, n_cols, True
            )
            # Second term: -(A x - B) = B - A x, with A x one spmm at the same
            # descriptor arrays. Detached, as the legacy backward saved a
            # detached A: the residual term contributes no higher-order path.
            Ax = torch.ops.tsgu.spmm(values.detach(), rowptr, col, x_matrix.unsqueeze(0), 1, n_rows, n_cols)
            residual_neg = B_matrix.unsqueeze(0) - Ax
            Apgb = ctx.lstsq(A, gradB)
            if Apgb.dim() == 1:
                Apgb = Apgb.unsqueeze(1)
            grad_values = grad_values + torch.ops.tsgu.sddmm(
                rowptr, col, residual_neg, Apgb.unsqueeze(0), 1, n_rows, n_cols, False
            )

        grad_rhs = None
        if ctx.needs_input_grad[1]:
            grad_rhs = gradB.squeeze(1) if grad.ndim == 1 else gradB

        return grad_values, grad_rhs, None, None, None, None, None
