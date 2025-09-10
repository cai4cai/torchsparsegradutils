from typing import Callable, Optional, cast

import torch


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
    SparseGenericLstsq : Autograd implementation.

    References
    ----------
    .. [1f] Gene H. Golub and Victor Pereyra. The Differentiation of
        Pseudo-Inverses and Nonlinear Least Squares Problems Whose Variables Separate.
        SIAM Journal on Numerical Analysis, 10(2):413-432, 1973.

    Examples
    --------
    >>> # Simple sparse least squares example:
    >>> import torch
    >>> from torchsparsegradutils import sparse_generic_lstsq
    >>> indices = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3],
    ...                         [0, 0, 0, 0, 1, 1, 1, 2]])
    >>> values = torch.tensor([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0])
    >>> A = torch.sparse_coo_tensor(indices, values, (5, 3)).coalesce()
    >>> B = torch.randn(5)
    >>> x = sparse_generic_lstsq(A, B)
    >>> x.shape
    torch.Size([3])

    >>> # Multiple RHS:
    >>> Bm = torch.randn(5, 4)
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
    if lstsq is None or transpose_lstsq is None:
        from .utils import lsmr

        if lstsq is None:

            def lstsq_default(AA, BB):
                # Handle multiple RHS by solving each column separately
                if BB.dim() == 1:
                    return lsmr(AA, BB)[0]
                else:
                    solutions = []
                    for i in range(BB.shape[1]):
                        sol = lsmr(AA, BB[:, i])[0]
                        solutions.append(sol)
                    return torch.stack(solutions, dim=1)

            lstsq = lstsq_default
        if transpose_lstsq is None:

            def transpose_lstsq_default(AA, BB):
                # Handle multiple RHS by solving each column separately
                if BB.dim() == 1:
                    return lsmr(torch.adjoint(AA), BB, AA)[0]
                else:
                    solutions = []
                    for i in range(BB.shape[1]):
                        sol = lsmr(torch.adjoint(AA), BB[:, i], AA)[0]
                        solutions.append(sol)
                    return torch.stack(solutions, dim=1)

            transpose_lstsq = transpose_lstsq_default

    # Autograd Function.apply is typed as Any; cast for type checkers.
    return cast(torch.Tensor, SparseGenericLstsq.apply(A, B, lstsq, transpose_lstsq))


class SparseGenericLstsq(torch.autograd.Function):
    r"""Autograd kernel for sparse least squares with sparse-aware gradients.

    See Also
    --------
    sparse_generic_lstsq : User wrapper.

    """

    @staticmethod
    def forward(ctx, A, B, lstsq, transpose_lstsq):
        grad_flag = A.requires_grad or B.requires_grad
        ctx.lstsq = lstsq
        ctx.transpose_lstsq = transpose_lstsq

        x = lstsq(A.detach(), B.detach())

        x.requires_grad = grad_flag

        if B.dim() == 1:
            if x.dim() == 2:
                x = x.squeeze()
        else:
            if x.dim() == 1:
                x = x.unsqueeze(1)

        ctx.save_for_backward(A.detach(), B.detach(), x.detach())
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        A, B, x = ctx.saved_tensors
        if B.ndim == 1:
            B = B.unsqueeze(1)
        if x.ndim == 1:
            x = x.unsqueeze(1)

        # Backprop rule: gradB = (A^T)^{+} grad
        gradB = ctx.transpose_lstsq(A, grad)
        if gradB.ndim == 1:
            gradB = gradB.unsqueeze(1)

        # We make use of equation 4.12 in https://www.jstor.org/stable/2156365
        # but assume A is tall and full rank to get A^+ A = Id and simplify the derivation.
        # We don't try and compute the rank of A for computational reason but at least check
        # that A is a tall matrix
        if A.shape[1] > A.shape[0]:
            raise ValueError(f"A should be a tall full-rank matrix. Got A.shape={A.shape}")
        # Following the derivation in https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
        # but using the pseudo-inverse instead of the inverse:
        # The gradient with respect to the matrix A seen as a dense matrix would
        # lead to a backprop rule as follows
        # gradA = -((A^T)^{+} grad)(A^{+} B) - (Ax-B)(A^+ (A^T)^{+} grad )
        #       = - gradB @ x.T - (Ax-B) @ (A^+ gradB).T
        # but we are only interested in the gradient with respect to
        # the (non-zero) values of A. To save memory, instead of computing the full
        # dense matrices gradB @ x.T and (Ax-B) @ (A^+ gradB).T
        # and then subsampling at the nnz locations in A,
        # we can directly only compute the required values:
        # gradA_u1[i,j] = - dotprod(gradB[i,:], x[j,:])
        # gradA_u2[i,j] = - dotprod(residuals[i,:], (A^+ gradB)[j,:])

        # Dense equivalent
        # gradA_u1 = - gradB @ torch.t(x)
        # mresiduals = B - A@x
        # Apgb = ctx.lstsq(A,gradB)
        # if Apgb.dim() == 1:
        #     Apgb = Apgb.unsqueeze(1)
        # gradA_u2 = mresiduals @ torch.t(Apgb)
        # gradA = gradA_u1 + gradA_u2
        # return gradA, gradB, None, None

        # We start by getting the i and j indices:
        if A.layout == torch.sparse_coo:
            A_row_idx = A.indices()[0, :]
            A_col_idx = A.indices()[1, :]
        else:
            A_col_idx = A.col_indices()
            A_crow_idx = A.crow_indices()
            # Uncompress row indices:
            A_row_idx = torch.repeat_interleave(
                torch.arange(A.size()[0], device=A.device), A_crow_idx[1:] - A_crow_idx[:-1]
            )

        mgradbselect = -gradB.index_select(0, A_row_idx)  # -gradB[i, :]
        xselect = x.index_select(0, A_col_idx)  # x[j, :]

        # Dot product:
        mgbx = mgradbselect * xselect
        gradA_u1 = torch.sum(mgbx, dim=1)

        # residuals
        mresiduals = B - A @ x
        mresidualsselect = mresiduals.index_select(0, A_row_idx)
        Apgb = ctx.lstsq(A, gradB)
        if Apgb.dim() == 1:
            Apgb = Apgb.unsqueeze(1)
        Apgbselect = Apgb.index_select(0, A_col_idx)

        # Dot product:
        mresApgb = mresidualsselect * Apgbselect
        gradA_u2 = torch.sum(mresApgb, dim=1)

        gradA = gradA_u1 + gradA_u2
        if A.layout == torch.sparse_coo:
            gradA = torch.sparse_coo_tensor(torch.stack([A_row_idx, A_col_idx]), gradA, A.shape)
        else:
            gradA = torch.sparse_csr_tensor(A.crow_indices(), A_col_idx, gradA, A.shape)

        if grad.ndim == 1:
            gradB = gradB.squeeze()

        return gradA, gradB, None, None
