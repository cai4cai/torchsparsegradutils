import warnings
from typing import Callable, Optional, cast

import torch


def sparse_generic_solve(
    A: torch.Tensor,
    B: torch.Tensor,
    solve: Optional[Callable[..., torch.Tensor]] = None,
    transpose_solve: Optional[Callable[..., torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    r"""Sparse linear solve with iterative methods and sparse-aware gradients.

     Solves :math:`\mathbf{A}\,\mathbf{x} = \mathbf{B}` with sparse :math:`\mathbf{A} \in \mathbb{R}^{n\times n}`
     (COO/CSR) and dense :math:`\mathbf{B} \in \mathbb{R}^{n\times p}` using iterative methods, while
     preserving sparsity in :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{A}}`. Supports single
     (vector) and multiple (matrix) right-hand sides and works with non-differentiable solvers via
     the implicit function theorem.

     Let :math:`\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}}` be the upstream gradient and
     :math:`\mathbf{x}` the solution. The dense-form gradients are

     Gradient with respect to B (dense):

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{B}} \;=\; \mathbf{A}^{-\top} \, \mathbf{G}
         \;\equiv\; \mathbf{G}_B.

     Gradient with respect to A (sparse):

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{A}} \;=\; -\, \mathbf{G}_B\, \mathbf{x}^{\top}.

     We evaluate only the entries corresponding to nonzeros of :math:`\mathbf{A}`, yielding a
     sparse gradient tensor with memory proportional to ``nnz(A)``. Equivalently, for a nonzero
     :math:`\mathbf{A}_{ij}` the contribution is

     .. math::
         \bigg[\frac{\partial \mathcal{L}}{\partial \mathbf{A}}\bigg]_{ij}
         \;=\; -\, (\mathbf{G}_B)_{i,:} \,\cdot\, \mathbf{x}_{j,:}.

    Parameters
    ----------
    A : torch.Tensor, sparse COO or CSR, shape ``(n, n)``
        Sparse square coefficient matrix. Must be invertible (or suitable) for the
        chosen solver. All tensors must be on the same device.
    B : torch.Tensor, dense (strided), shape ``(n,)`` or ``(n, k)``
        Right-hand side(s). ``B.shape[0]`` must equal ``A.shape[0]``.
    solve : callable, optional
        Forward solver with signature ``solve(A, B, **kwargs) -> X``.
        If ``None``, uses ``minres`` (recommended for symmetric indefinite).
        Other typical choices include:

        * ``linear_cg`` (SPD matrices)
        * ``bicgstab`` (general non-symmetric)

    transpose_solve : callable, optional
        Solver for the transpose system used in backprop, with signature
        ``transpose_solve(A, G, **kwargs) -> Y`` that solves :math:`A^\top Y = G` in the
        least-squares / iterative sense. If ``None``, defaults to ``solve``.
    **kwargs : dict
        Extra keyword arguments forwarded to the solvers (e.g., tolerances,
        iteration caps, or solver-specific settings objects).

    Returns
    -------
    torch.Tensor
        Solution tensor ``X`` with the same shape as ``B``: ``(n,)`` or ``(n, k)``.

    Raises
    ------
    ValueError
        If inputs are not tensors; shapes are incompatible; ranks are invalid.
    TypeError
        If ``A`` is not COO/CSR or if ``B`` is not dense (strided).
    UserWarning
        If ``A`` and ``B`` use different dtypes (may affect solver behavior).

    Notes
    -----
    Only entries at the nonzeros of :math:`\mathbf{A}` are computed, keeping the gradient
    sparse and memory-efficient.

    See Also
    --------
    sparse_triangular_solve : Triangular systems with sparse-aware gradients.
    sparse_generic_lstsq : Overdetermined least-squares with sparse-aware gradients.

    Examples
    --------
    >>> import torch
    >>> from torchsparsegradutils import sparse_generic_solve
    >>> from torchsparsegradutils.utils import linear_cg, bicgstab, minres
    >>> # Symmetric positive definite example
    >>> indices = torch.tensor([[0, 0, 1, 1, 2],
    ...                         [0, 1, 0, 1, 2]])
    >>> values = torch.tensor([4.0, -1.0, -1.0, 4.0, 2.0])
    >>> A = torch.sparse_coo_tensor(indices, values, (3, 3))
    >>> B = torch.tensor([1.0, 2.0, 3.0])
    >>> x = sparse_generic_solve(A, B, solve=linear_cg)
    >>> x.shape
    torch.Size([3])

    >>> # Multiple RHS with BiCGSTAB
    >>> X = sparse_generic_solve(A, torch.randn(3, 5), solve=bicgstab)
    >>> X.shape
    torch.Size([3, 5])

    >>> # Default solver (MINRES)
    >>> x = sparse_generic_solve(A, B)

    >>> # With custom solver settings:
    >>> from torchsparsegradutils.solvers.cg import LinearCGSettings
    >>> settings = LinearCGSettings(max_cg_iterations=1000, cg_tolerance=1e-8)
    >>> x = sparse_generic_solve(A, B, solve=linear_cg, settings=settings)

    >>> # With gradients (A.grad is sparse)
    >>> A.requires_grad_(True)  # doctest: +ELLIPSIS
    tensor(...)
    >>> B.requires_grad_(True)  # doctest: +ELLIPSIS
    tensor(...)
    >>> x = sparse_generic_solve(A, B)
    >>> x.sum().backward()
    >>> A.grad.is_sparse
    True
    """

    # Input validation
    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise ValueError("Both A and B should be instances of torch.Tensor")

    if A.layout not in (torch.sparse_coo, torch.sparse_csr):
        raise TypeError(f"Unsupported sparse layout: {A.layout}. Only COO and CSR are supported.")

    if A.dim() != 2:
        raise ValueError("A must be a 2D tensor")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    if B.dim() not in (1, 2):
        raise ValueError("B must be a 1D or 2D tensor")
    if B.shape[0] != A.shape[0]:
        raise ValueError(f"Incompatible dimensions: A has shape {tuple(A.shape)}, B has shape {tuple(B.shape)}")

    if B.layout != torch.strided:
        raise TypeError("B must be a dense (strided) tensor")

    if A.dtype != B.dtype:
        warnings.warn(
            f"A and B have different dtypes: A={A.dtype}, B={B.dtype}. This may affect solver behavior.",
            UserWarning,
            stacklevel=2,
        )

    # ---------- default solvers ----------
    if solve is None and transpose_solve is None:
        from ..utils import minres

        solve = minres
        transpose_solve = minres
    elif solve is None:
        solve = transpose_solve
    elif transpose_solve is None:
        transpose_solve = solve

    return _legacy_sparse_generic_solve(A, B, solve, transpose_solve, kwargs)


# deleted by its kernel commit (spec/commit.md Phase 3)
def _legacy_sparse_generic_solve(A, B, solve, transpose_solve, kwargs):
    X = cast(torch.Tensor, SparseGenericSolve.apply(A, B, solve, transpose_solve, kwargs))

    # Ensure output rank matches B (solver might return (n,1) for 1D B, etc.)
    if B.dim() == 1 and X.dim() == 2 and X.shape[1] == 1:
        X = X.squeeze(-1)
    elif B.dim() == 2 and X.dim() == 1:
        X = X.unsqueeze(-1)

    return X


class SparseGenericSolve(torch.autograd.Function):
    r"""
    Autograd function for sparse linear system solving with iterative methods.

    See Also
    --------
    sparse_generic_solve : User-facing function that calls this autograd function.
    """

    @staticmethod
    def forward(ctx, A, B, solve, transpose_solve, kwargs):
        grad_flag = A.requires_grad or B.requires_grad
        ctx.solve = solve
        ctx.transpose_solve = transpose_solve
        ctx.kwargs = kwargs  # Store kwargs for backward pass

        x = solve(A.detach(), B.detach(), **kwargs)

        # Ensure output dtype matches input dtype
        if x.dtype != A.dtype:
            x = x.to(dtype=A.dtype)

        x.requires_grad = grad_flag

        ctx.save_for_backward(A, x)
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        A, x = ctx.saved_tensors

        # Unsqueeze, if necessary
        is_vector = x.ndim == 1
        if is_vector:
            x = x.unsqueeze(-1)
            grad = grad.unsqueeze(-1)

        # Backprop rule: gradB = A^{-T} grad
        gradB = sparse_generic_solve(
            A,
            grad,
            solve=ctx.transpose_solve,
            transpose_solve=ctx.solve,
            **ctx.kwargs,
        )

        # Ensure gradient dtype matches input dtype
        if gradB.dtype != A.dtype:
            gradB = gradB.to(dtype=A.dtype)

        # The gradient with respect to the matrix A seen as a dense matrix would
        # lead to a backprop rule as follows
        # gradA = -(A^{-T} grad)(A^{-1} B) = - gradB @ x.T
        # but we are only interested in the gradient with respect to
        # the (non-zero) values of A. To save memory, instead of computing the full
        # dense matrix gradB @ x.T and then subsampling at the nnz locations in a,
        # we can directly only compute the required values:
        # gradA[i,j] = - dotprod(gradB[i,:], x[j,:])

        # We start by getting the i and j indices:
        if A.layout == torch.sparse_coo:
            A_coalesced = A.coalesce()  # Ensure tensor is coalesced before accessing indices
            A_row_idx = A_coalesced.indices()[0, :]
            A_col_idx = A_coalesced.indices()[1, :]
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
        gradA = torch.sum(mgbx, dim=1)

        # Ensure gradient dtype matches input dtype
        if gradA.dtype != A.dtype:
            gradA = gradA.to(dtype=A.dtype)

        if A.layout == torch.sparse_coo:
            gradA = torch.sparse_coo_tensor(torch.stack([A_row_idx, A_col_idx]), gradA, A.shape)
        else:
            gradA = torch.sparse_csr_tensor(A.crow_indices(), A_col_idx, gradA, A.shape)

        # Squeeze gradB back to original shape if it was a vector
        if is_vector:
            gradB = gradB.squeeze(-1)

        return gradA, gradB, None, None, None
