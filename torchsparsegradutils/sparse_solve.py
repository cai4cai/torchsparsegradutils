import warnings
from typing import Callable, Optional, cast

import torch

from torchsparsegradutils.utils import convert_coo_to_csr, sparse_block_diag, sparse_block_diag_split, stack_csr


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
            B = torch.cat([*B])

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
            grad = torch.cat([*grad])

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
    >>> from torchsparsegradutils.utils.linear_cg import LinearCGSettings
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
        from .utils import minres

        solve = minres
        transpose_solve = minres
    elif solve is None:
        solve = transpose_solve
    elif transpose_solve is None:
        transpose_solve = solve

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
        ctx.transpose_solve = transpose_solve
        ctx.kwargs = kwargs  # Store kwargs for backward pass

        x = solve(A.detach(), B.detach(), **kwargs)

        # Ensure output dtype matches input dtype
        if x.dtype != A.dtype:
            x = x.to(dtype=A.dtype)

        x.requires_grad = grad_flag

        ctx.save_for_backward(A, x.detach())
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
        gradB = ctx.transpose_solve(A, grad, **ctx.kwargs)

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
