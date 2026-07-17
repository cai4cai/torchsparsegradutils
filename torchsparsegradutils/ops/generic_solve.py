import warnings
from typing import Callable, Optional, cast

import torch

from torchsparsegradutils._batched import BatchedCSR


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
    The backward routes gradA through the CUDA-only ``tsgu::sddmm`` op
    (architecture.md §4) -- these examples run on CUDA tensors.

    >>> import torch
    >>> from torchsparsegradutils import sparse_generic_solve
    >>> from torchsparsegradutils.utils import linear_cg, bicgstab, minres
    >>> # Symmetric positive definite example
    >>> indices = torch.tensor([[0, 0, 1, 1, 2],
    ...                         [0, 1, 0, 1, 2]])
    >>> values = torch.tensor([4.0, -1.0, -1.0, 4.0, 2.0])
    >>> A = torch.sparse_coo_tensor(indices, values, (3, 3)).cuda()
    >>> B = torch.tensor([1.0, 2.0, 3.0]).cuda()
    >>> x = sparse_generic_solve(A, B, solve=linear_cg)
    >>> x.shape
    torch.Size([3])

    >>> # Multiple RHS with BiCGSTAB
    >>> X = sparse_generic_solve(A, torch.randn(3, 5).cuda(), solve=bicgstab)
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

    return _tsgu_sparse_generic_solve(A, B, solve, transpose_solve, kwargs)


def _tsgu_sparse_generic_solve(A, B, solve, transpose_solve, kwargs):
    """Host-loop forward + ``tsgu::sddmm`` backward (spec/commit.md Phase 3
    commit 17; map.md routing: ``sparse_generic_solve`` forward = host loop →
    ``tsgu::spmm`` — the default solvers route their matvecs through the
    kernel via ``solvers/_matvec.py`` — backward = host ``transpose_solve`` →
    gradA ``tsgu::sddmm``; replaces ``_legacy_sparse_generic_solve`` /
    ``SparseGenericSolve``, deleted in this commit).

    Unwraps ``A`` into its ``BatchedCSR`` descriptor at the boundary
    (architecture.md §2) and threads the descriptor's stored values through
    :class:`_GenericSolveIFT`, so the sampled gradient the sddmm kernel
    produces flows back through ``from_torch``'s differentiable extraction
    chain and rewraps as a sparse gradient at ``A``'s own pattern in ``A``'s
    own layout (COO in → COO grad out; map.md invariant 3) — the same
    mechanism ``_tsgu_sparse_mm`` documents.

    The solve itself cannot be a ``tsgu::`` custom op: its defining inputs
    are the pluggable ``solve``/``transpose_solve`` **callables** (map.md
    contract: user solvers must keep working), which cannot cross a
    ``torch.library`` schema. The implicit-function-theorem adjoint is
    therefore wired with a thin private ``autograd.Function`` whose backward
    is built from differentiable pieces only (a recursive transposed solve
    and ``tsgu::sddmm``), preserving the higher-order-gradient guarantee
    (testing.md: gradgradcheck for solve ops, PR #85).
    """
    descriptor = BatchedCSR.from_torch(A)
    return cast(
        torch.Tensor,
        _GenericSolveIFT.apply(
            descriptor.values, B, descriptor.rowptr, descriptor.col, A, solve, transpose_solve, kwargs
        ),
    )


class _GenericSolveIFT(torch.autograd.Function):
    """Implicit-function-theorem adjoint for the pluggable host solve.

    Forward runs the (non-differentiable) user solve on detached operands.
    Backward: gradB solves the transposed system with ``transpose_solve``
    (recursively through ``sparse_generic_solve`` so the node chain supports
    higher-order gradients), and the sparse gradient w.r.t. ``A``'s stored
    values is exactly one ``tsgu::sddmm`` at ``A``'s pattern with the fused
    negate epilogue: gradA[i, j] = -dot(gradB[i, :], x[j, :]).
    """

    @staticmethod
    def forward(ctx, values, B, rowptr, col, A, solve, transpose_solve, kwargs):
        x = solve(A.detach(), B.detach(), **kwargs)

        # Ensure output dtype matches input dtype
        if x.dtype != A.dtype:
            x = x.to(dtype=A.dtype)

        # Ensure output rank matches B (solver might return (n, 1) for 1D B, etc.)
        if B.dim() == 1 and x.dim() == 2 and x.shape[1] == 1:
            x = x.squeeze(-1)
        elif B.dim() == 2 and x.dim() == 1:
            x = x.unsqueeze(-1)

        ctx.save_for_backward(rowptr, col, A, x)
        ctx.solve = solve
        ctx.transpose_solve = transpose_solve
        ctx.kwargs = kwargs
        ctx.rhs_was_vector = B.dim() == 1
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        rowptr, col, A, x = ctx.saved_tensors
        n_rows, n_cols = A.shape[-2], A.shape[-1]

        grad_matrix = grad.unsqueeze(-1) if ctx.rhs_was_vector else grad
        x_matrix = x.unsqueeze(-1) if ctx.rhs_was_vector else x

        # Backprop rule: gradB = A^{-T} grad — a recursive sparse_generic_solve
        # with the two solvers swapped, so the backward is itself the same
        # differentiable node (higher-order gradients, PR #85).
        gradB = sparse_generic_solve(
            A,
            grad_matrix,
            solve=ctx.transpose_solve,
            transpose_solve=ctx.solve,
            **ctx.kwargs,
        )

        # Ensure gradient dtype matches input dtype
        if gradB.dtype != A.dtype:
            gradB = gradB.to(dtype=A.dtype)

        grad_values = None
        if ctx.needs_input_grad[0]:
            # gradA at A's own sparsity pattern, negate epilogue fused in the
            # kernel (map.md routing) — no dense materialisation:
            # gradA[i, j] = -dot(gradB[i, :], x[j, :]).
            grad_values = torch.ops.tsgu.sddmm(
                rowptr, col, gradB.unsqueeze(0), x_matrix.unsqueeze(0), 1, n_rows, n_cols, True
            )

        grad_rhs = None
        if ctx.needs_input_grad[1]:
            grad_rhs = gradB.squeeze(-1) if ctx.rhs_was_vector else gradB

        return grad_values, grad_rhs, None, None, None, None, None, None
