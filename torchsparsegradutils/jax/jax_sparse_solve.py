from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import torch

from .jax_bindings import j2t as _j2t, t2j as _t2j, t2j_coo as _t2j_coo, t2j_csr as _t2j_csr


def sparse_solve_j4t(
    A: torch.Tensor,
    B: torch.Tensor,
    solve: Optional[Callable[..., Tuple["jax.Array", Any]]] = None,
    transpose_solve: Optional[Callable[..., Tuple["jax.Array", Any]]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    r"""Solve sparse linear systems using JAX solvers with PyTorch autograd.

    Computes :math:`A x = B` by dispatching to a JAX iterative solver while
    preserving gradients via a custom autograd function. Supports COO/CSR
    inputs, single or multiple RHS, and CPU/GPU (device inferred from inputs).

    Parameters
    ----------
    A : torch.Tensor
        Sparse square matrix ``(n,n)`` (COO or CSR).
    B : torch.Tensor
        RHS of shape ``(n,)`` or ``(n,k)``.
    solve : callable, optional
        JAX solver ``(A,B,**kw)->(x,info)`` (default BiCGSTAB).
    transpose_solve : callable, optional
        Solver for transpose system in backward (defaults to same solver on ``A.T``).
    **kwargs : Any
        Extra args forwarded to the JAX solver (``tol``, ``atol``, ``maxiter`` ...).

    Returns
    -------
    torch.Tensor
        Solution with same shape as ``B``.

    Raises
    ------
    TypeError
        If layout unsupported.
    ValueError
        If shape/dtype constraints violated.

    Notes
    -----
    * Auto-enables JAX x64 when inputs are ``float64``.
    * Backward solves :math:`A^T y = g` (implicit differentiation) and forms
      sparse gradients w.r.t. non-zero entries only.

    See Also
    --------
    torchsparsegradutils.jax.jax_bindings : Conversion helpers.
    SparseSolveJ4T : Underlying autograd function.

    Examples
    --------
    Basic (BiCGSTAB)::
        >>> import torch
        >>> from torchsparsegradutils.jax.jax_sparse_solve import sparse_solve_j4t
        >>> idx = torch.tensor([[0,1,1],[0,0,1]])
        >>> val = torch.tensor([2.0,-1.0,2.0])
        >>> A = torch.sparse_coo_tensor(idx, val, (2,2))
        >>> b = torch.tensor([1.0, 3.0])
        >>> x = sparse_solve_j4t(A, b)
        >>> x.shape
        torch.Size([2])

    CG solver::
        >>> import jax.scipy.sparse.linalg as jaxla
        >>> _ = sparse_solve_j4t(A, b, solve=jaxla.cg, tol=1e-10)

    Multi-RHS::
        >>> B = torch.randn(2,3)
        >>> X = sparse_solve_j4t(A, B)
        >>> X.shape
        torch.Size([2, 3])

    Gradient flow::
        >>> A.requires_grad_(True)  # doctest: +ELLIPSIS
        tensor(...)
        >>> b.requires_grad_(True)  # doctest: +ELLIPSIS
        tensor(...)
        >>> x = sparse_solve_j4t(A, b)
        >>> loss = (x**2).sum(); loss.backward()
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
        raise ValueError(f"Incompatible dimensions: A has shape {A.shape}, B has shape {B.shape}")

    # Check dtype compatibility
    if A.dtype != B.dtype:
        raise ValueError(f"A and B must have the same dtype. Got A: {A.dtype}, B: {B.dtype}")

    # Set default solvers
    if solve is None or transpose_solve is None:
        # Use bicgstab by default
        if solve is None:
            solve = lambda A, B, **kw: jax.scipy.sparse.linalg.bicgstab(A, B, **kw)
        if transpose_solve is None:
            transpose_solve = lambda A, B, **kw: jax.scipy.sparse.linalg.bicgstab(A.transpose(), B, **kw)

    # Enable JAX x64 mode for double precision
    if A.dtype == torch.float64 or B.dtype == torch.float64:
        # Use double precision for JAX
        jax.config.update("jax_enable_x64", True)

    # Type: ignore used because autograd.Function.apply has dynamic return typing.
    return SparseSolveJ4T.apply(A, B, solve, transpose_solve, kwargs)  # type: ignore[return-value]


class SparseSolveJ4T(torch.autograd.Function):
    r"""Autograd function: JAX-backed sparse solve with sparse gradients.

    Forward: solve :math:`A x = B` via JAX iterative solver.
    Backward: solve :math:`A^T y = g` then compute gradients only at non-zero
    entries of ``A`` without densifying.

    Notes
    -----
    * Conversions via :mod:`torchsparsegradutils.jax.jax_bindings` (COO/CSR preserved).
    * Gradient w.r.t. ``A`` is sparse (original sparsity pattern).

    See Also
    --------
    sparse_solve_j4t : Public wrapper.
    """

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        solve: Callable[..., Tuple["jax.Array", Any]],
        transpose_solve: Callable[..., Tuple["jax.Array", Any]],
        kwargs: dict,
    ) -> torch.Tensor:
        grad_flag = A.requires_grad or B.requires_grad
        ctx.transpose_solve = transpose_solve
        ctx.kwargs = kwargs  # Store kwargs for backward pass

        if A.layout == torch.sparse_coo:
            A_j = _t2j_coo(A.detach())
        elif A.layout == torch.sparse_csr:
            A_j = _t2j_csr(A.detach())
        else:
            raise TypeError(f"Unsupported layout type: {A.layout}")
        B_j = _t2j(B.detach())

        # pin JAX arrays to the same device as A
        if A.device.type == "cpu":
            jax_dev = jax.devices("cpu")[0]
        else:
            jax_dev = jax.devices("gpu")[A.device.index]
        ctx.jax_device = jax_dev
        A_j = jax.device_put(A_j, jax_dev)
        B_j = jax.device_put(B_j, jax_dev)

        x_j, exit_code = solve(A_j, B_j, **kwargs)

        x = _j2t(x_j)

        # Ensure output dtype matches input dtype
        if x.dtype != A.dtype:
            x = x.to(dtype=A.dtype)

        ctx.save_for_backward(A, x)
        ctx.A_j = A_j
        x.requires_grad = grad_flag
        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        A, x = ctx.saved_tensors

        # Unsqueeze, if necessary
        is_vector = x.ndim == 1
        if is_vector:
            x = x.unsqueeze(-1)
            grad = grad.unsqueeze(-1)

        grad_j = _t2j(grad.detach())
        # pin gradient to the same JAX device
        grad_j = jax.device_put(grad_j, ctx.jax_device)

        # Backprop rule: gradB = A^{-T} grad
        gradB_j, exit_code = ctx.transpose_solve(ctx.A_j.transpose(), grad_j, **ctx.kwargs)
        gradB = _j2t(gradB_j)

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
            # Ensure A is coalesced before accessing indices
            if not A.is_coalesced():
                A = A.coalesce()
            A_row_idx = A.indices()[0, :]
            A_col_idx = A.indices()[1, :]
            A_crow_idx = None  # for type checkers
        else:
            A_col_idx = A.col_indices()
            A_crow_idx = A.crow_indices()
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
            assert A_crow_idx is not None
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)

        # Squeeze gradB back to original shape if it was a vector
        if is_vector:
            gradB = gradB.squeeze(-1)

        return gradA, gradB, None, None, None
