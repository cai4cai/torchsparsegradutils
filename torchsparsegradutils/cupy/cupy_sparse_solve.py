import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

import torchsparsegradutils.cupy as tsgucupy

# from cupyx.scipy.sparse.linalg import cg, cgs, minres, gmres, spsolve


def sparse_solve_c4t(
    A: torch.Tensor,
    B: torch.Tensor,
    solve: Optional[Union[str, Callable[..., Any]]] = None,
    transpose_solve: Optional[Union[str, Callable[..., Any]]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    r"""
    Solve sparse linear systems using CuPy / SciPy with automatic backend selection.

    Solves :math:`A X = B` using CPU (NumPy / SciPy) or GPU (CuPy / cupyx.scipy)
    backends, chosen from the device of the input PyTorch sparse tensor ``A``.
    Supports selected iterative solvers and a direct sparse solve with automatic
    COO / CSR format conversion and an autograd-compatible backward pass via a
    transpose solve.

    Parameters
    ----------
    A : torch.Tensor
        Sparse square matrix of shape ``(n, n)`` in ``torch.sparse_coo`` or
        ``torch.sparse_csr`` layout.
    B : torch.Tensor
        Right-hand side(s). Shape ``(n,)`` (vector RHS) or ``(n, k)`` (multi-RHS).
    solve : {"cg", "cgs", "minres", "gmres", "spsolve"} or callable, optional
        Solver selector or a custom callable ``solve(A, b, **kwargs) -> x``.
        Built-ins:

        - ``"cg"`` : Conjugate Gradient (SPD; vector RHS only)
        - ``"cgs"`` : Conjugate Gradient Squared (vector RHS only)
        - ``"minres"`` : MINRES (symmetric; vector RHS only)
        - ``"gmres"`` : GMRES (vector RHS only)
        - ``"spsolve"`` : Direct sparse solve (supports multi-RHS)

        If ``None`` (default):

        - vector RHS → direct ``spsolve``
        - multi-RHS → factorize then solve (SciPy/CuPy factorized)
    transpose_solve : {"cg", "cgs", "minres", "gmres", "spsolve"} or callable, optional
        Solver for the transpose system :math:`A^T y = g` used in backprop.
        Defaults to using the same selection as ``solve`` (or factorized).
    **kwargs : dict
        Additional solver parameters passed through to the chosen backend:

        - Iterative solvers commonly accept ``tol``/``rtol``, ``atol``, ``maxiter``,
          and optionally ``x0``, ``M``, ``callback``; unsupported kwargs are ignored.
        - Direct ``spsolve`` ignores iteration controls.

    Returns
    -------
    torch.Tensor
        Solution tensor ``X`` with the same shape and dtype as ``B`` (or cast to match
        ``A.dtype`` if necessary) and on the same device as ``A``.

    Raises
    ------
    TypeError
        If ``A`` is not ``torch.sparse_coo`` or ``torch.sparse_csr``.
    ValueError
        If ``A`` is not square; if ``B`` has incompatible shape; or if an iterative
        solver is requested for a multi-RHS input.

    Notes
    -----
    Backend selection
        - CPU tensors → NumPy/SciPy
        - CUDA tensors → CuPy/cupyx.scipy
    Solver compatibility
        - Iterative solvers (``cg``, ``cgs``, ``minres``, ``gmres``): **vector RHS only**
        - Direct solver (``spsolve``): supports vector **and** multi-RHS
    Performance considerations
        - CSR is typically more efficient than COO for these solvers.
        - Backends may internally convert to CSC/CSR and emit efficiency warnings.
        - SciPy ``minres`` may upcast float32 to float64 on CPU.
    Gradients
        The backward pass solves :math:`A^T y = \\mathrm{grad}` using the same backend
        and forms sparse gradients for ``A`` by only computing entries at its nonzero
        positions.

    Examples
    --------
    Basic solve (default direct solver)
    >>> import torch
    >>> from torchsparsegradutils.cupy import sparse_solve_c4t
    >>> idx = torch.tensor([[0, 1, 1], [0, 0, 1]])
    >>> val = torch.tensor([2.0, -1.0, 2.0])
    >>> A = torch.sparse_coo_tensor(idx, val, (2, 2))
    >>> b = torch.tensor([1.0, 3.0])
    >>> x = sparse_solve_c4t(A, b)
    >>> x.shape
    torch.Size([2])

    Iterative solver
    >>> x_cg = sparse_solve_c4t(A, b, solve="cg", tol=1e-8)

    Multi-RHS with direct solve
    >>> B = torch.randn(2, 3)
    >>> X = sparse_solve_c4t(A, B, solve="spsolve")
    >>> X.shape
    torch.Size([2, 3])

    Note: CUDA backend (CuPy) is selected automatically when tensors are on GPU.

    See Also
    --------
    torchsparsegradutils.jax.sparse_solve_j4t :
        JAX-backed sparse solver with autograd support.
    torchsparsegradutils.cupy.t2c_coo, torchsparsegradutils.cupy.t2c_csr :
        PyTorch→CuPy/NumPy sparse converters used internally.
    torchsparsegradutils.cupy.c2t_coo, torchsparsegradutils.cupy.c2t_csr :
        CuPy/NumPy→PyTorch sparse converters used internally.
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

    # Check for iterative solver compatibility with multi-RHS
    vector_solvers = {"cg", "cgs", "minres", "gmres"}
    is_multi_rhs = B.ndim == 2 and B.shape[1] > 1

    if solve in vector_solvers and is_multi_rhs:
        raise ValueError(
            f"Solver '{solve}' does not support multi-RHS (B.shape={B.shape}). "
            f"Use solve='spsolve' or solve=None for multi-RHS problems, or reshape B to a vector."
        )

    if transpose_solve in vector_solvers and is_multi_rhs:
        raise ValueError(
            f"Transpose solver '{transpose_solve}' does not support multi-RHS (B.shape={B.shape}). "
            f"Use transpose_solve='spsolve' or transpose_solve=None for multi-RHS problems."
        )

    # Convert string solver names to actual solver functions
    def _get_solver_function(solver_name, xsp):
        """Get the appropriate solver function based on the backend."""
        if solver_name is None or callable(solver_name):
            return solver_name

        def _wrap_iterative_solver(base_solver, backend_type, solver_name=None):
            """Wrap an iterative solver to handle parameter mapping and return format."""

            def wrapped_solver(A, b, **solver_kwargs):
                # Create a copy to avoid modifying the original
                filtered_kwargs = solver_kwargs.copy()

                # Extract tolerance parameter and map to correct name for the backend
                tolerance = filtered_kwargs.pop("tol", None)
                atol = filtered_kwargs.pop("atol", None)

                # Define solver-specific parameter support
                solver_params = {
                    "cg": {"rtol", "maxiter", "atol"} if backend_type == "scipy" else {"tol", "maxiter", "atol"},
                    "cgs": {"rtol", "maxiter", "atol"} if backend_type == "scipy" else {"tol", "maxiter", "atol"},
                    "minres": {"rtol", "maxiter", "shift"} if backend_type == "scipy" else {"tol", "maxiter"},
                    "gmres": {"rtol", "maxiter", "atol"} if backend_type == "scipy" else {"tol", "maxiter", "atol"},
                }

                # Get supported parameters for this solver
                supported_params = solver_params.get(solver_name, set())

                if tolerance is not None:
                    # Map tolerance parameter based on backend and solver support
                    if backend_type == "scipy" and "rtol" in supported_params:
                        filtered_kwargs["rtol"] = tolerance
                    elif backend_type == "cupy" and "tol" in supported_params:
                        filtered_kwargs["tol"] = tolerance

                # Handle atol parameter if supported
                if atol is not None and "atol" in supported_params:
                    filtered_kwargs["atol"] = atol

                # Filter out unsupported parameters
                final_kwargs = {
                    k: v
                    for k, v in filtered_kwargs.items()
                    if k in supported_params
                    or k in {"x0", "M", "callback", "show", "check"}  # Always allow common parameters
                }

                # Call the base solver
                result = base_solver(A, b, **final_kwargs)

                # Handle return format - some solvers return (solution, info) tuples
                if isinstance(result, tuple):
                    return result[0]  # Return just the solution
                return result

            return wrapped_solver

        def _wrap_direct_solver(base_solver):
            """Wrap a direct solver to ignore tolerance parameters."""

            def wrapped_solver(A, b, **solver_kwargs):
                # Direct solvers don't use iterative solver parameters, so ignore them all
                filtered_kwargs = {
                    k: v
                    for k, v in solver_kwargs.items()
                    if k not in ["tol", "tolerance", "atol", "rtol", "maxiter", "matvec_max"]
                }
                return base_solver(A, b, **filtered_kwargs)

            return wrapped_solver

        # Determine backend type
        backend_type = "scipy" if A.device.type == "cpu" else "cupy"

        solver_map = {
            "cg": _wrap_iterative_solver(xsp.linalg.cg, backend_type, "cg"),
            "cgs": _wrap_iterative_solver(xsp.linalg.cgs, backend_type, "cgs"),
            "minres": _wrap_iterative_solver(xsp.linalg.minres, backend_type, "minres"),
            "gmres": _wrap_iterative_solver(xsp.linalg.gmres, backend_type, "gmres"),
            "spsolve": _wrap_direct_solver(xsp.linalg.spsolve),
        }

        if solver_name not in solver_map:
            raise ValueError(f"Unknown solver: {solver_name}. Supported solvers: {list(solver_map.keys())}")

        return solver_map[solver_name]

    # Get the appropriate backend modules
    xp, xsp = tsgucupy._get_array_modules(A.data)

    # Warn about dtype issues with minres on CPU
    if solve == "minres" and A.device.type == "cpu":
        warnings.warn(
            "Using 'minres' solver on CPU may change the output dtype to float64 "
            "even with float32 inputs due to SciPy implementation. Consider using "
            "'cg' or 'spsolve' for consistent dtype behavior.",
            UserWarning,
            stacklevel=2,
        )

    if transpose_solve == "minres" and A.device.type == "cpu":
        warnings.warn(
            "Using 'minres' transpose solver on CPU may change the output dtype to float64 "
            "even with float32 inputs due to SciPy implementation.",
            UserWarning,
            stacklevel=2,
        )

    # Convert string solver names to functions
    solve_func = _get_solver_function(solve, xsp)
    transpose_solve_func = _get_solver_function(transpose_solve, xsp)

    return SparseSolveC4T.apply(A, B, solve_func, transpose_solve_func, kwargs)


class SparseSolveC4T(torch.autograd.Function):
    r"""
    Autograd function for CuPy / SciPy–backed sparse solves.

    Forward: converts PyTorch sparse ``A`` (COO / CSR) and dense ``B`` to backend
    sparse / dense types, then calls either an iterative solver, direct
    ``spsolve``, or a cached factorized solve for multi-RHS.

    Backward: solves :math:`A^T y = \mathrm{grad}_X` and reconstructs
    :math:`\nabla_A` only at the nonzero positions of ``A`` via
    :math:`\nabla_A = -(A^{-T} \, \mathrm{grad}_X) \, X^T` (sampled at existing
    sparsity pattern) while returning :math:`\nabla_B = A^{-T} \mathrm{grad}_X`.

    See Also
    --------
    torchsparsegradutils.jax.sparse_solve_j4t : JAX-backed sparse solver.
    """

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B: torch.Tensor,
        solve: Optional[Callable[..., Any]],
        transpose_solve: Optional[Callable[..., Any]],
        kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        xp, xsp = tsgucupy._get_array_modules(A.data)
        grad_flag = A.requires_grad or B.requires_grad
        ctx.transpose_solve = transpose_solve

        # Transfer data to cupy/scipy
        if A.layout == torch.sparse_coo:
            A_c = tsgucupy.t2c_coo(A.detach())
        elif A.layout == torch.sparse_csr:
            A_c = tsgucupy.t2c_csr(A.detach())
        else:
            raise TypeError(f"Unsupported layout type: {A.layout}")
        B_c = xp.asarray(B.detach())

        # Solve the sparse system
        ctx.factorisedsolver = None
        ctx.kwargs = kwargs  # Store kwargs for backward pass
        if solve is not None:
            x_c = solve(A_c, B_c, **kwargs)
        elif (B.ndim == 1) or (B.shape[1] == 1):
            # xp.sparse.linalg.spsolve only works if B is a vector but is fully on GPU with cupy
            # TODO: Is this still true?
            # see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html#scipy.sparse.linalg.spsolve
            x_c = xsp.linalg.spsolve(A_c, B_c)
        else:
            # Make use of a factorisation (only the solver is then on the GPU with cupy)
            # We store it in ctx to reuse it in the backward pass
            ctx.factorisedsolver = xsp.linalg.factorized(A_c)
            x_c = ctx.factorisedsolver(B_c)

        if isinstance(x_c, tuple):
            # If the solver returns a tuple, we assume the first element is the solution
            x_c = x_c[0]

        x = torch.as_tensor(x_c, device=A.device)

        # Ensure output dtype matches input dtype
        if x.dtype != A.dtype:
            x = x.to(dtype=A.dtype)

        if (B.ndim == 2) and (x.ndim == 1):
            x = x.unsqueeze(-1)

        ctx.save_for_backward(A, x)
        ctx.A_c = A_c
        x.requires_grad = grad_flag
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None, None]:
        A, x = ctx.saved_tensors
        xp, xsp = tsgucupy._get_array_modules(A.data)

        # Unsqueeze, if necessary
        is_vector = x.ndim == 1
        if is_vector:
            x = x.unsqueeze(-1)
            grad = grad.unsqueeze(-1)

        grad_c = xp.asarray(grad.detach())

        # Backprop rule: gradB = A^{-T} grad
        if ctx.transpose_solve is not None:
            gradB_c = ctx.transpose_solve(ctx.A_c, grad_c, **ctx.kwargs)
        elif ctx.factorisedsolver is None:
            gradB_c = xsp.linalg.spsolve(xp.transpose(ctx.A_c), grad_c)
        else:
            # Re-use factorised solver from forward pass
            gradB_c = ctx.factorisedsolver(grad_c, trans="T")

        if isinstance(gradB_c, tuple):
            # If the solver returns a tuple, we assume the first element is the gradient
            gradB_c = gradB_c[0]

        gradB = torch.as_tensor(gradB_c, device=A.device)

        # Ensure gradient dtype matches input dtype
        if gradB.dtype != A.dtype:
            gradB = gradB.to(dtype=A.dtype)

        if (grad.ndim == 2) and (gradB.ndim == 1):
            gradB = gradB.unsqueeze(-1)

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
        gradA = torch.sum(mgbx, dim=1)

        # Ensure gradient dtype matches input dtype
        if gradA.dtype != A.dtype:
            gradA = gradA.to(dtype=A.dtype)

        if A.layout == torch.sparse_coo:
            gradA = torch.sparse_coo_tensor(torch.stack([A_row_idx, A_col_idx]), gradA, A.shape)
        else:
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)

        # Squeeze gradB back to original shape if it was a vector
        if is_vector:
            gradB = gradB.squeeze(-1)

        return gradA, gradB, None, None, None
