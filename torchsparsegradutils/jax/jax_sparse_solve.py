import torch
import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg

from .jax_bindings import j2t as _j2t
from .jax_bindings import t2j as _t2j
from .jax_bindings import t2j_coo as _t2j_coo
from .jax_bindings import t2j_csr as _t2j_csr


def sparse_solve_j4t(A, B, solve=None, transpose_solve=None, **kwargs):
    """
    Solve the sparse linear system Ax = B using JAX backends.

    This function uses JAX's sparse linear algebra solvers and supports both
    iterative solvers and automatic device placement (CPU/GPU) based on input tensors.

    Args:
        A (torch.Tensor): A 2D sparse square tensor in COO or CSR format. Must have
                         shape (n, n) where n is the number of rows/columns.
        B (torch.Tensor): A 1D or 2D tensor with shape (n,) or (n, k) where n matches
                         the dimension of A and k is the number of right-hand sides.
                         JAX solvers generally support both vector and multi-RHS.
        solve (callable, optional): Solver function to use. Should be a JAX solver function such as:
            - None: Use default solver (jax.scipy.sparse.linalg.bicgstab)
            - jax.scipy.sparse.linalg.cg: Conjugate Gradient (requires symmetric positive definite A)
            - jax.scipy.sparse.linalg.bicgstab: Biconjugate Gradient Stabilized (default)
            - jax.scipy.sparse.linalg.gmres: Generalized Minimal Residual
            - Custom JAX solver function that takes (A, B) and returns (solution, info)
        transpose_solve (callable, optional): Solver for the transpose system A^T x = b
                                            used in backpropagation. Same options as solve.
                                            If None, uses the same solver as solve on A.transpose().

        **kwargs: Additional keyword arguments passed to the solver functions.
                 Common parameters:
                 - tol (float): Tolerance for iterative solvers (default 1e-5)
                 - maxiter (int): Maximum number of iterations
                 - atol (float): Absolute tolerance for some solvers

    Returns:
        torch.Tensor: Solution tensor X with the same shape as B.

    Raises:
        TypeError: If A is not a sparse tensor with supported layout (COO or CSR).
        ValueError: If A is not square, if B has incompatible dimensions, or if inputs
                   have mismatched dtypes.

    Note:
        - JAX automatically handles device placement based on input tensor device
        - For float64 inputs, JAX x64 mode is automatically enabled
        - JAX solvers typically return (solution, info) tuples; info contains convergence details
        - Both vector (1D or 2D with shape[1]==1) and multi-RHS (2D with shape[1]>1) are supported
        - GPU memory may be preallocated by JAX even for CPU computations
        - JAX uses different convergence criteria and may have different numerical behavior
          compared to SciPy/CuPy solvers

    Example:
        >>> import torch
        >>> from jax.scipy.sparse.linalg import cg, bicgstab
        >>> # Create sparse system
        >>> A = torch.sparse_coo_tensor([[0,1,1],[0,0,1]], [2.0,1.0,3.0], (2,2))
        >>> B = torch.tensor([1.0, 2.0])
        >>> # Solve with different solvers
        >>> X1 = sparse_solve_j4t(A, B, solve=cg)          # Conjugate gradient
        >>> X2 = sparse_solve_j4t(A, B, solve=bicgstab)    # BiCGSTAB (default)
        >>> X3 = sparse_solve_j4t(A, B)                    # Uses bicgstab by default
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

    return SparseSolveJ4T.apply(A, B, solve, transpose_solve, kwargs)


class SparseSolveJ4T(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, solve, transpose_solve, kwargs):
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
    def backward(ctx, grad):
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
