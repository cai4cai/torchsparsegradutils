import torch
from .utils import convert_coo_to_csr, sparse_block_diag, sparse_block_diag_split, stack_csr


def sparse_triangular_solve(A, B, upper=True, unitriangular=False, transpose=False):
    """
    Solves a system of equations given by AX = B, where A is a sparse triangular matrix,
    and B is a dense right-hand side matrix.

    This function accepts both batched and unbatched inputs, and can work with either upper
    or lower triangular matrices.

    A can be in either COO (Coordinate Format) or CSR (Compressed Sparse Row) format. However,
    if it is in COO format, it will be converted to CSR format before solving as the
    triangular solve operation doesn't work with COO format.

    This function supports backpropagation, and preserves the sparsity of the gradient during
    the backpass.

    Args:
        A (torch.Tensor): The left-hand side sparse triangular matrix. Must be a 2-dimensional
                          (matrix) or 3-dimensional (batch of matrices) tensor, and must be in
                          either COO or CSR format.
        B (torch.Tensor): The right-hand side dense matrix. Must be a 2-dimensional (matrix) or
                          3-dimensional (batch of matrices) tensor.
        upper (bool, optional): If True, A is assumed to be an upper triangular matrix and the
                                lower triangular elements are not accessed. If False,
                                A is assumed to be a lower triangular matrix. Default is True.
        unitriangular (bool, optional): If True, the diagonal elements of A are assumed to be 1
                                        and are not used in the solve operation. Default is False.
        transpose (bool, optional): If True, solves A^T X = B. If False, solves
                                    AX = B. Default is False.

    Returns:
        torch.Tensor: The solution of the system of equations.

    Raises:
        ValueError: If A and B are not both torch.Tensor instances, or if they don't have the same
                    number of dimensions, or if they are not at least 2-dimensional, or if A is not
                    in COO or CSR format, or if A and B are batched but don't have the same batch size.

    Note:
        The gradient with respect to the sparse matrix A is computed only for its
        non-zero values to save memory.

    References:
        https://github.com/pytorch/pytorch/issues/87358
        https://github.com/pytorch/pytorch/issues/88890
    """

    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise ValueError("Both A and B should be instances of torch.Tensor")

    if A.dim() < 2 or B.dim() < 2:
        raise ValueError("Both A and B should be at least 2-dimensional tensors")

    if A.dim() != B.dim():
        raise ValueError("Both A and B should have the same number of dimensions")

    if A.layout not in {torch.sparse_coo, torch.sparse_csr}:
        raise ValueError("A should be in either COO or CSR format")

    if A.dim() == 3 and A.size(0) != B.size(0):
        raise ValueError("If A and B have a leading batch dimension, they should have the same batch size")

    return SparseTriangularSolve.apply(A, B, upper, unitriangular, transpose)


class SparseTriangularSolve(torch.autograd.Function):
    """
    Solves a system of equations with a square upper or lower triangular
    invertible sparse matrix A and dense right-hand side matrix B,
    with backpropagation support

    Solves: Ax = B

    A can be in either COO or CSR format.
    But, COO will internally be converted to CSR before solving.

    This implementation preserves the sparsity of the gradient calculated during a
    backpass through torch.triangular_solve, as detailed here:
    https://github.com/pytorch/pytorch/issues/87358
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
    def backward(ctx, grad):
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


def sparse_generic_solve(A, B, solve=None, transpose_solve=None, **kwargs):
    """
    Solve the sparse linear system Ax = B using custom iterative solvers.

    This function provides a generic interface for sparse linear system solving using
    custom solver functions. It supports both vector and multi-RHS systems and maintains
    gradient computation through the sparse structure during backpropagation.

    Args:
        A (torch.Tensor): A 2D sparse square tensor in COO or CSR format. Must have
                         shape (n, n) where n is the number of rows/columns.
        B (torch.Tensor): A 1D or 2D tensor with shape (n,) or (n, k) where n matches
                         the dimension of A and k is the number of right-hand sides.
        solve (callable, optional): Solver function to use. Should be a function that takes
                                   (A, B) and returns the solution X. Can be:
            - None: Use default solver (minres from utils)
            - torchsparsegradutils.utils.linear_cg: Conjugate Gradient (requires symmetric positive definite A)
            - torchsparsegradutils.utils.bicgstab: Biconjugate Gradient Stabilized
            - torchsparsegradutils.utils.minres: Minimum Residual (requires symmetric A, default)
            - Custom solver function with signature solve(A, B) -> X
        transpose_solve (callable, optional): Solver for the transpose system A^T x = b
                                            used in backpropagation. Same options as solve.
                                            If None, uses the same solver as solve.
                                            For symmetric matrices (like with minres), can use the same solver.
        **kwargs: Additional keyword arguments passed to the solver functions.
                 Common parameters depend on the specific solver used:
                 - tolerance (float): Tolerance for convergence (linear_cg)
                 - settings: Settings objects for different solvers (BICGSTABSettings, MINRESSettings, etc.)

    Returns:
        torch.Tensor: Solution tensor X with the same shape as B.

    Raises:
        TypeError: If A is not a sparse tensor with supported layout (COO or CSR).
        ValueError: If A is not square, if B has incompatible dimensions, or if inputs
                   have mismatched dtypes.

    Note:
        - Both COO and CSR sparse formats are supported
        - The function preserves the sparsity pattern during gradient computation
        - Solver functions should be differentiable or the gradients will be computed
          using the implicit function theorem
        - For symmetric matrices, minres is often the best choice
        - For general matrices, bicgstab or gmres-like solvers work better
        - The transpose_solve is used during backpropagation; for symmetric matrices
          it can be the same as the forward solve

    Example:
        >>> import torch
        >>> from torchsparsegradutils.utils import linear_cg, bicgstab, minres
        >>> # Create sparse system
        >>> A = torch.sparse_coo_tensor([[0,1,1],[0,0,1]], [2.0,1.0,3.0], (2,2))
        >>> B = torch.tensor([1.0, 2.0])
        >>> # Solve with different solvers
        >>> X1 = sparse_generic_solve(A, B, solve=linear_cg)    # Conjugate gradient
        >>> X2 = sparse_generic_solve(A, B, solve=bicgstab)     # BiCGSTAB
        >>> X3 = sparse_generic_solve(A, B)                     # Uses minres by default
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

    # Check dtype compatibility (optional warning, not strict requirement)
    if A.dtype != B.dtype:
        import warnings

        warnings.warn(
            f"A and B have different dtypes: A={A.dtype}, B={B.dtype}. "
            "This may cause unexpected behavior in some solvers.",
            UserWarning,
            stacklevel=2,
        )

    # Set default solvers
    if solve is None or transpose_solve is None:
        from .utils import minres

        if solve is None:
            solve = minres
        if transpose_solve is None:
            # MINRES assumes A to be symmetric -> no need to transpose A
            transpose_solve = minres

    return SparseGenericSolve.apply(A, B, solve, transpose_solve, kwargs)


class SparseGenericSolve(torch.autograd.Function):
    """
    Solves a system of equations with a square
    invertible sparse matrix A and dense right-hand side matrix B,
    with backpropagation support

    Solves: Ax = B

    A can be in either COO or CSR format.
    solve: higher level function that solves for solution to the linear equation. This function need not be differentiable.
    transpose_solve: higher level function for solving the transpose linear equation. This function need not be differentiable.

    This implementation preserves the sparsity of the gradient
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
    def backward(ctx, grad):
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
