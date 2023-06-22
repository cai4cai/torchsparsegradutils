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

        For the backpropagation, a workaround is implemented for a known issue with
        torch.triangular_solve on the CPU for lower triangular matrices. This issue and the
        subsequent workaround are relevant only for PyTorch versions lower than 2.0 (see PyTorch
        issue #88890).

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
            A = convert_coo_to_csr(A)  # triangular solve doesn't work with sparse coo
            ctx.csr = False

        # Check if a workaround for https://github.com/pytorch/pytorch/issues/88890 is needed
        workaround88890 = (
            A.device == torch.device("cpu")
            and (not ctx.upper)
            and ctx.unitriangular
            and (torch.__version__ < (2,))
            and ctx.transpose
        )
        if not workaround88890:
            x = torch.triangular_solve(
                B.detach(), A.detach(), upper=upper, unitriangular=unitriangular, transpose=transpose
            ).solution
        else:
            n = A.shape[0]
            id_csr = torch.sparse_csr_tensor(
                torch.arange(n + 1),
                torch.arange(n),
                torch.ones(n, device=A.device, dtype=A.dtype),
                (n, n),
                device=A.device,
            )
            x = torch.triangular_solve(B.detach(), A.detach() + id_csr, upper=ctx.upper, transpose=transpose).solution

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
        # Check if a workaround for https://github.com/pytorch/pytorch/issues/88890 is needed
        workaround88890 = (
            A.device == torch.device("cpu")
            and (not ctx.upper)
            and ctx.unitriangular
            and (torch.__version__ < (2,))
            and not ctx.transpose
        )
        if not workaround88890:
            gradB = torch.triangular_solve(
                grad, A, upper=ctx.upper, transpose=not ctx.transpose, unitriangular=ctx.unitriangular
            ).solution
        else:
            # Not sure which workaround it best but for now let's assume we don't want to explicitly transpose A
            # but prefer to let torch.triangular_solve do this inernally
            n = A.shape[0]
            id_csr = torch.sparse_csr_tensor(
                torch.arange(n + 1),
                torch.arange(n),
                torch.ones(n, device=A.device, dtype=A.dtype),
                (n, n),
                device=A.device,
            )
            gradB = torch.triangular_solve(grad, A + id_csr, upper=ctx.upper, transpose=not ctx.transpose).solution

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


def sparse_generic_solve(A, B, solve=None, transpose_solve=None):
    if solve is None or transpose_solve is None:
        from .utils import minres

        if solve is None:
            solve = minres
        if transpose_solve is None:
            # MINRES assumes A to be symmetric -> no need to transpose A
            transpose_solve = minres

    return SparseGenericSolve.apply(A, B, solve, transpose_solve)


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
    def forward(ctx, A, B, solve, transpose_solve):
        grad_flag = A.requires_grad or B.requires_grad
        ctx.transpose_solve = transpose_solve

        x = solve(A.detach(), B.detach())

        x.requires_grad = grad_flag

        ctx.save_for_backward(A, x.detach())
        return x

    @staticmethod
    def backward(ctx, grad):
        A, x = ctx.saved_tensors

        # Backprop rule: gradB = A^{-T} grad
        gradB = ctx.transpose_solve(A, grad)

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

        if A.layout == torch.sparse_coo:
            gradA = torch.sparse_coo_tensor(torch.stack([A_row_idx, A_col_idx]), gradA, A.shape)
        else:
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)

        return gradA, gradB, None, None
