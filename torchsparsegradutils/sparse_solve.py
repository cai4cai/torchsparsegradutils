import torch
from .utils import convert_coo_to_csr, sparse_block_diag, sparse_block_diag_split, stack_csr


def sparse_triangular_solve(A, B, upper=True, unitriangular=False):
    return SparseTriangularSolve.apply(A, B, upper, unitriangular)


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
    def forward(ctx, A, B, upper, unitriangular):
        ctx.batch_size = B.size()[0] if B.dim() == 3 else None
        ctx.A_shape = A.size()  # (b), m, m
        ctx.B_shape = B.size()  # (b), m, p
        ctx.csr = True
        ctx.upper = upper
        ctx.ut = unitriangular
        
        grad_flag = A.requires_grad or B.requires_grad
        
        if ctx.batch_size is not None:
            A = sparse_block_diag(*A)
            B = torch.cat([*B])

        if A.layout == torch.sparse_coo:
            A = convert_coo_to_csr(A)  # triangular solve doesn't work with sparse coo
            ctx.csr = False

        x = torch.triangular_solve(B.detach(), A.detach(), upper=upper, unitriangular=unitriangular).solution

        if ctx.batch_size is not None:
            x = x.view(ctx.batch_size, ctx.A_shape[-2], ctx.B_shape[-1])
        
        x.requires_grad = grad_flag
        ctx.save_for_backward(A, x.detach())
        return x

    @staticmethod
    def backward(ctx, grad):
        if ctx.batch_size is not None:
            grad = torch.cat([*grad])
            
        A, x = ctx.saved_tensors

        # Backprop rule: gradB = A^{-T} grad
        # Check if a workaround for https://github.com/pytorch/pytorch/issues/88890 is needed
        workaround88890 = A.device == torch.device("cpu") and (not ctx.upper) and ctx.ut
        if not workaround88890:
            gradB = torch.triangular_solve(grad, A, upper=ctx.upper, transpose=True, unitriangular=ctx.ut).solution
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
            gradB = torch.triangular_solve(grad, A + id_csr, upper=ctx.upper, transpose=True).solution

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

        mgradbselect = -gradB.index_select(0, A_row_idx)  # -gradB[i, :]
        xselect = x.index_select(0, A_col_idx)  # x[j, :]

        if ctx.ut is True and torch.any(A_row_idx == A_col_idx):
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
            if A.layout == torch.sparse_coo:
                gradA = torch.stack([*gradA])
            else:
                gradA = stack_csr([*gradA])

                gradB = gradB.view(ctx.B_shape)

        return gradA, gradB, None, None


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
