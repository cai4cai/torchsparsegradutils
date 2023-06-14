import torch
from torchsparsegradutils.utils import sparse_block_diag, sparse_block_diag_split, stack_csr


def sparse_mm(A, B):
    """
    Performs a matrix multiplication between a sparse matrix A and a dense matrix B,
    preserving the sparsity of the gradient with respect to A, permitting sparse backpropagation.

    The sparse matrix A can be in either COO or CSR format, and is expected
    to be 2-dimensional, with an optional leading batch dimension. The dense matrix B
    should also be 2-dimensional, with a matching optional leading batch dimension.
    The batch size must be the same for both A and B.

    Args:
        A (torch.Tensor): The sparse matrix in COO or CSR format.
        B (torch.Tensor): The dense matrix.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
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

    return SparseMatMul.apply(A, B)


class SparseMatMul(torch.autograd.Function):
    """
    Matrix multuplication between sparse matrix (A)
    and dense matrix (B), with support for backpropagation
    Matrix A can be in either COO or CSR format

    This implementation provides a memory efficient version of
    torch.sparse.mm on the backward pass, working around the issue described in:
    https://github.com/pytorch/pytorch/issues/41128
    """

    @staticmethod
    def forward(ctx, A, B):
        ctx.batch_size = B.size()[0] if B.dim() == 3 else None
        ctx.A_shape = A.size()  # (b), n, m
        ctx.B_shape = B.size()  # (b), m, p

        grad_flag = A.requires_grad or B.requires_grad

        A, B = A.detach(), B.detach()

        if ctx.batch_size is not None:
            A = sparse_block_diag(*A)
            B = torch.cat([*B])

        x = torch.sparse.mm(A, B)

        ctx.save_for_backward(A, B)

        if ctx.batch_size is not None:
            x = x.view(ctx.batch_size, ctx.A_shape[-2], ctx.B_shape[-1])

        x.requires_grad = grad_flag
        return x

    @staticmethod
    def backward(ctx, grad):
        A, B = ctx.saved_tensors

        # The gradient with respect to the matrix A, seen as a dense matrix, would
        # lead to a backprop rule as follows: gradA = grad @ b.T
        # but we are only interested in the gradient with respect to
        # the (non-zero) values of A. To save memory, instead of computing the full
        # dense matrix prev_grad @ b and then subsampling at the nnz locations in A,
        # we can directly only compute the required values:
        # grad_a[i,j] = dotprod(grad[i,:], b[j,:])

        # We start by getting the i and j indices:

        if A.layout == torch.sparse_coo:
            A_row_idx, A_col_idx = A._indices()
        elif A.layout == torch.sparse_csr:
            A_col_idx = A.col_indices()
            A_crow_idx = A.crow_indices()
            # Uncompress row indices:
            A_row_idx = torch.repeat_interleave(
                torch.arange(A.size()[0], device=A.device), A_crow_idx[1:] - A_crow_idx[:-1]
            )
        else:
            raise ValueError(f"Unsupported layout: {A.layout}")

        if ctx.batch_size is not None:
            grad = torch.cat([*grad])

        grad_select = grad.index_select(0, A_row_idx)  # grad[i, :]
        B_select = B.index_select(0, A_col_idx)  # B[j, :]

        # Dot product:
        gradB_ewise = grad_select * B_select
        gradA = torch.sum(gradB_ewise, dim=1)

        # Create a sparse matrix of the gradient with respect to the nnz of A
        if A.layout == torch.sparse_coo:
            gradA = torch.sparse_coo_tensor(A._indices(), gradA, A.shape)
        elif A.layout == torch.sparse_csr:
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)

        # Now compute the dense gradient with respect to B
        gradB = torch.sparse.mm(A.t(), grad)

        if ctx.batch_size is not None:
            shapes = ctx.A_shape[0] * (ctx.A_shape[-2:],)
            gradA = sparse_block_diag_split(gradA, *shapes)
            if A.layout == torch.sparse_coo:
                gradA = torch.stack([*gradA])
            else:
                gradA = stack_csr([*gradA])  # NOTE: torch.stack does not work for csr tensors

            gradB = gradB.view(ctx.B_shape)
        return gradA, gradB
