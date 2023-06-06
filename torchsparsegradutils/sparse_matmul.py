import torch


def sparse_mm(A, B):
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
        grad_flag = A.requires_grad or B.requires_grad
        A, B = A.detach(), B.detach()
        x = torch.sparse.mm(A, B)
        x.requires_grad = grad_flag
        ctx.save_for_backward(A, B)
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
            A_row_idx, A_col_idx = A.indices()
        elif A.layout == torch.sparse_csr:
            A_col_idx = A.col_indices()
            A_crow_idx = A.crow_indices()
            # Uncompress row indices:
            A_row_idx = torch.repeat_interleave(
                torch.arange(A.size()[0], device=A.device), A_crow_idx[1:] - A_crow_idx[:-1]
            )
        else:
            raise ValueError(f"Unsupported layout: {A.layout}")

        grad_select = grad.index_select(0, A_row_idx)  # grad[i, :]
        B_select = B.index_select(0, A_col_idx)  # B[j, :]

        # Dot product:
        gradB_ewise = grad_select * B_select
        gradA = torch.sum(gradB_ewise, dim=1)

        # Create a sparse matrix of the gradient with respect to the nnz of A
        if A.layout == torch.sparse_coo:
            gradA = torch.sparse_coo_tensor(A.indices(), gradA, A.shape)
        elif A.layout == torch.sparse_csr:
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)

        # Now compute the dense gradient with respect to B
        gradB = torch.sparse.mm(A.t(), grad)
        return gradA, gradB
    
    
def sparse_bmm(A, B):
    pass
