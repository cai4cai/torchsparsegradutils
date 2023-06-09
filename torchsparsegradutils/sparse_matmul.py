import torch
from torchsparsegradutils.utils import sparse_block_diag


def sparse_mm(A, B):
    """
    Performs matrix multiplication between a sparse tensor (A) and a dense tensor (B), with sparse gradient support for backpropagation.
    The sparse tensor A should be in COO or CSR format. This function does not support hybrid sparse tensors.
    
    Parameters:
    A (torch.Tensor): A 2D tensor in either COO or CSR format, representing a sparse matrix. The first dimension is the number of rows and the second is the number of columns.
    B (torch.Tensor): A 2D tensor, representing a dense matrix. The first dimension is the number of rows (should be equal to the number of columns of A) and the second is the number of columns.
    
    Returns:
    torch.Tensor: The result of the matrix multiplication, a 2D tensor with the first dimension equal to the number of rows in A and the second equal to the number of columns in B.

    """
    
    if A.layout != torch.sparse_coo and A.layout != torch.sparse_csr:
        raise ValueError('A must be sparse matrix in COO or CSR format')
    
    if A.dense_dim() != 0:
        raise ValueError('This function does not support hybrid sparse tensors, where dense_dim() > 0, the sparse values are expected to be scalars')
    
    if A.dim() != 2:
        raise ValueError('A must be a 2D tensor, representing a sparse matrix')
    
    if B.dim() != 2:
        raise ValueError('B must be a 2D tensor, representing a dense matrix')
    
    return SparseMatMul.apply(A, B)


def sparse_bmm_old(A, B):
    """
    Performs batched matrix multiplication between a sparse tensor (A) and a dense tensor (B), with sparse gradient support for backpropagation.
    
    The sparse tensor A should be in COO or CSR format.
    
    Parameters:
    A (torch.Tensor): A 3D tensor in either COO or CSR format. The first dimension is the batch size, the second dimension is the number of rows and the third is the number of columns.
    B (torch.Tensor): A 3D tensor, representing a batched dense matrix. The first dimension is the batch size, the second dimension is the number of rows (should be equal to the number of columns of A) and the third is the number of columns.
    
    Returns:
    torch.Tensor: The result of the batched matrix multiplication, a 3D tensor with the first dimension being the batch size, the second dimension equal to the number of rows in A and the third equal to the number of columns in B.
    
    
    NOTE: Batched sparse COO tensors are not currently technically supported in pytorch. As they appear as 3 sparse dimensions.
          Whereas, sparse CSR does support batched tensors, where batchsize = tensor.shape[:-tensor.sparse_dim() - tensor.dense_dim()]
    """

    if A.layout != torch.sparse_coo and A.layout != torch.sparse_csr:
        raise ValueError('A must be batched sparse matrix in COO or CSR format')
    
    if A.dense_dim() != 0:
        raise ValueError('This function does not support hybrid sparse tensors, where dense_dim() > 0, the sparse values are expected to be scalars')
    
    if A.dim() != 3:
        raise ValueError('A must be a 3D tensor, representing a batched sparse matrix')
    
    if B.dim() != 3:
        raise ValueError('B must be a 3D tensor, representing a batched dense matrix')
    
    if A.shape[0] != B.shape[0]:
        raise ValueError('Batch sizes of A and B must match')
    
    batch_size, N, _ = A.shape
    _, _, K = B.shape
    
    A = sparse_block_diag(*A)
    B = torch.cat([*B])
    
    return SparseMatMul.apply(A, B).reshape(batch_size, N, K)





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
            if not A.is_coalesced():
                A = A.coalesce()
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


def sparse_bmm_func(A, B):
    return SparseMatMulVMap().apply(A, B)

def sparse_bmm(A, B):
    return torch.vmap(sparse_bmm_func, in_dims=(0, 0))(A, B)

class SparseMatMulVMap(torch.autograd.Function):
    """
    Matrix multuplication between sparse matrix (A)
    and dense matrix (B), with support for backpropagation
    Matrix A can be in either COO or CSR format

    This implementation provides a memory efficient version of
    torch.sparse.mm on the backward pass, working around the issue described in:
    https://github.com/pytorch/pytorch/issues/41128
    """
    # generate_vmap_rules = True

    @staticmethod
    def forward(A, B):
        grad_flag = A.requires_grad or B.requires_grad
        A, B = A.detach(), B.detach()
        x = torch.sparse.mm(A, B)
        x.requires_grad = grad_flag
        return x
    
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        A, B = inputs
        ctx.save_for_backward(A, B)
    
    @staticmethod
    def vmap(info, in_dims, A, B):
        A_bdim, B_bdim = in_dims
        return SparseMatMulVMap.apply(A[A_bdim], B[B_bdim]), 0

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
            if not A.is_coalesced():
                A = A.coalesce()
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
    
    
