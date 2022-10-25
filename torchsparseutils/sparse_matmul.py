import torch

def sparse_matmul(A, B):
    return SparseMatMul.apply(A, B)

class SparseMatMul(torch.autograd.Function):
    """
    Matrix multuplication between sparse matrix (A) 
    and dense matrix (B), which supports backpropagation
    Matrix A can be in either COO or CSR format
    """
    @staticmethod
    def forward(ctx, A, B):
        A, B = A.detach(), B.detach()
        x = torch.sparse.mm(A, B)
        x.requires_grad = True
        ctx.save_for_backward(A, B)
        return x

    @staticmethod
    def backward(ctx, grad):
        A, B = ctx.saved_tensors
        
        if A.layout == torch.sparse_coo:
            A_row_idx, A_col_idx = A.indices()
        elif A.layout == torch.sparse_csr:
            A_col_idx = A.col_indices()
            A_crow_idx = A.crow_indices()
            A_row_idx = torch.repeat_interleave(torch.arange(A.size()[0], device=A.device), 
                                                A_crow_idx[1:]-A_crow_idx[:-1])
        else:
            raise ValueError(f"Unsupported layout: {A.layout}")

        grad_select = grad.index_select(0, A_row_idx)
        B_select = B.index_select(0, A_col_idx)
        
        gradB_ewise = grad_select * B_select
        gradA = torch.sum(gradB_ewise, dim=1)
        
        if A.layout == torch.sparse_coo:
            gradA = torch.sparse_coo_tensor(A.indices(), gradA, A.shape)
        elif A.layout == torch.sparse_csr:
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)
            
        gradB = torch.sparse.mm(A.t(), grad)
        return gradA, gradB