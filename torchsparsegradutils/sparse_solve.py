import torch

def sparse_triangular_solve(A, B, upper=True):
    return SparseTriangularSolve.apply(A, B, upper)
    

class SparseTriangularSolve(torch.autograd.Function):
    """
    Solves a system of equations with a square upper or lower triangular 
    invertible sparse matrix A and dense right-hand side matrix B, 
    with backpropagation support
    
    Solves: Ax = B

    A can be in either COO or CSR format.
    But, COO will internally be converted to CSR before solving.
    """
    @staticmethod
    def forward(ctx, A, B, upper):
        ctx.csr = True
        ctx.upper = upper
        if A.layout == torch.sparse_coo:  
            A = A.to_sparse_csr() # triangular solve doesn't work with sparse coo
            ctx.csr = False      
        x = torch.triangular_solve(B.detach(), A.detach(), upper=upper)[0]
        x.requires_grad = True
        ctx.save_for_backward(A, x.detach())
        return x

    @staticmethod
    def backward(ctx, grad):
        A, x = ctx.saved_tensors
        gradB = torch.triangular_solve(grad, A, upper=ctx.upper, transpose=True)[0]
        
        A_col_idx = A.col_indices()
        A_crow_idx = A.crow_indices()
        A_row_idx = torch.repeat_interleave(torch.arange(A.size()[0], device=A.device), 
                                            A_crow_idx[1:]-A_crow_idx[:-1])     
        
        mgradbselect = -gradB.index_select(0, A_row_idx)
        xselect = x.index_select(0, A_col_idx)
        
        mgbx = mgradbselect * xselect
        gradA = torch.sum(mgbx, dim=1)
            
        if ctx.csr is False:
            gradA = torch.sparse_coo_tensor(torch.stack([A_row_idx, A_col_idx]), gradA, A.shape)
        else:
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)

        return gradA, gradB, None