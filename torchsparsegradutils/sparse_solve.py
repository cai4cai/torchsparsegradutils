import torch

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
        grad_flag = A.requires_grad or B.requires_grad
        ctx.csr = True
        ctx.upper = upper
        ctx.ut = unitriangular
        
        if A.layout == torch.sparse_coo:  
            A = A.to_sparse_csr() # triangular solve doesn't work with sparse coo
            ctx.csr = False     
             
        x = torch.triangular_solve(B.detach(), A.detach(), upper=upper, unitriangular=unitriangular).solution
        
        x.requires_grad = grad_flag
        ctx.save_for_backward(A, x.detach())
        return x

    @staticmethod
    def backward(ctx, grad):
        A, x = ctx.saved_tensors
        
        # Backprop rule: gradB = a^{-T} grad
        gradB = torch.triangular_solve(grad, A, upper=ctx.upper, transpose=True, unitriangular=ctx.ut).solution
        
        # The gradient with respect to the matrix a seen as a dense matrix would
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
        A_row_idx = torch.repeat_interleave(torch.arange(A.size()[0], device=A.device), 
                                            A_crow_idx[1:]-A_crow_idx[:-1])     
        
        mgradbselect = -gradB.index_select(0, A_row_idx)   # -gradB[i, :]
        xselect = x.index_select(0, A_col_idx)  # x[j, :]
        
        if torch.any(A_row_idx == A_col_idx) and ctx.ut is True:
            raise ValueError(f"First input should be strictly triangular (i.e. unit diagonals is implicit)")
        
        # Dot product:
        mgbx = mgradbselect * xselect
        gradA = torch.sum(mgbx, dim=1)
            
        if ctx.csr is False:
            gradA = torch.sparse_coo_tensor(torch.stack([A_row_idx, A_col_idx]), gradA, A.shape)
        else:
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)

        return gradA, gradB, None, None