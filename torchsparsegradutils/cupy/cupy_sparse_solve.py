import torchsparsegradutils.cupy as tsgucupy
import torch


def sparse_solve_c4t(A, B):
    return SparseSolveC4T.apply(A, B)


class SparseSolveC4T(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        xp, xsp = tsgucupy._get_array_modules(A.data)
        grad_flag = A.requires_grad or B.requires_grad

        # Transfer data to cupy/scipy
        if A.layout == torch.sparse_coo:
            A_c = tsgucupy.t2c_coo(A.detach())
        elif A.layout == torch.sparse_csr:
            A_c = tsgucupy.t2c_csr(A.detach())
        else:
            raise TypeError(f"Unsupported layout type: {A.layout}")
        B_c = xp.asarray(B.detach())

        # Solve the sparse system
        ctx.factorisedsolver = None
        if (B.ndim == 1) or (B.shape[1] == 1):
            # xp.sparse.linalg.spsolve only works if B is a vector but is fully on GPU with cupy
            x_c = xsp.linalg.spsolve(A_c, B_c)
        else:
            # Make use of a factorisation (only the solver is then on the GPU with cupy)
            # We store it in ctx to reuse it in the backward pass
            ctx.factorisedsolver = xsp.linalg.factorized(A_c)
            x_c = ctx.factorisedsolver(B_c)

        x = torch.as_tensor(x_c, device=A.device)

        ctx.save_for_backward(A, x)
        x.requires_grad = grad_flag
        return x

    @staticmethod
    def backward(ctx, grad):
        A, x = ctx.saved_tensors
        xp, xsp = tsgucupy._get_array_modules(A.data)

        if A.layout == torch.sparse_coo:
            A_c = tsgucupy.t2c_coo(A.detach())
        elif A.layout == torch.sparse_csr:
            A_c = tsgucupy.t2c_csr(A.detach())
        else:
            raise TypeError(f"Unsupported layout type: {A.layout}")

        x_c = xp.asarray(x.detach())
        grad_c = xp.asarray(grad.detach())

        # Backprop rule: gradB = A^{-T} grad
        if ctx.factorisedsolver is None:
            gradB_c = xsp.linalg.spsolve(xp.transpose(A_c), grad_c)
        else:
            # Re-use factorised solver from forward pass
            grad_c = xp.asarray(grad)
            gradB_c = ctx.factorisedsolver(grad_c, trans="T")

        gradB = torch.as_tensor(gradB_c, device=A.device)

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

        return gradA, gradB
