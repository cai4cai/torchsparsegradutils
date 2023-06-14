import torch


def sparse_generic_lstsq(A, B, lstsq=None, transpose_lstsq=None):
    if lstsq is None or transpose_lstsq is None:
        from .utils import lsmr

        if lstsq is None:
            lstsq = lambda AA, BB: lsmr(AA, BB)[0]
        if transpose_lstsq is None:
            # MINRES assumes A to be symmetric -> no need to transpose A
            transpose_lstsq = lambda AA, BB: lsmr(torch.adjoint(AA), BB, AA)[0]

    return SparseGenericLstsq.apply(A, B, lstsq, transpose_lstsq)


class SparseGenericLstsq(torch.autograd.Function):
    """
    Solves a linear least squares problem with a full-rank, tall
    sparse matrix A and dense right-hand side matrix B,
    with backpropagation support

    Solves: min_x || Ax - B ||^2

    A can be in either COO or CSR format.
    lstsq: higher level function that solves for the linear least squares problem. This function need not be differentiable.
    transpose_lstsq: higher level function for solving the transpose linear least squares problem. This function need not be differentiable.

    This implementation preserves the sparsity of the gradient. We make use of the derivation in
    Golub GH, Pereyra V. The differentiation of pseudo-inverses and nonlinear least squares problems whose variables separate.
    SIAM Journal on numerical analysis. 1973 Apr;10(2):413-32.
    We also assume that A is tall and full-rank so that A^+ A = Id where A^+ is the pseudo-inverse of A
    """

    @staticmethod
    def forward(ctx, A, B, lstsq, transpose_lstsq):
        grad_flag = A.requires_grad or B.requires_grad
        ctx.lstsq = lstsq
        ctx.transpose_lstsq = transpose_lstsq

        x = lstsq(A.detach(), B.detach())

        x.requires_grad = grad_flag

        if B.dim() == 1:
            if x.dim() == 2:
                x = x.squeeze()
        else:
            if x.dim() == 1:
                x = x.unsqueeze(1)

        ctx.save_for_backward(A.detach(), B.detach(), x.detach())
        return x

    @staticmethod
    def backward(ctx, grad):
        A, B, x = ctx.saved_tensors
        if B.dim() == 1:
            B = B.unsqueeze(1)
        if x.dim() == 1:
            x = x.unsqueeze(1)

        # Backprop rule: gradB = (A^T)^{+} grad
        gradB = ctx.transpose_lstsq(A, grad)
        if gradB.dim() == 1:
            gradB = gradB.unsqueeze(1)

        # We make use of equation 4.12 in https://www.jstor.org/stable/2156365
        # but assume A is tall and full rank to get A^+ A = Id and simplify the derivation.
        # We don't try and compute the rank of A for computational reason but at least check
        # that A is a tall matrix
        if A.shape[1] > A.shape[0]:
            raise ValueError(f"A should be a tall full-rank matrix. Got A.shape={A.shape}")
        # Following the derivation in https://blog.flaport.net/solving-sparse-linear-systems-in-pytorch.html
        # but using the pseudo-inverse instead of the inverse:
        # The gradient with respect to the matrix A seen as a dense matrix would
        # lead to a backprop rule as follows
        # gradA = -((A^T)^{+} grad)(A^{+} B) - (Ax-B)(A^+ (A^T)^{+} grad )
        #       = - gradB @ x.T - (Ax-B) @ (A^+ gradB).T
        # but we are only interested in the gradient with respect to
        # the (non-zero) values of A. To save memory, instead of computing the full
        # dense matrices gradB @ x.T and (Ax-B) @ (A^+ gradB).T
        # and then subsampling at the nnz locations in A,
        # we can directly only compute the required values:
        # gradA_u1[i,j] = - dotprod(gradB[i,:], x[j,:])
        # gradA_u2[i,j] = - dotprod(residuals[i,:], (A^+ gradB)[j,:])

        # Dense equivalent
        # gradA_u1 = - gradB @ torch.t(x)
        # mresiduals = B - A@x
        # Apgb = ctx.lstsq(A,gradB)
        # if Apgb.dim() == 1:
        #     Apgb = Apgb.unsqueeze(1)
        # gradA_u2 = mresiduals @ torch.t(Apgb)
        # gradA = gradA_u1 + gradA_u2
        # return gradA, gradB, None, None

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
        gradA_u1 = torch.sum(mgbx, dim=1)

        # residuals
        mresiduals = B - A @ x
        mresidualsselect = mresiduals.index_select(0, A_row_idx)
        Apgb = ctx.lstsq(A, gradB)
        if Apgb.dim() == 1:
            Apgb = Apgb.unsqueeze(1)
        Apgbselect = Apgb.index_select(0, A_col_idx)

        # Dot product:
        mresApgb = mresidualsselect * Apgbselect
        gradA_u2 = torch.sum(mresApgb, dim=1)

        gradA = gradA_u1 + gradA_u2

        if A.layout == torch.sparse_coo:
            gradA = torch.sparse_coo_tensor(torch.stack([A_row_idx, A_col_idx]), gradA, A.shape)
        else:
            gradA = torch.sparse_csr_tensor(A_crow_idx, A_col_idx, gradA, A.shape)

        return gradA, gradB, None, None
