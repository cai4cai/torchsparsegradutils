import torch

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg

from .jax_bindings import j2t as _j2t
from .jax_bindings import t2j as _t2j
from .jax_bindings import t2j_coo as _t2j_coo
from .jax_bindings import t2j_csr as _t2j_csr


def sparse_solve_j4t(A, B, solve=None, transpose_solve=None):
    if solve is None or transpose_solve is None:
        # Use bicgstab by default
        if solve is None:
            solve = jax.scipy.sparse.linalg.bicgstab
        if transpose_solve is None:
            transpose_solve = lambda A, B: jax.scipy.sparse.linalg.bicgstab(A.transpose(), B)

    return SparseSolveJ4T.apply(A, B, solve, transpose_solve)


class SparseSolveJ4T(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, solve, transpose_solve):
        grad_flag = A.requires_grad or B.requires_grad
        ctx.transpose_solve = transpose_solve

        if A.layout == torch.sparse_coo:
            A_j = _t2j_coo(A.detach())
        elif A.layout == torch.sparse_csr:
            A_j = _t2j_csr(A.detach())
        else:
            raise TypeError(f"Unsupported layout type: {A.layout}")
        B_j = _t2j(B.detach())

        x_j, exit_code = solve(A_j, B_j)

        x = _j2t(x_j)

        ctx.save_for_backward(A, x)
        ctx.A_j = A_j
        x.requires_grad = grad_flag
        return x

    @staticmethod
    def backward(ctx, grad):
        A, x = ctx.saved_tensors

        grad_j = _t2j(grad.detach())

        # Backprop rule: gradB = A^{-T} grad
        gradB_j, exit_code = ctx.transpose_solve(ctx.A_j.transpose(), grad_j)
        gradB = _j2t(gradB_j)

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
