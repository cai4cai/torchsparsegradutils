from typing import cast

import torch

from torchsparsegradutils._compat import sparse_mm_compat
from torchsparsegradutils.utils import sparse_block_diag, sparse_block_diag_split, stack_csr


def sparse_mm(A: torch.Tensor, B: torch.Tensor, *, assume_unique: bool = False) -> torch.Tensor:
    r"""Sparse–dense matrix multiplication with memory-efficient gradients.

     Computes :math:`\mathbf{C} = \mathbf{A}\,\mathbf{B}` where
     :math:`\mathbf{A} \in \mathbb{R}^{n\times m}` is sparse (COO/CSR),
     :math:`\mathbf{B} \in \mathbb{R}^{m\times p}` is dense, and
     :math:`\mathbf{C} \in \mathbb{R}^{n\times p}`. Gradients preserve the sparsity pattern
     of :math:`\mathbf{A}`. Supports unbatched 2D ``(n,m) @ (m,p)`` and batched 3D inputs by
     block-diagonalising the batch of sparse matrices and concatenating dense matrices along
     the batch dimension.

     Let the upstream gradient be :math:`\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{C}} \in \mathbb{R}^{n\times p}`.
     The gradients are:

     Gradient with respect to B (dense):

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{B}} \;=\; \mathbf{A}^{\top} \, \mathbf{G}.

     Gradient with respect to A (sparse): For a dense view one has

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{A}} \;=\; \mathbf{G}\, \mathbf{B}^{\top},

     but we evaluate only the entries at the nonzeros of :math:`\mathbf{A}`. Equivalently,
     for a nonzero entry :math:`\mathbf{A}_{ij}` the contribution is

     .. math::
         \bigg[\frac{\partial \mathcal{L}}{\partial \mathbf{A}}\bigg]_{ij}
         \;=\; \sum_{k=1}^{p} \mathbf{G}_{ik} \, \mathbf{B}_{jk}
         \;=\; \mathbf{G}_{i,:} \,\cdot\, \mathbf{B}_{j,:},

     where the dot denotes a row-wise inner product across the :math:`p` right-hand sides.

    Parameters
    ----------
    A : torch.Tensor, sparse COO or CSR, shape ``(n, m)`` or ``(b, n, m)``
        Left operand. For batched input, all batch items must share ``(n, m)``. All tensors
        must be on the same device.
    B : torch.Tensor, dense (strided), shape ``(m, p)`` or ``(b, m, p)``
        Right operand. Must have the same number of dimensions as ``A`` and
        matching batch size / inner dimension ``m``.

    assume_unique : bool, keyword-only, default=False
        For COO inputs, skip the coalesced-input check when True: the caller
        guarantees unique coordinates, which may be unsorted. This promise is
        not checked; duplicates can produce incorrect gradients for A.
        By default, COO inputs must be flagged coalesced when gradient recording
        is enabled and A.requires_grad is True. Call A.coalesce() before passing
        it here, or use assume_unique=True only for known-unique coordinates.
        Uncoalesced inputs are accepted for inference and for B-only
        differentiation when A.requires_grad is False.
        Ignored for CSR after validating that the option is a bool.

    Returns
    -------
    torch.Tensor
        Dense result of shape ``(n, p)`` or ``(b, n, p)``.

    Raises
    ------
    ValueError
        If ``A`` or ``B`` are not tensors; if ranks are < 2 or not both 2D/3D;
        if layouts are incompatible (``A`` not COO/CSR or ``B`` not dense);
        if shapes are incompatible (batch or inner dims); if assume_unique is not a bool;
        or if an uncoalesced COO input requires a gradient without assume_unique=True.
    RuntimeError
        If the underlying sparse matmul fails.

    Notes
    -----
    This avoids dense gradients for sparse matrices [1a]_ (a known issue with
    :func:`torch.sparse.mm` backprop), computing only gradients at the nonzero
    entries of :math:`A` to reduce memory use.

    The gradient returned for a valid COO input preserves its stored coordinates,
    entry count, ordering and coalesced flag, including explicit zero entries.
    This applies to the tensor passed to sparse_mm, not to an earlier tensor
    containing duplicates that the caller coalesced. To train individual duplicate
    values, keep the dense values tensor as the leaf parameter and pass
    torch.sparse_coo_tensor(indices, values, shape).coalesce() here. PyTorch
    distributes the coordinate derivative to each original value during backward.

    Compared with earlier releases, uncoalesced trainable COO inputs now raise
    by default. Pass A.coalesce(), or assume_unique=True for known-unique indices.

    Outside inference_mode(), the default check uses torch.is_grad_enabled()
    and A.requires_grad. It is disabled under no_grad(), but cannot anticipate a later
    autograd.grad call that requests only B while A also requires gradients.
    With reentrant checkpointing, validation occurs during backward recomputation.
    Inference mode takes precedence even if enable_grad() is nested inside it.

    Neither option mutates the input or explicitly coalesces unbatched operands.
    Native kernels may still sort internally. For COO batches with more than
    one item, block-diagonal assembly coalesces extracted items that are not
    flagged as coalesced. Extracted items may lose that flag, so pre-coalescing
    a batch may not avoid further per-item coalescing. The original COO indices
    are retained for its gradient, independently of the assembly's ordering.
    Broader batching performance and memory improvements are separate work.

    See Also
    --------
    torch.sparse.mm : PyTorch's native sparse ``@`` dense.
    sparse_generic_lstsq : Sparse least-squares with sparse-aware gradients.

    References
    ----------
    .. [1a] PyTorch issue on dense gradients for sparse ops:
           https://github.com/pytorch/pytorch/issues/41128

    Examples
    --------
    Basic (unbatched)::

        >>> indices = torch.tensor([[0, 0, 1, 1, 2, 2],
        ...                         [0, 2, 1, 3, 0, 2]])
        >>> values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> A = torch.sparse_coo_tensor(indices, values, (3, 4)).coalesce()
        >>> B = torch.randn(4, 2)
        >>> out = sparse_mm(A, B)
        >>> out.shape
        torch.Size([3, 2])

    Batched::

        >>> A_batch = torch.stack([A, A]).coalesce()          # (2, 3, 4) — COO stack
        >>> B_batch = torch.randn(2, 4, 2)         # (2, 4, 2)
        >>> out = sparse_mm(A_batch, B_batch)
        >>> out.shape
        torch.Size([2, 3, 2])

    With gradients::

        >>> A.requires_grad_(True)  # doctest: +ELLIPSIS
        tensor(...)
        >>> B.requires_grad_(True)  # doctest: +ELLIPSIS
        tensor(...)
        >>> out = sparse_mm(A, B)
        >>> out.sum().backward()
        >>> A.grad.is_sparse
        True
    """

    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise ValueError("Both A and B should be instances of torch.Tensor")
    if A.dim() < 2 or B.dim() < 2:
        raise ValueError("Both A and B should be at least 2-dimensional tensors")
    if A.dim() != B.dim() or A.dim() not in (2, 3):
        raise ValueError("A and B must both be 2D or both be 3D tensors")
    if A.layout not in {torch.sparse_coo, torch.sparse_csr}:
        raise ValueError("A should be in either COO or CSR sparse format")
    if B.layout != torch.strided:
        raise ValueError("B must be a dense (strided) tensor")
    if A.dim() == 3 and A.size(0) != B.size(0):
        raise ValueError("If batched, A and B must have the same batch size")
    if A.size(-1) != B.size(-2):
        raise ValueError(f"Incompatible inner dimensions: A[..., {A.size(-1)}] vs B[..., {B.size(-2)}]")

    if not isinstance(assume_unique, bool):
        raise ValueError("assume_unique must be a bool")
    if torch.is_inference_mode_enabled():
        # Unlike native ops, Function.apply can attach a grad_fn when enable_grad
        # is nested inside inference_mode. Keep inference calls gradient-free.
        with torch.no_grad():
            return cast(torch.Tensor, SparseMatMul.apply(A, B))
    if (
        A.layout == torch.sparse_coo
        and torch.is_grad_enabled()
        and A.requires_grad
        and not assume_unique
        and not A.is_coalesced()
    ):
        raise ValueError(
            "Uncoalesced COO input requiring a gradient: call A.coalesce() first "
            "or pass assume_unique=True only if coordinates are unique."
        )
    return cast(torch.Tensor, SparseMatMul.apply(A, B))


class SparseMatMul(torch.autograd.Function):
    r"""Autograd kernel for memory-efficient sparse matrix multiplication.

    See Also
    --------
    sparse_mm : User-facing function that calls this autograd function.
    torch.sparse.mm : PyTorch's native sparse matrix multiplication.
    """

    @staticmethod
    def forward(ctx, A, B):
        ctx.batch_size = B.size()[0] if B.dim() == 3 else None
        ctx.A_shape = A.size()  # (b), n, m
        ctx.B_shape = B.size()  # (b), m, p

        # Preserve COO storage order before batch assembly may reorder the slices.
        # Save only indices, not another copy of the original values.
        coo_indices = A._indices() if A.layout == torch.sparse_coo and ctx.needs_input_grad[0] else None
        ctx.coo_is_coalesced = A.is_coalesced() if A.layout == torch.sparse_coo else None
        A, B = A.detach(), B.detach()
        if ctx.batch_size is not None:
            if torch.is_inference_mode_enabled() and not A.is_inference():
                # Sparse unbind can fail for tensors created outside inference_mode
                # (observed on 2.5 and 2.14 nightly). Keep assembly gradient-free.
                with torch.inference_mode(False), torch.no_grad():
                    A = sparse_block_diag(*A)
            else:
                A = sparse_block_diag(*A)
            B = B.reshape(-1, B.size(-1))

        x = sparse_mm_compat(A, B)

        ctx.save_for_backward(A, B, coo_indices)

        if ctx.batch_size is not None:
            x = x.view(ctx.batch_size, ctx.A_shape[-2], ctx.B_shape[-1])

        return x

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        A, B, coo_indices = ctx.saved_tensors

        gradA = None
        gradB = None

        # -------- Only compute gradA if needed --------
        if ctx.needs_input_grad[0]:
            # The gradient with respect to the matrix A, seen as a dense matrix, would
            # lead to a backprop rule as follows: gradA = grad @ b.T
            # but we are only interested in the gradient with respect to
            # the (non-zero) values of A. To save memory, instead of computing the full
            # dense matrix prev_grad @ b and then subsampling at the nnz locations in A,
            # we can directly only compute the required values:
            # grad_a[i,j] = dotprod(grad[i,:], b[j,:])

            # We start by getting the i and j indices:

            if A.layout == torch.sparse_coo:
                if ctx.batch_size is None:
                    A_row_idx, A_col_idx = coo_indices
                else:
                    batch_idx, row_idx, col_idx = coo_indices
                    # Address block-diagonal rows in the original stored order.
                    A_row_idx = batch_idx * ctx.A_shape[-2] + row_idx
                    A_col_idx = batch_idx * ctx.A_shape[-1] + col_idx
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
                grad_for_A = grad.reshape(-1, grad.size(-1))
            else:
                grad_for_A = grad

            grad_select = grad_for_A.index_select(0, A_row_idx)  # grad[i, :]
            B_select = B.index_select(0, A_col_idx)  # B[j, :]

            # Dot product:
            gradA = (grad_select * B_select).sum(dim=1)

            # Create a sparse matrix of the gradient with respect to the nnz of A
            if A.layout == torch.sparse_coo:
                gradA = torch.sparse_coo_tensor(coo_indices, gradA, ctx.A_shape, is_coalesced=ctx.coo_is_coalesced)
            elif A.layout == torch.sparse_csr:
                gradA = torch.sparse_csr_tensor(A.crow_indices(), A_col_idx, gradA, A.shape)

            if ctx.batch_size is not None and A.layout == torch.sparse_csr:
                shapes = ctx.A_shape[0] * (ctx.A_shape[-2:],)
                gradA = stack_csr(list(sparse_block_diag_split(gradA, *shapes)))

        # -------- Only compute gradB if needed --------
        if ctx.needs_input_grad[1]:
            if ctx.batch_size is not None:
                grad_for_B = grad.reshape(-1, grad.size(-1))
            else:
                grad_for_B = grad

            # Now compute the dense gradient with respect to B
            gradB = sparse_mm_compat(A.t(), grad_for_B)

            if ctx.batch_size is not None:
                gradB = gradB.view(ctx.B_shape)

        return gradA, gradB
