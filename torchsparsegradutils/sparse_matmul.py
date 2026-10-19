from typing import Literal, cast

import torch

from torchsparsegradutils.utils import stack_csr

CoalesceStrategy = Literal["error", "when_needed", "assume_unique", "forward"]


def sparse_mm(A: torch.Tensor, B: torch.Tensor, *, coalesce_strategy: CoalesceStrategy = "error") -> torch.Tensor:
    r"""Sparse–dense matrix multiplication with memory-efficient gradients.

     Computes :math:`\mathbf{C} = \mathbf{A}\,\mathbf{B}` where
     :math:`\mathbf{A} \in \mathbb{R}^{n\times m}` is sparse (COO/CSR),
     :math:`\mathbf{B} \in \mathbb{R}^{m\times p}` is dense, and
     :math:`\mathbf{C} \in \mathbb{R}^{n\times p}`. Gradients preserve the sparsity pattern
     of :math:`\mathbf{A}`. Supports unbatched 2D ``(n,m) @ (m,p)`` and batched 3D inputs by
     mapping the batch into a block-diagonal sparse matrix without sorting its items.
     Sparse-input gradients are calculated one item at a time to bound temporary memory.

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

    coalesce_strategy : {"error", "when_needed", "assume_unique", "forward"}, keyword-only
        COO preparation policy; ignored for CSR after validating the option.
        The default, ``"error"``, rejects an uncoalesced COO input, even if its
        coordinates happen to be unique. Call ``A.coalesce()`` once and reuse it,
        or explicitly choose another strategy.
        ``"when_needed"`` coalesces for the sparse-input gradient on CPU and
        before forward multiplication on CUDA, reusing native kernel preparation.
        ``"assume_unique"`` performs no explicit coalescing: the caller guarantees
        unique coordinates, which may be unsorted. Duplicates can give incorrect
        gradients; this promise is not checked. Native kernels may still sort.
        ``"forward"`` coalesces before multiplication on every device and reuses
        that operand for backward. None of these policies mutates the input.

    Returns
    -------
    torch.Tensor
        Dense result of shape ``(n, p)`` or ``(b, n, p)``.

    Raises
    ------
    ValueError
        If ``A`` or ``B`` are not tensors; if ranks are < 2 or not both 2D/3D;
        if layouts are incompatible (``A`` not COO/CSR or ``B`` not dense);
        if shapes are incompatible (batch or inner dims); if the strategy is invalid;
        or if ``"error"`` receives an uncoalesced COO tensor.
    RuntimeError
        If the underlying sparse matmul fails.

    Notes
    -----
    This avoids dense gradients for sparse matrices [1a]_ (a known issue with
    :func:`torch.sparse.mm` backprop), computing only gradients at the nonzero
    entries of :math:`A` to reduce memory use.

    Unlike earlier versions, uncoalesced COO inputs raise by default. Preparation
    is independent of whether gradients are required. Batched COO inputs are
    checked before slicing, and a known coalesced flag is preserved for each
    slice without sorting again. Forward and dense-input backward use a single
    block-diagonal multiplication. Sparse-input backward processes one item at a
    time to limit temporary memory, at the cost of extra kernel launches.
    Pre-coalescing a reused operator avoids repeated sorting in safe strategies.

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

    if coalesce_strategy not in ("error", "when_needed", "assume_unique", "forward"):
        raise ValueError("coalesce_strategy must be 'error', 'when_needed', 'assume_unique', or 'forward'")
    if A.layout == torch.sparse_coo and coalesce_strategy == "error" and not A.is_coalesced():
        raise ValueError(
            "Uncoalesced COO input: call A.coalesce() first or explicitly select "
            "coalesce_strategy='when_needed', 'forward', or 'assume_unique' "
            "(only if coordinates are unique)."
        )
    return cast(torch.Tensor, SparseMatMul.apply(A, B, coalesce_strategy))


def _batch_to_block_diag(A: torch.Tensor) -> torch.Tensor:
    """Map a fixed-shape batch without sorting or copying its values."""
    batch, rows, cols = A.shape
    if batch == 1:
        if A.layout == torch.sparse_coo:
            return torch.sparse_coo_tensor(A._indices()[1:], A._values(), (rows, cols), is_coalesced=A.is_coalesced())
        return A[0]
    shape = (batch * rows, batch * cols)
    if A.layout == torch.sparse_coo:
        batch_idx, row_idx, col_idx = A._indices()
        indices = torch.stack((batch_idx * rows + row_idx, batch_idx * cols + col_idx))
        # This injective mapping preserves lexicographic ordering when known.
        return torch.sparse_coo_tensor(indices, A._values(), shape, is_coalesced=A.is_coalesced())
    nnz = A.col_indices().size(-1)
    offsets = torch.arange(batch, dtype=A.crow_indices().dtype, device=A.device)
    crow = torch.cat(
        ((A.crow_indices()[:, :-1] + offsets[:, None] * nnz).reshape(-1), A.crow_indices().new_full((1,), batch * nnz))
    )
    col = (A.col_indices() + offsets[:, None] * cols).reshape(-1)
    return torch.sparse_csr_tensor(crow, col, A.values().reshape(-1), shape)


def _batch_items(A: torch.Tensor):
    if A.size(0) == 1:
        yield _batch_to_block_diag(A)
        return
    if A.layout == torch.sparse_coo and A.is_coalesced():
        indices, values = A.indices(), A.values()
        # Known ordering makes each batch a contiguous range. Transfer only the
        # boundaries once, avoiding a filtering kernel/synchronization per item.
        boundaries = torch.searchsorted(indices[0].contiguous(), torch.arange(A.size(0) + 1, device=A.device)).tolist()
        for start, end in zip(boundaries, boundaries[1:]):
            yield torch.sparse_coo_tensor(indices[1:, start:end], values[start:end], A.shape[1:], is_coalesced=True)
    else:
        for i in range(A.size(0)):
            yield A[i]


def _sparse_gradient(A: torch.Tensor, B: torch.Tensor, grad: torch.Tensor, strategy: CoalesceStrategy) -> torch.Tensor:
    if A.layout == torch.sparse_coo:
        if strategy != "assume_unique":
            A = A.coalesce()
        rows, cols = A._indices()
    else:
        cols = A.col_indices()
        crow = A.crow_indices()
        rows = torch.repeat_interleave(
            torch.arange(A.size(0), device=A.device), crow[1:] - crow[:-1], output_size=cols.numel()
        )
    values = (grad.index_select(0, rows) * B.index_select(0, cols)).sum(dim=1)
    if A.layout == torch.sparse_coo:
        return torch.sparse_coo_tensor(A._indices(), values, A.shape, is_coalesced=A.is_coalesced())
    return torch.sparse_csr_tensor(A.crow_indices(), cols, values, A.shape)


class SparseMatMul(torch.autograd.Function):
    """Sparse-aware autograd with per-item sparse-gradient temporary storage."""

    @staticmethod
    def forward(ctx, A, B, coalesce_strategy):
        ctx.coalesce_strategy = coalesce_strategy
        A, B = A.detach(), B.detach()
        if A.layout == torch.sparse_coo and (
            coalesce_strategy == "forward" or (coalesce_strategy == "when_needed" and A.is_cuda)
        ):
            A = A.coalesce()
        batched = A.dim() == 3
        flat_A = _batch_to_block_diag(A) if batched else A
        ctx.save_for_backward(A, B, flat_A)
        # PyTorch 2.5 CUDA CSR multiplication misread strided single-column operands
        # in our regression tests; the same cases passed on PyTorch 2.14 nightly.
        flat_B = B.reshape(-1, B.size(-1)) if batched else B
        result = torch.sparse.mm(flat_A, flat_B.contiguous())
        return result.reshape(A.size(0), A.size(1), B.size(2)) if batched else result

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        A, B, flat_A = ctx.saved_tensors
        need_A, need_B = ctx.needs_input_grad[:2]
        strategy = ctx.coalesce_strategy
        if A.dim() == 2:
            gradA = _sparse_gradient(A, B, grad, strategy) if need_A else None
            gradB = torch.sparse.mm(A.t(), grad.contiguous()) if need_B else None
            return gradA, gradB, None

        gradB = None
        gradA = None
        if need_A:
            parts = [_sparse_gradient(item, B[i], grad[i], strategy) for i, item in enumerate(_batch_items(A))]
            if not parts:
                gradA = A.clone()  # A zero-length batch has no stored values.
            elif A.layout == torch.sparse_csr:
                gradA = stack_csr(parts)
            else:
                indices = torch.cat(
                    [
                        torch.cat((torch.full_like(part._indices()[:1], i), part._indices()))
                        for i, part in enumerate(parts)
                    ],
                    dim=1,
                )
                gradA = torch.sparse_coo_tensor(
                    indices,
                    torch.cat([part._values() for part in parts]),
                    A.shape,
                    is_coalesced=all(part.is_coalesced() for part in parts),
                )
        if need_B:
            flat_grad = grad.reshape(-1, grad.size(-1))
            gradB = torch.sparse.mm(flat_A.t(), flat_grad.contiguous()).reshape(B.shape)
        return gradA, gradB, None
