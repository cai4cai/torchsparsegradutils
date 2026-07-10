from typing import Sequence, Union

import torch
from torch import Tensor

__all__ = ["sparse_logsumexp"]


def _scatter_logsumexp(
    values: Tensor,
    scatter_index: Tensor,
    n_groups: int,
    n_zeros_per_group: Union[Tensor, None],
) -> Tensor:
    r"""Numerically stable log-sum-exp over scatter-reduced groups.

    Reduces ``values`` into ``n_groups`` groups, where ``scatter_index[k]`` gives
    the group of ``values[k]``. This is a *scatter* reduction, not a segmented one:
    ``scatter_index`` is an arbitrary per-value output index (via
    ``Tensor.scatter_reduce_``), so a group's members need not be contiguous. Uses
    the standard max-shift trick so that :math:`\exp` never overflows.

    Parameters
    ----------
    values : Tensor, shape ``(nnz,)``
        The explicit (nonzero) values to reduce.
    scatter_index : Tensor, shape ``(nnz,)``
        Output group index for each value.
    n_groups : int
        Number of output groups.
    n_zeros_per_group : Tensor or None, shape ``(n_groups,)``
        Count of structural zeros contributing ``exp(0) = 1`` to each group, or
        ``None`` to ignore structural zeros entirely.

    Returns
    -------
    Tensor, shape ``(n_groups,)``
        Per-group log-sum-exp. Empty groups (no values and no zeros) are ``-inf``.
    """
    device, dtype = values.device, values.dtype

    # Per-group max, detached — a stability shift the result is invariant to.
    max_val = torch.full((n_groups,), float("-inf"), device=device, dtype=dtype)
    max_val.scatter_reduce_(0, scatter_index, values, reduce="amax", include_self=True)
    shift = max_val.detach().clone()
    shift[shift == float("-inf")] = 0.0  # empty groups

    sum_exp = torch.zeros(n_groups, device=device, dtype=dtype)
    sum_exp.scatter_reduce_(0, scatter_index, (values - shift[scatter_index]).exp(), reduce="sum", include_self=True)

    # Structural zeros each contribute exp(0 - shift) = exp(-shift).
    if n_zeros_per_group is not None:
        sum_exp = sum_exp + n_zeros_per_group.to(dtype) * (-shift).exp()

    # Un-shift. Groups with no contribution at all (sum_exp == 0) stay -inf.
    result = shift + sum_exp.log()
    return torch.where(sum_exp == 0.0, float("-inf"), result)


def _row_col_val(input: Tensor, nrows: int, ncols: int):
    """Per-nnz ``(rows, cols, vals, row_nnz, col_nnz)`` for a 2-D sparse tensor.

    CSR/CSC expose their explicit values without duplicate cells, so we read the
    index arrays directly and only uncompress the compressed axis — avoiding the
    full sort that ``to_sparse_coo().coalesce()`` would pay for. COO may carry
    duplicate coordinates, so it is coalesced to merge them first.

    ``row_nnz`` / ``col_nnz`` are the per-row / per-column nonzero counts when the
    layout yields them for free (the compressed axis' segment lengths), else
    ``None`` — letting the caller skip a redundant ``bincount`` on that axis.
    """
    if input.layout == torch.sparse_csr:
        vals = input.values()
        crow = input.crow_indices()
        row_nnz = crow[1:] - crow[:-1]
        rows = torch.repeat_interleave(torch.arange(nrows, device=vals.device), row_nnz)
        return rows, input.col_indices().long(), vals, row_nnz, None
    if input.layout == torch.sparse_csc:
        vals = input.values()
        ccol = input.ccol_indices()
        col_nnz = ccol[1:] - ccol[:-1]
        cols = torch.repeat_interleave(torch.arange(ncols, device=vals.device), col_nnz)
        return input.row_indices().long(), cols, vals, None, col_nnz
    coo = input.coalesce()
    rows, cols = coo.indices()
    return rows, cols, coo.values(), None, None


def _logsumexp_2d(input: Tensor, dims, keepdim: bool, include_zeros: bool) -> Tensor:
    """Reduction for an unbatched 2-D sparse tensor (see :func:`sparse_logsumexp`)."""
    nrows, ncols = input.shape

    # One scatter path for every layout/axis: segment_reduce fits only the natural
    # CSR/CSC axis and measured 1.5x-13x slower than scatter on GPU, worst with the
    # many-small-segments common case.
    rows, cols, vals, row_nnz, col_nnz = _row_col_val(input, nrows, ncols)

    if dims == [0, 1]:
        # Reduce everything -> scalar. n_zeros structural zeros == one entry of log(n_zeros).
        flat = vals
        if include_zeros:
            n_zeros = nrows * ncols - vals.numel()
            if n_zeros > 0:
                log_nz = torch.full((1,), float(n_zeros), device=vals.device, dtype=vals.dtype).log()
                flat = torch.cat([vals, log_nz])
        result = torch.logsumexp(flat, dim=0)
        return result.reshape(1, 1) if keepdim else result

    # dims == [1]: reduce columns -> one value per row.
    # dims == [0]: reduce rows    -> one value per column.
    if dims == [1]:
        scatter_idx, n_groups, axis_nnz, full, keep_ax = rows, nrows, row_nnz, ncols, 1
    else:
        scatter_idx, n_groups, axis_nnz, full, keep_ax = cols, ncols, col_nnz, nrows, 0

    if include_zeros:
        axis_nnz = axis_nnz if axis_nnz is not None else torch.bincount(scatter_idx, minlength=n_groups)
        n_zeros = (full - axis_nnz).to(vals.dtype)
    else:
        n_zeros = None
    result = _scatter_logsumexp(vals, scatter_idx, n_groups, n_zeros)
    return result.unsqueeze(keep_ax) if keepdim else result


def _logsumexp_batched(input: Tensor, dims, keepdim: bool, include_zeros: bool) -> Tensor:
    """Reduction within each batch slice ``input[k]`` of a batched 3-D sparse tensor.

    Each ``(rows, cols)`` matrix ``input[k]`` is reduced independently: the batch
    index is folded into the scatter index, so a single scatter reduces every slice
    at once. Batched inputs go through COO (batched CSR/CSC require equal nnz per
    slice in PyTorch).
    """
    b, nrows, ncols = input.shape
    coo = input if (input.layout == torch.sparse_coo and input.is_coalesced()) else input.to_sparse_coo().coalesce()
    bidx, rows, cols = coo.indices()
    vals = coo.values()

    if dims == [1, 2]:
        # Reduce each whole slice -> (b,).
        n_zeros = (nrows * ncols - torch.bincount(bidx, minlength=b)).to(vals.dtype) if include_zeros else None
        result = _scatter_logsumexp(vals, bidx, b, n_zeros)
        return result.reshape(b, 1, 1) if keepdim else result

    # dims == [2]: reduce columns -> (b, rows).
    # dims == [1]: reduce rows    -> (b, cols).
    if dims == [2]:
        scatter_idx, n_groups, full, out_shape, keep_ax = bidx * nrows + rows, b * nrows, ncols, (b, nrows), 2
    else:
        scatter_idx, n_groups, full, out_shape, keep_ax = bidx * ncols + cols, b * ncols, nrows, (b, ncols), 1

    n_zeros = (full - torch.bincount(scatter_idx, minlength=n_groups)).to(vals.dtype) if include_zeros else None
    result = _scatter_logsumexp(vals, scatter_idx, n_groups, n_zeros).reshape(out_shape)
    return result.unsqueeze(keep_ax) if keepdim else result


def sparse_logsumexp(
    input: Tensor,
    dim: Union[int, Sequence[int]],
    keepdim: bool = False,
    include_zeros: bool = True,
) -> Tensor:
    r"""Sparse-aware log-sum-exp, mirroring :func:`torch.logsumexp`.

    Computes :math:`\log \sum \exp(x)` along ``dim`` directly on the explicit
    (nonzero) values of a 2-D (or batched 3-D) sparse tensor, without materialising
    the dense equivalent. The reduction is numerically stable via the max-shift trick.

    Parameters
    ----------
    input : Tensor
        A sparse tensor with layout ``torch.sparse_coo``, ``torch.sparse_csr`` or
        ``torch.sparse_csc``. Either an unbatched 2-D matrix, or a batched 3-D
        ``(batch, rows, cols)`` tensor whose leading dimension is an independent
        batch (each ``(rows, cols)`` slice is reduced on its own). Any other layout
        or rank raises ``NotImplementedError``.
    dim : int or sequence of int
        Dimension(s) to reduce. For a 2-D input: ``dim=1`` reduces the columns (one
        value per row), ``dim=0`` reduces the rows (one value per column), and
        ``[0, 1]`` reduces to a scalar. For a batched 3-D input the batch axis
        (``0``) cannot be reduced; ``dim`` must select from ``{1, 2}``.
    keepdim : bool, default ``False``
        If ``True``, the reduced dimension(s) are retained with size 1.
    include_zeros : bool, default ``True``
        If ``True``, structural zeros (absent entries) are treated as genuine
        zero-valued entries, each contributing ``exp(0) = 1`` to the sum. This makes
        the result match :func:`torch.logsumexp` on ``input.to_dense()``. If
        ``False``, absent entries are treated as ``-inf`` and only explicit values
        participate (appropriate when the sparsity pattern is a support mask).

    Returns
    -------
    Tensor
        Dense result following :func:`torch.logsumexp` shape conventions.

    Raises
    ------
    NotImplementedError
        If ``input`` is not a supported sparse layout, is not 2-D or 3-D, or if a
        batched 3-D input's ``dim`` includes the batch axis.

    Examples
    --------
    >>> i = torch.tensor([[0, 1, 1], [1, 0, 2]])
    >>> v = torch.tensor([1.0, 2.0, 3.0])
    >>> x = torch.sparse_coo_tensor(i, v, (3, 3))
    >>> sparse_logsumexp(x, dim=1, include_zeros=False)
    tensor([1.0000, 3.3133,   -inf])
    >>> sparse_logsumexp(x, dim=1, include_zeros=True)
    tensor([1.5514, 3.3490, 1.0986])

    ``include_zeros=True`` agrees with the dense reference:

    >>> torch.logsumexp(x.to_dense(), dim=1)
    tensor([1.5514, 3.3490, 1.0986])

    Batched 3-D input, reducing within each slice:

    >>> xb = torch.stack([x.to_dense(), x.to_dense()]).to_sparse_coo()
    >>> sparse_logsumexp(xb, dim=2)
    tensor([[1.5514, 3.3490, 1.0986],
            [1.5514, 3.3490, 1.0986]])
    """
    if input.ndim not in (2, 3):
        raise NotImplementedError(
            f"sparse_logsumexp supports 2-D or batched 3-D sparse tensors, got ndim={input.ndim}."
        )

    supported = {torch.sparse_coo, torch.sparse_csr, torch.sparse_csc}
    if input.layout not in supported:
        raise NotImplementedError(f"sparse_logsumexp does not support layout {input.layout}. Supported: {supported}.")

    # Normalise dim to a sorted set of non-negative ints for this rank.
    dims_list = [dim] if isinstance(dim, int) else list(dim)
    dims = sorted({d % input.ndim for d in dims_list})

    if input.ndim == 2:
        return _logsumexp_2d(input, dims, keepdim, include_zeros)

    if 0 in dims:
        raise NotImplementedError("Cannot reduce the batch dimension (0) of a batched 3-D sparse tensor.")
    return _logsumexp_batched(input, dims, keepdim, include_zeros)
