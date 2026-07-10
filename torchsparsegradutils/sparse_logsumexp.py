from typing import Sequence, Union

import torch
from torch import Tensor

__all__ = ["sparse_logsumexp"]


def _segment_logsumexp(
    values: Tensor,
    segment_ids: Tensor,
    n_segments: int,
    n_zeros_per_segment: Union[Tensor, None],
) -> Tensor:
    r"""Numerically stable log-sum-exp over scattered segments.

    Reduces ``values`` into ``n_segments`` groups, where ``segment_ids[k]`` gives
    the group of ``values[k]``. Uses the standard max-shift trick so that
    :math:`\exp` never overflows.

    Parameters
    ----------
    values : Tensor, shape ``(nnz,)``
        The explicit (nonzero) values to reduce.
    segment_ids : Tensor, shape ``(nnz,)``
        Output segment index for each value.
    n_segments : int
        Number of output segments.
    n_zeros_per_segment : Tensor or None, shape ``(n_segments,)``
        Count of structural zeros contributing ``exp(0) = 1`` to each segment, or
        ``None`` to ignore structural zeros entirely.

    Returns
    -------
    Tensor, shape ``(n_segments,)``
        Per-segment log-sum-exp. Empty segments (no values and no zeros) are ``-inf``.
    """
    device, dtype = values.device, values.dtype

    # Per-segment maximum, used purely as a numerical-stability shift. It is
    # detached: the log-sum-exp is exactly invariant to the shift, so gradients
    # must not flow through the max (otherwise autograd tracks a spurious path
    # through argmax). Empty segments start at -inf.
    max_val = torch.full((n_segments,), float("-inf"), device=device, dtype=dtype)
    max_val.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    shift = max_val.detach().clone()
    shift[shift == float("-inf")] = 0.0  # safe shift for empty segments

    # Shifted sum-exp over the explicit values.
    sum_exp = torch.zeros(n_segments, device=device, dtype=dtype)
    sum_exp.scatter_reduce_(0, segment_ids, (values - shift[segment_ids]).exp(), reduce="sum", include_self=True)

    # Structural zeros each contribute exp(0 - shift) = exp(-shift).
    if n_zeros_per_segment is not None:
        sum_exp = sum_exp + n_zeros_per_segment.to(dtype) * (-shift).exp()

    # Un-shift. Segments with no contribution at all (sum_exp == 0) stay -inf.
    result = shift + sum_exp.log()
    return torch.where(sum_exp == 0.0, torch.full_like(result, float("-inf")), result)


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
        row_nnz = input.crow_indices()[1:] - input.crow_indices()[:-1]
        rows = torch.repeat_interleave(torch.arange(nrows, device=vals.device), row_nnz)
        return rows, input.col_indices().long(), vals, row_nnz, None
    if input.layout == torch.sparse_csc:
        vals = input.values()
        col_nnz = input.ccol_indices()[1:] - input.ccol_indices()[:-1]
        cols = torch.repeat_interleave(torch.arange(ncols, device=vals.device), col_nnz)
        return input.row_indices().long(), cols, vals, None, col_nnz
    coo = input.coalesce()
    rows, cols = coo.indices()
    return rows, cols, coo.values(), None, None


def sparse_logsumexp(
    input: Tensor,
    dim: Union[int, Sequence[int]],
    keepdim: bool = False,
    include_zeros: bool = True,
) -> Tensor:
    r"""Sparse-aware log-sum-exp, mirroring :func:`torch.logsumexp`.

    Computes :math:`\log \sum \exp(x)` along ``dim`` directly on the explicit
    (nonzero) values of a 2-D sparse tensor, without materialising the dense
    equivalent. The reduction is numerically stable via the max-shift trick.

    Parameters
    ----------
    input : Tensor
        A 2-D sparse tensor with layout ``torch.sparse_coo``, ``torch.sparse_csr``
        or ``torch.sparse_csc``. Any other layout raises ``NotImplementedError``.
    dim : int or sequence of int
        Dimension(s) to reduce. ``dim=1`` reduces the columns (one value per row),
        ``dim=0`` reduces the rows (one value per column), and ``[0, 1]`` reduces to
        a scalar.
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
        If ``input`` is not a supported 2-D sparse layout.

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
    """
    if input.ndim != 2:
        raise NotImplementedError(f"sparse_logsumexp supports 2-D sparse tensors only, got ndim={input.ndim}.")

    supported = {torch.sparse_coo, torch.sparse_csr, torch.sparse_csc}
    if input.layout not in supported:
        raise NotImplementedError(f"sparse_logsumexp does not support layout {input.layout}. Supported: {supported}.")

    # Normalise dim to a sorted set of non-negative ints.
    dims = [dim] if isinstance(dim, int) else list(dim)
    dims = sorted({d % 2 for d in dims})

    nrows, ncols = input.shape

    # Read per-nnz row/col/value directly (no COO copy for CSR/CSC). row_nnz /
    # col_nnz are the free per-axis nonzero counts when the layout provides them.
    rows, cols, vals, row_nnz, col_nnz = _row_col_val(input, nrows, ncols)

    if dims == [0, 1]:
        # Reduce everything -> scalar.
        n_zeros = (
            torch.tensor([nrows * ncols - vals.numel()], device=vals.device, dtype=vals.dtype)
            if include_zeros
            else None
        )
        result = _segment_logsumexp(vals, torch.zeros_like(rows), 1, n_zeros).squeeze(0)
        return result.reshape(1, 1) if keepdim else result

    if dims == [1]:
        # Reduce columns -> one value per row.
        if include_zeros:
            row_nnz = row_nnz if row_nnz is not None else torch.bincount(rows, minlength=nrows)
            n_zeros = (ncols - row_nnz).to(vals.dtype)
        else:
            n_zeros = None
        result = _segment_logsumexp(vals, rows, nrows, n_zeros)
        return result.unsqueeze(1) if keepdim else result

    # dims == [0]: reduce rows -> one value per column.
    if include_zeros:
        col_nnz = col_nnz if col_nnz is not None else torch.bincount(cols, minlength=ncols)
        n_zeros = (nrows - col_nnz).to(vals.dtype)
    else:
        n_zeros = None
    result = _segment_logsumexp(vals, cols, ncols, n_zeros)
    return result.unsqueeze(0) if keepdim else result
