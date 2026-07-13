from typing import Literal, Sequence, Union

import torch
from packaging.version import parse as parse_version
from torch import Tensor

__all__ = ["sparse_logsumexp", "sparse_bidir_logsumexp"]


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

    Any leading batch dimensions are supported: ``values`` / ``scatter_index`` of
    shape ``(*batch, nnz)`` are reduced independently along the last axis into an
    output of shape ``(*batch, n_groups)``. With no batch dims (1-D inputs) this is
    an ordinary scatter reduction.

    Parameters
    ----------
    values : Tensor, shape ``(*batch, nnz)``
        The explicit (nonzero) values to reduce.
    scatter_index : Tensor, shape ``(*batch, nnz)``
        Output group index for each value, within ``[0, n_groups)``.
    n_groups : int
        Number of output groups (size of the reduced last axis). A single value shared
        by every batch slice, since the output is a rectangular ``(*batch, n_groups)``.
        A caller whose slices need *different* numbers of groups passes the maximum over
        the slices and pads: the surplus groups of the shorter slices receive no values
        and no zeros, so they come back ``-inf`` and can be sliced off.
    n_zeros_per_group : Tensor or None, shape ``(*batch, n_groups)``
        Count of structural zeros contributing ``exp(0) = 1`` to each group, or
        ``None`` to ignore structural zeros entirely.

    Returns
    -------
    Tensor, shape ``(*batch, n_groups)``
        Per-group log-sum-exp. Empty groups (no values and no zeros) are ``-inf``.
    """
    device, dtype = values.device, values.dtype
    batch = values.shape[:-1]

    # Per-group max, detached — a stability shift the result is invariant to.
    max_val = torch.full((*batch, n_groups), float("-inf"), device=device, dtype=dtype)
    max_val.scatter_reduce_(-1, scatter_index, values, reduce="amax", include_self=True)
    if n_zeros_per_group is not None:
        max_val = torch.where(n_zeros_per_group > 0, max_val.clamp(min=0.0), max_val)
    shift = max_val.detach().clone()
    shift[~shift.isfinite()] = 0.0  # empty groups (-inf) and +inf values (avoid inf - inf)

    sum_exp = torch.zeros((*batch, n_groups), device=device, dtype=dtype)
    shifted_exp = (values - torch.gather(shift, -1, scatter_index)).exp()
    sum_exp.scatter_reduce_(-1, scatter_index, shifted_exp, reduce="sum", include_self=True)

    # Structural zeros each contribute exp(0 - shift) = exp(-shift).
    if n_zeros_per_group is not None:
        has_zeros = n_zeros_per_group > 0
        zeros_contrib = n_zeros_per_group.to(dtype) * (-shift).exp()
        sum_exp = sum_exp + torch.where(has_zeros, zeros_contrib, torch.zeros_like(sum_exp))

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
    coo = input if input.is_coalesced() else input.coalesce()
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


def _bidir_2d(input: Tensor, include_zeros: bool):
    """Row- and column-wise log-sum-exp of a 2-D sparse tensor in one batched scatter.

    Both reductions run over the *same* nonzero values, so instead of two passes we
    stack the two directions into a single batched ``_scatter_logsumexp``:
    ``batch 0`` scatters by column (reducing rows -> ``col_lse``) and ``batch 1``
    scatters by row (reducing columns -> ``row_lse``). ``values.expand(2, nnz)`` is a
    view (no copy); its backward correctly *sums* each nonzero's row and column
    gradient contributions.

    Returns ``(col_lse (ncols,), row_lse (nrows,), padded (2, G))`` where
    ``G = max(nrows, ncols)`` and the ``padded`` tail (beyond each axis' length) is
    ``-inf`` (empty groups). ``padded`` is the scatter's native output buffer, and
    ``col_lse`` / ``row_lse`` are basic-slice *views* into it — the three returns cost
    one allocation between them, not three.
    """
    nrows, ncols = input.shape
    rows, cols, vals, row_nnz, col_nnz = _row_col_val(input, nrows, ncols)
    G = max(nrows, ncols)

    batched_vals = vals.expand(2, vals.numel())  # (2, nnz) view, no copy
    batched_idx = torch.stack([cols, rows])  # batch 0 -> col_lse, batch 1 -> row_lse

    if include_zeros:
        col_nnz = col_nnz if col_nnz is not None else torch.bincount(cols, minlength=ncols)
        row_nnz = row_nnz if row_nnz is not None else torch.bincount(rows, minlength=nrows)
        n_zeros = torch.zeros(2, G, device=vals.device, dtype=vals.dtype)  # padded tail stays 0
        n_zeros[0, :ncols] = nrows - col_nnz.to(vals.dtype)
        n_zeros[1, :nrows] = ncols - row_nnz.to(vals.dtype)
    else:
        n_zeros = None

    padded = _scatter_logsumexp(batched_vals, batched_idx, G, n_zeros)  # (2, G)
    return padded[0, :ncols], padded[1, :nrows], padded


def _bidir_batched(input: Tensor, include_zeros: bool):
    """Per-slice row- and column-wise log-sum-exp of a batched 3-D sparse tensor.

    Same single batched scatter as :func:`_bidir_2d`, with the batch index folded into
    the group index (each slice padded to ``G = max(nrows, ncols)`` groups so both
    directions share one output). Batched inputs go through COO (batched CSR/CSC
    require equal nnz per slice in PyTorch).

    Returns ``(col_lse (b, ncols), row_lse (b, nrows), padded (2, b, G))``. As in
    :func:`_bidir_2d` the three share one allocation: ``padded`` is the scatter's native
    buffer and the other two are views into it.

    Note ``padded`` comes back in that native ``(direction, batch, group)`` order, not
    the ``(batch, direction, group)`` order the public ``output_layout="padded"``
    documents. Transposing it is a copy, so the caller does it only on the path that
    actually asks for that layout, rather than every call paying for a buffer the tuple
    and nested paths discard.
    """
    b, nrows, ncols = input.shape
    coo = input if (input.layout == torch.sparse_coo and input.is_coalesced()) else input.to_sparse_coo().coalesce()
    bidx, rows, cols = coo.indices()
    vals = coo.values()
    G = max(nrows, ncols)

    batched_vals = vals.expand(2, vals.numel())  # (2, nnz) view, no copy
    batched_idx = torch.stack([bidx * G + cols, bidx * G + rows])  # into b*G groups

    if include_zeros:
        col_nnz = torch.bincount(bidx * ncols + cols, minlength=b * ncols).reshape(b, ncols)
        row_nnz = torch.bincount(bidx * nrows + rows, minlength=b * nrows).reshape(b, nrows)
        n_zeros = torch.zeros(2, b, G, device=vals.device, dtype=vals.dtype)  # padded tail stays 0
        n_zeros[0, :, :ncols] = nrows - col_nnz.to(vals.dtype)
        n_zeros[1, :, :nrows] = ncols - row_nnz.to(vals.dtype)
        n_zeros = n_zeros.reshape(2, b * G)
    else:
        n_zeros = None

    padded = _scatter_logsumexp(batched_vals, batched_idx, b * G, n_zeros).reshape(2, b, G)
    return padded[0, :, :ncols], padded[1, :, :nrows], padded  # (b, ncols), (b, nrows), (2, b, G)


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
    ValueError
        If ``input`` is a hybrid sparse tensor (has dense dimensions).

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

    if input.dense_dim() != 0:
        raise ValueError("sparse_logsumexp requires a sparse tensor with zero dense dimensions.")

    # Validate against the raw dims (before normalising), matching torch.logsumexp.
    dims_list = [dim] if isinstance(dim, int) else list(dim)
    if not dims_list:
        raise RuntimeError("sparse_logsumexp: dim must not be an empty sequence.")
    for d in dims_list:
        if not -input.ndim <= d < input.ndim:
            raise IndexError(
                f"Dimension out of range (expected to be in range of "
                f"[{-input.ndim}, {input.ndim - 1}], but got {d})"
            )
    normalised = [d % input.ndim for d in dims_list]
    if len(set(normalised)) != len(normalised):
        raise RuntimeError("sparse_logsumexp: dim contains a repeated dimension.")
    dims = sorted(normalised)

    if input.ndim == 2:
        return _logsumexp_2d(input, dims, keepdim, include_zeros)

    if 0 in dims:
        raise NotImplementedError("Cannot reduce the batch dimension (0) of a batched 3-D sparse tensor.")
    return _logsumexp_batched(input, dims, keepdim, include_zeros)


def sparse_bidir_logsumexp(
    input: Tensor,
    keepdim: bool = False,
    include_zeros: bool = True,
    output_layout: Literal["tuple", "padded", "nested"] = "tuple",
) -> Union[Tensor, "tuple[Tensor, Tensor]"]:
    r"""Simultaneous row- and column-wise log-sum-exp of a sparse tensor.

    Computes the column-wise reduction (over rows) *and* the row-wise reduction (over
    columns) of a 2-D (or batched 3-D) sparse tensor in a single traversal of the sparse
    structure. For a 2-D input this equals::

        (sparse_logsumexp(input, dim=0, ...), sparse_logsumexp(input, dim=1, ...))

    For a batched 3-D input the per-slice reductions use ``dim=1`` (over rows) and
    ``dim=2`` (over columns) instead — ``dim=0`` is the batch axis and cannot be reduced.

    Every nonzero contributes to both outputs, so a single extraction feeds one batched
    :func:`~torch.Tensor.scatter_reduce_` (via ``values.expand(2, nnz)``, a view) rather
    than two separate passes; the shared extraction and single autograd graph make it
    faster than two calls (at a modestly higher peak memory, since both reductions are
    live at once). Numerically stable via the same max-shift trick as
    :func:`sparse_logsumexp`.

    Parameters
    ----------
    input : Tensor
        A sparse tensor (``torch.sparse_coo`` / ``sparse_csr`` / ``sparse_csc``), either
        an unbatched 2-D ``(rows, cols)`` matrix or a batched 3-D ``(batch, rows, cols)``
        tensor whose leading axis is an independent batch. Any other layout or rank
        raises ``NotImplementedError``.
    keepdim : bool, default ``False``
        If ``True``, the reduced dimension is retained with size 1 in each returned
        vector. Only supported with ``output_layout="tuple"`` (raises otherwise).
    include_zeros : bool, default ``True``
        As in :func:`sparse_logsumexp`: if ``True``, structural zeros contribute
        ``exp(0) = 1`` (matching :func:`torch.logsumexp` on the dense tensor); if
        ``False``, only explicit values participate.
    output_layout : {"tuple", "padded", "nested"}, default ``"tuple"``
        How to return the two reductions:

        - ``"tuple"``: ``(col_lse, row_lse)``. ``col_lse`` is the reduction over rows
          (2-D ``dim=0``; batched ``dim=1``), shape ``(cols,)`` / ``(batch, cols)``;
          ``row_lse`` is the reduction over columns (2-D ``dim=1``; batched ``dim=2``),
          shape ``(rows,)`` / ``(batch, rows)``. **Note the order: column result first.**
        - ``"padded"``: a single dense tensor of shape ``(2, G)`` (unbatched) or
          ``(batch, 2, G)`` (batched) with ``G = max(rows, cols)``. Row ``0`` is
          ``col_lse``, row ``1`` is ``row_lse``; each is padded to ``G`` with ``-inf``.
        - ``"nested"``: ``torch.nested.as_nested_tensor([col_lse, row_lse])`` — a single
          container preserving the two (possibly different) lengths. Requires
          PyTorch >= 2.4, and (as a prototype API) is meant to be consumed via
          ``.unbind()``; whole-tensor ops such as ``.sum()`` / ``.shape`` are not
          supported on it. Emits PyTorch's nested-tensor prototype ``UserWarning``.

    Returns
    -------
    tuple[Tensor, Tensor] or Tensor
        Depending on ``output_layout`` (see above).

    Raises
    ------
    NotImplementedError
        If ``input`` is not a supported sparse layout / rank, or ``output_layout=
        "nested"`` on a PyTorch older than 2.4.
    ValueError
        If ``output_layout`` is unknown, ``keepdim=True`` with a non-tuple layout, or
        ``input`` is a hybrid sparse tensor (has dense dimensions).

    Examples
    --------
    >>> i = torch.tensor([[0, 1, 1], [1, 0, 2]])
    >>> v = torch.tensor([1.0, 2.0, 3.0])
    >>> x = torch.sparse_coo_tensor(i, v, (3, 3))
    >>> col_lse, row_lse = sparse_bidir_logsumexp(x)
    >>> col_lse  # dim=0 reduction (one value per column)
    tensor([2.2395, 1.5514, 3.0949])
    >>> row_lse  # dim=1 reduction (one value per row)
    tensor([1.5514, 3.3490, 1.0986])

    Agrees with two separate :func:`sparse_logsumexp` calls:

    >>> torch.equal(col_lse, sparse_logsumexp(x, dim=0)) and torch.equal(row_lse, sparse_logsumexp(x, dim=1))
    True

    The ``"padded"`` layout stacks both into one ``(2, max(rows, cols))`` tensor:

    >>> sparse_bidir_logsumexp(x, output_layout="padded")
    tensor([[2.2395, 1.5514, 3.0949],
            [1.5514, 3.3490, 1.0986]])
    """
    if input.ndim not in (2, 3):
        raise NotImplementedError(
            f"sparse_bidir_logsumexp supports 2-D or batched 3-D sparse tensors, got ndim={input.ndim}."
        )

    supported = {torch.sparse_coo, torch.sparse_csr, torch.sparse_csc}
    if input.layout not in supported:
        raise NotImplementedError(
            f"sparse_bidir_logsumexp does not support layout {input.layout}. Supported: {supported}."
        )

    if input.dense_dim() != 0:
        raise ValueError("sparse_bidir_logsumexp requires a sparse tensor with zero dense dimensions.")

    if output_layout not in ("tuple", "padded", "nested"):
        raise ValueError(
            f"sparse_bidir_logsumexp: unknown output_layout {output_layout!r}. "
            "Expected one of 'tuple', 'padded', 'nested'."
        )
    if keepdim and output_layout != "tuple":
        raise ValueError("sparse_bidir_logsumexp: keepdim is only supported with output_layout='tuple'.")
    # Gate the nested prototype up front, before paying the forward cost.
    if output_layout == "nested" and parse_version(torch.__version__) < parse_version("2.4"):
        raise NotImplementedError("PyTorch version is too old for nested tensors")

    batched = input.ndim == 3
    col_lse, row_lse, padded = _bidir_batched(input, include_zeros) if batched else _bidir_2d(input, include_zeros)

    if output_layout == "padded":
        # The batched scatter's native buffer is (2, b, G); the documented layout is
        # (b, 2, G). That transpose is a copy, and this is the only path that needs it.
        return padded.permute(1, 0, 2).contiguous() if batched else padded

    if output_layout == "nested":
        return torch.nested.as_nested_tensor([col_lse, row_lse])

    if keepdim:
        # col_lse reduced dim=0 (2-D) / dim=1 (batched); row_lse reduced dim=1 / dim=2.
        col_ax, row_ax = (1, 2) if batched else (0, 1)
        col_lse, row_lse = col_lse.unsqueeze(col_ax), row_lse.unsqueeze(row_ax)
    return col_lse, row_lse
