from typing import Literal, Sequence, Union

import torch
from packaging.version import parse as parse_version
from torch import Tensor

__all__ = ["sparse_logsumexp", "sparse_bidir_logsumexp"]

# ---------------------------------------------------------------------------
# tsgu::seglse / tsgu::seglse_bwd / tsgu::seglse_bidir / tsgu::seglse_bidir_bwd
# — op schemas, fake kernels, autograd registration.
# spec/commit.md Phase 1 #9; routing verbatim from spec/map.md "Kernel
# routing". Schemas take plain dense tensors only (architecture.md §2).
#
# spec/commit.md Phase 3 commit 12: `sparse_logsumexp` dispatches to
# `tsgu::seglse` / `tsgu::seglse_bwd` (CUDA implementation registered in
# cuda/csrc/kernels/logsumexp/seglse.cu) -- its `_legacy_*` body was deleted
# there.
#
# spec/commit.md Phase 3 commit 13: `sparse_bidir_logsumexp` now dispatches
# to `tsgu::seglse_bidir` / `tsgu::seglse_bidir_bwd` (CUDA implementation
# registered in cuda/csrc/kernels/logsumexp/seglse_bidir.cu) -- its
# `_legacy_*` body (`_bidir_2d` / `_bidir_batched` / `_scatter_logsumexp`) is
# deleted in this commit.
# ---------------------------------------------------------------------------


@torch.library.custom_op("tsgu::seglse", mutates_args=())
def seglse(vals: Tensor, rowptr: Tensor, B: int, n: int, m: int, include_zeros: bool) -> Tensor:
    r"""Segmented log-sum-exp forward (map.md / kernels.md Family 2 flagship)
    — the shared forward kernel behind ``sparse_logsumexp``. Reduces the
    *stored* values within each of the ``B * n`` folded segments
    (naming.md §2 folded row ``row_global = b * n + r``) to one log-sum-exp
    value per segment.

    The wrapper selects which logical axis this reduces (``dim`` of
    ``sparse_logsumexp``) by passing either ``A``'s own BatchedCSR (segments
    = rows, reducing columns) or its BatchedCSC (segments = columns, reducing
    rows) — this op only ever sees a generic folded-row/segment structure,
    never COO/CSC directly.

    Parameters
    ----------
    vals : Tensor, shape ``(nse_total,)``
        Stored values to reduce (naming.md §2 ``vals``).
    rowptr : Tensor, shape ``(B * n + 1,)``
        Absolute CSR-style pointer over the ``B * n`` folded segments
        (naming.md §2 ``rowptr``).
    B, n : int
        ``batch_size`` and number of segments per batch item (``n_rows``
        when reducing columns, ``n_cols`` when reducing rows).
    m : int
        Full size of the *reduced* axis (``n_cols`` when reducing columns,
        ``n_rows`` when reducing rows) — needed only to count structural
        zeros when ``include_zeros=True`` (``m - segment_nse`` per segment).
    include_zeros : bool
        Whether structural zeros contribute ``exp(0) = 1`` to each segment's
        sum (map.md contract: matches ``torch.logsumexp`` on the dense
        equivalent when ``True``).

    Returns
    -------
    Tensor, shape ``(B, n)``
        Per-segment log-sum-exp; also the ``lse`` saved for
        ``tsgu::seglse_bwd`` (kernels.md: "no recompute").
    """
    raise NotImplementedError(
        "tsgu::seglse has no implementation registered yet — lands in spec/commit.md Phase 3 (commit 12)."
    )


@seglse.register_fake
def _seglse_fake(vals: Tensor, rowptr: Tensor, B: int, n: int, m: int, include_zeros: bool) -> Tensor:
    # Value-independent: shape derives only from the B, n shape ints.
    return vals.new_empty(B, n)


@torch.library.custom_op("tsgu::seglse_bwd", mutates_args=())
def seglse_bwd(vals: Tensor, rowptr: Tensor, lse: Tensor, gout: Tensor, B: int, n: int) -> Tensor:
    r"""Backward of :func:`seglse` (map.md / kernels.md Family 2):
    embarrassingly parallel per specified entry,
    ``gradA_val = exp(v - lse[seg]) * gout[seg]``, using the ``lse`` saved
    from the forward pass (no recompute).

    Parameters
    ----------
    vals : Tensor, shape ``(nse_total,)``
        The forward's stored values (naming.md §2 ``vals``).
    rowptr : Tensor, shape ``(B * n + 1,)``
        Same folded-segment pointer as the forward call (naming.md §2
        ``rowptr``).
    lse : Tensor, shape ``(B, n)``
        :func:`seglse`'s saved output.
    gout : Tensor, shape ``(B, n)``
        Upstream gradient w.r.t. :func:`seglse`'s output.
    B, n : int
        ``batch_size`` and segment count, as in the forward call.

    Returns
    -------
    Tensor, shape ``(nse_total,)``
        Gradient values aligned with ``vals`` / the pattern's ``rowptr``.
    """
    raise NotImplementedError(
        "tsgu::seglse_bwd has no implementation registered yet — lands in spec/commit.md Phase 3 (commit 12)."
    )


@seglse_bwd.register_fake
def _seglse_bwd_fake(vals: Tensor, rowptr: Tensor, lse: Tensor, gout: Tensor, B: int, n: int) -> Tensor:
    return vals.new_empty(vals.shape[0])


# tsgu::seglse_bwd gets no register_autograd: in this commit it is only ever
# used as sparse_logsumexp's backward primitive; any higher-order-gradient
# support for it is decided at its own kernel commit (commit 12), not
# invented here.


def _seglse_setup_context(ctx, inputs, output):
    vals, rowptr, B, n, m, include_zeros = inputs
    ctx.save_for_backward(vals, rowptr, output)
    ctx.B, ctx.n = B, n
    ctx.vals_requires_grad = vals.requires_grad


def _seglse_backward(ctx, grad_output):
    vals, rowptr, lse = ctx.saved_tensors
    grad_vals = None
    if ctx.vals_requires_grad:
        # grad_output can arrive as a broadcast/expanded view (e.g. a stride-0
        # tensor from a reduction's backward, such as .sum()) rather than a
        # materialized (B, n) buffer -- the CUDA kernel indexes it with a
        # plain contiguous pointer, so it must be made contiguous here first
        # (never inside the kernel launcher, which has no autograd context to
        # do this safely).
        grad_vals = torch.ops.tsgu.seglse_bwd(vals, rowptr, lse, grad_output.contiguous(), ctx.B, ctx.n)
    return grad_vals, None, None, None, None, None


seglse.register_autograd(_seglse_backward, setup_context=_seglse_setup_context)


@torch.library.custom_op("tsgu::seglse_bidir", mutates_args=())
def seglse_bidir(vals: Tensor, rowptr: Tensor, col: Tensor, B: int, n: int, m: int, include_zeros: bool) -> Tensor:
    r"""Fused row-and-column segmented log-sum-exp (map.md:
    ``sparse_bidir_logsumexp`` forward) — a single traversal of
    ``(rowptr, col, vals)`` updating both the per-row and per-column
    accumulators together (kernels.md Family 2: "the entire point is a
    single traversal ... instead of two"). Must equal two separate
    :func:`seglse` calls (map.md contract).

    Parameters
    ----------
    vals : Tensor, shape ``(nse_total,)``
        Stored values (naming.md §2 ``vals``).
    rowptr : Tensor, shape ``(B * n + 1,)``
        Absolute CSR pointer over folded rows (naming.md §2 ``rowptr``).
    col : Tensor, shape ``(nse_total,)``
        Local column indices in ``[0, m)`` (naming.md §2 ``col``) — needed
        (unlike :func:`seglse`) because both axes are reduced at once.
    B, n, m : int
        ``batch_size``, ``n_rows``, ``n_cols``.
    include_zeros : bool
        As in :func:`seglse`.

    Returns
    -------
    Tensor, shape ``(2, B, G)`` where ``G = max(n, m)``
        The op's native "padded" buffer (map.md / ``sparse_bidir_logsumexp``'s
        ``output_layout="padded"``): index ``0`` along dim 0 is the
        column-reduction (``col_lse``, one value per column, padded to
        ``G``), index ``1`` is the row-reduction (``row_lse``). The
        ``tuple``/``nested`` output layouts are assembled host-side from this
        buffer, outside the op (architecture.md §2 / map.md).
    """
    raise NotImplementedError(
        "tsgu::seglse_bidir has no implementation registered yet — lands in spec/commit.md Phase 3 (commit 13)."
    )


@seglse_bidir.register_fake
def _seglse_bidir_fake(
    vals: Tensor, rowptr: Tensor, col: Tensor, B: int, n: int, m: int, include_zeros: bool
) -> Tensor:
    # Value-independent: shape derives only from B and G = max(n, m).
    G = max(n, m)
    return vals.new_empty(2, B, G)


@torch.library.custom_op("tsgu::seglse_bidir_bwd", mutates_args=())
def seglse_bidir_bwd(
    vals: Tensor, rowptr: Tensor, col: Tensor, padded: Tensor, gout: Tensor, B: int, n: int, m: int
) -> Tensor:
    r"""Backward of :func:`seglse_bidir`. Every specified entry receives a
    gradient contribution from *both* directions it participates in:
    ``gradA_val = exp(v - col_lse[col]) * gout[0, col] + exp(v - row_lse[row])
    * gout[1, row]``, where ``col_lse = padded[0]``, ``row_lse = padded[1]``
    are the forward's saved output and ``row`` is recovered from ``rowptr``
    (kernels.md Family 2).

    Parameters
    ----------
    vals, rowptr, col : Tensor
        Same pattern as the forward call (naming.md §2 ``vals``/``rowptr``/
        ``col``).
    padded : Tensor, shape ``(2, B, G)``
        :func:`seglse_bidir`'s saved output.
    gout : Tensor, shape ``(2, B, G)``
        Upstream gradient in the same padded layout as ``padded`` (the
        wrapper converts a user-facing ``tuple``/``nested`` upstream
        gradient into this layout before autograd sees it).
    B, n, m : int
        ``batch_size``, ``n_rows``, ``n_cols``.

    Returns
    -------
    Tensor, shape ``(nse_total,)``
        Gradient values aligned with ``vals``.
    """
    raise NotImplementedError(
        "tsgu::seglse_bidir_bwd has no implementation registered yet — lands in spec/commit.md Phase 3 (commit 13)."
    )


@seglse_bidir_bwd.register_fake
def _seglse_bidir_bwd_fake(
    vals: Tensor, rowptr: Tensor, col: Tensor, padded: Tensor, gout: Tensor, B: int, n: int, m: int
) -> Tensor:
    return vals.new_empty(vals.shape[0])


# tsgu::seglse_bidir_bwd gets no register_autograd: in this commit it is only
# ever used as sparse_bidir_logsumexp's backward primitive; any
# higher-order-gradient support is decided at its own kernel commit
# (commit 13), not invented here.


def _seglse_bidir_setup_context(ctx, inputs, output):
    vals, rowptr, col, B, n, m, include_zeros = inputs
    ctx.save_for_backward(vals, rowptr, col, output)
    ctx.B, ctx.n, ctx.m = B, n, m
    ctx.vals_requires_grad = vals.requires_grad


def _seglse_bidir_backward(ctx, grad_output):
    vals, rowptr, col, padded = ctx.saved_tensors
    grad_vals = None
    if ctx.vals_requires_grad:
        # Same landmine as tsgu::seglse_bwd's wrapper (_seglse_backward above,
        # kernels.md Family 2 backward note carried into commit 13): grad_output
        # can arrive as a broadcast/expanded (stride-0) view from a reduction's
        # backward (e.g. a `.sum()` on the padded output) rather than a
        # materialized (2, B, G) buffer -- the CUDA kernel indexes it with a
        # plain contiguous pointer, so it must be made contiguous here first.
        grad_vals = torch.ops.tsgu.seglse_bidir_bwd(
            vals, rowptr, col, padded, grad_output.contiguous(), ctx.B, ctx.n, ctx.m
        )
    return grad_vals, None, None, None, None, None, None


seglse_bidir.register_autograd(_seglse_bidir_backward, setup_context=_seglse_bidir_setup_context)


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


# ---------------------------------------------------------------------------
# sparse_logsumexp's tsgu::seglse-backed reduction (spec/commit.md Phase 3
# commit 12). sparse_bidir_logsumexp's tsgu::seglse_bidir-backed reduction
# (via _bidir_csr / _bidir_2d / _bidir_batched) follows further down, wired
# in commit 13.
# ---------------------------------------------------------------------------


def _seglse_dispatch(vals: Tensor, rowptr: Tensor, num_segments: int, full_size: int, include_zeros: bool) -> Tensor:
    """Thin call-through to ``torch.ops.tsgu.seglse`` (kernels.md Family 2):
    ``B=1`` is passed regardless of any logical batching -- the op only ever
    needs ``total_segs = B * n`` and the reduced axis's full size, and
    folding batch structure into ``rowptr``/``num_segments`` up front (as
    every caller below does) is exactly equivalent for its purposes.
    """
    index_dtype = rowptr.dtype if rowptr.dtype in (torch.int32, torch.int64) else torch.int64
    rowptr = rowptr.to(index_dtype)
    lse = torch.ops.tsgu.seglse(vals, rowptr, 1, num_segments, full_size, include_zeros)
    return lse.reshape(num_segments)


def _seglse_sorted(
    seg_idx: Tensor,
    vals: Tensor,
    num_segments: int,
    full_size: int,
    include_zeros: bool,
    counts: Union[Tensor, None] = None,
) -> Tensor:
    """``_seglse_dispatch`` for a segment axis that is *not* already
    contiguous in ``vals`` (e.g. dim=0 on a CSR input, dim=1 on a CSC input,
    or any axis of a COO input) -- sorts ``vals`` into contiguous per-segment
    groups first (kernels.md: "CSR rows are sorted for free ... COO/CSC must
    not assume sorted order"). ``vals.index_select`` is autograd-differentiable,
    so gradients flow back to ``vals``'s original order automatically -- no
    manual un-permutation needed.
    """
    order = torch.argsort(seg_idx, stable=True)
    vals_sorted = vals.index_select(0, order)
    if counts is None:
        counts = torch.bincount(seg_idx, minlength=num_segments)
    index_dtype = torch.int32 if max(num_segments, vals.numel()) < 2**31 else torch.int64
    rowptr = torch.zeros(num_segments + 1, dtype=index_dtype, device=vals.device)
    rowptr[1:] = torch.cumsum(counts, dim=0).to(index_dtype)
    return _seglse_dispatch(vals_sorted, rowptr, num_segments, full_size, include_zeros)


def _logsumexp_2d(input: Tensor, dims, keepdim: bool, include_zeros: bool) -> Tensor:
    """Reduction for an unbatched 2-D sparse tensor (see :func:`sparse_logsumexp`),
    dispatching to ``tsgu::seglse``."""
    nrows, ncols = input.shape

    if dims == [0, 1]:
        # Reduce everything -> scalar; one segment holding every stored value.
        vals = (
            input.values()
            if input.layout != torch.sparse_coo
            else (input if input.is_coalesced() else input.coalesce()).values()
        )
        n_specified = vals.numel()
        rowptr = torch.tensor([0, n_specified], dtype=torch.int64, device=vals.device)
        result = _seglse_dispatch(vals, rowptr, 1, nrows * ncols, include_zeros)[0]
        return result.reshape(1, 1) if keepdim else result

    if dims == [1]:
        # Reduce columns -> one value per row. CSR rows are already
        # contiguous (rowptr = crow_indices, no sort needed); every other
        # layout sorts by row first.
        if input.layout == torch.sparse_csr:
            result = _seglse_dispatch(input.values(), input.crow_indices(), nrows, ncols, include_zeros)
        else:
            rows, _cols, vals, row_nnz, _col_nnz = _row_col_val(input, nrows, ncols)
            result = _seglse_sorted(rows, vals, nrows, ncols, include_zeros, counts=row_nnz)
        return result.unsqueeze(1) if keepdim else result

    # dims == [0]: reduce rows -> one value per column. CSC columns are
    # already contiguous (colptr = ccol_indices); every other layout sorts.
    if input.layout == torch.sparse_csc:
        result = _seglse_dispatch(input.values(), input.ccol_indices(), ncols, nrows, include_zeros)
    else:
        _rows, cols, vals, _row_nnz, col_nnz = _row_col_val(input, nrows, ncols)
        result = _seglse_sorted(cols, vals, ncols, nrows, include_zeros, counts=col_nnz)
    return result.unsqueeze(0) if keepdim else result


def _logsumexp_batched(input: Tensor, dims, keepdim: bool, include_zeros: bool) -> Tensor:
    """Reduction within each batch slice ``input[k]`` of a batched 3-D sparse
    tensor, dispatching to ``tsgu::seglse``.

    Each ``(rows, cols)`` matrix ``input[k]`` is reduced independently: the
    batch index is folded into the segment index (naming.md §2 folded row),
    so a single kernel launch reduces every slice at once. Batched inputs go
    through COO (batched CSR/CSC require equal nnz per slice in PyTorch).
    """
    b, nrows, ncols = input.shape
    coo = input if (input.layout == torch.sparse_coo and input.is_coalesced()) else input.to_sparse_coo().coalesce()
    bidx, rows, cols = coo.indices()
    vals = coo.values()

    if dims == [1, 2]:
        # Reduce each whole slice -> (b,); segments = batch index.
        result = _seglse_sorted(bidx, vals, b, nrows * ncols, include_zeros)
        return result.reshape(b, 1, 1) if keepdim else result

    # dims == [2]: reduce columns -> (b, rows); segments = folded row b*nrows+r.
    # dims == [1]: reduce rows    -> (b, cols); segments = folded col b*ncols+c.
    if dims == [2]:
        seg_idx, num_segments, full_size, out_shape, keep_ax = bidx * nrows + rows, b * nrows, ncols, (b, nrows), 2
    else:
        seg_idx, num_segments, full_size, out_shape, keep_ax = bidx * ncols + cols, b * ncols, nrows, (b, ncols), 1

    result = _seglse_sorted(seg_idx, vals, num_segments, full_size, include_zeros).reshape(out_shape)
    return result.unsqueeze(keep_ax) if keepdim else result


def _bidir_csr(input: Tensor, nrows: int, ncols: int):
    """``(rowptr, col, vals)`` for a 2-D sparse tensor, sorted by row (the layout
    ``tsgu::seglse_bidir`` requires — kernels.md Family 2: row segments must be
    contiguous). CSR gives this for free; CSC sorts by row first; COO's
    ``coalesce()`` already sorts lexicographically (row-major), so no extra sort
    is needed there either.
    """
    if input.layout == torch.sparse_csr:
        return input.crow_indices(), input.col_indices(), input.values()

    if input.layout == torch.sparse_csc:
        rows = input.row_indices()
        ccol = input.ccol_indices()
        col_nnz = ccol[1:] - ccol[:-1]
        cols_local = torch.repeat_interleave(torch.arange(ncols, device=rows.device, dtype=rows.dtype), col_nnz)
        vals = input.values()
        order = torch.argsort(rows, stable=True)
        col_sorted = cols_local.index_select(0, order)
        vals_sorted = vals.index_select(0, order)
        counts = torch.bincount(rows.long(), minlength=nrows)
        rowptr = torch.zeros(nrows + 1, dtype=rows.dtype, device=vals.device)
        rowptr[1:] = torch.cumsum(counts, dim=0).to(rows.dtype)
        return rowptr, col_sorted, vals_sorted

    # COO: coalesce() sorts lexicographically (row-major) -- already row-sorted.
    coo = input if input.is_coalesced() else input.coalesce()
    rows, cols = coo.indices()
    vals = coo.values()
    index_dtype = torch.int32 if max(nrows, vals.numel()) < 2**31 else torch.int64
    counts = torch.bincount(rows, minlength=nrows)
    rowptr = torch.zeros(nrows + 1, dtype=index_dtype, device=vals.device)
    rowptr[1:] = torch.cumsum(counts, dim=0).to(index_dtype)
    return rowptr, cols.to(index_dtype), vals


def _bidir_2d(input: Tensor, include_zeros: bool):
    """Row- and column-wise log-sum-exp of a 2-D sparse tensor via a single
    ``tsgu::seglse_bidir`` traversal (kernels.md Family 2 "Bidirectional":
    "one read of values/indices instead of two").

    Returns ``(col_lse (ncols,), row_lse (nrows,), padded (2, G))`` where
    ``G = max(nrows, ncols)`` and the ``padded`` tail (beyond each axis' length) is
    ``-inf`` (empty groups). ``padded`` is the op's native output buffer, and
    ``col_lse`` / ``row_lse`` are basic-slice *views* into it — the three returns cost
    one allocation between them, not three.
    """
    nrows, ncols = input.shape
    rowptr, col, vals = _bidir_csr(input, nrows, ncols)
    padded = torch.ops.tsgu.seglse_bidir(vals, rowptr, col, 1, nrows, ncols, include_zeros).reshape(2, -1)  # (2, G)
    return padded[0, :ncols], padded[1, :nrows], padded


def _bidir_batched(input: Tensor, include_zeros: bool):
    """Per-slice row- and column-wise log-sum-exp of a batched 3-D sparse tensor,
    via the same single-traversal ``tsgu::seglse_bidir`` call as :func:`_bidir_2d`,
    with the batch index folded into the row segment (naming.md §2 folded row).
    Batched inputs go through COO (batched CSR/CSC require equal nnz per slice
    in PyTorch).

    Returns ``(col_lse (b, ncols), row_lse (b, nrows), padded (2, b, G))``. As in
    :func:`_bidir_2d` the three share one allocation: ``padded`` is the op's native
    buffer and the other two are views into it, and the direction is axis 0 in both.
    """
    b, nrows, ncols = input.shape
    coo = input if (input.layout == torch.sparse_coo and input.is_coalesced()) else input.to_sparse_coo().coalesce()
    bidx, rows, cols = coo.indices()
    vals = coo.values()

    # coalesce() sorts lexicographically over (batch, row, col) -- the folded
    # row seg = bidx * nrows + rows is therefore already non-decreasing, no
    # argsort/index_select needed (unlike CSC's per-layout sort above).
    seg = bidx * nrows + rows
    num_segments = b * nrows
    index_dtype = torch.int32 if max(num_segments, vals.numel()) < 2**31 else torch.int64
    counts = torch.bincount(seg, minlength=num_segments)
    rowptr = torch.zeros(num_segments + 1, dtype=index_dtype, device=vals.device)
    rowptr[1:] = torch.cumsum(counts, dim=0).to(index_dtype)

    padded = torch.ops.tsgu.seglse_bidir(
        vals, rowptr, cols.to(index_dtype), b, nrows, ncols, include_zeros
    )  # (2, b, G)
    return padded[0, :, :ncols], padded[1, :, :nrows], padded


def sparse_logsumexp(
    input: Tensor,
    dim: Union[int, Sequence[int]],
    keepdim: bool = False,
    include_zeros: bool = True,
) -> Tensor:
    r"""Sparse-aware log-sum-exp, mirroring :func:`torch.logsumexp`.

    Computes :math:`\log \sum \exp(x)` along ``dim`` directly on the explicit
    (nonzero) values of an unbatched or batched sparse tensor, without materialising
    the dense equivalent. The reduction is numerically stable via the max-shift trick.

    Parameters
    ----------
    input : Tensor
        A sparse tensor with layout ``torch.sparse_coo``, ``torch.sparse_csr`` or
        ``torch.sparse_csc``. Either an unbatched ``[r, c]`` matrix, or a batched
        ``[b, r, c]`` tensor whose leading dimension is an independent batch (each
        ``[r, c]`` slice is reduced on its own). Any other layout or rank raises
        ``NotImplementedError``.

        Batched CSR/CSC inputs are supported, but PyTorch requires every slice of a
        batched compressed tensor to hold the same number of specified elements --
        a ragged batch can only be expressed as COO.
    dim : int or sequence of int
        Dimension(s) to reduce. For an unbatched input: ``dim=1`` reduces the columns
        (one value per row), ``dim=0`` reduces the rows (one value per column), and
        ``[0, 1]`` reduces to a scalar. For a batched input the batch axis (``0``)
        cannot be reduced; ``dim`` must select from ``{1, 2}``.
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
    ``tsgu::seglse`` (spec/commit.md Phase 3 commit 12) is CUDA-only
    (architecture.md §4) -- these examples run on a CUDA tensor.

    >>> i = torch.tensor([[0, 1, 1], [1, 0, 2]])
    >>> v = torch.tensor([1.0, 2.0, 3.0])
    >>> x = torch.sparse_coo_tensor(i, v, (3, 3)).cuda()
    >>> sparse_logsumexp(x, dim=1, include_zeros=False)
    tensor([1.0000, 3.3133,   -inf], device='cuda:0')
    >>> sparse_logsumexp(x, dim=1, include_zeros=True)
    tensor([1.5514, 3.3490, 1.0986], device='cuda:0')

    ``include_zeros=True`` agrees with the dense reference:

    >>> torch.logsumexp(x.to_dense(), dim=1)
    tensor([1.5514, 3.3490, 1.0986], device='cuda:0')

    Batched 3-D input, reducing within each slice:

    >>> xb = torch.stack([x.to_dense(), x.to_dense()]).to_sparse_coo()
    >>> sparse_logsumexp(xb, dim=2)
    tensor([[1.5514, 3.3490, 1.0986],
            [1.5514, 3.3490, 1.0986]], device='cuda:0')
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
                f"Dimension out of range (expected to be in range of [{-input.ndim}, {input.ndim - 1}], but got {d})"
            )
    normalised = [d % input.ndim for d in dims_list]
    if len(set(normalised)) != len(normalised):
        raise RuntimeError("sparse_logsumexp: dim contains a repeated dimension.")
    dims = sorted(normalised)

    # tsgu::seglse live (spec/commit.md Phase 3 commit 12) -- _legacy_sparse_logsumexp
    # deleted in this commit; sparse_bidir_logsumexp (below) is untouched.
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
    columns) of an unbatched or batched sparse tensor in a single traversal of the sparse
    structure. For an unbatched ``[r, c]`` input this equals::

        (sparse_logsumexp(input, dim=0, ...), sparse_logsumexp(input, dim=1, ...))

    For a batched ``[b, r, c]`` input the per-slice reductions use ``dim=1`` (over rows)
    and ``dim=2`` (over columns) instead — ``dim=0`` is the batch axis and cannot be
    reduced.

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
        an unbatched ``[r, c]`` matrix or a batched ``[b, r, c]`` tensor whose leading
        axis is an independent batch. Any other layout or rank raises
        ``NotImplementedError``.

        Batched CSR/CSC inputs are supported, but PyTorch requires every slice of a
        batched compressed tensor to hold the same number of specified elements --
        a ragged batch can only be expressed as COO.
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
          (unbatched ``dim=0``; batched ``dim=1``), shape ``(c,)`` / ``(b, c)``;
          ``row_lse`` is the reduction over columns (unbatched ``dim=1``; batched
          ``dim=2``), shape ``(r,)`` / ``(b, r)``. **Note the order: column result first.**
        - ``"padded"``: a single dense tensor of shape ``(2, G)`` (unbatched) or
          ``(2, b, G)`` (batched) with ``G = max(r, c)``. Index ``0`` along the
          leading axis is ``col_lse``, index ``1`` is ``row_lse``; each is padded to ``G``
          with ``-inf``. The direction leads in both cases, so ``out[0]`` / ``out[1]``
          select the two reductions regardless of rank. This is the scatter's native
          output buffer, returned without a copy.

          .. warning::
             Experimental. The axis order, the ``G = max(r, c)`` group padding and
             the ``-inf`` fill mirror the internal scatter buffer, and may change without
             a deprecation cycle if that buffer does. Use ``"tuple"`` for a stable layout.

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
    This example's second block compares against :func:`sparse_logsumexp`
    (``tsgu::seglse`` as of spec/commit.md Phase 3 commit 12), which is
    CUDA-only (architecture.md §4) -- run on a CUDA tensor throughout.

    >>> i = torch.tensor([[0, 1, 1], [1, 0, 2]])
    >>> v = torch.tensor([1.0, 2.0, 3.0])
    >>> x = torch.sparse_coo_tensor(i, v, (3, 3)).cuda()
    >>> col_lse, row_lse = sparse_bidir_logsumexp(x)
    >>> col_lse  # dim=0 reduction (one value per column)
    tensor([2.2395, 1.5514, 3.0949], device='cuda:0')
    >>> row_lse  # dim=1 reduction (one value per row)
    tensor([1.5514, 3.3490, 1.0986], device='cuda:0')

    Agrees with two separate :func:`sparse_logsumexp` calls:

    >>> torch.equal(col_lse, sparse_logsumexp(x, dim=0)) and torch.equal(row_lse, sparse_logsumexp(x, dim=1))
    True

    The ``"padded"`` layout stacks both into one ``(2, max(r, c))`` tensor:

    >>> sparse_bidir_logsumexp(x, output_layout="padded")
    tensor([[2.2395, 1.5514, 3.0949],
            [1.5514, 3.3490, 1.0986]], device='cuda:0')
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

    # tsgu::seglse_bidir live (spec/commit.md Phase 3 commit 13) -- via
    # _bidir_2d / _bidir_batched, both dispatching to the fused single-traversal
    # kernel (cuda/csrc/kernels/logsumexp/seglse_bidir.cu).
    batched = input.ndim == 3
    col_lse, row_lse, padded = _bidir_batched(input, include_zeros) if batched else _bidir_2d(input, include_zeros)

    if output_layout == "padded":
        return padded

    if output_layout == "nested":
        return torch.nested.as_nested_tensor([col_lse, row_lse])

    if keepdim:
        # col_lse reduced dim=0 (2-D) / dim=1 (batched); row_lse reduced dim=1 / dim=2.
        col_ax, row_ax = (1, 2) if batched else (0, 1)
        col_lse, row_lse = col_lse.unsqueeze(col_ax), row_lse.unsqueeze(row_ax)
    return col_lse, row_lse
