from typing import List, Tuple

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# tsgu::coo2csr — op schema, fake kernel. spec/commit.md Phase 1 #9; routing
# verbatim from spec/map.md "Kernel routing": ``convert_coo_to_csr*`` and
# ``BatchedCSR.from_torch``'s COO path both route here (architecture.md §3
# says the kernel replaces that path's pure-torch internals in commit 19 —
# nothing calls this op yet). Schema takes plain dense (index) tensors only
# (architecture.md §2).
#
# No CUDA/CPU implementation is registered in this commit — the op exists
# only as schema + fake (meta) kernel and raises NotImplementedError if
# actually invoked. Per map.md's routing table ("— (index-only, no grad)")
# this op gets no register_autograd, here or ever: it only rearranges
# integer index tensors, which carry no gradient.
# ---------------------------------------------------------------------------


@torch.library.custom_op("tsgu::coo2csr", mutates_args=())
def coo2csr(batch: Tensor, row: Tensor, col: Tensor, B: int, n: int) -> List[Tensor]:
    r"""Fused sort+compress of batched COO coordinates into folded CSR
    (map.md: ``convert_coo_to_csr*`` / ``BatchedCSR.from_torch``'s COO path,
    architecture.md §3 — this kernel replaces that path's internals,
    spec/commit.md commit 19).

    Parameters
    ----------
    batch : Tensor, shape ``(nse_total,)``, integer dtype
        Batch index of each COO entry, in ``[0, B)``.
    row : Tensor, shape ``(nse_total,)``, integer dtype
        Local row index of each COO entry, in ``[0, n)``.
    col : Tensor, shape ``(nse_total,)``, integer dtype
        Local column index of each COO entry (naming.md §2 ``col``); not
        bounds-checked against ``n_cols`` here (that shape isn't needed to
        sort+compress by row).
    B, n : int
        ``batch_size`` and ``n_rows``.

    Returns
    -------
    list of Tensor, length 3 — ``(rowptr, col_sorted, permutation)``
        ``rowptr``, shape ``(B * n + 1,)``: absolute CSR pointer over folded
        rows ``row_global = b * n + r`` (naming.md §2 ``rowptr``).
        ``col_sorted``, shape ``(nse_total,)``: ``col`` reordered to match
        ``rowptr`` (naming.md §2 ``col``).
        ``permutation``, shape ``(nse_total,)``: the sort permutation —
        ``values[permutation]`` reorders a values tensor aligned with the
        input coordinate order to match ``rowptr``/``col_sorted``.
    """
    raise NotImplementedError(
        "tsgu::coo2csr has no implementation registered yet — lands in spec/commit.md Phase 3 (commit 19)."
    )


@coo2csr.register_fake
def _coo2csr_fake(batch: Tensor, row: Tensor, col: Tensor, B: int, n: int) -> List[Tensor]:
    # Value-independent (architecture.md §2): every output shape derives
    # only from B, n, and col's length — never from batch/row/col contents.
    nse_total = col.shape[0]
    rowptr = row.new_empty(B * n + 1)
    col_sorted = col.new_empty(nse_total)
    permutation = row.new_empty(nse_total)
    return [rowptr, col_sorted, permutation]


def stack_csr(
    tensors: List[torch.Tensor],
    dim: int = 0,
) -> torch.Tensor:
    """
    Stack CSR sparse tensors along a new dimension.

    This function is analogous to :func:`torch.stack`, but specifically
    designed for CSR (Compressed Sparse Row) tensors. Unlike COO tensors,
    CSR tensors are **not** currently supported by :func:`torch.stack`,
    hence this helper provides the missing functionality.

    Parameters
    ----------
    tensors : list of torch.Tensor
        List of 2D CSR sparse tensors to be stacked. All tensors must have
        the same shape and layout.
    dim : int, default=0
        Dimension along which to stack the tensors.

    Returns
    -------
    torch.Tensor
        A CSR sparse tensor with an additional dimension of size
        ``len(tensors)`` inserted at position ``dim``.

    Raises
    ------
    TypeError
        If ``tensors`` is not a list or tuple.
    ValueError
        If ``tensors`` is empty, contain tensors of different shapes,
        are not in CSR format, or are not 2D.

    Notes
    -----
    - :func:`torch.stack` supports COO sparse tensors but not CSR.
      This function fills that gap by implementing stacking logic
      for CSR tensors.

    Examples
    --------
    Stack multiple 2D CSR tensors:

    >>> import torch
    >>> from torchsparsegradutils.utils import stack_csr
    >>> crow = torch.tensor([0, 1, 2])
    >>> col = torch.tensor([0, 1])
    >>> A = torch.sparse_csr_tensor(crow, col, torch.tensor([1.0, 2.0]), (2, 2))
    >>> B = torch.sparse_csr_tensor(crow, col, torch.tensor([3.0, 4.0]), (2, 2))
    >>> stacked = stack_csr([A, B])
    >>> stacked.shape
    torch.Size([2, 2, 2])

    The new dimension is a CSR batch dimension:

    >>> stacked.crow_indices().shape
    torch.Size([2, 3])
    """
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("Expected a list of tensors, but got {}.".format(type(tensors)))

    if len(tensors) == 0:
        raise ValueError("Cannot stack empty list of tensors.")

    if not all([tensor.shape == tensors[0].shape for tensor in tensors]):
        raise ValueError("All tensors must have the same shape.")

    if not all([tensor.layout == torch.sparse_csr for tensor in tensors]):
        raise ValueError("All tensors must be in CSR layout.")

    if not all([tensor.ndim == 2 for tensor in tensors]):
        raise ValueError("All tensors must be 2D.")

    crow_indices = torch.stack([tensor.crow_indices() for tensor in tensors], dim=dim)
    col_indices = torch.stack([tensor.col_indices() for tensor in tensors], dim=dim)
    values = torch.stack([tensor.values() for tensor in tensors], dim=dim)

    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    shape = tuple(shape)

    return torch.sparse_csr_tensor(crow_indices, col_indices, values, shape)


def _sort_coo_indices(
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sort COO indices in ascending lexicographic order with permutation tracking.

    This function sorts COO (Coordinate List) format indices and returns both
    the sorted indices and a permutation tensor mapping the original order to
    the sorted order. It performs a similar role to ``.coalesce()`` but works
    directly on index tensors and supports both int32 and int64 indices.

    Parameters
    ----------
    indices : torch.Tensor
        COO indices with shape ``(2, nnz)`` for unbatched tensors or
        ``(3, nnz)`` for batched tensors.

    Returns
    -------
    indices_sorted : torch.Tensor
        Sorted indices in the same shape and dtype as the input.
    permutation : torch.Tensor
        Tensor of shape ``(nnz,)`` giving the permutation mapping from the
        original indices to the sorted indices.

    Notes
    -----
    - Sorting is lexicographic: first by row indices, then by column indices.
    - For batched tensors (3, nnz), sorting is performed across all batches
      jointly. If independent per-batch sorting is required, call this
      function separately on each batch slice.

    Examples
    --------
    Sort unbatched COO indices:

    >>> import torch
    >>> from torchsparsegradutils.utils.convert import _sort_coo_indices
    >>> indices = torch.tensor([[1, 0, 1], [2, 1, 0]])
    >>> sorted_indices, perm = _sort_coo_indices(indices)
    >>> sorted_indices
    tensor([[0, 1, 1],
            [1, 0, 2]])

    Sort batched COO indices:

    >>> batched_indices = torch.tensor([
    ...     [0, 0, 1],   # batch indices
    ...     [1, 0, 1],   # row indices
    ...     [2, 1, 0]    # col indices
    ... ])
    >>> sorted_idx, perm = _sort_coo_indices(batched_indices)

    See Also
    --------
    torch.Tensor.coalesce : Built-in method for sorting/merging duplicate COO indices.
    """
    indices_sorted, permutation = torch.unique(indices, dim=-1, sorted=True, return_inverse=True)
    return indices_sorted.contiguous(), torch.argsort(permutation)


def _compress_row_indices(
    row_indices: torch.Tensor,
    num_rows: int,
    *,
    _validate: bool = True,
) -> torch.Tensor:
    """
    Convert COO row indices to CSR crow-indices.

    Computes CSR (Compressed Sparse Row) ``crow_indices`` from a 1D tensor of
    COO row indices by counting non-zeros per row and taking a cumulative sum.

    Parameters
    ----------
    row_indices : torch.Tensor
        1D tensor of non-negative integer row indices with shape ``(nnz,)``.
        Values must be in ``[0, num_rows - 1]``.
    num_rows : int
        Total number of rows in the matrix.

    Returns
    -------
    torch.Tensor
        CSR crow-indices of shape ``(num_rows + 1,)`` on the same device as
        ``row_indices``. By definition:
        ``crow[0] = 0``,
        ``crow[i+1] - crow[i]`` equals the number of non-zeros in row ``i``,
        and ``crow[-1] = nnz``.

    Raises
    ------
    ValueError
        If ``row_indices`` is not 1D, contains out-of-range/negative values,
        or if ``num_rows`` is not positive.
    TypeError
        If ``row_indices`` is not an integer tensor.

    Notes
    -----
    - Rows with zero non-zeros are handled naturally (the cumulative count
      repeats for those rows).
    - The output dtype matches ``row_indices.dtype`` (commonly ``int64`` or ``int32``).

    Examples
    --------
    Basic compression:

    >>> import torch
    >>> from torchsparsegradutils.utils.convert import _compress_row_indices
    >>> row_indices = torch.tensor([0, 0, 2, 2])
    >>> _compress_row_indices(row_indices, num_rows=3)
    tensor([0, 2, 2, 4])

    Empty rows:

    >>> row_indices = torch.tensor([0, 2])
    >>> _compress_row_indices(row_indices, num_rows=3)
    tensor([0, 1, 1, 2])

    See Also
    --------
    convert_coo_to_csr_indices_values : Convert full COO (row,col,values) to CSR arrays.
    torch.sparse_csr_tensor : Construct a CSR sparse tensor from (crow, col, values).

    Notes on ``_validate``
    ----------------------
    ``_validate=False`` (internal use only, e.g. ``_batched.py``'s
    ``_fold_coo_to_csr``) skips the two ``torch.any(...)`` content checks
    below for callers that pass already-trusted, internally-derived indices
    (never raw user input). This isn't just an optimisation: those checks
    convert a tensor to a Python ``bool`` via ``if``, which commit 15
    (spec/commit.md Phase 3; ``BatchedCSR.transposed``, the first caller to
    run this function inside a differentiable backward) surfaced as a
    ``GuardOnDataDependentSymNode`` failure under opcheck's AOTAutograd
    dynamic-shape test — the tensor's *size* there is already data-dependent
    (an unbacked SymInt, from an upstream ``torch.repeat_interleave`` with a
    data-dependent repeat count), and PyTorch's functionalization pass can't
    resolve a content-dependent boolean on such a tensor during tracing.
    Skipping validation for trusted-internal callers avoids the guard
    entirely rather than working around it; public/user-facing callers keep
    ``_validate=True`` (the default) and its full error-checking contract.
    """
    if not isinstance(row_indices, torch.Tensor):
        raise TypeError("row_indices must be a torch.Tensor.")
    if row_indices.ndim != 1:
        raise ValueError(f"row_indices must be 1D, got shape {tuple(row_indices.shape)}.")
    if row_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("row_indices must have integer dtype (torch.int32 or torch.int64).")
    if not isinstance(num_rows, int) or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    if _validate:
        if torch.any(row_indices < 0):
            raise ValueError("row_indices contains negative entries.")
        if torch.any(row_indices >= num_rows):
            raise ValueError("row_indices contains entries >= num_rows.")
    # Compute the number of non-zero elements in each row. Not
    # torch.bincount(row_indices, minlength=num_rows): bincount's output
    # length is max(minlength, row_indices.max() + 1) -- by this function's
    # own contract that's always exactly num_rows (every value is < num_rows,
    # enforced above when _validate=True, guaranteed by construction
    # otherwise), but PyTorch can't know that *statically* when
    # row_indices's own length is an unbacked SymInt (commit 15,
    # spec/commit.md Phase 3: BatchedCSR.transposed's internal
    # _validate=False call chain), making bincount's output length itself an
    # unbacked SymInt and any later shape comparison against the Python int
    # `num_rows` a GuardOnDataDependentSymNode. A `zeros(num_rows) +
    # scatter_add_` computes the identical per-row counts with an output
    # shape that is `num_rows` unconditionally (a plain Python int, never
    # data-dependent) -- friendly to dynamic-shape/export tracing regardless
    # of row_indices's own shape.
    ones = torch.ones(row_indices.shape[0], dtype=row_indices.dtype, device=row_indices.device)
    counts = torch.zeros(num_rows, dtype=row_indices.dtype, device=row_indices.device).scatter_add_(
        0, row_indices.to(torch.int64), ones
    )

    # Compute the cumulative sum of counts to get CSR indices
    crow_indices = torch.cat([torch.zeros(1, dtype=row_indices.dtype, device=counts.device), counts.cumsum_(dim=0)])

    return crow_indices


def convert_coo_to_csr_indices_values(coo_indices, num_rows, values=None):
    """
    Convert COO indices to CSR format with optional value permutation.

    Converts COO (Coordinate List Format) row and column indices to CSR
    (Compressed Sparse Row) format. Supports both batched and unbatched
    indices. The function sorts the COO indices lexicographically and
    compresses row indices to CSR crow format.

    Parameters
    ----------
    coo_indices : torch.Tensor
        COO indices tensor with shape (2, nnz) for unbatched or (3, nnz)
        for batched format. Rows are [row_idx, col_idx] or
        [batch_idx, row_idx, col_idx].
    num_rows : int
        Number of rows in the matrix.
    values : torch.Tensor, optional
        Values tensor corresponding to COO indices. If provided, values
        are reordered according to the index sorting permutation.

    Returns
    -------
    crow_indices : torch.Tensor
        CSR crow indices. For unbatched: shape (num_rows + 1,).
        For batched: shape (num_batches, num_rows + 1).
    col_indices : torch.Tensor
        CSR column indices. Same shape as original column indices but
        reordered according to sorting.
    values_or_permutation : torch.Tensor
        If values provided: reordered values tensor.
        If values is None: permutation indices from sorting.

    Raises
    ------
    ValueError
        If indices tensor has wrong number of dimensions, row indices exceed
        num_rows, or values shape doesn't match indices.

    Examples
    --------
    Unbatched COO to CSR conversion:

    >>> import torch
    >>> from torchsparsegradutils.utils import convert_coo_to_csr_indices_values
    >>> # COO indices for 3x3 matrix
    >>> coo_indices = torch.tensor([[0, 1, 2], [1, 0, 2]])
    >>> values = torch.tensor([1.0, 2.0, 3.0])
    >>> crow, col, vals = convert_coo_to_csr_indices_values(
    ...     coo_indices, num_rows=3, values=values)
    >>> crow
    tensor([0, 1, 2, 3])

    Batched conversion:

    >>> # Batched COO indices: 2 batches, each with 2 elements
    >>> batch_coo = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]])
    >>> batch_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> crow, col, vals = convert_coo_to_csr_indices_values(
    ...     batch_coo, num_rows=2, values=batch_values)

    Without values (get permutation):

    >>> crow, col, perm = convert_coo_to_csr_indices_values(
    ...     coo_indices, num_rows=3)
    """
    if coo_indices.shape[0] < 2:
        raise ValueError(
            f"Indices tensor must have at least 2 rows (row and column indices). Got {coo_indices.shape[0]} rows."
        )
    elif coo_indices.shape[0] > 3:
        raise ValueError(
            f"Current implementation only supports single batch diomension, therefore indices tensor must have at most 3 rows (batch, row and column indices). Got {coo_indices.shape[0]} rows."
        )

    if coo_indices[-2].max() >= num_rows:
        raise ValueError(
            f"Row indices must be less than num_rows ({num_rows}). Got max row index {coo_indices[-2].max()}"
        )

    if values is not None and values.shape[0] != coo_indices.shape[1]:
        raise ValueError(
            f"Number of values ({values.shape[0]}) does not match number of indices ({coo_indices.shape[1]})"
        )

    coo_indices, permutation = _sort_coo_indices(coo_indices)

    if coo_indices.shape[0] == 2:
        row_indices, col_indices = coo_indices
        crow_indices = _compress_row_indices(row_indices, num_rows)

        values = values[permutation] if values is not None else permutation

    else:
        batch_indices, row_indices, col_indices = coo_indices
        crow_indices = torch.cat(
            [
                _compress_row_indices(row_indices[batch_indices == batch], num_rows)
                for batch in torch.unique(batch_indices)
            ]
        )
        num_batches = torch.unique(batch_indices).shape[0]

        crow_indices = crow_indices.reshape(num_batches, -1)
        col_indices = col_indices.reshape(num_batches, -1)

        values = values[permutation] if values is not None else permutation

        values = values.reshape(num_batches, -1)

    return crow_indices, col_indices, values


def convert_coo_to_csr(sparse_coo_tensor):
    """
    Convert COO sparse tensor to CSR format.

    Converts a COO (Coordinate List Format) sparse tensor to CSR (Compressed
    Sparse Row) format. Handles both unbatched and batched tensors with
    optional leading batch dimension.

    Parameters
    ----------
    sparse_coo_tensor : torch.Tensor
        COO sparse tensor to convert. Must have layout torch.sparse_coo.
        Can be 2D (m, n) or 3D (b, m, n) with single batch dimension.

    Returns
    -------
    torch.Tensor
        CSR sparse tensor with same shape and values as input.

    Raises
    ------
    ValueError
        If input tensor layout is not torch.sparse_coo.

    Notes
    -----
    The function automatically coalesces the COO tensor if not already
    coalesced, ensuring proper handling of duplicate indices.

    Examples
    --------
    Convert 2D COO to CSR:

    >>> import torch
    >>> from torchsparsegradutils.utils import convert_coo_to_csr
    >>> # Create COO tensor
    >>> indices = torch.tensor([[0, 1, 1], [1, 0, 2]])
    >>> values = torch.tensor([1.0, 2.0, 3.0])
    >>> coo_tensor = torch.sparse_coo_tensor(indices, values, (2, 3))
    >>> csr_tensor = convert_coo_to_csr(coo_tensor)
    >>> csr_tensor.layout
    torch.sparse_csr

    Convert batched COO to CSR:

    >>> # Batched COO tensor: 2 batches, each with 2 elements
    >>> batch_indices = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0]])
    >>> batch_values = torch.tensor([1.0, 2.0, 3.0, 4.0])
    >>> batched_coo = torch.sparse_coo_tensor(batch_indices, batch_values, (2, 2, 2))
    >>> batched_csr = convert_coo_to_csr(batched_coo)
    >>> batched_csr.shape
    torch.Size([2, 2, 2])
    """
    if sparse_coo_tensor.layout == torch.sparse_coo:
        if sparse_coo_tensor.is_coalesced() is False:
            sparse_coo_tensor = sparse_coo_tensor.coalesce()
        crow_indices, col_indices, values = convert_coo_to_csr_indices_values(
            sparse_coo_tensor.indices(), sparse_coo_tensor.size()[-2], sparse_coo_tensor.values()
        )
        return torch.sparse_csr_tensor(crow_indices, col_indices, values, sparse_coo_tensor.size())
    else:
        raise ValueError(f"Unsupported layout: {sparse_coo_tensor.layout}")


def _demcompress_crow_indices(crow_indices, num_rows):
    """
    Decompress CSR crow indices to COO row indices.

    Converts CSR (Compressed Sparse Row) crow indices back to COO row indices
    by expanding the compressed representation to individual row indices for
    each non-zero element.

    Parameters
    ----------
    crow_indices : torch.Tensor
        CSR crow indices tensor of shape (num_rows + 1,). Contains cumulative
        counts of non-zero elements up to each row.
    num_rows : int
        Total number of rows in the matrix.

    Returns
    -------
    torch.Tensor
        COO row indices tensor of shape (nnz,) where each element indicates
        the row index of the corresponding non-zero value.

    Notes
    -----
    This function performs the inverse operation of _compress_row_indices.
    Each row i is repeated (crow_indices[i+1] - crow_indices[i]) times in
    the output.

    Examples
    --------
    Basic decompression:

    >>> import torch
    >>> from torchsparsegradutils.utils.convert import _demcompress_crow_indices
    >>> # CSR crow indices for matrix with pattern:
    >>> # [X X .] -> 2 elements in row 0
    >>> # [. . .] -> 0 elements in row 1
    >>> # [X . X] -> 2 elements in row 2
    >>> crow_indices = torch.tensor([0, 2, 2, 4])
    >>> row_indices = _demcompress_crow_indices(crow_indices, num_rows=3)
    >>> row_indices
    tensor([0, 0, 2, 2])

    Single element per row:

    >>> # Each row has one element
    >>> crow_indices = torch.tensor([0, 1, 2, 3])
    >>> row_indices = _demcompress_crow_indices(crow_indices, num_rows=3)
    >>> row_indices
    tensor([0, 1, 2])
    """

    row_indices = torch.repeat_interleave(
        torch.arange(num_rows, dtype=crow_indices.dtype, device=crow_indices.device),
        crow_indices[1:] - crow_indices[:-1],
    )

    return row_indices


def sparse_eye(
    size: Tuple[int, ...],
    *,
    layout: torch.layout = torch.sparse_coo,
    values_dtype: torch.dtype = torch.float64,
    indices_dtype: torch.dtype = torch.int64,
    device: torch.device = torch.device("cpu"),
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Create a sparse identity matrix.

    Constructs an identity matrix in sparse format (COO or CSR). Supports both
    unbatched and batched square matrices.

    Parameters
    ----------
    size : tuple of int
        Shape of the identity matrix. Must be either ``(n, n)`` for unbatched
        or ``(batch_size, n, n)`` for batched. Rows and columns must be equal.
    layout : torch.layout, default=torch.sparse_coo
        Sparse tensor layout. Must be either ``torch.sparse_coo`` or
        ``torch.sparse_csr``.
    values_dtype : torch.dtype, default=torch.float64
        Data type of the values. Only ``torch.float32`` and ``torch.float64``
        are supported.
    indices_dtype : torch.dtype, default=torch.int64
        Data type of the indices. Must be ``torch.int32`` or ``torch.int64``.
    device : torch.device, default=torch.device("cpu")
        Device on which to create the tensor.
    requires_grad : bool, default=False
        Whether autograd should record operations on the returned tensor.

    Returns
    -------
    torch.Tensor
        Sparse identity matrix of shape ``(n, n)`` or batched identity matrix
        of shape ``(batch_size, n, n)``, in the requested sparse layout.

    Raises
    ------
    ValueError
        If size is not 2D or 3D, matrix is not square, or if dtypes/layout are
        unsupported.

    Notes
    -----
    - For batched inputs, each batch element is an independent identity matrix.

    Examples
    --------
    Unbatched identity (COO):

    >>> from torchsparsegradutils.utils import sparse_eye
    >>> I = sparse_eye((3, 3), layout=torch.sparse_coo)
    >>> I.to_dense()  # doctest: +ELLIPSIS
    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]]...)

    Batched identity (CSR):

    >>> I_batched = sparse_eye((2, 4, 4), layout=torch.sparse_csr)
    >>> I_batched.shape
    torch.Size([2, 4, 4])
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    if len(size) > 3:
        raise ValueError("size must have at most 3 dimensions (supports 1 batch dimension)")

    if size[-2] != size[-1]:
        raise ValueError("size must define a square matrix (n, n) or batched square matrix (b, n, n)")

    if values_dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Values dtype {values_dtype} not supported. Use torch.float32 or torch.float64.")

    if indices_dtype not in (torch.int32, torch.int64):
        raise ValueError(f"indices_dtype {indices_dtype} not supported. Use torch.int32 or torch.int64.")

    values = torch.ones(size[-1], dtype=values_dtype, device=device)

    if layout == torch.sparse_coo:
        if indices_dtype not in [torch.int32, torch.int64]:
            raise ValueError("For sparse_coo layout, indices_dtype can either be torch.int32 or torch.int64.")

        indices = torch.arange(0, size[-1], dtype=indices_dtype, device=device)
        indices = torch.stack([indices, indices], dim=0)

        if len(size) == 3:
            batch_dim_indices = (
                torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(size[-1]).unsqueeze(0)
            )
            sparse_dim_indices = torch.cat([indices] * size[0], dim=-1)
            indices = torch.cat([batch_dim_indices, sparse_dim_indices])
            values = values.repeat(size[0])

        # NOTE: is_coalesced=True since there are no duplicate indices in identity matrix, flag avails in PyTorch 2.1+
        return torch.sparse_coo_tensor(
            indices, values, size, dtype=values_dtype, device=device, requires_grad=requires_grad, is_coalesced=True
        )

    elif layout == torch.sparse_csr:
        if indices_dtype not in [torch.int32, torch.int64]:
            raise ValueError("For sparse_csr layout, indices_dtype can either be torch.int32 or torch.int64.")

        crow_indices = torch.arange(0, size[-1] + 1, dtype=indices_dtype, device=device)
        col_indices = torch.arange(0, size[-1], dtype=indices_dtype, device=device)

        if len(size) == 3:
            crow_indices = crow_indices.repeat(size[0], 1)
            col_indices = col_indices.repeat(size[0], 1)
            values = values.repeat(size[0], 1)

        return torch.sparse_csr_tensor(
            crow_indices, col_indices, values, size, dtype=values_dtype, device=device, requires_grad=requires_grad
        )

    else:
        raise ValueError("Layout {} not supported. Only sparse_coo and sparse_csr are supported.".format(layout))
