from typing import List, Tuple

import torch


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

    Stack along a different dimension:

    >>> stacked_dim1 = stack_csr([A, B], dim=1)
    >>> stacked_dim1.shape
    torch.Size([2, 2, 2])
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
    >>> from torchsparsegradutils.utils.utils import _sort_coo_indices
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
    >>> from torchsparsegradutils.utils.utils import _compress_row_indices
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
    """
    if not isinstance(row_indices, torch.Tensor):
        raise TypeError("row_indices must be a torch.Tensor.")
    if row_indices.ndim != 1:
        raise ValueError(f"row_indices must be 1D, got shape {tuple(row_indices.shape)}.")
    if row_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("row_indices must have integer dtype (torch.int32 or torch.int64).")
    if not isinstance(num_rows, int) or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    if row_indices.numel() > 0:
        if torch.any(row_indices < 0):
            raise ValueError("row_indices contains negative entries.")
        if torch.any(row_indices >= num_rows):
            raise ValueError("row_indices contains entries >= num_rows.")
    # Compute the number of non-zero elements in each row
    counts = torch.bincount(row_indices, minlength=num_rows).to(row_indices.dtype)

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
    >>> from torchsparsegradutils.utils.utils import _demcompress_crow_indices
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


# use @torch.jit.script ?
def sparse_block_diag(*sparse_tensors: torch.Tensor) -> torch.Tensor:
    """
    Construct a block-diagonal sparse matrix from COO/CSR inputs.

    Builds a block-diagonal sparse tensor from a sequence of **2D** sparse tensors
    that are all in the same layout (either COO or CSR). The result has blocks
    placed on the diagonal and zeros elsewhere, analogous to :func:`torch.block_diag`
    but for sparse inputs.

    Parameters
    ----------
    *sparse_tensors : torch.Tensor
        Variable number of 2D sparse tensors (all COO or all CSR). Each tensor must
        have exactly 2 sparse dimensions and 0 dense dimensions.

    Returns
    -------
    torch.Tensor
        A sparse tensor in the same layout as the inputs with shape
        ``(sum_i n_i, sum_i m_i)``, where each block ``i`` has shape ``(n_i, m_i)``.

    Raises
    ------
    TypeError
        If any input is not a :class:`torch.Tensor`.
    ValueError
        If no tensors are provided; if layouts are mixed; or if any input does not
        have exactly 2 sparse dims and 0 dense dims.

    Notes
    -----
    The resulting block structure is

    .. code-block:: text

        [A₁  0   0  ... 0 ]
        [0   A₂  0  ... 0 ]
        [0   0   A₃ ... 0 ]
        [⋮   ⋮   ⋮  ⋱  ⋮ ]
        [0   0   0  ... Aₙ]

    Offsets for each block are computed using **cumulative** row/column sizes of
    all preceding blocks (not simply ``i * size``), so inputs may have different
    shapes.

    Examples
    --------
    COO inputs:

    >>> import torch
    >>> from torchsparsegradutils.utils import sparse_block_diag
    >>> A = torch.sparse_coo_tensor(torch.tensor([[0, 1], [0, 1]]), torch.tensor([1., 2.]), size=(2, 2))
    >>> B = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), torch.tensor([3.]), size=(1, 1))
    >>> C = sparse_block_diag(A, B)
    >>> C.shape
    torch.Size([3, 3])
    >>> C.layout
    torch.sparse_coo

    CSR inputs:

    >>> A_csr = A.to_sparse_csr()
    >>> B_csr = B.to_sparse_csr()
    >>> D = sparse_block_diag(A_csr, B_csr)
    >>> D.layout
    torch.sparse_csr

    See Also
    --------
    torch.block_diag : Dense block-diagonal construction for dense inputs.
    stack_csr : Stack CSR matrices along a new batch dimension.
    """
    # ---- validation ----
    for i, t in enumerate(sparse_tensors):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"TypeError: expected Tensor as element {i} in argument 0, but got {type(t).__name__}")

    if len(sparse_tensors) == 0:
        raise ValueError("At least one sparse tensor must be provided.")

    if all(t.layout == torch.sparse_coo for t in sparse_tensors):
        layout = torch.sparse_coo
    elif all(t.layout == torch.sparse_csr for t in sparse_tensors):
        layout = torch.sparse_csr
    else:
        raise ValueError("Sparse tensors must either be all sparse_coo or all sparse_csr.")

    if not all(t.sparse_dim() == 2 for t in sparse_tensors):
        raise ValueError("All sparse tensors must have exactly two sparse dimensions.")
    if not all(t.dense_dim() == 0 for t in sparse_tensors):
        raise ValueError("All sparse tensors must have zero dense dimensions.")

    if len(sparse_tensors) == 1:
        return sparse_tensors[0]

    # ---- COO path ----
    if layout == torch.sparse_coo:
        row_parts = []
        col_parts = []
        val_parts = []
        total_rows = 0
        total_cols = 0

        row_offset = 0
        col_offset = 0
        for t in sparse_tensors:
            t = t.coalesce() if not t.is_coalesced() else t
            rows, cols = t.indices()
            vals = t.values()

            # apply cumulative offsets
            row_parts.append(rows + row_offset)
            col_parts.append(cols + col_offset)
            val_parts.append(vals)

            # update offsets and totals
            n_i, m_i = t.size(-2), t.size(-1)
            row_offset += n_i
            col_offset += m_i
            total_rows += n_i
            total_cols += m_i

        rows_all = torch.cat(row_parts, dim=0)
        cols_all = torch.cat(col_parts, dim=0)
        vals_all = torch.cat(val_parts, dim=0)

        return torch.sparse_coo_tensor(
            torch.stack([rows_all, cols_all], dim=0), vals_all, size=(total_rows, total_cols)
        )

    # ---- CSR path ----
    # We need to stitch crow/col/values with cumulative offsets.
    crow_parts = []
    col_parts = []
    val_parts = []
    total_rows = 0
    total_cols = 0

    col_offset = 0
    crow_running_last = None  # last crow value of accumulated blocks

    for idx, t in enumerate(sparse_tensors):
        crow = t.crow_indices()
        col = t.col_indices()
        vals = t.values()

        # For the first block, we keep the full crow. For subsequent blocks,
        # drop the initial zero and shift by the previous cumulative nnz.
        if idx == 0:
            crow_acc = crow
        else:
            # shift crow by last value of previous crow
            crow_acc = crow[1:] + crow_running_last

        # shift columns by cumulative column offset
        col_acc = col + col_offset

        crow_parts.append(crow_acc)
        col_parts.append(col_acc)
        val_parts.append(vals)

        n_i, m_i = t.size(-2), t.size(-1)
        total_rows += n_i
        total_cols += m_i
        col_offset += m_i
        crow_running_last = (crow_parts[-1][-1] if idx == 0 else crow_parts[-1][-1]).clone()

    crow_all = torch.cat(crow_parts, dim=0)
    col_all = torch.cat(col_parts, dim=0)
    vals_all = torch.cat(val_parts, dim=0)

    return torch.sparse_csr_tensor(crow_all, col_all, vals_all, size=(total_rows, total_cols))


def sparse_block_diag_split(
    sparse_block_diag_tensor: torch.Tensor, *shapes: Tuple[int, int]
) -> tuple[torch.Tensor, ...]:
    """
    Split a block-diagonal sparse matrix back into its component blocks.

    Given a block-diagonal sparse tensor produced by :func:`sparse_block_diag`,
    return the original 2D sparse tensors (in the same layout) according to the
    provided shapes. Supports COO and CSR layouts.

    Parameters
    ----------
    sparse_block_diag_tensor : torch.Tensor
        Input block-diagonal sparse tensor (COO or CSR). Must be 2D and have
        exactly two sparse dimensions and zero dense dimensions.
    *shapes : tuple of int
        Sequence of shapes ``(rows_i, cols_i)`` for each block in the order they
        appear along the diagonal. The sums of rows and cols must match the
        input tensor's height and width, respectively.

    Returns
    -------
    tuple of torch.Tensor
        The recovered sparse blocks, each a 2D sparse tensor in the same layout
        as `sparse_block_diag_tensor`.

    Raises
    ------
    ValueError
        If the input layout is not COO or CSR; if any provided shape is not 2D;
        or if the sum of provided shapes does not match the input size.
    TypeError
        If `sparse_block_diag_tensor` is not a tensor.

    Notes
    -----
    - For COO inputs, this function assumes the tensor is **coalesced**. If it
      is not, it will be coalesced internally to avoid duplicate coordinates.
    - This is the inverse operation of :func:`sparse_block_diag` when given the
      correct shapes (order and sizes) of the original blocks.

    See Also
    --------
    sparse_block_diag : Construct a block-diagonal sparse matrix from 2D sparse blocks.
    """
    if not isinstance(sparse_block_diag_tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")

    if sparse_block_diag_tensor.layout == torch.sparse_coo:
        layout = torch.sparse_coo
    elif sparse_block_diag_tensor.layout == torch.sparse_csr:
        layout = torch.sparse_csr
    else:
        raise ValueError("Input tensor layout not supported. Only sparse_coo and sparse_csr are supported.")

    if not all(len(s) == 2 for s in shapes):
        raise ValueError("All shapes must be two-dimensional (rows, cols).")

    # Validate total shape matches the block-diagonal tensor
    total_rows = sum(s[0] for s in shapes)
    total_cols = sum(s[1] for s in shapes)
    in_rows, in_cols = sparse_block_diag_tensor.size(-2), sparse_block_diag_tensor.size(-1)
    if (total_rows, total_cols) != (in_rows, in_cols):
        raise ValueError(
            f"Sum of provided block shapes ({total_rows}, {total_cols}) does not match "
            f"input tensor size ({in_rows}, {in_cols})."
        )

    if layout == torch.sparse_coo:
        # Ensure coalesced for clean masking
        t = (
            sparse_block_diag_tensor.coalesce()
            if not sparse_block_diag_tensor.is_coalesced()
            else sparse_block_diag_tensor
        )
        row_idx, col_idx = t.indices()
        vals = t.values()

        blocks: list[torch.Tensor] = []
        row_offset = 0
        col_offset = 0

        for rows, cols in shapes:
            rmask = (row_idx >= row_offset) & (row_idx < row_offset + rows)
            cmask = (col_idx >= col_offset) & (col_idx < col_offset + cols)
            mask = rmask & cmask

            sub_rows = row_idx[mask] - row_offset
            sub_cols = col_idx[mask] - col_offset
            sub_vals = vals[mask]

            blocks.append(
                torch.sparse_coo_tensor(
                    torch.stack((sub_rows, sub_cols), dim=0),
                    sub_vals,
                    size=(rows, cols),
                    device=t.device,
                    dtype=sub_vals.dtype,
                )
            )

            row_offset += rows
            col_offset += cols

        return tuple(blocks)

    # CSR path
    t = sparse_block_diag_tensor
    crow = t.crow_indices()
    ccol = t.col_indices()
    vals = t.values()

    blocks: list[torch.Tensor] = []
    row_offset = 0
    col_offset = 0

    for rows, cols in shapes:
        # Pointer range for this row block in crow
        start_ptr = int(crow[row_offset].item())
        end_ptr = int(crow[row_offset + rows].item())

        # Slice the values/columns for this block and shift columns back
        sub_ccol = ccol[start_ptr:end_ptr] - col_offset
        sub_vals = vals[start_ptr:end_ptr]

        # Row pointers for this block: subtract start_ptr to rebase to 0
        sub_crow = crow[row_offset : row_offset + rows + 1] - crow[row_offset]

        blocks.append(
            torch.sparse_csr_tensor(
                sub_crow,
                sub_ccol,
                sub_vals,
                size=(rows, cols),
                device=t.device,
                dtype=sub_vals.dtype,
            )
        )

        row_offset += rows
        col_offset += cols

    return tuple(blocks)


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
    - In COO format, ``.coalesce()`` is called on the result to ensure
      duplicate indices are removed and sorted.
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

        return torch.sparse_coo_tensor(
            indices, values, size, dtype=values_dtype, device=device, requires_grad=requires_grad
        ).coalesce()

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
