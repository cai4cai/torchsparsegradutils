# extracted from f19d7b4 — parity Oracle A, never shipped; do not edit
#
# Frozen copy of the block-diagonal batching helpers that the pre-rewrite ops
# in this package relied on (``sparse_block_diag`` / ``sparse_block_diag_split``
# / ``stack_csr``). These are ⚫ RETIRE in map.md and are deleted from the live
# package, so the oracle carries its own copies rather than importing them.

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
