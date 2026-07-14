# ⚫ RETIRE — deleted by commit 21 (spec/commit.md Phase 4); only _legacy_* op code and its tests may use this
from typing import Tuple

import torch


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
