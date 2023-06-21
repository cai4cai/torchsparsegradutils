import torch


def stack_csr(tensors, dim=0):
    """
    Stacks a list of CSR tensors along the batch dimension.
    This function is analogous to torch.stack() but for CSR tensors.

    Args:
        tensors (list): List of CSR tensors to be stacked.
        dim (int): The axis to stack the tensors along. Default is 0.

    Returns:
        torch.Tensor: Stacked CSR tensor.
    """
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("Expected a list of tensors, but got {}.".format(type(tensors)))

    if len(tensors) == 0:
        raise ValueError("Cannot stack empty list of tensors.")

    if not all([tensor.shape == tensors[0].shape for tensor in tensors]):
        raise ValueError("All tensors must have the same shape.")

    if not all([tensor.layout == torch.sparse_csr for tensor in tensors]):
        raise ValueError("All tensors must be CSR.")

    if not all([tensor.ndim == 2 for tensor in tensors]):
        raise ValueError("All tensors must be 2D.")

    crow_indices = torch.stack([tensor.crow_indices() for tensor in tensors], dim=dim)
    col_indices = torch.stack([tensor.col_indices() for tensor in tensors], dim=dim)
    values = torch.stack([tensor.values() for tensor in tensors], dim=dim)

    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    shape = tuple(shape)

    return torch.sparse_csr_tensor(crow_indices, col_indices, values, shape)


def _sort_coo_indices(indices):
    """
    Sorts COO (Coordinate List Format) indices in ascending order and returns a permutation tensor that indicates the indices in the original data that result in a sorted tensor.

    This function can support both unbatched and batched COO indices, and essentially performs the same operation as .coalesce() called on a COO tensor.
    The advantage is that COO coordinates can be sorted prior to conversion to CSR, without having to use the torch.sparse_coo_tensor object which only supports int64 indices.

    Args:
        indices (torch.Tensor): The input indices in COO format to be sorted.

    Returns:
        torch.Tensor: A tensor containing sorted indices.
        torch.Tensor: A permutation tensor that contains the indices in the original tensor that give the sorted tensor.
    """
    indices_sorted, permutation = torch.unique(indices, dim=-1, sorted=True, return_inverse=True)
    return indices_sorted.contiguous(), torch.argsort(permutation)


def _compress_row_indices(row_indices, num_rows):
    """Compresses COO row indices to CSR crow indices.


    Args:
        row_indices (torch.Tensor): Tensor of COO row indices.
        num_rows (int): Number of rows in the matrix.


    Returns:
        torch.Tensor: Compressed CSR crow indices.
    """
    # Compute the number of non-zero elements in each row
    counts = torch.bincount(row_indices, minlength=num_rows).to(row_indices.dtype)

    # Compute the cumulative sum of counts to get CSR indices
    crow_indices = torch.cat([torch.zeros(1, dtype=row_indices.dtype, device=counts.device), counts.cumsum_(dim=0)])

    return crow_indices


def convert_coo_to_csr_indices_values(coo_indices, num_rows, values=None):
    """Converts COO row and column indices to CSR crow and col indices.
    Supports batched indices, which would have shape [3, nnz]. Or, [2, nnz] for unbatched indices.
    This function sorts the row and column indices (similar to torch.sparse_coo_tensor.coalesce()) and then compresses the row indices to CSR crow indices.
    If values are provided, the values tensor is permuted according to the sorted COO indices.
    If no values are provided, the permutation indices are returned.


    Args:
        coo_indices (torch.Tensor): Tensor of COO indices
        num_rows (int): Number of rows in the matrix.


    Returns:
        torch.Tensor: Compressed CSR crow indices.
        torch.Tensor: CSR Col indices.
        torch.Tensor: Permutation indices from sorting COO indices. Or permuted values if values are provided.
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
    """Converts a COO sparse tensor to CSR format. COO tensor can have optional single leading batch dimension.

    Args:
        sparse_coo_tensor (torch.Tensor): COO sparse tensor.


    Returns:
        torch.Tensor: CSR sparse tensor.
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
    """Decompresses CSR crow indices to COO row indices.


    Args:
        csr_crow_indices (torch.Tensor): Tensor of CSR crow indices.
        num_rows (int): Number of rows in the matrix.


    Returns:
        torch.Tensor: Decompressed COO row indices.
    """

    row_indices = torch.repeat_interleave(
        torch.arange(num_rows, dtype=crow_indices.dtype, device=crow_indices.device),
        crow_indices[1:] - crow_indices[:-1],
    )

    return row_indices


# use @torch.jit.script ?
def sparse_block_diag(*sparse_tensors):
    """
    Function to create a block diagonal sparse matrix from provided sparse tensors.
    This function is designed to replicate torch.block_diag(), but for sparse tensors,
    but only supports 2D sparse tensors
    (whereas torch.block_diag() supports dense tensors of 0, 1 or 2 dimensions).

    Args:
        *sparse_tensors (torch.Tensor): Variable length list of sparse tensors. All input tensors must either be all
                                        sparse_coo or all sparse_csr format. The input sparse tensors must have exactly
                                        two sparse dimensions and no dense dimensions.

    Returns:
        A block diagonal sparse tensor in the same format (either sparse_coo or sparse_csr) as the input tensors.

    Raises:
        TypeError: If the inputs are not provided as a list or tuple.
        ValueError: If no tensors are provided or if the provided tensors are not all of the same sparse format.
        ValueError: If all sparse tensors do not have two sparse dimensions and zero dense dimensions.
    """

    for i, sparse_tensor in enumerate(sparse_tensors):
        if not isinstance(sparse_tensor, torch.Tensor):
            raise TypeError(
                f"TypeError: expected Tensor as element {i} in argument 0, but got {type(sparse_tensor).__name__}"
            )

    if len(sparse_tensors) == 0:
        raise ValueError("At least one sparse tensor must be provided.")

    if all(sparse_tensor.layout == torch.sparse_coo for sparse_tensor in sparse_tensors):
        layout = torch.sparse_coo
    elif all(sparse_tensor.layout == torch.sparse_csr for sparse_tensor in sparse_tensors):
        layout = torch.sparse_csr
    else:
        raise ValueError("Sparse tensors must either be all sparse_coo or all sparse_csr.")

    if not all(sparse_tensor.sparse_dim() == 2 for sparse_tensor in sparse_tensors):
        raise ValueError("All sparse tensors must have two sparse dimensions.")

    if not all(sparse_tensor.dense_dim() == 0 for sparse_tensor in sparse_tensors):
        raise ValueError("All sparse tensors must have zero dense dimensions.")

    if len(sparse_tensors) == 1:
        return sparse_tensors[0]

    if layout == torch.sparse_coo:
        row_indices_list = []
        col_indices_list = []
        values_list = []

        num_row = 0
        num_col = 0

        for i, sparse_tensor in enumerate(sparse_tensors):
            sparse_tensor = sparse_tensor.coalesce() if not sparse_tensor.is_coalesced() else sparse_tensor

            row_indices, col_indices = sparse_tensor.indices()

            # calculate block offsets
            # not in-place addition to avoid modifying the original tensor indices
            row_indices = row_indices + i * sparse_tensor.size()[-2]
            col_indices = col_indices + i * sparse_tensor.size()[-1]

            # accumulate indices and values:
            row_indices_list.append(row_indices)
            col_indices_list.append(col_indices)
            values_list.append(sparse_tensor.values())

            # accumulate tensor sizes:
            num_row += sparse_tensor.size()[-2]
            num_col += sparse_tensor.size()[-1]

        row_indices = torch.cat(row_indices_list)
        col_indices = torch.cat(col_indices_list)
        values = torch.cat(values_list)

        return torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]), values, torch.Size([num_row, num_col]))

    elif layout == torch.sparse_csr:
        crow_indices_list = []
        col_indices_list = []
        values_list = []

        num_row = 0
        num_col = 0

        for i, sparse_tensor in enumerate(sparse_tensors):
            crow_indices = sparse_tensor.crow_indices()
            col_indices = sparse_tensor.col_indices()

            # Calculate block offsets
            # not in-place addition to avoid modifying the original tensor indices
            if i > 0:
                crow_indices = crow_indices[1:]
                crow_indices = crow_indices + crow_indices_list[-1][-1]
            col_indices = col_indices + i * sparse_tensor.size()[-1]

            # accumulate tensor sizes:
            num_row += sparse_tensor.size()[-2]
            num_col += sparse_tensor.size()[-1]

            # Accumulate indices and values:
            crow_indices_list.append(crow_indices)
            col_indices_list.append(col_indices)
            values_list.append(sparse_tensor.values())

        crow_indices = torch.cat(crow_indices_list)
        col_indices = torch.cat(col_indices_list)
        values = torch.cat(values_list)

        return torch.sparse_csr_tensor(crow_indices, col_indices, values, torch.Size([num_row, num_col]))


def sparse_block_diag_split(sparse_block_diag_tensor, *shapes):
    """
    Function to split a block diagonal sparse matrix into original sparse tensors.

    NOTE: Sparse COO tensors are assumed to already by coalesced.
    This is because newly created or indexed sparse COO tensors default to is_coalesced=False,
    and running coalesce() imposes an unnecessary performance penalty.

    Args:
        sparse_block_diag_tensor (torch.Tensor): The input block diagonal sparse tensor.
        *shapes (sequence of tuple): The shapes of the original tensors. This is required to correctly split the tensor.

    Returns:
        A list of original sparse tensors.

    Raises:
        TypeError: If the input tensor is not a sparse tensor.
    """

    if sparse_block_diag_tensor.layout == torch.sparse_coo:
        layout = torch.sparse_coo
    elif sparse_block_diag_tensor.layout == torch.sparse_csr:
        layout = torch.sparse_csr
    else:
        raise ValueError("Input tensor format not supported. Only sparse_coo and sparse_csr are supported.")

    if not all(len(shape) == 2 for shape in shapes):
        raise ValueError("All shapes must be two-dimensional.")

    if layout == torch.sparse_coo:
        tensors = []
        start_row = 0
        start_col = 0
        current_val_offset = 0

        sparse_block_diag_tensor = (
            sparse_block_diag_tensor.coalesce()
            if not sparse_block_diag_tensor.is_coalesced()
            else sparse_block_diag_tensor
        )

        row_indices, col_indices = sparse_block_diag_tensor.indices()
        values = sparse_block_diag_tensor.values()

        for shape in shapes:
            rows, cols = shape[-2], shape[-1]

            mask_row = (start_row <= row_indices) & (row_indices < start_row + rows)
            mask_col = (start_col <= col_indices) & (col_indices < start_col + cols)
            mask = mask_row & mask_col

            indices_sub = torch.stack((row_indices[mask] - start_row, col_indices[mask] - start_col))
            values_sub = values[mask]

            tensor_sub = torch.sparse_coo_tensor(indices_sub, values_sub, (rows, cols))
            tensors.append(tensor_sub)

            start_row += rows
            start_col += cols
            current_val_offset += rows * cols

        return tuple(tensors)

    elif layout == torch.sparse_csr:
        tensors = []
        start_row = 0
        current_col_offset = 0
        current_val_offset = 0

        crow_indices = sparse_block_diag_tensor.crow_indices()
        col_indices = sparse_block_diag_tensor.col_indices()
        values = sparse_block_diag_tensor.values()

        for shape in shapes:
            rows, cols = shape[-2], shape[-1]

            # Compute the number of values in the sub-block
            values_count = crow_indices[start_row + shape[0]] - crow_indices[start_row]

            # Find the starting and ending points in crow_indices
            start_ptr = crow_indices[start_row]

            # Apply the pointers to get the sub-block indices and values
            col_indices_sub = col_indices[current_val_offset : current_val_offset + values_count] - current_col_offset
            values_sub = values[current_val_offset : current_val_offset + values_count]

            # Create the sub-block crow_indices
            crow_indices_sub = crow_indices[start_row : start_row + shape[0] + 1] - start_ptr

            # Construct the sub-block as a CSR tensor
            tensor_sub = torch.sparse_csr_tensor(crow_indices_sub, col_indices_sub, values_sub, (rows, cols))

            tensors.append(tensor_sub)
            start_row += shape[0]
            current_col_offset += cols
            current_val_offset += values_count

        return tuple(tensors)


def sparse_eye(
    size,
    *,
    layout=torch.sparse_coo,
    values_dtype=torch.float64,
    indices_dtype=torch.int64,
    device=torch.device("cpu"),
    requires_grad=False,
):
    """
    Function to create a sparse identity matrix.

    Args:
        size (tuple): Tuple specifying the dimensions of the sparse matrix. The size can be either (num_rows, num_cols) for an unbatched matrix or (batch_size, num_rows, num_cols) for a batched matrix. The number of rows and columns must be equal.
        values_dtype (:class:`torch.dtype`, optional): The desired data type of values tensor. Default is torch.float64.
        indices_dtype (:class:`torch.dtype`, optional): The desired data type of indices tensor. Default is torch.int64.
        layout (:class:`torch.layout`, optional): The desired layout of returned SparseTensor. Default is torch.sparse_coo_tensor.
        device (:class:`torch.device`, optional): The desired device of returned tensor. Default is torch.device('cpu').
        requires_grad (bool, optional): If autograd should record operations on the returned tensor.

    Returns:
        A sparse identity matrix of shape (n, n).

    Raises:
        ValueError: If the provided data types or layout are not supported.
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")

    if size[-2] != size[-1]:
        raise ValueError("size must be a square matrix (n, n) or batched square matrix (b, n, n)")

    if values_dtype not in [torch.float32, torch.float64]:
        raise ValueError(
            "Values dtype {} not supported. Only torch.float32 and torch.float64 are supported.".format(values_dtype)
        )

    values = torch.ones(size[-1], dtype=values_dtype, device=device)

    if layout == torch.sparse_coo:
        if indices_dtype != torch.int64:
            raise ValueError("For sparse_coo layout, indices_dtype has to be torch.int64.")

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
