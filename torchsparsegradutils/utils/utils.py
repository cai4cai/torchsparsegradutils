import torch


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

    if values != None and values.shape[0] != coo_indices.shape[1]:
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


def sparse_block_diag(*sparse_tensors):
    if not isinstance(sparse_tensors, (list, tuple)):
        raise TypeError('sparse_tensors must be a list or a tuple')

    if len(sparse_tensors) == 0:
        raise ValueError("At least one sparse tensor must be provided.")
    elif len(sparse_tensors) == 1:
        return sparse_tensors[0]
    
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

    if layout == torch.sparse_coo:
        
        row_indices_list = []
        col_indices_list = []
        values_list = []

        num_row = 0
        num_col = 0
        
        for i, sparse_tensor in enumerate(sparse_tensors):            
            
            sparse_tensor = sparse_tensor.coalesce()
            row_indices, col_indices = sparse_tensor.indices()
            
            # calculate block offsets
            row_indices += i * sparse_tensor.size()[-2]
            col_indices += i * sparse_tensor.size()[-1]

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
            if i > 0:
                crow_indices = crow_indices[1:]
                crow_indices += crow_indices_list[-1][-1]
            col_indices += i * sparse_tensor.size()[-1]

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
