import torch


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
    crow_indices = torch.cat([torch.zeros(1, dtype=row_indices.dtype, device=counts.device), counts.cumsum(dim=0, dtype=row_indices.dtype)])


    return crow_indices


# TODO: add support for batched
def covert_coo_to_csr_indices_values(coo_indices, num_rows, values=None):
    """Converts COO row and column indices to CSR crow and col indices.
    This function compressed the row indices and sorts the column indices.
    If values are provided, the tensor is permuted according to the sorted column indices.
    If no values are provided, the permutation indices are returned.


    Args:
        row_indices (torch.Tensor): Tensor of COO row indices.
        col_indices (torch.Tensor): Tensor of COO column indices.
        num_rows (int): Number of rows in the matrix.


    Returns:
        torch.Tensor: Compressed CSR crow indices.
        torch.Tensor: Compressed CSR col indices.
        torch.Tensor: Permutation indices from sorting the col indices. Or permuted values if values are provided.
    """
    row_indices, col_indices = coo_indices
    crow_indices = _compress_row_indices(row_indices, num_rows)
    return crow_indices, col_indices, values

    # col_indices, permutation = torch.sort(col_indices)
   
    # if values == None:
    #     return crow_indices, col_indices, permutation
    # else:
    #     return crow_indices, col_indices, values[permutation]


# TODO: add support for batched
def convert_coo_to_csr(sparse_coo_tensor):
    """Converts a COO sparse tensor to CSR format.


    Args:
        sparse_coo_tensor (torch.Tensor): COO sparse tensor.


    Returns:
        torch.Tensor: CSR sparse tensor.
    """
    if sparse_coo_tensor.layout == torch.sparse_coo:
        crow_indices, col_indices, values = covert_coo_to_csr_indices_values(sparse_coo_tensor.indices(), sparse_coo_tensor.size()[0], sparse_coo_tensor.values())
        return torch.sparse_csr_tensor(crow_indices, col_indices, values, sparse_coo_tensor.size())
    else:
        raise ValueError(f"Unsupported layout: {sparse_coo_tensor.layout}")
   


def demcompress_crow_indices(crow_indices, num_rows):
    """Decompresses CSR crow indices to COO row indices.


    Args:
        csr_crow_indices (torch.Tensor): Tensor of CSR crow indices.
        num_rows (int): Number of rows in the matrix.


    Returns:
        torch.Tensor: Decompressed COO row indices.
    """
   
    row_indices = torch.repeat_interleave(
                torch.arange(num_rows, dtype=crow_indices.dtype, device=crow_indices.device), crow_indices[1:] - crow_indices[:-1]
            )
   
    return row_indices