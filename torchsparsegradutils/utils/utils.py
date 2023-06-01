import torch

def compress_row_indices(row_indices, num_rows):
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