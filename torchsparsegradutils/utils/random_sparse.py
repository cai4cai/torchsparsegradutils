"""
utility functions for generating random sparse matrices

NOTE: sparse COO tensors have indices tensor of size (ndim, nse) and with element type torch.int64
NOTE: Sparse CSR The index tensors crow_indices and col_indices should have element type either torch.int64 (default) or torch.int32. 
      If you want to use MKL-enabled matrix operations, use torch.int32. 
      This is as a result of the default linking of pytorch being with MKL LP64, which uses 32 bit integer indexing
"""

import torch
import random

def _gencoordinates_2d_coo(nr, nc, nnz, *, dtype=torch.int64, device=torch.device("cpu")):
    """Used to genererate nnz random unique coordinates
    
    Args:
        nr (int): number of rows
        nc (int): number of columns
        nnz (int): number of pairs of coordinates to generate
        device (torch.device, optional): device to generate coordinates on. Defaults to torch.device("cpu").

    Returns:
        torch.tensor: tensor of shape [2, nnz] containing the generated coordinates
    """
    assert nnz <= nr * nc, "Number of elements (nnz) must be less than or equal to the total number of elements (nr * nc)."
    
    coordinates = set()
    while True:
        r, c = random.randrange(nr), random.randrange(nc)
        coordinates.add((r, c))
        if len(coordinates) == nnz:
            return torch.stack([torch.tensor(co, dtype=dtype) for co in coordinates], dim=-1).to(device)


def _gencoordinates_2d_csr(nr, nc, nnz, *, dtype=torch.int64, device=torch.device("cpu")):
    """Used to genererate nnz random unique csr coordinates
    
    Args:
        nr (int): number of rows
        nc (int): number of columns
        nnz (int): number of elements the generated csr indices represent
        device (torch.device, optional): device to generate coordinates on. Defaults to torch.device("cpu").

    Returns:
        crow_indices (torch.tensor): tensor of shape [nr+1] containing the generated crow_indices
        col_indices (torch.tensor): tensor of shape [nnz] containing the generated col_indices
    """
        
    assert nnz <= nr * nc, "Number of elements (nnz) must be less than or equal to the total number of elements (nr * nc)."
    
    crow_indices = torch.zeros(nr+1, dtype=dtype, device=device)
    col_indices = torch.zeros(nnz, dtype=dtype, device=device)
    
    for i in range(nr):
        if i == nr - 1:
            step = nnz - crow_indices[i]  # if we reach the last step, we set step size to ensure we reach nnz
        else:
            step = torch.randint(0, min(nc+1, nnz - crow_indices[i] - (nr - i)), (1,))
        crow_indices[i+1] = crow_indices[i] + step
        
        col_indices[crow_indices[i]:crow_indices[i+1]] = torch.sort(torch.randperm(nc)[:step])[0]

    return crow_indices, col_indices

  
def gencoordinates(size, nnz, *, layout=torch.sparse_coo, dtype=torch.int64, device=torch.device("cpu")):
    """Used to genererate nnz random unique COO or CSR coordinates for batched sparse matrix specified by size ((b), nr, nc).
    As per the PyTorch documentation, batched sparse matrices must have the same number of elements (nnz) per batch element.
    Currently, for simplicity, this implementation uses the same indices for each batch element.
    Therefore, the coordinates are unique for each batch element, but not across batch elements.

    Args:
        size (tuple): Tuple specifying the dimensions of the batched sparse matrix. The size can be either ((nr, nc)) for a single batch or ((b), nr, nc) for a batched matrix, where (b) represents the number of batch elements.
        nnz (int): Number of elements the generated COO or CSR indices represent. This must be less than or equal to the total number of elements in the matrix.
        layout (torch.layout, optional): Layout of the sparse matrix, either torch.sparse_coo or torch.sparse_csr. Defaults to torch.sparse_coo.
        dtype (torch.dtype, optional): Data type of the generated tensor. Defaults to torch.int64.
        device (torch.device, optional): Device to generate the coordinates on. Defaults to torch.device("cpu").


    Raises:
        ValueError: Raised if size has less than 2 dimensions.
        ValueError: Raised if size has more than 3 dimensions, as this implementation only supports 1 batch dimension.
        ValueError: Raised if nnz is greater than the total number of elements (nr * nc).
        ValueError: Raised if the layout is not torch.sparse_coo or torch.sparse_csr.

    Returns:
        torch.Tensor: returns a tensor of shape [ndim, nnz] for COO layout or [nr+1] and [nnz] for CSR layout
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")
    
    if nnz > size[-2] * size[-1]:
        raise ValueError("nnz must be less than or equal to nr * nc")
    
    if layout == torch.sparse_coo:
        unbatched_coo_coordinates = _gencoordinates_2d_coo(size[-2], size[-1], nnz, dtype=dtype, device=device)
        if len(size) == 2:
            return unbatched_coo_coordinates
        else:
            sparse_dim_coordinates = torch.cat([unbatched_coo_coordinates] * size[0], dim=-1)
            batch_dim_coordinates = torch.arange(size[0], dtype=dtype, device=device).repeat(nnz).flatten().unsqueeze(0)
            return torch.cat([batch_dim_coordinates, sparse_dim_coordinates])
        
    elif layout == torch.sparse_csr:
        crow_indices, col_indices = _gencoordinates_2d_csr(size[-2], size[-1], nnz, dtype=dtype, device=device)
        if len(size) == 2:
            return crow_indices, col_indices
        else:
            crow_indices = crow_indices.repeat(size[0], 1)
            col_indices = col_indices.repeat(size[0], 1)
            return crow_indices, col_indices
    else:
        raise ValueError(f"layout must be torch.sparse_coo or torch.sparse_csr, but got layout {layout}")


# Square strictly Triangular:

def _gencoordinates_2d_coo_strictly_tri(n, nnz, upper=True, *, dtype=torch.int64, device=torch.device("cpu")):
    """Used to generate nnz random unique COO coordinates for a square matrix with either strictly lower triangular or strictly upper triangular coordinates.

    Args:
        n (int): Size of the square matrix (number of rows and columns).
        nnz (int): Number of elements the generated COO indices represent.
        upper (bool, optional): Flag indicating whether to generate strictly upper triangular coordinates. Defaults to True.
        dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.int64.
        device (torch.device, optional): Device to generate coordinates on. Defaults to torch.device("cpu").

    Returns:
        indices (torch.Tensor): Tensor of shape [2, nnz] containing the generated COO indices.
    """
    assert nnz <= n * (n - 1) // 2, "Number of elements (nnz) must be less than or equal to the total number of elements (n * (n - 1) // 2)."

    coordinates = set()
    while True:
            r, c = random.randrange(n), random.randrange(n)
            if (r < c and upper) or (r > c and not upper):
                coordinates.add((r, c))
            if len(coordinates) == nnz:
                return torch.stack([torch.tensor(co) for co in coordinates], dim=-1).to(device)

    
def _gencoordinates_2d_csr_strictly_tri(n, nnz, upper=True, *, dtype=torch.int64, device=torch.device("cpu")):
    """Used to generate nnz random unique csr coordinates for a square matrix with either strictly lower triangular or strictly upper triangular coordinates.
    
    Args:
        n (int): Number of rows and columns (since it's a square matrix).
        nnz (int): Number of elements the generated csr indices represent.
        upper (bool, optional): Flag indicating whether to generate strictly upper triangular coordinates. Defaults to True.
        dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.int64.
        device (torch.device, optional): Device to generate coordinates on. Defaults to torch.device("cpu").

    Returns:
        crow_indices (torch.Tensor): Tensor of shape [nr+1] containing the generated crow_indices.
        col_indices (torch.Tensor): Tensor of shape [nnz] containing the generated col_indices.
    """
    assert nnz <= n * n, "Number of elements (nnz) must be less than or equal to the total number of elements (n * n)."
    assert n == n, "Number of rows (nr) must be equal to the number of columns (nc) to create a square matrix."
    
    crow_indices = torch.zeros(n+1, dtype=dtype, device=device)
    col_indices = torch.zeros(nnz, dtype=dtype, device=device)
    
    for i in range(n):
        if i == n - 1:
            step = nnz - crow_indices[i]  # If we reach the last step, we set the step size to ensure we reach nnz.
        else:
            max_step = min(n-i, nnz - crow_indices[i] - (n - i))
            step = random.randint(0, max_step+1) if upper else torch.randint(1, max_step+1)
        crow_indices[i+1] = crow_indices[i] + step
        
        if upper:
            col_indices[crow_indices[i]:crow_indices[i+1]] = torch.arange(i+1, i+1+step, device=device)
        else:
            col_indices[crow_indices[i]:crow_indices[i+1]] = torch.arange(0, step, device=device)

    return crow_indices, col_indices
        
        
def gencoordinates_square_strictly_tri(size, nnz, *, upper=True, layout=torch.sparse_coo, dtype=torch.int64, device=torch.device("cpu")):
    """Used to genererate nnz random unique COO or CSR coordinates for sparse strictly lower triangular or strictly upper triangular matrices with shape ((b), n, n).
    As per the PyTorch documentation, batched sparse matrices must have the same number of elements (nnz) per batch element.
    Currently, for simplicity, this implementation uses the same indices for each batch element.
    Therefore, the coordinates are unique for each batch element, but not across batch elements.

    Args:
        size (tuple): _description_
        nnz (int): _description_
        layout (torch.layout, optional): _description_. Defaults to torch.sparse_coo.
        dtype (torch.dtype, optional): _description_. Defaults to torch.int64.
        device (torch.device, optional): _description_. Defaults to torch.device("cpu").

    Raises:
        ValueError: Raised if size has less than 2 dimensions.
        ValueError: Raised if size has more than 3 dimensions, as this implementation only supports 1 batch dimension.
        ValueError: Raise if size is not a square matrix (n, n) or batched square matrix (b, n, n).
        ValueError: Raised if nnz is greater than the total number of elements (nr * nc).
        ValueError: Raised if the layout is not torch.sparse_coo or torch.sparse_csr.

    Returns:
        torch.Tensor: returns a tensor of shape [ndim, nnz] for COO layout or [nr+1] and [nnz] for CSR layout
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")
    
    if size[-2] != size[-1]:
        raise ValueError("size must be a square matrix (n, n) or batched square matrix (b, n, n)")
    
    if nnz > size[-2] * size[-1]:
        raise ValueError("nnz must be less than or equal to n * n")
    
    if layout == torch.sparse_coo:
        unbatched_coo_coordinates = _gencoordinates_2d_coo_strictly_tri(size[-2], nnz, dtype=dtype, device=device)
        if len(size) == 2:
            return unbatched_coo_coordinates
        else:
            sparse_dim_coordinates = torch.cat([unbatched_coo_coordinates] * size[0], dim=-1)
            batch_dim_coordinates = torch.arange(size[0], dtype=dtype, device=device).repeat(nnz).flatten().unsqueeze(0)
            return torch.cat([batch_dim_coordinates, sparse_dim_coordinates])
        
    elif layout == torch.sparse_csr:
        crow_indices, col_indices = _gencoordinates_2d_csr_strictly_tri(size[-2], nnz, dtype=dtype, device=device)
        if len(size) == 2:
            return crow_indices, col_indices
        else:
            crow_indices = crow_indices.repeat(size[0], 1)
            col_indices = col_indices.repeat(size[0], 1)
            return crow_indices, col_indices
    else:
        raise ValueError(f"layout must be torch.sparse_coo or torch.sparse_csr, but got layout {layout}")
    












# unit test and move to contraints
def is_strictly_triangular(crow_indices, col_indices, upper=True):
    """Check whether the given crow_indices and col_indices represent strictly upper triangular or strictly lower triangular indices.
    
    Args:
        crow_indices (torch.Tensor): Tensor of shape [nr+1] containing the crow_indices.
        col_indices (torch.Tensor): Tensor of shape [nnz] containing the col_indices.
        upper (bool, optional): Flag indicating whether to check for strictly upper triangular indices. Defaults to True.
    
    Returns:
        bool: True if the indices are strictly upper triangular or strictly lower triangular, False otherwise.
    """
    assert crow_indices.dim() == 1 and col_indices.dim() == 1, "Input tensors must be 1-dimensional."
    assert crow_indices.size(0) == crow_indices[-1], "crow_indices must represent a valid compressed sparse row structure."
    assert col_indices.size(0) == crow_indices[-1], "col_indices must have the same length as crow_indices[-1]."
    
    if upper:
        for i in range(crow_indices.size(0) - 1):
            if (col_indices[crow_indices[i]:crow_indices[i+1]] <= i).any():
                return False
        return True
    else:
        for i in range(crow_indices.size(0) - 1):
            if (col_indices[crow_indices[i]:crow_indices[i+1]] >= i+1).any():
                return False
        return True