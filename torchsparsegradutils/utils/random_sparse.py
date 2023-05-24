"""
utility functions for generating random sparse matrices

NOTE: sparse COO tensors have indices tensor of size (ndim, nse) and with element type torch.int64

NOTE: Sparse CSR The index tensors crow_indices and col_indices should have element type either torch.int64 (default) or torch.int32. 
If you want to use MKL-enabled matrix operations, use torch.int32. 
This is as a result of the default linking of pytorch being with MKL LP64, which uses 32 bit integer indexing
"""

import torch
from random import randrange

def _gencoordinates_2d(nr, nc, nnz, *, dtype=torch.int64, device=torch.device("cpu")):
    """Used to genererate nnz random unique coordinates
    
    Args:
        nr (int): number of rows
        nc (int): number of columns
        nnz (int): number of pairs of coordinates to generate
        device (torch.device, optional): device to generate coordinates on. Defaults to torch.device("cpu").

    Returns:
        torch.tensor: tensor of shape [2, nnz] containing the generated coordinates
    """
    coordinates = set()
    while True:
        r, c = randrange(nr), randrange(nc)
        coordinates.add((r, c))
        if len(coordinates) == nnz:
            return torch.stack([torch.tensor(co, dtype=dtype) for co in coordinates], dim=-1).to(device)


def gencoordinates(size, nnz, *, layout=torch.sparse_coo, dtype=torch.int64, device=torch.device("cpu")):
    """
    Used to genererate nnz random unique COO or CSR coordinates for sparse matrix specified by size
    As per the PyTorch documentation, batched sparse matrices must have the same number of elements (nnz) per batch element.
    Currently, for simplicity, this implementation uses the same indices for each batch element.
    Therefore, the coordinates are unique for each bathc element, but not across batch elements.
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")
            
    coo_coordinates = _gencoordinates_2d(size[-2], size[-1], nnz, dtype=dtype, device=device)
    
    if layout == torch.sparse_coo:
        if len(size) == 2:
            return coo_coordinates
        else:
            sparse_dim_coordinates = torch.cat([coo_coordinates] * size[0], dim=-1)
            batch_dim_coordinates = torch.arange(size[0], dtype=dtype, device=device).repeat(nnz).flatten().unsqueeze(0)
            return torch.cat([batch_dim_coordinates, sparse_dim_coordinates])
    
    
    