"""
utility functions for generating random sparse matrices

NOTE: sparse COO tensors have indices tensor of size (ndim, nse) and with element type torch.int64
NOTE: Sparse CSR The index tensors crow_indices and col_indices should have element type either torch.int64 (default) or torch.int32. 
      If you want to use MKL-enabled matrix operations, use torch.int32. 
      This is as a result of the default linking of pytorch being with MKL LP64, which uses 32 bit integer indexing
"""
import warnings
import torch
import random
from torchsparsegradutils.utils.utils import compress_row_indices

def _gen_indices_2d_coo(nr, nc, nnz, *, dtype=torch.int64, device=torch.device("cpu")):
    """Generates nnz random unique coordinates in COO format.

    Args:
        nr (int): Number of rows.
        nc (int): Number of columns.
        nnz (int): Number of pairs of coordinates to generate.
        device (torch.device, optional): Device to generate coordinates on. Defaults to torch.device("cpu").

    Returns:
        torch.tensor: Tensor of shape [2, nnz] containing the generated coordinates.
    """
    assert nnz <= nr * nc, "Number of elements (nnz) must be less than or equal to the total number of elements (nr * nc)."    
  
    coordinates = set()
    while True:
        r, c = random.randrange(nr), random.randrange(nc)
        coordinates.add((r, c))
        if len(coordinates) == nnz:
            return torch.stack([torch.tensor(co, dtype=dtype, device=device) for co in coordinates], dim=-1)
    
    # Alternatively, could do:
    # indices = torch.randperm(nr * nc)[:nnz]
    # return torch.stack([indices // nc, indices % nc]).to(device)
    
    
def generate_random_sparse_coo_matrix(size, nnz, *, indices_dtype=torch.int64, values_dtype=torch.float32, device=torch.device("cpu")):
    """Generates a random sparse COO matrix of the specified size and number of non-zero elements.

    Args:
        size (tuple): Tuple specifying the dimensions of the batched sparse matrix. The size can be either (num_rows, num_cols) for a unbatched matrix or (batch_size, num_rows, num_cols) for a batched matrix.
        nnz (int): Number of non-zero values in sparse matrix (number of sparse elements). This must be less than or equal to the total number of elements in the matrix.
        indices_dtype (torch.dtype, optional): Data type for indices of sparse tensor. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values of sparse tensor. Defaults to torch.float32.
        device (torch.device, optional): Device to generate the tensor on. Defaults to torch.device("cpu").

    Raises:
        ValueError: Raised if size has less than 2 dimensions.
        ValueError: Raised if size has more than 3 dimensions, as this implementation only supports 1 batch dimension.
        ValueError: Raised if nnz is greater than the total number of elements (size[-2] * size[-3]).
        UserWarning: Raised when `indices_dtype` is not `torch.int64`, as this is the only indices dtype supported for sparse COO tensors. Any other index dtype will be converted to `torch.int64`.

    Returns:
        torch.Tensor: Returns a sparse COO tensor of shape size with nnz non-zero elements.
    """
   
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")

    if nnz > size[-2] * size[-1]:
        raise ValueError("nnz must be less than or equal to nr * nc")
    
    if indices_dtype != torch.int64:
        warnings.warn(f"Only indices of type torch.int64 supported for sparse COO tensors. Indices of type {indices_dtype} will be cast to torch.int64.", UserWarning)
    
    if len(size) == 2:
        coo_indices = _gen_indices_2d_coo(size[-2], size[-1], nnz, dtype=indices_dtype, device=device)
        values = torch.rand(nnz, dtype=values_dtype, device=device)
    else:
        sparse_dim_indices = torch.cat([_gen_indices_2d_coo(size[-2], size[-1], nnz, dtype=indices_dtype, device=device) for _ in range(size[0])], dim=-1)
        batch_dim_indices = torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(nnz).unsqueeze(0)
        coo_indices = torch.cat([batch_dim_indices, sparse_dim_indices])
        values = torch.rand(nnz*size[0], dtype=values_dtype, device=device)        

    return torch.sparse_coo_tensor(coo_indices, values, size, device=device).coalesce()


def generate_random_sparse_csr_matrix(size, nnz, *, indices_dtype=torch.int64, values_dtype=torch.float32, device=torch.device("cpu")):
    """Generates a random sparse CSR matrix of the specified size and number of non-zero elements.

    Args:
        size (tuple): Tuple specifying the dimensions of the batched sparse matrix. The size can be either (num_rows, num_cols) for a unbatched matrix or (batch_size, num_rows, num_cols) for a batched matrix.
        nnz (int): Number of non-zero values in sparse matrix (number of sparse elements). This must be less than or equal to the total number of elements in the matrix.
        indices_dtype (torch.dtype, optional): Data type for indices of sparse tensor. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values of sparse tensor. Defaults to torch.float32.
        device (torch.device, optional): Device to generate the tensor on. Defaults to torch.device("cpu").

    Raises:
        ValueError: Raised if size has less than 2 dimensions.
        ValueError: Raised if size has more than 3 dimensions, as this implementation only supports 1 batch dimension.
        ValueError: Raised if nnz is greater than the total number of elements (size[-2] * size[-3]).
        UserWarning: Raised when `indices_dtype` has a bit depth less than `torch.int32`, as this is not recommended.

    Returns:
        torch.Tensor: Returns a sparse CSR tensor of shape size with nnz non-zero elements.
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")

    if nnz > size[-2] * size[-1]:
        raise ValueError("nnz must be less than or equal to nr * nc")
    
    if (indices_dtype != torch.int64) and (indices_dtype != torch.int32):
        warnings.warn(f"A bit depth of less than torch.int32 is not recommended for sparse CSR tensors", UserWarning)

    row_indices, col_indices = _gen_indices_2d_coo(size[-2], size[-1], nnz, dtype=indices_dtype, device=device)
    crow_indices = compress_row_indices(row_indices, size[-2])
    
    if len(size) == 2:
        values = torch.rand(nnz, dtype=values_dtype, device=device)
    else:
        crow_indices = crow_indices.repeat(size[0], 1)
        col_indices = col_indices.repeat(size[0], 1)
        values = torch.rand(nnz*size[0], dtype=values_dtype, device=device)
        
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size, device=device)


# Square strictly Triangular:

def _gen_indices_2d_coo_strictly_tri(n, nnz, *, upper=True, dtype=torch.int64, device=torch.device("cpu")):
    """Generates nnz random unique COO coordinates for a square matrix with either strictly lower triangular or strictly upper triangular coordinates.

    Args:
        n (int): Size of the square matrix (number of rows and columns).
        nnz (int): Number of elements the generated COO indices represent.
        upper (bool, optional): Flag indicating whether to generate strictly upper triangular coordinates. Defaults to True.
        dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.int64.
        device (torch.device, optional): Device to generate coordinates on. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: Tensor of shape [2, nnz] containing the generated COO indices.
    """
    assert nnz <= n * (n - 1) // 2, "Number of elements (nnz) must be less than or equal to the total number of elements (n * (n - 1) // 2)."

    coordinates = set()
    while True:
        r, c = random.randrange(n), random.randrange(n)
        if (r < c and upper) or (r > c and not upper):
            coordinates.add((r, c))
        if len(coordinates) == nnz:
            return torch.stack([torch.tensor(co, dtype=dtype, device=device) for co in coordinates], dim=-1).to(device)
                    
        
def generate_random_sparse_strictly_triangular_coo_matrix(size, nnz, *, upper=True, indices_dtype=torch.int64, values_dtype=torch.float32, device=torch.device("cpu")):
    """Generates a random sparse COO square matrix with strictly upper or lower triangular coordinates.

    Args:
        size (tuple): Tuple specifying the dimensions of the batched sparse matrix. The size can be either (num_rows, num_cols) for a unbatched matrix or (batch_size, num_rows, num_cols) for a batched matrix. The number of rows and columns must be equal.
        nnz (int): Number of non-zero values in sparse matrix (number of sparse elements). nnz must be less than or equal to (n * n-1)/2, where n is the number of rows or columns. # TODO: per batch element...
        upper (bool, optional): If True, generates strictly upper triangular indices. If False, generates strictly lower triangular indices. Defaults to True.
        indices_dtype (torch.dtype, optional): Data type for indices of sparse tensor. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values of sparse tensor. Defaults to torch.float32.
        device (torch.device, optional): Device to generate the coordinates on. Defaults to torch.device("cpu").

    Raises:
        ValueError: Raised if size has less than 2 dimensions.
        ValueError: Raised if size has more than 3 dimensions, as this implementation only supports 1 batch dimension.
        ValueError: Raised if size is not a square matrix (n, n) or batched square matrix (b, n, n).
        ValueError: Raised if nnz is greater than (n * n-1)/2, where n is the number of rows or columns.
        UserWarning: Raised when `indices_dtype` is not `torch.int64`, as this is the only indices dtype supported for sparse COO tensors. Any other index dtype will be converted to `torch.int64`.

    Returns:
        torch.Tensor: Returns a square strictly upper or lower sparse COO tensor of shape size with nnz non-zero elements.
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")

    if size[-2] != size[-1]:
        raise ValueError("size must be a square matrix (n, n) or batched square matrix (b, n, n)")

    if nnz > size[-2] * (size[-2] - 1) // 2:
        raise ValueError("nnz must be less than or equal to (n * n-1)/2, where n is the number of rows or columns")
    
    if indices_dtype != torch.int64:
        warnings.warn(f"Only indices of type torch.int64 supported for sparse COO tensors. Indices of type {indices_dtype} will be cast to torch.int64.", UserWarning)

    if len(size) == 2:
        coo_indices = _gen_indices_2d_coo_strictly_tri(size[-2], nnz, upper=upper, dtype=indices_dtype, device=device)
        values = torch.rand(nnz, dtype=values_dtype, device=device)
    else:
        sparse_dim_indices = torch.cat([_gen_indices_2d_coo_strictly_tri(size[-2], nnz, upper=upper, dtype=indices_dtype, device=device) for _ in range(size[0])], dim=-1)
        batch_dim_indices = batch_dim_indices = torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(nnz).unsqueeze(0)
        coo_indices = torch.cat([batch_dim_indices, sparse_dim_indices])
        values = torch.rand(nnz*size[0], dtype=values_dtype, device=device)        

    return torch.sparse_coo_tensor(coo_indices, values, size, device=device).coalesce()
    

def generate_random_sparse_strictly_triangular_csr_matrix(size, nnz, *, upper=True, indices_dtype=torch.int64, values_dtype=torch.float32, device=torch.device("cpu")):
    """Generates a random sparse CSR square matrix with strictly upper or lower triangular coordinates.

    Args:
        size (tuple): Tuple specifying the dimensions of the batched sparse matrix. The size can be either (num_rows, num_cols) for a unbatched matrix or (batch_size, num_rows, num_cols) for a batched matrix. The number of rows and columns must be equal.
        nnz (int): Number of non-zero values in sparse matrix (number of sparse elements). nnz must be less than or equal to (n * n-1)/2, where n is the number of rows or columns.
        upper (bool, optional): If True, generates strictly upper triangular indices. If False, generates strictly lower triangular indices. Defaults to True.
        indices_dtype (torch.dtype, optional): Data type for indices of sparse tensor. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values of sparse tensor. Defaults to torch.float32.
        device (torch.device, optional): Device to generate the coordinates on. Defaults to torch.device("cpu").

    Raises:
        ValueError: Raised if size has less than 2 dimensions.
        ValueError: Raised if size has more than 3 dimensions, as this implementation only supports 1 batch dimension.
        ValueError: Raised if size is not a square matrix (n, n) or batched square matrix (b, n, n).
        ValueError: Raised if nnz is greater than (n * n-1)/2, where n is the number of rows or columns.
        UserWarning: Raised when `indices_dtype` has a bit depth less than `torch.int32`, as this is not recommended.

    Returns:
        torch.Tensor: Returns a square strictly upper or lower sparse CSR tensor of shape size with nnz non-zero elements.
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")

    if size[-2] != size[-1]:
        raise ValueError("size must be a square matrix (n, n) or batched square matrix (b, n, n)")

    if nnz > size[-2] * (size[-2] - 1) // 2:
        raise ValueError("nnz must be less than or equal to (n * n-1)/2, where n is the number of rows or columns")
    
    if (indices_dtype != torch.int64) and (indices_dtype != torch.int32):
        warnings.warn(f"A bit depth of less than torch.int32 is not recommended for sparse CSR tensors", UserWarning)

    row_indices, col_indices = _gen_indices_2d_coo_strictly_tri(size[-2], nnz, upper=upper, dtype=indices_dtype, device=device)
    crow_indices = compress_row_indices(row_indices, size[-2])
    
    if len(size) == 2:
        values = torch.rand(nnz, dtype=values_dtype, device=device)
    else:
        crow_indices = crow_indices.repeat(size[0], 1)
        col_indices = col_indices.repeat(size[0], 1)
        values = torch.rand(nnz*size[0], dtype=values_dtype, device=device)
        
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size, device=device)
    












# # unit test and move to contraints
# def is_strictly_triangular(crow_indices, col_indices, upper=True):
#     """Check whether the given crow_indices and col_indices represent strictly upper triangular or strictly lower triangular indices.
    
#     Args:
#         crow_indices (torch.Tensor): Tensor of shape [nr+1] containing the crow_indices.
#         col_indices (torch.Tensor): Tensor of shape [nnz] containing the col_indices.
#         upper (bool, optional): Flag indicating whether to check for strictly upper triangular indices. Defaults to True.
    
#     Returns:
#         bool: True if the indices are strictly upper triangular or strictly lower triangular, False otherwise.
#     """
#     assert crow_indices.dim() == 1 and col_indices.dim() == 1, "Input tensors must be 1-dimensional."
#     assert crow_indices.size(0) == crow_indices[-1], "crow_indices must represent a valid compressed sparse row structure."
#     assert col_indices.size(0) == crow_indices[-1], "col_indices must have the same length as crow_indices[-1]."
    
#     if upper:
#         for i in range(crow_indices.size(0) - 1):
#             if (col_indices[crow_indices[i]:crow_indices[i+1]] <= i).any():
#                 return False
#         return True
#     else:
#         for i in range(crow_indices.size(0) - 1):
#             if (col_indices[crow_indices[i]:crow_indices[i+1]] >= i+1).any():
#                 return False
#         return True