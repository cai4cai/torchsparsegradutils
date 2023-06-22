"""
utility functions for generating random sparse matrices

NOTE: sparse COO tensors have indices tensor of size (ndim, nse) and with indices type torch.int64
NOTE: Sparse CSR The index tensors crow_indices and col_indices should have element type either torch.int64 (default) or torch.int32.
      If you want to use MKL-enabled matrix operations, use torch.int32.
      This is as a result of the default linking of pytorch being with MKL LP64, which uses 32 bit integer indexing
NOTE: The batches of sparse CSR tensors are dependent: the number of specified elements in all batches must be the same.
      This somewhat  artificial constraint allows efficient storage of the indices of different CSR batches.

TODO: This code needs reformatting into just rand_sparse and rand_sparse_tri
TODO: Add support for non-strict triangular matrices
"""
import warnings
import torch
import random
from torchsparsegradutils.utils.utils import convert_coo_to_csr_indices_values


def rand_sparse(
    size,
    nnz,
    layout=torch.sparse_coo,
    *,
    indices_dtype=torch.int64,
    values_dtype=torch.float32,
    device=torch.device("cpu"),
):
    if layout == torch.sparse_coo:
        return generate_random_sparse_coo_matrix(
            size, nnz, indices_dtype=indices_dtype, values_dtype=values_dtype, device=device
        )
    elif layout == torch.sparse_csr:
        return generate_random_sparse_csr_matrix(
            size, nnz, indices_dtype=indices_dtype, values_dtype=values_dtype, device=device
        )
    else:
        raise ValueError("Unsupported layout type. It should be either torch.sparse_coo or torch.sparse_csr")


def rand_sparse_tri(
    size,
    nnz,
    layout=torch.sparse_coo,
    *,
    upper=True,
    strict=False,
    indices_dtype=torch.int64,
    values_dtype=torch.float32,
    device=torch.device("cpu"),
    value_range=(0, 1),
):
    if layout == torch.sparse_coo:
        return generate_random_sparse_strictly_triangular_coo_matrix(
            size,
            nnz,
            upper=upper,
            indices_dtype=indices_dtype,
            values_dtype=values_dtype,
            device=device,
            value_range=value_range,
        )
    elif layout == torch.sparse_csr:
        return generate_random_sparse_strictly_triangular_csr_matrix(
            size,
            nnz,
            upper=upper,
            indices_dtype=indices_dtype,
            values_dtype=values_dtype,
            device=device,
            value_range=value_range,
        )
    else:
        raise ValueError("Unsupported layout type. It should be either torch.sparse_coo or torch.sparse_csr")


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
    assert (
        nnz <= nr * nc
    ), "Number of elements (nnz) must be less than or equal to the total number of elements (nr * nc)."

    coordinates = set()
    while True:
        r, c = random.randrange(nr), random.randrange(nc)
        coordinates.add((r, c))
        if len(coordinates) == nnz:
            return torch.stack([torch.tensor(co, dtype=dtype, device=device) for co in coordinates], dim=-1)

    # Alternatively, could do:
    # indices = torch.randperm(nr * nc)[:nnz]
    # return torch.stack([indices // nc, indices % nc]).to(device)


def generate_random_sparse_coo_matrix(
    size, nnz, *, indices_dtype=torch.int64, values_dtype=torch.float32, device=torch.device("cpu")
):
    """Generates a random sparse COO matrix of the specified size and number of non-zero elements.

    Args:
        size (tuple): Tuple specifying the dimensions of the sparse matrix. The size can be either (num_rows, num_cols) for an unbatched matrix or (batch_size, num_rows, num_cols) for a batched matrix.
        nnz (int): Number of non-zero values in sparse matrix (number of sparse elements), per batch element. This must be less than or equal to the total number of elements in the matrix of each batch element.
        indices_dtype (torch.dtype, optional): Data type for indices of sparse tensor. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values of sparse tensor. Defaults to torch.float32.
        device (torch.device, optional): Device to generate the tensor on. Defaults to torch.device("cpu").

    Raises:
        ValueError: Raised if size has less than 2 dimensions.
        ValueError: Raised if size has more than 3 dimensions, as this implementation only supports 1 batch dimension.
        ValueError: Raised if nnz is greater than the total number of elements (size[-2] * size[-3]).
        ValueError: Raised if indices_dtype is not torch.int64 for sparse COO tensors.

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
        raise ValueError("indices_dtype must be torch.int64 for sparse COO tensors")

    if len(size) == 2:
        coo_indices = _gen_indices_2d_coo(size[-2], size[-1], nnz, dtype=indices_dtype, device=device)
        values = torch.rand(nnz, dtype=values_dtype, device=device)
    else:
        sparse_dim_indices = torch.cat(
            [_gen_indices_2d_coo(size[-2], size[-1], nnz, dtype=indices_dtype, device=device) for _ in range(size[0])],
            dim=-1,
        )
        batch_dim_indices = (
            torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(nnz).unsqueeze(0)
        )
        coo_indices = torch.cat([batch_dim_indices, sparse_dim_indices])
        values = torch.rand(nnz * size[0], dtype=values_dtype, device=device)

    return torch.sparse_coo_tensor(coo_indices, values, size, device=device).coalesce()


def generate_random_sparse_csr_matrix(
    size, nnz, *, indices_dtype=torch.int64, values_dtype=torch.float32, device=torch.device("cpu")
):
    """Generates a random sparse CSR matrix of the specified size and number of non-zero elements.

    Args:
        size (tuple): Tuple specifying the dimensions of the sparse matrix. The size can be either (num_rows, num_cols) for an unbatched matrix or (batch_size, num_rows, num_cols) for a batched matrix.
        nnz (int): Number of non-zero values in sparse matrix (number of sparse elements), per batch element. This must be less than or equal to the total number of elements in the matrix.
        indices_dtype (torch.dtype, optional): Data type for indices of sparse tensor. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values of sparse tensor. Defaults to torch.float32.
        device (torch.device, optional): Device to generate the tensor on. Defaults to torch.device("cpu").

    Raises:
        ValueError: Raised if size has less than 2 dimensions.
        ValueError: Raised if size has more than 3 dimensions, as this implementation only supports 1 batch dimension.
        ValueError: Raised if nnz is greater than the total number of elements (size[-2] * size[-3]).
        ValueError: Raised if indices_dtype is not torch.int64 or torch.int32, as these are the only indices dtypes supported for sparse CSR tensors.

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
        raise ValueError("indices_dtype must be torch.int64 or torch.int32 for sparse CSR tensors")

    if len(size) == 2:
        coo_indices = _gen_indices_2d_coo(size[-2], size[-1], nnz, dtype=indices_dtype, device=device)
        crow_indices, col_indices, _ = convert_coo_to_csr_indices_values(coo_indices, size[-2], values=None)
        values = torch.rand(nnz, dtype=values_dtype, device=device)
    else:
        sparse_dim_indices = torch.cat(
            [_gen_indices_2d_coo(size[-2], size[-1], nnz, dtype=indices_dtype, device=device) for _ in range(size[0])],
            dim=-1,
        )
        batch_dim_indices = (
            torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(nnz).unsqueeze(0)
        )
        coo_indices = torch.cat([batch_dim_indices, sparse_dim_indices])

        crow_indices, col_indices, _ = convert_coo_to_csr_indices_values(coo_indices, size[-2], values=None)
        values = torch.rand((size[0], nnz), dtype=values_dtype, device=device)

    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size, device=device)


# Square strictly Triangular:


def _gen_indices_2d_coo_strictly_tri(n, nnz, *, upper=True, dtype=torch.int64, device=torch.device("cpu")):
    """Generates nnz random unique COO coordinates for a square matrix with either strictly lower triangular or strictly upper triangular coordinates.

    Args:
        n (int): Size of the square matrix (number of rows and columns).
        nnz (int): Number of elements the generated COO indices represent.
        upper (bool, optional): Flag indicating whether to generate strictly upper triangular coordinates, or strictly lower triangular coordinates. Defaults to True.
        dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.int64.
        device (torch.device, optional): Device to generate coordinates on. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: Tensor of shape [2, nnz] containing the generated COO indices.
    """
    assert (
        nnz <= n * (n - 1) // 2
    ), "Number of elements (nnz) must be less than or equal to the total number of elements (n * (n - 1) // 2)."

    coordinates = set()
    while True:
        r, c = random.randrange(n), random.randrange(n)
        if (r < c and upper) or (r > c and not upper):
            coordinates.add((r, c))
        if len(coordinates) == nnz:
            return torch.stack([torch.tensor(co, dtype=dtype, device=device) for co in coordinates], dim=-1).to(device)


def generate_random_sparse_strictly_triangular_coo_matrix(
    size,
    nnz,
    *,
    upper=True,
    indices_dtype=torch.int64,
    values_dtype=torch.float32,
    device=torch.device("cpu"),
    value_range=(0, 1),
):
    """Generates a random sparse COO square matrix with strictly upper or lower triangular coordinates.

    Args:
        size (tuple): Tuple specifying the dimensions of the sparse matrix. The size can be either (num_rows, num_cols) for an unbatched matrix or (batch_size, num_rows, num_cols) for a batched matrix. The number of rows and columns must be equal.
        nnz (int): Number of non-zero values in sparse matrix (number of sparse elements), per batch element. nnz must be less than or equal to (n * n-1)/2, where n is the number of rows or columns.
        upper (bool, optional): If True, generates strictly upper triangular indices. If False, generates strictly lower triangular indices. Defaults to True.
        indices_dtype (torch.dtype, optional): Data type for indices of sparse tensor. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values of sparse tensor. Defaults to torch.float32.
        device (torch.device, optional): Device to generate the coordinates on. Defaults to torch.device("cpu").
        value_range (tuple, optional): Tuple specifying the range of values to generate for the values of the sparse tensor. Defaults to [0, 1).

    Raises:
        ValueError: Raised if size has less than 2 dimensions.
        ValueError: Raised if size has more than 3 dimensions, as this implementation only supports 1 batch dimension.
        ValueError: Raised if size is not a square matrix (n, n) or batched square matrix (b, n, n).
        ValueError: Raised if nnz is greater than (n * n-1)/2, where n is the number of rows or columns.
        ValueError: Raised if indices_dtype is not torch.int64 for sparse COO tensors.

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
        raise ValueError("indices_dtype must be torch.int64 for sparse COO tensors")

    if len(size) == 2:
        coo_indices = _gen_indices_2d_coo_strictly_tri(size[-2], nnz, upper=upper, dtype=indices_dtype, device=device)
        values = torch.rand(nnz, dtype=values_dtype, device=device)
    else:
        sparse_dim_indices = torch.cat(
            [
                _gen_indices_2d_coo_strictly_tri(size[-2], nnz, upper=upper, dtype=indices_dtype, device=device)
                for _ in range(size[0])
            ],
            dim=-1,
        )
        batch_dim_indices = batch_dim_indices = (
            torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(nnz).unsqueeze(0)
        )
        coo_indices = torch.cat([batch_dim_indices, sparse_dim_indices])
        values = torch.rand(nnz * size[0], dtype=values_dtype, device=device)

    values = values * (value_range[1] - value_range[0]) + value_range[0]
    return torch.sparse_coo_tensor(coo_indices, values, size, device=device).coalesce()


def generate_random_sparse_strictly_triangular_csr_matrix(
    size,
    nnz,
    *,
    upper=True,
    indices_dtype=torch.int64,
    values_dtype=torch.float32,
    device=torch.device("cpu"),
    value_range=(0, 1),
):
    """Generates a random sparse CSR square matrix with strictly upper or lower triangular coordinates.

    Args:
        size (tuple): Tuple specifying the dimensions of the sparse matrix. The size can be either (num_rows, num_cols) for an unbatched matrix or (batch_size, num_rows, num_cols) for a batched matrix. The number of rows and columns must be equal.
        nnz (int): Number of non-zero values in sparse matrix (number of sparse elements), per batch element. nnz must be less than or equal to (n * n-1)/2, where n is the number of rows or columns.
        upper (bool, optional): If True, generates strictly upper triangular indices. If False, generates strictly lower triangular indices. Defaults to True.
        indices_dtype (torch.dtype, optional): Data type for indices of sparse tensor. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values of sparse tensor. Defaults to torch.float32.
        device (torch.device, optional): Device to generate the coordinates on. Defaults to torch.device("cpu").
        value_range (tuple, optional): Tuple specifying the range of values to generate for the values of the sparse tensor. Defaults to [0, 1).

    Raises:
        ValueError: Raised if size has less than 2 dimensions.
        ValueError: Raised if size has more than 3 dimensions, as this implementation only supports 1 batch dimension.
        ValueError: Raised if size is not a square matrix (n, n) or batched square matrix (b, n, n).
        ValueError: Raised if nnz is greater than (n * n-1)/2, where n is the number of rows or columns.
        ValueError: Raised if indices_dtype is not torch.int64 or torch.int32, as these are the only indices dtypes supported for sparse CSR tensors.

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
        raise ValueError("indices_dtype must be torch.int64 or torch.int32 for sparse CSR tensors")

    if len(size) == 2:
        coo_indices = _gen_indices_2d_coo_strictly_tri(size[-2], nnz, upper=upper, dtype=indices_dtype, device=device)
        crow_indices, col_indices, _ = convert_coo_to_csr_indices_values(coo_indices, size[-2], values=None)
        values = torch.rand(nnz, dtype=values_dtype, device=device)
    else:
        sparse_dim_indices = torch.cat(
            [
                _gen_indices_2d_coo_strictly_tri(size[-2], nnz, upper=upper, dtype=indices_dtype, device=device)
                for _ in range(size[0])
            ],
            dim=-1,
        )
        batch_dim_indices = batch_dim_indices = (
            torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(nnz).unsqueeze(0)
        )
        coo_indices = torch.cat([batch_dim_indices, sparse_dim_indices])
        crow_indices, col_indices, _ = convert_coo_to_csr_indices_values(coo_indices, size[-2], values=None)
        values = torch.rand((size[0], nnz), dtype=values_dtype, device=device)

    values = values * (value_range[1] - value_range[0]) + value_range[0]
    return torch.sparse_csr_tensor(crow_indices, col_indices, values, size, device=device)
