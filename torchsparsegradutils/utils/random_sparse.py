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

import torch
import random
from torchsparsegradutils.utils.utils import convert_coo_to_csr_indices_values, convert_coo_to_csr

__all__ = [
    "rand_sparse",
    "rand_sparse_tri",
    "make_spd_sparse",
    "generate_random_sparse_coo_matrix",
    "generate_random_sparse_csr_matrix",
    "generate_random_sparse_strictly_triangular_coo_matrix",
    "generate_random_sparse_strictly_triangular_csr_matrix",
    "generate_random_sparse_triangular_coo_matrix",
    "generate_random_sparse_triangular_csr_matrix",
]


def rand_sparse(
    size,
    nnz,
    layout=torch.sparse_coo,
    *,
    indices_dtype=torch.int64,
    values_dtype=torch.float32,
    device=torch.device("cpu"),
    well_conditioned=False,
    min_diag_value=1.0,
):
    """Generate a random sparse matrix with the specified layout and properties.

    This is a convenience function that delegates to the appropriate matrix generator
    based on the requested layout (COO or CSR).

    Args:
        size (tuple): Tuple specifying the dimensions of the sparse matrix. Can be either
            (num_rows, num_cols) for an unbatched matrix or (batch_size, num_rows, num_cols)
            for a batched matrix.
        nnz (int): Number of non-zero values in sparse matrix (number of sparse elements),
            per batch element. Must be less than or equal to the total number of elements
            in the matrix.
        layout (torch.layout, optional): The desired sparse tensor layout. Must be either
            torch.sparse_coo or torch.sparse_csr. Defaults to torch.sparse_coo.
        indices_dtype (torch.dtype, optional): Data type for indices of sparse tensor.
            Must be torch.int64 or torch.int32. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values of sparse tensor.
            Defaults to torch.float32.
        device (torch.device, optional): Device to generate the tensor on.
            Defaults to torch.device("cpu").
        well_conditioned (bool, optional): If True and the matrix is square, ensures
            diagonal elements are sufficiently large for numerical stability.
            Defaults to False.
        min_diag_value (float, optional): Minimum value for diagonal elements when
            well_conditioned=True. Only used for square matrices. Defaults to 1.0.

    Returns:
        torch.Tensor: A sparse tensor of the specified layout with the requested properties.

    Raises:
        ValueError: If layout is not torch.sparse_coo or torch.sparse_csr.

    Examples:
        >>> # Generate a 100x100 COO matrix with 500 non-zeros
        >>> A = rand_sparse((100, 100), 500)
        >>>
        >>> # Generate a well-conditioned CSR matrix for numerical stability
        >>> A = rand_sparse((50, 50), 200, layout=torch.sparse_csr, well_conditioned=True)
    """
    if layout == torch.sparse_coo:
        return generate_random_sparse_coo_matrix(
            size,
            nnz,
            indices_dtype=indices_dtype,
            values_dtype=values_dtype,
            device=device,
            well_conditioned=well_conditioned,
            min_diag_value=min_diag_value,
        )
    elif layout == torch.sparse_csr:
        return generate_random_sparse_csr_matrix(
            size,
            nnz,
            indices_dtype=indices_dtype,
            values_dtype=values_dtype,
            device=device,
            well_conditioned=well_conditioned,
            min_diag_value=min_diag_value,
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
    well_conditioned=False,
    min_diag_value=1.0,
):
    """Generate a random sparse triangular matrix with the specified layout and properties.

    This function generates either strictly triangular (excluding diagonal) or
    non-strictly triangular (including diagonal) sparse matrices.

    Important Note for Non-Strict Triangular Matrices:
        When strict=False, this function automatically includes ALL diagonal elements
        in the generated matrix. Therefore, nnz must be >= n (matrix size) to
        accommodate the full diagonal. The remaining (nnz - n) elements will be
        randomly placed in the triangular region.

    Args:
        size (tuple): Tuple specifying the dimensions of the sparse matrix. Must be
            (n, n) for an unbatched square matrix or (batch_size, n, n) for a batched
            square matrix.
        nnz (int): Number of non-zero values in sparse matrix per batch element.
            - For strict=True: Must be <= n*(n-1)/2 (strictly triangular region)
            - For strict=False: Must be >= n and <= n*(n+1)/2 (includes diagonal)
        layout (torch.layout, optional): The desired sparse tensor layout. Must be either
            torch.sparse_coo or torch.sparse_csr. Defaults to torch.sparse_coo.
        upper (bool, optional): If True, generates upper triangular matrix. If False,
            generates lower triangular matrix. Defaults to True.
        strict (bool, optional): If True, generates strictly triangular matrix (excludes
            diagonal). If False, generates non-strict triangular matrix (includes diagonal).
            Defaults to False.
        indices_dtype (torch.dtype, optional): Data type for indices of sparse tensor.
            Must be torch.int64 or torch.int32. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values of sparse tensor.
            Defaults to torch.float32.
        device (torch.device, optional): Device to generate the tensor on.
            Defaults to torch.device("cpu").
        value_range (tuple, optional): Range (min, max) for generating random values.
            Defaults to (0, 1).
        well_conditioned (bool, optional): If True, ensures diagonal elements are
            sufficiently large for numerical stability. Only applicable when strict=False.
            Defaults to False.
        min_diag_value (float, optional): Minimum value for diagonal elements when
            well_conditioned=True. Only used when strict=False. Defaults to 1.0.

    Returns:
        torch.Tensor: A sparse triangular tensor of the specified layout.

    Raises:
        ValueError: If layout is not torch.sparse_coo or torch.sparse_csr.
        ValueError: If nnz constraints are violated based on strict parameter.

    Examples:
        >>> # Generate a 100x100 upper triangular matrix with diagonal
        >>> A = rand_sparse_tri((100, 100), 500, upper=True, strict=False)
        >>>
        >>> # Generate a strictly lower triangular matrix (no diagonal)
        >>> A = rand_sparse_tri((50, 50), 200, upper=False, strict=True)
        >>>
        >>> # Generate a well-conditioned triangular matrix for solving
        >>> A = rand_sparse_tri((100, 100), 300, well_conditioned=True, min_diag_value=1.0)
    """
    if layout == torch.sparse_coo:
        if strict:
            return generate_random_sparse_strictly_triangular_coo_matrix(
                size,
                nnz,
                upper=upper,
                indices_dtype=indices_dtype,
                values_dtype=values_dtype,
                device=device,
                value_range=value_range,
            )
        else:
            return generate_random_sparse_triangular_coo_matrix(
                size,
                nnz,
                upper=upper,
                indices_dtype=indices_dtype,
                values_dtype=values_dtype,
                device=device,
                value_range=value_range,
                well_conditioned=well_conditioned,
                min_diag_value=min_diag_value,
            )
    elif layout == torch.sparse_csr:
        if strict:
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
            return generate_random_sparse_triangular_csr_matrix(
                size,
                nnz,
                upper=upper,
                indices_dtype=indices_dtype,
                values_dtype=values_dtype,
                device=device,
                value_range=value_range,
                well_conditioned=well_conditioned,
                min_diag_value=min_diag_value,
            )
    else:
        raise ValueError("Unsupported layout type. It should be either torch.sparse_coo or torch.sparse_csr")


def _gen_indices_2d_coo(nr, nc, nnz, *, dtype=torch.int64, device=torch.device("cpu")):
    """Generates nnz random unique coordinates in COO format for a 2D matrix.

    This function generates random (row, column) coordinate pairs without replacement
    until exactly nnz unique coordinates are obtained.

    Args:
        nr (int): Number of rows in the matrix.
        nc (int): Number of columns in the matrix.
        nnz (int): Number of coordinate pairs to generate. Must be <= nr * nc.
        dtype (torch.dtype, optional): Data type for the coordinate indices.
            Defaults to torch.int64.
        device (torch.device, optional): Device to generate coordinates on.
            Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: Tensor of shape [2, nnz] containing the generated coordinates
            where the first row contains row indices and the second row contains
            column indices.

    Raises:
        AssertionError: If nnz > nr * nc (more coordinates requested than possible).

    Note:
        This function uses rejection sampling and may be slow for very dense matrices
        where nnz is close to nr * nc.
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
    size,
    nnz,
    *,
    indices_dtype=torch.int64,
    values_dtype=torch.float32,
    device=torch.device("cpu"),
    well_conditioned=False,
    min_diag_value=1.0,
):
    """Generate a random sparse COO matrix with the specified dimensions and density.

    This function creates general sparse matrices (not specifically triangular) by
    randomly placing nnz non-zero elements throughout the matrix structure.

    Args:
        size (tuple): Dimensions of the sparse matrix. Can be (num_rows, num_cols) for
            an unbatched matrix or (batch_size, num_rows, num_cols) for a batched matrix.
        nnz (int): Number of non-zero values per batch element. Must be <= total matrix
            elements (num_rows * num_cols).
        indices_dtype (torch.dtype, optional): Data type for indices. Must be torch.int64
            or torch.int32. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values. Defaults to torch.float32.
        device (torch.device, optional): Device for tensor creation. Defaults to "cpu".
        well_conditioned (bool, optional): If True and matrix is square, ensures diagonal
            elements are sufficiently large for numerical stability. Defaults to False.
        min_diag_value (float, optional): Minimum value for diagonal elements when
            well_conditioned=True. Only used for square matrices. Defaults to 1.0.

    Returns:
        torch.Tensor: Sparse COO tensor with randomly distributed non-zero elements.

    Raises:
        ValueError: If size has invalid dimensions (< 2 or > 3).
        ValueError: If nnz > total matrix elements.
        ValueError: If indices_dtype is not torch.int64 or torch.int32.

    Examples:
        >>> # Generate a 100x50 matrix with 200 random non-zeros
        >>> A = generate_random_sparse_coo_matrix((100, 50), 200)
        >>>
        >>> # Generate a well-conditioned square matrix
        >>> A = generate_random_sparse_coo_matrix((100, 100), 500, well_conditioned=True)

    Note:
        For triangular matrices, use generate_random_sparse_triangular_coo_matrix or
        generate_random_sparse_strictly_triangular_coo_matrix instead.
    """

    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")

    if nnz > size[-2] * size[-1]:
        raise ValueError("nnz must be less than or equal to nr * nc")

    if (indices_dtype != torch.int64) and (indices_dtype != torch.int32):
        raise ValueError("indices_dtype must be torch.int64 or torch.int32 for sparse COO tensors")

    if len(size) == 2:
        coo_indices = _gen_indices_2d_coo(size[-2], size[-1], nnz, dtype=indices_dtype, device=device)
        values = torch.rand(nnz, dtype=values_dtype, device=device)

        if well_conditioned and size[-2] == size[-1]:  # Only for square matrices
            # Ensure diagonal elements are sufficiently large for well-conditioning
            diagonal_mask = coo_indices[0] == coo_indices[1]
            if diagonal_mask.any():
                values[diagonal_mask] = (
                    torch.rand(diagonal_mask.sum(), dtype=values_dtype, device=device) * 0.5 + min_diag_value
                )
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

        if well_conditioned and size[-2] == size[-1]:  # Only for square matrices
            # Ensure diagonal elements are sufficiently large for well-conditioning
            diagonal_mask = coo_indices[1] == coo_indices[2]  # For batched case, check row vs col indices
            if diagonal_mask.any():
                values[diagonal_mask] = (
                    torch.rand(diagonal_mask.sum(), dtype=values_dtype, device=device) * 0.5 + min_diag_value
                )

    return torch.sparse_coo_tensor(coo_indices, values, size, device=device).coalesce()


def generate_random_sparse_csr_matrix(
    size,
    nnz,
    *,
    indices_dtype=torch.int64,
    values_dtype=torch.float32,
    device=torch.device("cpu"),
    well_conditioned=False,
    min_diag_value=1.0,
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
        values = torch.rand(nnz, dtype=values_dtype, device=device)

        if well_conditioned and size[-2] == size[-1]:  # Only for square matrices
            # Ensure diagonal elements are sufficiently large for well-conditioning
            diagonal_mask = coo_indices[0] == coo_indices[1]
            if diagonal_mask.any():
                values[diagonal_mask] = (
                    torch.rand(diagonal_mask.sum(), dtype=values_dtype, device=device) * 0.5 + min_diag_value
                )

        crow_indices, col_indices, values = convert_coo_to_csr_indices_values(coo_indices, size[-2], values=values)
    else:
        sparse_dim_indices = torch.cat(
            [_gen_indices_2d_coo(size[-2], size[-1], nnz, dtype=indices_dtype, device=device) for _ in range(size[0])],
            dim=-1,
        )
        batch_dim_indices = (
            torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(nnz).unsqueeze(0)
        )
        coo_indices = torch.cat([batch_dim_indices, sparse_dim_indices])

        values = torch.rand((size[0], nnz), dtype=values_dtype, device=device)

        if well_conditioned and size[-2] == size[-1]:  # Only for square matrices
            # Ensure diagonal elements are sufficiently large for well-conditioning
            diagonal_mask = coo_indices[1] == coo_indices[2]  # For batched case, check row vs col indices
            if diagonal_mask.any():
                # For batched case, we need to handle the values tensor shape differently
                flat_diagonal_mask = diagonal_mask.repeat(size[0])
                values_flat = values.view(-1)
                values_flat[flat_diagonal_mask] = (
                    torch.rand(flat_diagonal_mask.sum(), dtype=values_dtype, device=device) * 0.5 + min_diag_value
                )
                values = values_flat.view(size[0], nnz)

        crow_indices, col_indices, values = convert_coo_to_csr_indices_values(
            coo_indices, size[-2], values=values.view(-1)
        )

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
        ValueError: Raised if indices_dtype is not torch.int64 or torch.int32, as these are the only indices dtypes supported for sparse COO tensors.

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

    if (indices_dtype != torch.int64) and (indices_dtype != torch.int32):
        raise ValueError("indices_dtype must be torch.int64 or torch.int32 for sparse COO tensors")

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


# helper for non-strict triangular coordinates
def _gen_indices_2d_coo_nonstrict_tri(n, nnz, *, upper=True, dtype=torch.int64, device=torch.device("cpu")):
    """Generates nnz random unique COO coordinates for non-strict triangular matrices.

    This function generates coordinates for triangular matrices that INCLUDE the diagonal.
    It automatically populates ALL diagonal elements (0,0), (1,1), ..., (n-1,n-1) first,
    then fills the remaining (nnz - n) coordinates randomly in the triangular region.

    Critical Requirement:
        nnz must be >= n because this function ALWAYS includes all n diagonal elements.
        The diagonal cannot be excluded for non-strict triangular matrices.

    Args:
        n (int): Size of the square matrix (number of rows and columns).
        nnz (int): Number of coordinate pairs to generate. Must satisfy n <= nnz <= n*(n+1)/2.
        upper (bool, optional): If True, generates upper triangular coordinates (r <= c).
            If False, generates lower triangular coordinates (r >= c). Defaults to True.
        dtype (torch.dtype, optional): Data type for the coordinate indices.
            Defaults to torch.int64.
        device (torch.device, optional): Device to generate coordinates on.
            Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: Tensor of shape [2, nnz] containing the triangular coordinates
            with all diagonal elements guaranteed to be included.

    Raises:
        AssertionError: If nnz < n or nnz > n*(n+1)/2.

    Note:
        This function always includes the complete diagonal. If you need triangular
        matrices without diagonal elements, use _gen_indices_2d_coo_strictly_tri instead.

    Examples:
        >>> # Generate lower triangular coordinates for 4x4 matrix with 8 elements
        >>> coords = _gen_indices_2d_coo_nonstrict_tri(4, 8, upper=False)
        >>> # Result will include (0,0), (1,1), (2,2), (3,3) plus 4 random lower elements
    """
    assert nnz <= n * (n + 1) // 2 and nnz >= n, "nnz must be >= n and <= n*(n+1)/2 for non-strict triangular"
    coords = set((i, i) for i in range(n))  # includes ALL diagonal elements
    import random

    while len(coords) < nnz:
        r, c = random.randrange(n), random.randrange(n)
        if (r < c and upper) or (r > c and not upper) or (r == c):
            coords.add((r, c))
    return torch.stack([torch.tensor(x, dtype=dtype, device=device) for x in coords], dim=-1)


def generate_random_sparse_triangular_coo_matrix(
    size,
    nnz,
    *,
    upper=True,
    indices_dtype=torch.int64,
    values_dtype=torch.float32,
    device=torch.device("cpu"),
    value_range=(0, 1),
    well_conditioned=False,
    min_diag_value=1.0,
):
    """Generate a random sparse COO matrix with non-strict triangular structure.

    This function creates triangular matrices that INCLUDE the diagonal elements.
    All n diagonal elements are automatically included, so nnz must be >= n.

    Key Features:
        - Always includes ALL diagonal elements (cannot be excluded)
        - Remaining (nnz - n) elements are randomly placed in triangular region
        - Supports both upper and lower triangular matrices
        - Optional well-conditioning for numerical stability

    Args:
        size (tuple): Dimensions of the sparse matrix. Must be (n, n) for unbatched
            or (batch_size, n, n) for batched square matrices.
        nnz (int): Number of non-zero elements per batch. Must satisfy n <= nnz <= n*(n+1)/2
            where n is the matrix size. The constraint nnz >= n exists because this
            function always includes all diagonal elements.
        upper (bool, optional): If True, generates upper triangular matrix (r <= c).
            If False, generates lower triangular matrix (r >= c). Defaults to True.
        indices_dtype (torch.dtype, optional): Data type for indices. Must be
            torch.int64 or torch.int32. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values. Defaults to torch.float32.
        device (torch.device, optional): Device for tensor creation. Defaults to "cpu".
        value_range (tuple, optional): Range (min, max) for random values. Defaults to (0, 1).
        well_conditioned (bool, optional): If True, ensures diagonal elements are
            sufficiently large for numerical stability. Defaults to False.
        min_diag_value (float, optional): Minimum value for diagonal elements when
            well_conditioned=True. Defaults to 1.0.

    Returns:
        torch.Tensor: Sparse COO tensor with triangular structure including diagonal.

    Raises:
        ValueError: If size dimensions are invalid.
        ValueError: If nnz constraints are violated (nnz < n or nnz > n*(n+1)/2).
        ValueError: If indices_dtype is not supported.

    Examples:
        >>> # Generate 100x100 lower triangular matrix with 300 elements (includes all 100 diagonal)
        >>> A = generate_random_sparse_triangular_coo_matrix((100, 100), 300, upper=False)
        >>>
        >>> # Generate well-conditioned upper triangular matrix for linear solving
        >>> A = generate_random_sparse_triangular_coo_matrix(
        ...     (50, 50), 200, upper=True, well_conditioned=True, min_diag_value=1.0
        ... )

    Note:
        For strictly triangular matrices (excluding diagonal), use
        generate_random_sparse_strictly_triangular_coo_matrix instead.
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions")
    if size[-2] != size[-1]:
        raise ValueError("size must be a square matrix or batched square matrix")
    n = size[-2]
    if nnz > n * (n + 1) // 2 or nnz < n:
        raise ValueError("nnz must be between n and n*(n+1)/2")
    if (indices_dtype != torch.int64) and (indices_dtype != torch.int32):
        raise ValueError("indices_dtype must be torch.int64 or torch.int32 for sparse COO tensors")

    if len(size) == 2:
        coo_idx = _gen_indices_2d_coo_nonstrict_tri(n, nnz, upper=upper, dtype=indices_dtype, device=device)
        values = torch.rand(nnz, dtype=values_dtype, device=device)

        if well_conditioned:
            # Ensure diagonal elements are sufficiently large for well-conditioning
            diagonal_mask = coo_idx[0] == coo_idx[1]
            if diagonal_mask.any():
                values[diagonal_mask] = torch.rand(diagonal_mask.sum(), dtype=values_dtype, device=device) * (
                    value_range[1] - value_range[0]
                ) + max(min_diag_value, value_range[0])
    else:
        coo_idx = torch.cat(
            [
                _gen_indices_2d_coo_nonstrict_tri(n, nnz, upper=upper, dtype=indices_dtype, device=device)
                for _ in range(size[0])
            ],
            dim=-1,
        )
        batch_idx = torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(nnz).unsqueeze(0)
        coo_idx = torch.cat([batch_idx, coo_idx], dim=0)
        values = torch.rand(nnz * size[0], dtype=values_dtype, device=device)

        if well_conditioned:
            # Ensure diagonal elements are sufficiently large for well-conditioning
            diagonal_mask = coo_idx[1] == coo_idx[2]  # For batched case, check row vs col indices
            if diagonal_mask.any():
                diagonal_values = (
                    torch.rand(diagonal_mask.sum(), dtype=values_dtype, device=device) * 0.5 + min_diag_value
                )
                # Apply value_range to non-diagonal values only
                values = values * (value_range[1] - value_range[0]) + value_range[0]
                # Set well-conditioned diagonal values
                values[diagonal_mask] = diagonal_values
            else:
                values = values * (value_range[1] - value_range[0]) + value_range[0]
        else:
            values = values * (value_range[1] - value_range[0]) + value_range[0]

    return torch.sparse_coo_tensor(coo_idx, values, size, device=device).coalesce()


def generate_random_sparse_triangular_csr_matrix(
    size,
    nnz,
    *,
    upper=True,
    indices_dtype=torch.int64,
    values_dtype=torch.float32,
    device=torch.device("cpu"),
    value_range=(0, 1),
    well_conditioned=False,
    min_diag_value=1.0,
):
    """Generate a random sparse CSR matrix with non-strict triangular structure.

    This function creates triangular CSR matrices that INCLUDE the diagonal elements.
    All n diagonal elements are automatically included, so nnz must be >= n.
    The CSR format is often preferred for matrix operations and linear algebra routines.

    Key Features:
        - Always includes ALL diagonal elements (cannot be excluded)
        - Remaining (nnz - n) elements are randomly placed in triangular region
        - CSR format for efficient row-based operations
        - Optional well-conditioning for numerical stability

    Args:
        size (tuple): Dimensions of the sparse matrix. Must be (n, n) for unbatched
            or (batch_size, n, n) for batched square matrices.
        nnz (int): Number of non-zero elements per batch. Must satisfy n <= nnz <= n*(n+1)/2
            where n is the matrix size. The constraint nnz >= n exists because this
            function always includes all diagonal elements.
        upper (bool, optional): If True, generates upper triangular matrix (r <= c).
            If False, generates lower triangular matrix (r >= c). Defaults to True.
        indices_dtype (torch.dtype, optional): Data type for indices. Must be
            torch.int64 or torch.int32. Defaults to torch.int64.
        values_dtype (torch.dtype, optional): Data type for values. Defaults to torch.float32.
        device (torch.device, optional): Device for tensor creation. Defaults to "cpu".
        value_range (tuple, optional): Range (min, max) for random values. Defaults to (0, 1).
        well_conditioned (bool, optional): If True, ensures diagonal elements are
            sufficiently large for numerical stability. Defaults to False.
        min_diag_value (float, optional): Minimum value for diagonal elements when
            well_conditioned=True. Defaults to 1.0.

    Returns:
        torch.Tensor: Sparse CSR tensor with triangular structure including diagonal.

    Raises:
        ValueError: If size dimensions are invalid.
        ValueError: If nnz constraints are violated (nnz < n or nnz > n*(n+1)/2).
        ValueError: If indices_dtype is not supported.

    Examples:
        >>> # Generate 100x100 upper triangular CSR matrix with 300 elements
        >>> A = generate_random_sparse_triangular_csr_matrix((100, 100), 300, upper=True)
        >>>
        >>> # Generate well-conditioned lower triangular CSR matrix for solving
        >>> A = generate_random_sparse_triangular_csr_matrix(
        ...     (50, 50), 200, upper=False, well_conditioned=True, min_diag_value=2.0
        ... )

    Note:
        - CSR format is efficient for row-based operations and many BLAS routines
        - For strictly triangular matrices (excluding diagonal), use
          generate_random_sparse_strictly_triangular_csr_matrix instead
        - For COO format, use generate_random_sparse_triangular_coo_matrix instead
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions")
    if size[-2] != size[-1]:
        raise ValueError("size must be a square matrix or batched square matrix")
    n = size[-2]
    if nnz > n * (n + 1) // 2 or nnz < n:
        raise ValueError("nnz must be between n and n*(n+1)/2")
    if (indices_dtype != torch.int64) and (indices_dtype != torch.int32):
        raise ValueError("indices_dtype must be torch.int64 or torch.int32 for sparse CSR tensors")

    if len(size) == 2:
        coo_idx = _gen_indices_2d_coo_nonstrict_tri(n, nnz, upper=upper, dtype=indices_dtype, device=device)
        values = torch.rand(nnz, dtype=values_dtype, device=device)

        if well_conditioned:
            # Ensure diagonal elements are sufficiently large for well-conditioning
            diagonal_mask = coo_idx[0] == coo_idx[1]
            if diagonal_mask.any():
                diagonal_values = (
                    torch.rand(diagonal_mask.sum(), dtype=values_dtype, device=device) * 0.5 + min_diag_value
                )
                # Apply value_range to all values first
                values = values * (value_range[1] - value_range[0]) + value_range[0]
                # Then set well-conditioned diagonal values
                values[diagonal_mask] = diagonal_values
            else:
                values = values * (value_range[1] - value_range[0]) + value_range[0]
        else:
            values = values * (value_range[1] - value_range[0]) + value_range[0]

        crow, col, values = convert_coo_to_csr_indices_values(coo_idx, n, values=values)
    else:
        coo_idx = torch.cat(
            [
                _gen_indices_2d_coo_nonstrict_tri(n, nnz, upper=upper, dtype=indices_dtype, device=device)
                for _ in range(size[0])
            ],
            dim=-1,
        )
        batch_idx = torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(nnz).unsqueeze(0)
        coo_idx = torch.cat([batch_idx, coo_idx], dim=0)
        values = torch.rand((size[0], nnz), dtype=values_dtype, device=device)

        if well_conditioned:
            # Ensure diagonal elements are sufficiently large for well-conditioning
            diagonal_mask = coo_idx[1] == coo_idx[2]  # For batched case, check row vs col indices
            if diagonal_mask.any():
                # The diagonal_mask already has the correct shape [batch_size * nnz]
                # since coo_idx was created by concatenating batch_size matrices each with nnz elements
                diagonal_values = (
                    torch.rand(diagonal_mask.sum(), dtype=values_dtype, device=device) * 0.5 + min_diag_value
                )
                # Apply value_range to all values first
                values = values * (value_range[1] - value_range[0]) + value_range[0]
                # Then set well-conditioned diagonal values
                values_flat = values.view(-1)
                values_flat[diagonal_mask] = diagonal_values
                values = values_flat.view(size[0], nnz)
            else:
                values = values * (value_range[1] - value_range[0]) + value_range[0]
        else:
            values = values * (value_range[1] - value_range[0]) + value_range[0]

        crow, col, values = convert_coo_to_csr_indices_values(coo_idx, n, values=values.view(-1))

    return torch.sparse_csr_tensor(crow, col, values, size, device=device)


def make_spd_sparse(n, layout, value_dtype, index_dtype, device, sparsity_ratio=0.5, nz=None):
    """
    Generate a random sparse symmetric positive definite (SPD) matrix.

    This function returns unbatched sparse tensors only (2D matrices).

    Args:
        n (int): Matrix size (n x n)
        layout (torch.layout): Sparse tensor layout (torch.sparse_coo or torch.sparse_csr)
        value_dtype (torch.dtype): Data type for values
        index_dtype (torch.dtype): Data type for indices (may not be preserved due to PyTorch limitations)
        device (torch.device): Device to create tensors on
        sparsity_ratio (float, optional): Approximate fraction of upper-triangular off-diagonal elements to zero out.
            Each zeroed upper-triangular element also zeros its symmetric lower-triangular counterpart.
            Only used if nz is None. Defaults to 0.5.
        nz (int, optional): If provided, randomly zero out exactly this many symmetric pairs of off-diagonal elements.
            Each pair consists of elements (i,j) and (j,i) where i != j. Total elements zeroed = 2*nz.
            If None, use sparsity_ratio instead. Defaults to None.

    Returns:
        tuple: (A_sparse, A_dense) - sparse and dense versions of the SPD matrix

    Note:
        The matrix is constructed as M @ M.T + n*I where M is random and I is identity.
        This ensures positive definiteness. Then we randomly zero out symmetric pairs of
        off-diagonal elements to create sparsity while preserving both positive definiteness
        and symmetry.

        Symmetric zeroing: When element (i,j) is zeroed, element (j,i) is also zeroed to
        maintain matrix symmetry. This is critical for maintaining the SPD property.

        This function returns unbatched (2D) sparse tensors only. For batched operations,
        use the appropriate functions from this module.

        Warning: PyTorch may reset index dtypes during tensor operations (like coalesce),
        so the final tensor may not preserve the requested index_dtype.
    """
    # Generate random matrix and make it SPD
    M = torch.randn(n, n, dtype=value_dtype, device=device)
    A_dense = M @ M.t() + n * torch.eye(n, dtype=value_dtype, device=device)

    # Create sparsity by zeroing out random off-diagonal elements SYMMETRICALLY
    if nz is not None:
        # Use exact number of elements to zero out
        if nz > 0:
            # Get upper triangular off-diagonal mask (we'll mirror to lower triangle)
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
            upper_off_diag_indices = torch.nonzero(mask, as_tuple=False)

            # Ensure we don't try to zero out more elements than exist
            # Each upper triangular element corresponds to a pair (i,j) and (j,i)
            n_to_zero = min(nz // 2, upper_off_diag_indices.size(0))  # Divide by 2 since we zero pairs
            if n_to_zero > 0:
                selected_indices = upper_off_diag_indices[
                    torch.randperm(upper_off_diag_indices.size(0), device=device)[:n_to_zero]
                ]
                # Zero out both (i,j) and (j,i) to maintain symmetry
                A_dense[selected_indices[:, 0], selected_indices[:, 1]] = 0
                A_dense[selected_indices[:, 1], selected_indices[:, 0]] = 0
    elif sparsity_ratio > 0:
        # Use sparsity ratio to determine how many elements to zero out
        # Get upper triangular off-diagonal mask (we'll mirror to lower triangle)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)
        upper_off_diag_indices = torch.nonzero(mask, as_tuple=False)

        # Randomly select elements to zero out (each selection zeros a symmetric pair)
        n_to_zero = int(sparsity_ratio * upper_off_diag_indices.size(0))
        if n_to_zero > 0:
            selected_indices = upper_off_diag_indices[
                torch.randperm(upper_off_diag_indices.size(0), device=device)[:n_to_zero]
            ]
            # Zero out both (i,j) and (j,i) to maintain symmetry
            A_dense[selected_indices[:, 0], selected_indices[:, 1]] = 0
            A_dense[selected_indices[:, 1], selected_indices[:, 0]] = 0

    # Convert to sparse format
    idx = A_dense.nonzero(as_tuple=False).t()
    vals = A_dense[idx[0], idx[1]]

    if layout == torch.sparse_coo:
        # For COO, create tensor and coalesce
        # Note: PyTorch may reset index dtypes during coalesce operations
        A_sparse = torch.sparse_coo_tensor(idx, vals, (n, n), dtype=value_dtype, device=device).coalesce()
    elif layout == torch.sparse_csr:
        # For CSR, first create COO then convert to CSR
        A_coo = torch.sparse_coo_tensor(idx, vals, (n, n), dtype=value_dtype, device=device).coalesce()
        A_sparse = convert_coo_to_csr(A_coo)
    else:
        raise ValueError(f"Unsupported layout: {layout}. Use torch.sparse_coo or torch.sparse_csr.")

    return A_sparse, A_dense
