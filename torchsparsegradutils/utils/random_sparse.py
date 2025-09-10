r"""
Utility functions for generating random sparse matrices.

Notes
-----
* Sparse COO tensors have indices tensor of size ``(ndim, nse)`` with index dtype ``torch.int64``.
* Sparse CSR tensors store structure via ``crow_indices`` and ``col_indices`` whose dtype may be
    ``torch.int64`` (default) or ``torch.int32``. If you want MKL-enabled matrix operations,
    prefer ``torch.int32`` (PyTorch is typically linked with MKL LP64 which uses 32-bit integer indexing).
* Batched sparse CSR tensors currently require each batch to have the **same number of specified elements**;
    this constraint enables efficient storage of batched CSR indices.
"""

import random
from typing import Optional, Tuple, Union

import torch

from torchsparsegradutils.utils.utils import convert_coo_to_csr, convert_coo_to_csr_indices_values

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
    size: Union[Tuple[int, int], Tuple[int, int, int]],
    nnz: int,
    layout: torch.layout = torch.sparse_coo,
    *,
    indices_dtype: torch.dtype = torch.int64,
    values_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    well_conditioned: bool = False,
    min_diag_value: float = 1.0,
) -> torch.Tensor:
    r"""
    Generate a random sparse matrix.

    A convenience wrapper around
    :func:`generate_random_sparse_coo_matrix` and
    :func:`generate_random_sparse_csr_matrix`, dispatching based on
    the requested ``layout``.

    Parameters
    ----------
    size : tuple of int
        Shape of the matrix, either ``(n_r, n_c)`` or ``(b, n_r, n_c)``.
    nnz : int
        Number of nonzeros per batch item.
    layout : torch.layout, default=torch.sparse_coo
        Sparse format. Must be ``torch.sparse_coo`` or ``torch.sparse_csr``.
    indices_dtype : torch.dtype, default=torch.int64
        Index dtype (``torch.int64`` or ``torch.int32``).
    values_dtype : torch.dtype, default=torch.float32
        Data type of the nonzero values.
    device : torch.device, default=torch.device("cpu")
        Device for tensor allocation.
    well_conditioned : bool, default=False
        If True and square, diagonal values are boosted for stability.
        See :func:`generate_random_sparse_coo_matrix`.
    min_diag_value : float, default=1.0
        Minimum diagonal value when ``well_conditioned=True``.

    Returns
    -------
    torch.Tensor
        A sparse COO or CSR tensor.

    Raises
    ------
    ValueError
        If ``layout`` is not supported.

    See Also
    --------
    generate_random_sparse_coo_matrix : Generate a random sparse COO matrix.
    generate_random_sparse_csr_matrix : Generate a random sparse CSR matrix.

    Examples
    --------
    >>> A = rand_sparse((100, 100), 500)
    >>> A.layout
    torch.sparse_coo
    >>> B = rand_sparse((50, 50), 200, layout=torch.sparse_csr, well_conditioned=True)
    >>> B.layout
    torch.sparse_csr
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
    size: Union[Tuple[int, int], Tuple[int, int, int]],
    nnz: int,
    layout: torch.layout = torch.sparse_coo,
    *,
    upper: bool = True,
    strict: bool = False,
    indices_dtype: torch.dtype = torch.int64,
    values_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    value_range: Tuple[float, float] = (0, 1),
    well_conditioned: bool = False,
    min_diag_value: float = 1.0,
) -> torch.Tensor:
    r"""
    Generate a random sparse triangular matrix.

    A convenience wrapper around the triangular sparse matrix generators:
    - :func:`generate_random_sparse_triangular_coo_matrix`
    - :func:`generate_random_sparse_triangular_csr_matrix`
    - :func:`generate_random_sparse_strictly_triangular_coo_matrix`
    - :func:`generate_random_sparse_strictly_triangular_csr_matrix`

    Parameters
    ----------
    size : tuple of int
        Shape of the square matrix, ``(n, n)`` or ``(b, n, n)``.
    nnz : int
        Number of nonzeros per batch item.
        - If ``strict=True``: ``nnz <= n*(n-1)/2``.
        - If ``strict=False``: ``n <= nnz <= n*(n+1)/2`` (includes diagonal).
    layout : torch.layout, default=torch.sparse_coo
        Sparse format. Must be ``torch.sparse_coo`` or ``torch.sparse_csr``.
    upper : bool, default=True
        If True, generate upper-triangular. If False, lower-triangular.
    strict : bool, default=False
        If True, exclude diagonal. If False, include diagonal.
    indices_dtype : torch.dtype, default=torch.int64
        Index dtype.
    values_dtype : torch.dtype, default=torch.float32
        Data type of nonzero values.
    device : torch.device, default=torch.device("cpu")
        Device for tensor allocation.
    value_range : tuple of float, default=(0, 1)
        Range for random values.
    well_conditioned : bool, default=False
        If True, diagonal values are boosted. Only used when ``strict=False``.
    min_diag_value : float, default=1.0
        Minimum diagonal value when ``well_conditioned=True``.

    Returns
    -------
    torch.Tensor
        A sparse triangular COO or CSR tensor.

    Raises
    ------
    ValueError
        If ``layout`` is not supported.
    ValueError
        If ``nnz`` does not satisfy the constraints for strict/non-strict.

    Examples
    --------
    >>> A = rand_sparse_tri((100, 100), 500)
    >>> A.layout
    torch.sparse_coo

    See Also
    --------
    generate_random_sparse_triangular_coo_matrix : Non-strict triangular COO generator.
    generate_random_sparse_triangular_csr_matrix : Non-strict triangular CSR generator.
    generate_random_sparse_strictly_triangular_coo_matrix : Strict triangular COO generator.
    generate_random_sparse_strictly_triangular_csr_matrix : Strict triangular CSR generator.
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


def _gen_indices_2d_coo(
    nr: int,
    nc: int,
    nnz: int,
    *,
    dtype: torch.dtype = torch.int64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    r"""
    Generate random COO indices for a 2D matrix.

    Produces ``nnz`` unique coordinate pairs sampled without replacement
    from an ``nr × nc`` matrix. Coordinates are returned as a tensor of
    shape ``(2, nnz)`` where the first row contains row indices and the
    second row contains column indices.

    Parameters
    ----------
    nr : int
        Number of rows in the matrix.
    nc : int
        Number of columns in the matrix.
    nnz : int
        Number of nonzero coordinates to generate. Must satisfy
        ``0 <= nnz <= nr * nc``.
    dtype : torch.dtype, default=torch.int64
        Data type of the returned indices.
    device : torch.device, default=torch.device("cpu")
        Device on which to allocate the tensor.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2, nnz)`` containing the random coordinates.

    Raises
    ------
    AssertionError
        If ``nnz`` exceeds ``nr * nc``.

    Notes
    -----
    - Sampling is performed with rejection (by drawing until unique). For
      very dense cases where ``nnz`` is close to ``nr * nc``, this can be slow.

    See Also
    --------
    _gen_indices_2d_coo_strictly_tri : Generate strictly triangular COO indices.
    _gen_indices_2d_coo_nonstrict_tri : Generate non-strict triangular COO indices.

    Examples
    --------
    >>> coords = _gen_indices_2d_coo(4, 5, 6)
    >>> coords.shape
    torch.Size([2, 6])
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
    size: Union[Tuple[int, int], Tuple[int, int, int]],
    nnz: int,
    *,
    indices_dtype: torch.dtype = torch.int64,
    values_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    well_conditioned: bool = False,
    min_diag_value: float = 1.0,
) -> torch.Tensor:
    """
    Generate a random sparse COO matrix.

    Creates an unbatched ``(n_r, n_c)`` or batched ``(b, n_r, n_c)`` COO tensor
    with exactly ``nnz`` nonzeros per batch item, sampled uniformly at random
    across all entries. Optionally boosts diagonal entries for square matrices
    when ``well_conditioned=True``.

    Parameters
    ----------
    size : tuple of int
        Shape of the output matrix. Must be ``(n_r, n_c)`` or ``(b, n_r, n_c)``.
    nnz : int
        Number of nonzero entries **per batch item**. Must satisfy
        ``0 <= nnz <= n_r * n_c``.
    indices_dtype : torch.dtype, default=torch.int64
        Index dtype for the COO indices (``torch.int64`` or ``torch.int32``).
    values_dtype : torch.dtype, default=torch.float32
        Data type of the nonzero values.
    device : torch.device, default=torch.device("cpu")
        Device to allocate the tensors on.
    well_conditioned : bool, default=False
        If True and the matrix is square, diagonal entries (if present among the
        sampled nonzeros) are boosted to be at least ``min_diag_value``.
    min_diag_value : float, default=1.0
        Minimum diagonal value when ``well_conditioned=True``.

    Returns
    -------
    torch.Tensor
        A sparse COO tensor of shape ``size`` with ``nnz`` nonzeros per batch
        (or exactly ``nnz`` for the unbatched case).

    Raises
    ------
    ValueError
        If ``size`` has fewer than 2 dims, more than 3 dims, or ``nnz`` is out of range.
    ValueError
        If ``indices_dtype`` is not ``torch.int64`` or ``torch.int32``.

    Notes
    -----
    - Nonzeros are placed uniformly at random across all entries.

    See Also
    --------
    rand_sparse : Convenience dispatcher that selects COO/CSR generator based on ``layout``.
    generate_random_sparse_triangular_coo_matrix : Generate triangular COO matrices.
    generate_random_sparse_strictly_triangular_coo_matrix : Generate strictly triangular COO matrices.

    Examples
    --------
    Unbatched random COO (200 nonzeros):

    >>> A = generate_random_sparse_coo_matrix((100, 50), 200)

    Batched random COO (b=3, each with 500 nonzeros):

    >>> A = generate_random_sparse_coo_matrix((3, 100, 100), 500)

    Well-conditioned square matrix:

    >>> A = generate_random_sparse_coo_matrix((100, 100), 500, well_conditioned=True, min_diag_value=2.0)
    """

    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")

    n_r, n_c = size[-2], size[-1]

    # nnz bounds
    if nnz < 0:
        raise ValueError("nnz must be non-negative")
    if nnz > n_r * n_c:
        raise ValueError("nnz must be less than or equal to n_r * n_c")

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
    size: Union[Tuple[int, int], Tuple[int, int, int]],
    nnz: int,
    *,
    indices_dtype: torch.dtype = torch.int64,
    values_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    well_conditioned: bool = False,
    min_diag_value: float = 1.0,
) -> torch.Tensor:
    """
    Generate a random sparse CSR matrix.

    Creates an unbatched ``(n_r, n_c)`` or batched ``(b, n_r, n_c)`` CSR tensor
    with exactly ``nnz`` nonzeros per batch item, sampled uniformly at random.
    Optionally ensures larger diagonal entries for square matrices when
    ``well_conditioned=True``.

    Parameters
    ----------
    size : tuple of int
        Shape of the output matrix. Must be ``(n_r, n_c)`` or ``(b, n_r, n_c)``.
    nnz : int
        Number of nonzero entries **per batch item**. Must satisfy
        ``0 <= nnz <= n_r * n_c``.
    indices_dtype : torch.dtype, default=torch.int64
        Index dtype for the CSR structure (``torch.int64`` or ``torch.int32``).
    values_dtype : torch.dtype, default=torch.float32
        Data type of the nonzero values.
    device : torch.device, default=torch.device("cpu")
        Device to allocate the tensors on.
    well_conditioned : bool, default=False
        If True and the matrix is square, diagonal entries (if present among the
        sampled nonzeros) are boosted to be at least ``min_diag_value``.
    min_diag_value : float, default=1.0
        Minimum diagonal value when ``well_conditioned=True``.

    Returns
    -------
    torch.Tensor
        A sparse CSR tensor of shape ``size`` with ``nnz`` nonzeros per batch
        (or exactly ``nnz`` for the unbatched case).

    Raises
    ------
    ValueError
        If ``size`` has fewer than 2 dims, more than 3 dims, or ``nnz`` is out of range.
    ValueError
        If ``indices_dtype`` is not ``torch.int64`` or ``torch.int32``.

    Notes
    -----
    - Nonzeros are placed uniformly at random across all entries.

    See Also
    --------
    rand_sparse : Convenience dispatcher that selects COO/CSR generator based on ``layout``.
    generate_random_sparse_triangular_csr_matrix : Generate triangular CSR matrices.
    generate_random_sparse_strictly_triangular_csr_matrix : Generate strictly triangular CSR matrices.

    Examples
    --------
    Unbatched random CSR (80 nonzeros):

    >>> A = generate_random_sparse_csr_matrix((20, 30), 80)

    Batched random CSR (b=4, each with 120 nonzeros):

    >>> A = generate_random_sparse_csr_matrix((4, 50, 40), 120)

    Well-conditioned square matrix:

    >>> A = generate_random_sparse_csr_matrix((100, 100), 500, well_conditioned=True, min_diag_value=2.0)
    """
    if len(size) < 2:
        raise ValueError("size must have at least 2 dimensions")
    elif len(size) > 3:
        raise ValueError("size must have at most 3 dimensions, as this implementation only supports 1 batch dimension")

    n_r, n_c = size[-2], size[-1]

    # nnz bounds
    if nnz < 0:
        raise ValueError("nnz must be non-negative")
    if nnz > n_r * n_c:
        raise ValueError("nnz must be less than or equal to n_r * n_c")

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


def _gen_indices_2d_coo_strictly_tri(
    n: int,
    nnz: int,
    *,
    upper: bool = True,
    dtype: torch.dtype = torch.int64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate random COO indices for a strictly triangular matrix.

    Produces ``nnz`` unique coordinate pairs for a strictly upper- or
    lower-triangular part of an ``n × n`` matrix (i.e. excludes the diagonal).
    Coordinates are returned in COO format as a tensor of shape ``(2, nnz)``.

    Parameters
    ----------
    n : int
        Size of the square matrix (number of rows/columns).
    nnz : int
        Number of nonzero coordinates to generate. Must satisfy
        ``0 <= nnz <= n*(n-1)//2``.
    upper : bool, default=True
        If True, generate strictly **upper**-triangular coordinates (row < col).
        If False, generate strictly **lower**-triangular coordinates (row > col).
    dtype : torch.dtype, default=torch.int64
        Data type for the returned indices.
    device : torch.device, default=torch.device("cpu")
        Device on which to allocate the tensor.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2, nnz)`` containing the COO indices for the strictly
        triangular region.

    Raises
    ------
    AssertionError
        If ``nnz`` exceeds the maximum possible strictly triangular elements
        ``n*(n-1)//2``.

    Notes
    -----
    - Diagonal elements are **never included**.

    See Also
    --------
    _gen_indices_2d_coo_nonstrict_tri : Generate non-strict triangular COO indices.

    Examples
    --------
    Generate 6 strictly lower-triangular coordinates for a 4×4 matrix:

    >>> coords = _gen_indices_2d_coo_strictly_tri(4, 6, upper=False)
    >>> coords.shape
    torch.Size([2, 6])
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
    size: Union[Tuple[int, int], Tuple[int, int, int]],
    nnz: int,
    *,
    upper: bool = True,
    indices_dtype: torch.dtype = torch.int64,
    values_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Generate a random strictly triangular sparse COO matrix.

    Constructs a **strictly** upper- or lower-triangular sparse matrix in COO
    format. No diagonal entries are included. Supports unbatched ``(n, n)`` or
    batched ``(b, n, n)`` shapes. The number of nonzeros ``nnz`` is **per batch
    item** (for batched inputs).

    Parameters
    ----------
    size : tuple of int
        Shape of the output matrix. Must be ``(n, n)`` for unbatched or
        ``(batch_size, n, n)`` for batched matrices. The matrix must be square.
    nnz : int
        Number of nonzero entries **per batch**. Must satisfy
        ``0 <= nnz <= n*(n-1)/2`` for strictly triangular structure.
    upper : bool, default=True
        If True, generate strictly **upper**-triangular coordinates (row < col);
        otherwise generate strictly **lower**-triangular coordinates (row > col).
    indices_dtype : torch.dtype, default=torch.int64
        Index dtype for the COO indices (``torch.int64`` or ``torch.int32``).
    values_dtype : torch.dtype, default=torch.float32
        Data type of the nonzero values.
    device : torch.device, default=torch.device("cpu")
        Device to allocate the tensors on.
    value_range : tuple of float, default=(0.0, 1.0)
        Range ``(min, max)`` for uniformly sampled nonzero values.

    Returns
    -------
    torch.Tensor
        A strictly triangular sparse COO tensor of shape ``size`` with ``nnz``
        nonzeros (per batch item if batched).

    Raises
    ------
    ValueError
        If ``size`` has fewer than 2 dims, more than 3 dims, or is not square.
    ValueError
        If ``nnz`` is negative or exceeds ``n*(n-1)/2``.
    ValueError
        If ``indices_dtype`` is not ``torch.int64`` or ``torch.int32``.

    Notes
    -----
    - **Strictly** triangular means the diagonal is excluded by construction.

    See Also
    --------
    rand_sparse_tri : Convenience dispatcher for triangular matrices (strict or non-strict; COO/CSR).
    generate_random_sparse_triangular_coo_matrix : Generate non-strict triangular COO matrices.

    Examples
    --------
    Unbatched strictly upper-triangular COO (n=100, 300 nonzeros):

    >>> A = generate_random_sparse_strictly_triangular_coo_matrix((100, 100), 300, upper=True)

    Batched strictly lower-triangular COO (batch=2, n=50, 200 nonzeros per batch):

    >>> A = generate_random_sparse_strictly_triangular_coo_matrix((2, 50, 50), 200, upper=False)
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
        batch_dim_indices = (
            torch.arange(size[0], dtype=indices_dtype, device=device).repeat_interleave(nnz).unsqueeze(0)
        )
        coo_indices = torch.cat([batch_dim_indices, sparse_dim_indices])
        values = torch.rand(nnz * size[0], dtype=values_dtype, device=device)

    values = values * (value_range[1] - value_range[0]) + value_range[0]
    return torch.sparse_coo_tensor(coo_indices, values, size, device=device).coalesce()


def generate_random_sparse_strictly_triangular_csr_matrix(
    size: Union[Tuple[int, int], Tuple[int, int, int]],
    nnz: int,
    *,
    upper: bool = True,
    indices_dtype: torch.dtype = torch.int64,
    values_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    value_range: Tuple[float, float] = (0.0, 1.0),
) -> torch.Tensor:
    """
    Generate a random strictly triangular sparse CSR matrix.

    Constructs a **strictly** upper- or lower-triangular sparse matrix in CSR
    format. No diagonal entries are included. Supports unbatched ``(n, n)`` or
    batched ``(b, n, n)`` shapes. The number of nonzeros ``nnz`` is per batch
    item (for batched inputs).

    Parameters
    ----------
    size : tuple of int
        Shape of the output matrix. Must be ``(n, n)`` for unbatched or
        ``(batch_size, n, n)`` for batched matrices. The matrix must be square.
    nnz : int
        Number of nonzero entries **per batch**. Must satisfy
        ``0 <= nnz <= n*(n-1)/2`` for strictly triangular structure.
    upper : bool, default=True
        If True, generate strictly **upper**-triangular coordinates (row < col);
        otherwise generate strictly **lower**-triangular coordinates (row > col).
    indices_dtype : torch.dtype, default=torch.int64
        Index dtype for the CSR structure (``torch.int64`` or ``torch.int32``).
    values_dtype : torch.dtype, default=torch.float32
        Data type of the nonzero values.
    device : torch.device, default=torch.device("cpu")
        Device to allocate the tensors on.
    value_range : tuple of float, default=(0.0, 1.0)
        Range ``(min, max)`` for uniformly sampled nonzero values.

    Returns
    -------
    torch.Tensor
        A strictly triangular sparse CSR tensor of shape ``size`` with ``nnz``
        nonzeros (per batch item if batched).

    Raises
    ------
    ValueError
        If ``size`` has fewer than 2 dims, more than 3 dims, or is not square.
    ValueError
        If ``nnz`` exceeds ``n*(n-1)/2``.
    ValueError
        If ``indices_dtype`` is not ``torch.int64`` or ``torch.int32``.

    Notes
    -----
    - **Strictly** triangular means the diagonal is excluded by construction.

    See Also
    --------
    rand_sparse_tri : Convenience dispatcher for triangular matrices (strict or non-strict; COO/CSR).
    generate_random_sparse_triangular_csr_matrix : Generate non-strict triangular CSR matrices.

    Examples
    --------
    Unbatched strictly upper-triangular CSR (n=100, 300 nonzeros):

    >>> A = generate_random_sparse_strictly_triangular_csr_matrix((100, 100), 300, upper=True)

    Batched strictly lower-triangular CSR (batch=2, n=50, 200 nonzeros per batch):

    >>> A = generate_random_sparse_strictly_triangular_csr_matrix((2, 50, 50), 200, upper=False)
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
def _gen_indices_2d_coo_nonstrict_tri(
    n: int,
    nnz: int,
    *,
    upper: bool = True,
    dtype: torch.dtype = torch.int64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate random COO indices for non-strict triangular matrices (with diagonal).

    This helper generates ``nnz`` random unique coordinates for a triangular
    ``n × n`` matrix that **includes all diagonal elements**. The diagonal
    entries ``(0, 0), (1, 1), ..., (n-1, n-1)`` are always present, and the
    remaining ``nnz - n`` coordinates are sampled randomly within the triangular
    region.

    Parameters
    ----------
    n : int
        Size of the square matrix (number of rows/columns).
    nnz : int
        Number of coordinate pairs to generate. Must satisfy
        ``n <= nnz <= n*(n+1)/2``.
    upper : bool, default=True
        If True, generates upper triangular coordinates (row <= col).
        If False, generates lower triangular coordinates (row >= col).
    dtype : torch.dtype, default=torch.int64
        Data type of the returned indices tensor.
    device : torch.device, default=torch.device("cpu")
        Device on which to allocate the tensor.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(2, nnz)`` containing the triangular coordinates.
        All diagonal elements are guaranteed to be included.

    Raises
    ------
    AssertionError
        If ``nnz < n`` or ``nnz > n*(n+1)/2``.

    Notes
    -----
    - Always includes the **entire diagonal**.

    See Also
    --------
    _gen_indices_2d_coo_strictly_tri : Generate strictly triangular COO indices.

    Examples
    --------
    Generate lower triangular coordinates for a 4×4 matrix with 8 nonzeros:

    >>> coords = _gen_indices_2d_coo_nonstrict_tri(4, 8, upper=False)
    >>> coords.shape
    torch.Size([2, 8])
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
    size: Tuple[int, int] | Tuple[int, int, int],
    nnz: int,
    *,
    upper: bool = True,
    indices_dtype: torch.dtype = torch.int64,
    values_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    value_range: Tuple[float, float] = (0.0, 1.0),
    well_conditioned: bool = False,
    min_diag_value: float = 1.0,
) -> torch.Tensor:
    """
    Generate a random sparse COO matrix with non-strict triangular structure.

    Constructs a triangular sparse matrix in COO format that always includes all
    diagonal elements. Remaining entries are placed randomly in the upper or
    lower triangular region depending on ``upper``.

    Parameters
    ----------
    size : tuple of int
        Shape of the matrix. Must be ``(n, n)`` for unbatched or
        ``(batch_size, n, n)`` for batched matrices. The matrix must be square.
    nnz : int
        Number of nonzero elements per batch. Must satisfy ``n <= nnz <= n*(n+1)/2``.
        All diagonal elements are included, so ``nnz`` must be at least ``n``.
    upper : bool, default=True
        If True, generates an upper triangular matrix (row <= col).
        If False, generates a lower triangular matrix (row >= col).
    indices_dtype : torch.dtype, default=torch.int64
        Data type for sparse indices (``torch.int64`` or ``torch.int32``).
    values_dtype : torch.dtype, default=torch.float32
        Data type for nonzero values.
    device : torch.device, default=torch.device("cpu")
        Device on which to allocate tensors.
    value_range : tuple of float, default=(0.0, 1.0)
        Range ``(min, max)`` for random nonzero values.
    well_conditioned : bool, default=False
        If True, ensures diagonal elements are large enough for numerical stability.
    min_diag_value : float, default=1.0
        Minimum diagonal value if ``well_conditioned=True``.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor of shape ``size`` with triangular structure and diagonal
        included.

    Raises
    ------
    ValueError
        If ``size`` is not 2D or 3D, not square, or if ``nnz`` violates constraints.
    ValueError
        If ``indices_dtype`` is not ``torch.int64`` or ``torch.int32``.

    Notes
    -----
    - Always includes **all diagonal entries** (cannot be excluded).
    - Off-diagonal nonzeros are chosen randomly within the triangular region.
    - Use ``generate_random_sparse_strictly_triangular_coo_matrix`` for strictly
      triangular matrices (without diagonal).
    - Use ``generate_random_sparse_triangular_csr_matrix`` for CSR layout instead.

    See Also
    --------
    rand_sparse_tri : Convenience dispatcher for triangular matrices (non-strict; COO/CSR).

    Examples
    --------
    Generate a 100×100 lower triangular COO matrix with 300 nonzeros:


    >>> A = generate_random_sparse_triangular_coo_matrix((100, 100), 300, upper=False)

    Generate a well-conditioned upper triangular COO matrix with minimum diagonal value 2.0:

    >>> A = generate_random_sparse_triangular_coo_matrix(
    ...     (50, 50), 200, upper=True, well_conditioned=True, min_diag_value=2.0
    ... )
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
    size: Tuple[int, int] | Tuple[int, int, int],
    nnz: int,
    *,
    upper: bool = True,
    indices_dtype: torch.dtype = torch.int64,
    values_dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    value_range: Tuple[float, float] = (0.0, 1.0),
    well_conditioned: bool = False,
    min_diag_value: float = 1.0,
) -> torch.Tensor:
    """
    Generate a random sparse CSR matrix with non-strict triangular structure.

    Constructs a triangular sparse matrix in CSR format that always includes all
    diagonal elements. Remaining entries are placed randomly in the upper or
    lower triangular region depending on ``upper``.

    Parameters
    ----------
    size : tuple of int
        Shape of the matrix. Must be ``(n, n)`` for unbatched or
        ``(batch_size, n, n)`` for batched matrices. The matrix must be square.
    nnz : int
        Number of nonzero elements per batch. Must satisfy ``n <= nnz <= n*(n+1)/2``.
        All diagonal elements are included, so ``nnz`` must be at least ``n``.
    upper : bool, default=True
        If True, generates an upper triangular matrix (row <= col).
        If False, generates a lower triangular matrix (row >= col).
    indices_dtype : torch.dtype, default=torch.int64
        Data type for sparse indices (``torch.int64`` or ``torch.int32``).
    values_dtype : torch.dtype, default=torch.float32
        Data type for nonzero values.
    device : torch.device, default=torch.device("cpu")
        Device on which to allocate tensors.
    value_range : tuple of float, default=(0.0, 1.0)
        Range ``(min, max)`` for random nonzero values.
    well_conditioned : bool, default=False
        If True, ensures diagonal elements are large enough for numerical stability.
    min_diag_value : float, default=1.0
        Minimum diagonal value if ``well_conditioned=True``.

    Returns
    -------
    torch.Tensor
        Sparse CSR tensor of shape ``size`` with triangular structure and diagonal
        included.

    Raises
    ------
    ValueError
        If ``size`` is not 2D or 3D, not square, or if ``nnz`` violates constraints.
    ValueError
        If ``indices_dtype`` is not ``torch.int64`` or ``torch.int32``.

    Notes
    -----
    - Always includes **all diagonal entries** (cannot be excluded).
    - Off-diagonal nonzeros are chosen randomly within the triangular region.
    - Use ``generate_random_sparse_strictly_triangular_csr_matrix`` for strictly
      triangular matrices (without diagonal).
    - Use ``generate_random_sparse_triangular_coo_matrix`` for COO layout instead.

    See Also
    --------
    rand_sparse_tri : Convenience dispatcher for triangular matrices (non-strict; COO/CSR).

    Examples
    --------
    Generate a 100×100 upper triangular CSR matrix with 300 nonzeros:

    >>> A = generate_random_sparse_triangular_csr_matrix((100, 100), 300, upper=True)

    Generate a well-conditioned lower triangular CSR matrix with min diagonal value 2.0:

    >>> A = generate_random_sparse_triangular_csr_matrix(
    ...     (50, 50), 200, upper=False, well_conditioned=True, min_diag_value=2.0
    ... )
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


def make_spd_sparse(
    n: int,
    layout: torch.layout,
    value_dtype: torch.dtype,
    index_dtype: torch.dtype,
    device: torch.device,
    sparsity_ratio: float = 0.5,
    nz: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random sparse symmetric positive definite (SPD) matrix.

    Constructs a dense SPD matrix ``A = M Mᵀ + n I`` from a random Gaussian
    matrix ``M``, then applies structured sparsification by zeroing out
    symmetric pairs of off-diagonal entries. Converts the result to the
    requested sparse layout.

    Parameters
    ----------
    n : int
        Dimension of the matrix (produces an ``n × n`` SPD matrix).
    layout : torch.layout
        Sparse tensor layout to return. Must be ``torch.sparse_coo`` or
        ``torch.sparse_csr``.
    value_dtype : torch.dtype
        Data type for matrix values.
    index_dtype : torch.dtype
        Data type for sparse indices. Typically ``torch.int32`` or ``torch.int64``.
        Note: PyTorch may coerce COO indices to ``int64`` on coalesce, even if
        ``int32`` is requested. CSR tensors preserve the chosen dtype.
    device : torch.device
        Device on which to allocate the tensors (e.g., ``torch.device("cuda")``).
    sparsity_ratio : float, optional
        Fraction of off-diagonal upper-triangular elements to zero out. Each
        selection removes both ``(i, j)`` and ``(j, i)`` to maintain symmetry.
        Ignored if ``nz`` is provided. Default is ``0.5``.
    nz : int, optional
        Exact number of symmetric *pairs* of off-diagonal elements to zero out.
        Each pair corresponds to ``(i, j)`` and ``(j, i)`` for ``i ≠ j``.
        If ``None``, ``sparsity_ratio`` is used. Default is ``None``.

    Returns
    -------
    A_sparse : torch.Tensor
        Sparse SPD matrix in the requested layout (``torch.sparse_coo`` or
        ``torch.sparse_csr``) with shape ``(n, n)``.
    A_dense : torch.Tensor
        Dense SPD matrix with shape ``(n, n)`` before conversion to sparse.

    Raises
    ------
    ValueError
        If ``layout`` is not ``torch.sparse_coo`` or ``torch.sparse_csr``.

    Notes
    -----
    - The SPD property is guaranteed by construction as
      ``A = M Mᵀ + n I``, regardless of sparsification.
    - Sparsification is symmetric: whenever entry ``(i, j)`` is zeroed,
      ``(j, i)`` is also zeroed, preserving symmetry.
    - Only unbatched (2D) matrices are supported. For batched sparse SPD
      matrices, extend this function accordingly.
    - For COO tensors, PyTorch may coerce indices to ``int64`` during
      coalesce. For CSR tensors, the requested ``index_dtype`` is preserved.

    Examples
    --------
    Generate a sparse SPD matrix in COO format:

    >>> A_sp, A_dn = make_spd_sparse(
    ...     n=5,
    ...     layout=torch.sparse_coo,
    ...     value_dtype=torch.float32,
    ...     index_dtype=torch.int64,
    ...     device=torch.device("cpu"),
    ...     sparsity_ratio=0.6,
    ... )
    >>> A_sp.shape
    torch.Size([5, 5])
    >>> A_dn.shape
    torch.Size([5, 5])

    Generate a sparse SPD matrix in CSR format with exactly 4 zeroed pairs:

    >>> A_sp, A_dn = make_spd_sparse(
    ...     n=6,
    ...     layout=torch.sparse_csr,
    ...     value_dtype=torch.float64,
    ...     index_dtype=torch.int32,
    ...     device=torch.device("cpu"),
    ...     nz=4,
    ... )
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

    # Convert indices to requested dtype (though PyTorch may override this)
    idx = idx.to(dtype=index_dtype)

    if layout == torch.sparse_coo:
        # For COO, create tensor and coalesce
        # Note: PyTorch automatically converts int32 indices to int64 during coalesce()
        A_sparse = torch.sparse_coo_tensor(idx, vals, (n, n), dtype=value_dtype, device=device).coalesce()
    elif layout == torch.sparse_csr:
        # For CSR, first create COO then convert to CSR
        A_coo = torch.sparse_coo_tensor(idx, vals, (n, n), dtype=value_dtype, device=device).coalesce()
        A_sparse = convert_coo_to_csr(A_coo)
    else:
        raise ValueError(f"Unsupported layout: {layout}. Use torch.sparse_coo or torch.sparse_csr.")

    return A_sparse, A_dense
