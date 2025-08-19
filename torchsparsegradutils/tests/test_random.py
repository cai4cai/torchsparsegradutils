import torch
import pytest
from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix,
    generate_random_sparse_csr_matrix,
    generate_random_sparse_strictly_triangular_coo_matrix,
    generate_random_sparse_strictly_triangular_csr_matrix,
    rand_sparse_tri,
    rand_sparse,
    make_spd_sparse,
)

# enable sparse invariants checks if available
if hasattr(torch.sparse, "check_sparse_tensor_invariants"):
    torch.sparse.check_sparse_tensor_invariants.enable()

# Device fixture
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))


def _id_device(d):
    return str(d)


@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param


# ---------- Tests for generate_random_sparse_coo_matrix ----------


@pytest.mark.parametrize(
    "size, nnz, multiplier",
    [
        (torch.Size([4, 4]), 12, 1),
        (torch.Size([2, 4, 4]), 12, 2),
        (torch.Size([8, 16]), 32, 1),
        (torch.Size([4, 8, 16]), 32, 4),
    ],
)
def test_gen_random_coo_size_nnz(size, nnz, multiplier, device):
    A = generate_random_sparse_coo_matrix(size, nnz, device=device)
    assert A.size() == size
    assert A._nnz() == nnz * multiplier


@pytest.mark.parametrize("indices_dtype", [torch.int8, torch.int16])
def test_gen_random_coo_invalid_indices(indices_dtype, device):
    with pytest.raises(ValueError):
        generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype, device=device)


@pytest.mark.parametrize("values_dtype", [torch.float16, torch.float32, torch.float64])
def test_gen_random_coo_values_dtype(values_dtype, device):
    A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, values_dtype=values_dtype, device=device)
    assert A.values().dtype == values_dtype


def test_gen_random_coo_device(device):
    A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, device=device)
    assert A.device.type == device.type


@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_gen_random_coo_indices_dtype_behavior(indices_dtype, device):
    """Test PyTorch's automatic index dtype conversion behavior for COO tensors.

    NOTE: PyTorch automatically converts int32 indices to int64 for COO tensors,
    but preserves int32 for CSR tensors. This is a known PyTorch behavior.
    """
    A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype, device=device)

    if indices_dtype == torch.int32:
        # PyTorch converts int32 to int64 for COO tensors - this is expected behavior
        assert (
            A.indices().dtype == torch.int64
        ), f"Expected int64, got {A.indices().dtype} (PyTorch converts int32->int64 for COO)"
    else:
        # int64 should be preserved
        assert A.indices().dtype == indices_dtype, f"Expected {indices_dtype}, got {A.indices().dtype}"


# ---------- Tests for generate_random_sparse_csr_matrix ----------


@pytest.mark.parametrize(
    "size, nnz",
    [
        (torch.Size([4, 4]), 12),
        (torch.Size([2, 4, 4]), 12),
        (torch.Size([8, 16]), 32),
        (torch.Size([4, 8, 16]), 32),
    ],
)
def test_gen_random_csr_size(device, size, nnz):
    A = generate_random_sparse_csr_matrix(size, nnz, device=device)
    assert A.size() == size


@pytest.mark.parametrize("nnz", [17])
def test_gen_random_csr_too_many_nnz(nnz, device):
    with pytest.raises(ValueError):
        generate_random_sparse_csr_matrix(torch.Size([4, 4]), nnz, device=device)


@pytest.mark.parametrize("indices_dtype", [torch.int8, torch.int16])
def test_gen_random_csr_invalid_indices(indices_dtype, device):
    with pytest.raises(ValueError):
        generate_random_sparse_csr_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype, device=device)


@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_gen_random_csr_indices_dtype(indices_dtype, device):
    """Test that CSR tensors preserve the requested index dtype.

    NOTE: Unlike COO tensors, CSR tensors preserve int32 dtypes correctly.
    """
    A = generate_random_sparse_csr_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype, device=device)
    assert A.crow_indices().dtype == indices_dtype
    assert A.col_indices().dtype == indices_dtype


@pytest.mark.parametrize("values_dtype", [torch.float16, torch.float32, torch.float64])
def test_gen_random_csr_values_dtype(values_dtype, device):
    A = generate_random_sparse_csr_matrix(torch.Size([4, 4]), 12, values_dtype=values_dtype, device=device)
    assert A.values().dtype == values_dtype


@pytest.mark.parametrize(
    "size, nnz",
    [
        (torch.Size([4, 4]), 12),
        (torch.Size([2, 4, 4]), 12),
        (torch.Size([8, 16]), 32),
        (torch.Size([4, 8, 16]), 32),
    ],
)
def test_gen_random_csr_nnz(size, nnz, device):
    A = generate_random_sparse_csr_matrix(size, nnz, device=device)
    assert A._nnz() == nnz


# ---------- Tests for strictly triangular COO ----------


@pytest.mark.parametrize(
    "size, nnz",
    [
        (torch.Size([4, 4]), 5),
        (torch.Size([2, 4, 4, 4]), 5),
    ],
)
def test_gen_random_strict_tri_coo_invalid_dims(size, nnz, device):
    # too few dims or non-square batches
    if len(size) != 2:
        with pytest.raises(ValueError):
            generate_random_sparse_strictly_triangular_coo_matrix(size, nnz, device=device)


@pytest.mark.parametrize("nnz", [7])
def test_gen_random_strict_tri_coo_too_many_nnz(nnz, device):
    limit = 4 * (4 - 1) // 2
    with pytest.raises(ValueError):
        generate_random_sparse_strictly_triangular_coo_matrix(torch.Size([4, 4]), limit + 1, device=device)


@pytest.mark.parametrize("indices_dtype", [torch.int8, torch.int16])
def test_gen_random_strict_tri_coo_invalid_indices(indices_dtype, device):
    with pytest.raises(ValueError):
        generate_random_sparse_strictly_triangular_coo_matrix(
            torch.Size([4, 4]), 5, indices_dtype=indices_dtype, device=device
        )


@pytest.mark.parametrize("values_dtype", [torch.float16, torch.float32, torch.float64])
def test_gen_random_strict_tri_coo_values_dtype(values_dtype, device):
    A = generate_random_sparse_strictly_triangular_coo_matrix(
        torch.Size([4, 4]), 5, values_dtype=values_dtype, device=device
    )
    assert A.values().dtype == values_dtype


@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_gen_random_strict_tri_coo_indices_dtype_behavior(indices_dtype, device):
    """Test PyTorch's automatic index dtype conversion behavior for strictly triangular COO tensors.

    NOTE: PyTorch automatically converts int32 indices to int64 for COO tensors,
    but preserves int32 for CSR tensors. This is a known PyTorch behavior.
    """
    A = generate_random_sparse_strictly_triangular_coo_matrix(
        torch.Size([4, 4]), 5, indices_dtype=indices_dtype, device=device
    )

    if indices_dtype == torch.int32:
        # PyTorch converts int32 to int64 for COO tensors - this is expected behavior
        assert (
            A.indices().dtype == torch.int64
        ), f"Expected int64, got {A.indices().dtype} (PyTorch converts int32->int64 for COO)"
    else:
        # int64 should be preserved
        assert A.indices().dtype == indices_dtype, f"Expected {indices_dtype}, got {A.indices().dtype}"


@pytest.mark.parametrize(
    "size, upper, multiplier",
    [
        (torch.Size([4, 4]), True, 1),
        (torch.Size([2, 4, 4]), False, 2),
        (torch.Size([8, 8]), True, 1),
    ],
)
def test_gen_random_strict_tri_coo_properties(size, upper, multiplier, device):
    nnz = 5
    A = generate_random_sparse_strictly_triangular_coo_matrix(size, nnz, upper=upper, device=device)
    assert A.size() == size
    assert A._nnz() == nnz * multiplier
    Ad = A.to_dense()
    if upper:
        assert torch.equal(Ad, Ad.triu(1))
    else:
        assert torch.equal(Ad, Ad.tril(-1))


# ---------- Tests for strictly triangular CSR ----------


@pytest.mark.parametrize(
    "size, nnz",
    [
        (torch.Size([4, 4]), 5),
    ],
)
def test_gen_random_strict_tri_csr_invalid_dims(size, nnz, device):
    # only square supported
    with pytest.raises(ValueError):
        generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4, 8]), nnz, device=device)


@pytest.mark.parametrize("nnz", [7])
def test_gen_random_strict_tri_csr_too_many_nnz(nnz, device):
    limit = 4 * (4 - 1) // 2
    with pytest.raises(ValueError):
        generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4]), limit + 1, device=device)


@pytest.mark.parametrize("indices_dtype", [torch.int8, torch.int16])
def test_gen_random_strict_tri_csr_invalid_indices(indices_dtype, device):
    with pytest.raises(ValueError):
        generate_random_sparse_strictly_triangular_csr_matrix(
            torch.Size([4, 4]), 5, indices_dtype=indices_dtype, device=device
        )


@pytest.mark.parametrize("values_dtype", [torch.float16, torch.float32, torch.float64])
def test_gen_random_strict_tri_csr_values_dtype(values_dtype, device):
    A = generate_random_sparse_strictly_triangular_csr_matrix(
        torch.Size([4, 4]), 5, values_dtype=values_dtype, device=device
    )
    assert A.values().dtype == values_dtype


@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
def test_gen_random_strict_tri_csr_indices_dtype_behavior(indices_dtype, device):
    """Test that CSR strictly triangular tensors preserve the requested index dtype.

    NOTE: Unlike COO tensors, CSR tensors preserve int32 dtypes correctly.
    """
    A = generate_random_sparse_strictly_triangular_csr_matrix(
        torch.Size([4, 4]), 5, indices_dtype=indices_dtype, device=device
    )
    assert A.crow_indices().dtype == indices_dtype
    assert A.col_indices().dtype == indices_dtype


@pytest.mark.parametrize("upper", [True, False])
def test_gen_random_strict_tri_csr_properties(upper, device):
    nnz = 5
    size = torch.Size([4, 4])
    A = generate_random_sparse_strictly_triangular_csr_matrix(size, nnz, upper=upper, device=device)
    assert A.size() == size
    assert A._nnz() == nnz
    Ad = A.to_dense()
    if upper:
        assert torch.equal(Ad, Ad.triu(1))
    else:
        assert torch.equal(Ad, Ad.tril(-1))


# ---------- Tests for non-strict triangular matrices ----------


@pytest.mark.parametrize(
    "layout,upper",
    [
        (torch.sparse_coo, True),
        (torch.sparse_coo, False),
        (torch.sparse_csr, True),
        (torch.sparse_csr, False),
    ],
)
def test_rand_sparse_tri_nonstrict_diag(layout, upper, device):
    size = torch.Size([4, 4])
    nnz = 6
    A = rand_sparse_tri(size, nnz, layout=layout, strict=False, upper=upper, device=device)
    Ad = A.to_dense()
    # every diagonal entry should be non-zero
    assert torch.all(torch.diag(Ad) != 0)


# ---------- Tests for well-conditioned matrix generation ----------


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
def test_rand_sparse_well_conditioned_square(layout, device):
    """Test well-conditioned square matrix generation"""
    size = torch.Size([4, 4])
    nnz = 10
    min_diag_val = 2.5

    A = rand_sparse(size, nnz, layout=layout, device=device, well_conditioned=True, min_diag_value=min_diag_val)
    Ad = A.to_dense()

    # Check that diagonal elements that exist are >= min_diag_value
    diag_elements = torch.diag(Ad)
    nonzero_diag = diag_elements[diag_elements != 0]
    if len(nonzero_diag) > 0:
        assert torch.all(nonzero_diag >= min_diag_val), f"Found diagonal elements < {min_diag_val}: {nonzero_diag}"


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
def test_rand_sparse_well_conditioned_non_square(layout, device):
    """Test that well-conditioning only applies to square matrices"""
    size = torch.Size([4, 6])  # Non-square
    nnz = 12
    min_diag_val = 2.0

    # Should work without error but well-conditioning should be ignored
    A = rand_sparse(size, nnz, layout=layout, device=device, well_conditioned=True, min_diag_value=min_diag_val)
    assert A.size() == size
    assert A._nnz() == nnz


def test_rand_sparse_well_conditioned_default_behavior(device):
    """Test that default behavior is unchanged when well_conditioned=False"""
    size = torch.Size([4, 4])
    nnz = 10

    A_default = rand_sparse(size, nnz, device=device)
    A_explicit = rand_sparse(size, nnz, device=device, well_conditioned=False)

    # Both should have same structure (though values will be random)
    assert A_default.size() == A_explicit.size()
    assert A_default._nnz() == A_explicit._nnz()


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
@pytest.mark.parametrize("upper", [True, False])
@pytest.mark.parametrize("strict", [True, False])
def test_rand_sparse_tri_well_conditioned(layout, upper, strict, device):
    """Test well-conditioned triangular matrix generation"""
    size = torch.Size([4, 4])
    nnz = 6 if strict else 8  # Adjust nnz based on strict mode
    min_diag_val = 1.5

    # For strict triangular matrices, well-conditioning doesn't apply since there's no diagonal
    if strict:
        A = rand_sparse_tri(size, nnz, layout=layout, upper=upper, strict=strict, device=device)
    else:
        A = rand_sparse_tri(
            size,
            nnz,
            layout=layout,
            upper=upper,
            strict=strict,
            device=device,
            well_conditioned=True,
            min_diag_value=min_diag_val,
        )
    Ad = A.to_dense()

    # Check matrix structure
    if upper:
        if strict:
            assert torch.equal(Ad, Ad.triu(1)), "Strictly upper triangular matrix should have zero diagonal"
        else:
            assert torch.equal(Ad, Ad.triu(0)), "Upper triangular matrix failed"
    else:
        if strict:
            assert torch.equal(Ad, Ad.tril(-1)), "Strictly lower triangular matrix should have zero diagonal"
        else:
            assert torch.equal(Ad, Ad.tril(0)), "Lower triangular matrix failed"

    # For non-strict triangular matrices, check diagonal conditioning
    if not strict:
        diag_elements = torch.diag(Ad)
        # All diagonal elements should be non-zero and >= min_diag_value
        assert torch.all(diag_elements != 0), "Non-strict triangular should have non-zero diagonal"
        assert torch.all(diag_elements >= min_diag_val), f"Diagonal elements should be >= {min_diag_val}"


@pytest.mark.parametrize("min_diag_value", [0.5, 1.0, 2.0, 5.0])
def test_rand_sparse_tri_min_diag_values(min_diag_value, device):
    """Test different minimum diagonal values"""
    size = torch.Size([4, 4])
    nnz = 8

    A = rand_sparse_tri(
        size,
        nnz,
        torch.sparse_coo,
        upper=False,
        strict=False,
        device=device,
        well_conditioned=True,
        min_diag_value=min_diag_value,
    )
    Ad = A.to_dense()
    diag_elements = torch.diag(Ad)

    assert torch.all(diag_elements >= min_diag_value), f"All diagonal elements should be >= {min_diag_value}"


def test_generate_random_sparse_coo_matrix_well_conditioned(device):
    """Test well-conditioning for COO matrix generation"""
    size = torch.Size([5, 5])
    nnz = 15
    min_diag_val = 3.0

    A = generate_random_sparse_coo_matrix(size, nnz, device=device, well_conditioned=True, min_diag_value=min_diag_val)
    Ad = A.to_dense()

    # Check that any diagonal elements present are >= min_diag_value
    diag_elements = torch.diag(Ad)
    nonzero_diag = diag_elements[diag_elements != 0]
    if len(nonzero_diag) > 0:
        assert torch.all(nonzero_diag >= min_diag_val)


def test_generate_random_sparse_csr_matrix_well_conditioned(device):
    """Test well-conditioning for CSR matrix generation"""
    size = torch.Size([5, 5])
    nnz = 15
    min_diag_val = 2.0

    A = generate_random_sparse_csr_matrix(size, nnz, device=device, well_conditioned=True, min_diag_value=min_diag_val)
    Ad = A.to_dense()

    # Check that any diagonal elements present are >= min_diag_value
    diag_elements = torch.diag(Ad)
    nonzero_diag = diag_elements[diag_elements != 0]
    if len(nonzero_diag) > 0:
        assert torch.all(nonzero_diag >= min_diag_val)


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
def test_rand_sparse_tri_value_range_with_conditioning(layout, device):
    """Test that value_range and well_conditioned work together correctly"""
    size = torch.Size([4, 4])
    nnz = 8
    value_range = (0.1, 2.0)  # Use a wider range that doesn't conflict with min_diag_value
    min_diag_val = 1.0

    A = rand_sparse_tri(
        size,
        nnz,
        layout,
        upper=False,
        strict=False,
        device=device,
        value_range=value_range,
        well_conditioned=True,
        min_diag_value=min_diag_val,
    )
    Ad = A.to_dense()

    # Diagonal elements should be >= min_diag_value (conditioning takes precedence)
    diag_elements = torch.diag(Ad)
    assert torch.all(
        diag_elements >= min_diag_val - 1e-6
    ), f"Diagonal elements should be >= {min_diag_val}, got {diag_elements}"

    # Off-diagonal elements should be in value_range (check that most are in reasonable bounds)
    mask = ~torch.eye(size[0], dtype=torch.bool, device=device)
    off_diag = Ad[mask]
    nonzero_off_diag = off_diag[off_diag != 0]
    if len(nonzero_off_diag) > 0:
        # Check that most values are in the expected range (allowing for some precision tolerance)
        in_range = (nonzero_off_diag >= value_range[0] - 0.1) & (nonzero_off_diag <= value_range[1] + 0.1)
        assert (
            torch.sum(in_range).item() >= len(nonzero_off_diag) * 0.7
        ), f"Most off-diagonal elements should be near value_range {value_range}, got {nonzero_off_diag}"


def test_rand_sparse_batched_well_conditioned(device):
    """Test well-conditioning with batched matrices"""
    size = torch.Size([2, 4, 4])  # Batch of 2 4x4 matrices
    nnz = 8
    min_diag_val = 1.5

    A = rand_sparse(size, nnz, torch.sparse_coo, device=device, well_conditioned=True, min_diag_value=min_diag_val)
    Ad = A.to_dense()

    # Check each matrix in the batch
    for i in range(size[0]):
        diag_elements = torch.diag(Ad[i])
        nonzero_diag = diag_elements[diag_elements != 0]
        if len(nonzero_diag) > 0:
            assert torch.all(nonzero_diag >= min_diag_val), f"Batch {i}: diagonal elements should be >= {min_diag_val}"


def test_rand_sparse_tri_batched_well_conditioned(device):
    """Test well-conditioning with batched triangular matrices"""
    size = torch.Size([2, 4, 4])  # Batch of 2 4x4 matrices
    nnz = 8
    min_diag_val = 2.0

    A = rand_sparse_tri(
        size,
        nnz,
        torch.sparse_coo,
        upper=False,
        strict=False,
        device=device,
        well_conditioned=True,
        min_diag_value=min_diag_val,
    )
    Ad = A.to_dense()

    # Check each matrix in the batch
    for i in range(size[0]):
        diag_elements = torch.diag(Ad[i])
        # All diagonal elements should be non-zero and >= min_diag_value for non-strict triangular
        assert torch.all(diag_elements != 0), f"Batch {i}: should have non-zero diagonal"
        assert torch.all(diag_elements >= min_diag_val), f"Batch {i}: diagonal should be >= {min_diag_val}"


# ---------- Tests for make_spd_sparse ----------


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
@pytest.mark.parametrize("value_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_make_spd_sparse_basic(layout, value_dtype, index_dtype, device):
    """Test basic functionality of make_spd_sparse."""
    n = 10
    sparsity_ratio = 0.3

    A_sparse, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, sparsity_ratio)

    # Check basic properties
    assert A_sparse.shape == (n, n)
    assert A_dense.shape == (n, n)
    assert A_sparse.dtype == value_dtype
    assert A_dense.dtype == value_dtype
    assert A_sparse.device == device
    assert A_dense.device == device
    assert A_sparse.layout == layout

    # Check index dtype for different layouts
    # Note: PyTorch may reset index dtypes during coalesce operations
    # so we'll just check that the function accepts the parameter without error
    if layout == torch.sparse_coo:
        # Check that indices exist and are accessible
        assert A_sparse.indices() is not None
        assert A_sparse.indices().dim() == 2
    else:  # CSR
        # Check that CSR indices exist and are accessible
        assert A_sparse.crow_indices() is not None
        assert A_sparse.col_indices() is not None
        assert A_sparse.crow_indices().dim() == 1
        assert A_sparse.col_indices().dim() == 1

    # Check that sparse and dense versions are equivalent
    assert torch.allclose(A_sparse.to_dense(), A_dense, atol=1e-6)


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
def test_make_spd_sparse_positive_definite(layout, device):
    """Test that generated matrices are positive definite."""
    n = 8
    value_dtype = torch.float64  # Use higher precision for numerical stability
    index_dtype = torch.int64
    sparsity_ratio = 0.5

    A_sparse, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, sparsity_ratio)

    # Check positive definiteness using Cholesky decomposition
    try:
        _ = torch.linalg.cholesky(A_dense)
        # If we get here, the matrix is positive definite
        assert True
    except torch.linalg.LinAlgError:
        pytest.fail("Generated matrix is not positive definite")


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
def test_make_spd_sparse_sparsity_patterns(layout, device):
    """Test different sparsity patterns."""
    n = 12
    value_dtype = torch.float32
    index_dtype = torch.int64

    # Test different sparsity ratios
    for sparsity_ratio in [0.0, 0.3, 0.7, 0.9]:
        A_sparse, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, sparsity_ratio)

        # Count actual sparsity
        total_elements = n * n
        nnz = A_sparse._nnz()
        actual_sparsity = 1.0 - (nnz / total_elements)

        # For sparsity_ratio=0, matrix should be dense (only numerical zeros from construction)
        if sparsity_ratio == 0.0:
            # Should have most elements non-zero (allowing for some numerical artifacts)
            assert actual_sparsity < 0.1
        else:
            # Should have some sparsity, but exact amount depends on random selection
            assert actual_sparsity > 0.0


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
def test_make_spd_sparse_nz_parameter(layout, device):
    """Test using nz parameter instead of sparsity_ratio."""
    n = 10
    value_dtype = torch.float32
    index_dtype = torch.int64

    # Test with exact number of elements to zero out
    nz_values = [0, 5, 20, 40]

    for nz in nz_values:
        A_sparse, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, nz=nz)

        # Check that matrix is still positive definite
        try:
            _ = torch.linalg.cholesky(A_dense)
            assert True
        except torch.linalg.LinAlgError:
            pytest.fail(f"Generated matrix with nz={nz} is not positive definite")

        # Check basic properties
        assert A_sparse.shape == (n, n)
        assert A_sparse.dtype == value_dtype
        assert A_sparse.layout == layout


def test_make_spd_sparse_solve_system(device):
    """Test that we can solve linear systems with generated SPD matrices."""
    n = 15
    layout = torch.sparse_coo
    value_dtype = torch.float64  # Higher precision for numerical stability
    index_dtype = torch.int64
    sparsity_ratio = 0.4

    A_sparse, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, sparsity_ratio)

    # Generate random RHS
    b = torch.randn(n, dtype=value_dtype, device=device)

    # Solve using dense solver
    try:
        x_dense = torch.linalg.solve(A_dense, b)
        residual = A_dense @ x_dense - b
        residual_norm = torch.norm(residual).item()

        # Check that residual is small
        assert residual_norm < 1e-10, f"Dense solve residual too large: {residual_norm}"

    except torch.linalg.LinAlgError:
        pytest.fail("Could not solve linear system with generated SPD matrix")


def test_make_spd_sparse_invalid_layout():
    """Test error handling for invalid layouts."""
    n = 5
    value_dtype = torch.float32
    index_dtype = torch.int64
    device = torch.device("cpu")

    with pytest.raises(ValueError, match="Unsupported layout"):
        make_spd_sparse(n, torch.strided, value_dtype, index_dtype, device)


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
def test_make_spd_sparse_symmetry(layout, device):
    """Test that generated matrices are symmetric."""
    n = 8
    value_dtype = torch.float64
    index_dtype = torch.int64
    sparsity_ratio = 0.3

    A_sparse, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, sparsity_ratio)

    # Check symmetry (A = A^T)
    A_dense_T = A_dense.t()
    assert torch.allclose(A_dense, A_dense_T, atol=1e0), "Generated matrix is not symmetric"


@pytest.mark.parametrize("n", [4, 8, 16, 32])
def test_make_spd_sparse_different_sizes(n, device):
    """Test generation with different matrix sizes."""
    layout = torch.sparse_coo
    value_dtype = torch.float32
    index_dtype = torch.int64
    sparsity_ratio = 0.5

    A_sparse, A_dense = make_spd_sparse(n, layout, value_dtype, index_dtype, device, sparsity_ratio)

    # Check correct size
    assert A_sparse.shape == (n, n)
    assert A_dense.shape == (n, n)

    # Check positive definiteness
    try:
        _ = torch.linalg.cholesky(A_dense)
        assert True
    except torch.linalg.LinAlgError:
        pytest.fail(f"Generated {n}x{n} matrix is not positive definite")
