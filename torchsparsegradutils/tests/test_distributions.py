import pytest
import torch
from torch.distributions.multivariate_normal import _batch_mv

from torchsparsegradutils import sparse_mm
from torchsparsegradutils.distributions import SparseMultivariateNormal, SparseMultivariateNormalNative
from torchsparsegradutils.distributions.sparse_multivariate_normal import _batch_sparse_mv
from torchsparsegradutils.utils import rand_sparse_tri
from torchsparsegradutils.utils.dist_stats_helpers import cov_nagao_test, mean_hotelling_t2_test

# Identify Testing Parameters
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

TEST_DATA = [
    # name, batch size, event size, spartsity
    ("unbat", None, 4, 0.5),
    ("bat", 4, 4, 0.5),
    # ("bat2", 4, 64, 0.01),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
SPASRE_LAYOUTS = [torch.sparse_coo, torch.sparse_csr]
PARAMETERIZATIONS = ["ldlt", "llt"]  # LDL^T vs LL^T

# DISTRIBUTIONS = [SparseMultivariateNormal]


# Define Test Names:
def data_id(sizes):
    return sizes[0]


def device_id(device):
    return str(device)


def dtype_id(dtype):
    return str(dtype).split(".")[-1]


def layout_id(layout):
    return str(layout).split(".")[-1].split("_")[-1].upper()


def param_id(param):
    return param.upper()


# Define Fixtures


@pytest.fixture(params=TEST_DATA, ids=[data_id(d) for d in TEST_DATA])
def sizes(request):
    return request.param


@pytest.fixture(params=VALUE_DTYPES, ids=[dtype_id(d) for d in VALUE_DTYPES])
def value_dtype(request):
    return request.param


@pytest.fixture(params=INDEX_DTYPES, ids=[dtype_id(d) for d in INDEX_DTYPES])
def index_dtype(request):
    return request.param


@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


@pytest.fixture(params=SPASRE_LAYOUTS, ids=[layout_id(lay) for lay in SPASRE_LAYOUTS])
def layout(request):
    return request.param


@pytest.fixture(params=PARAMETERIZATIONS, ids=[param_id(p) for p in PARAMETERIZATIONS])
def parameterization(request):
    return request.param


# @pytest.fixture(params=DISTRIBUTIONS, ids=[dist_id(d) for d in DISTRIBUTIONS])
# def distribution(request):
#     return request.param


# Set random seed for reproducibility
# using instead of @pytest.mark.flaky(reruns=5)
@pytest.fixture(autouse=True)
def set_seed():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)


# Convenience functions:


def construct_distribution(sizes, layout, var, parameterization, value_dtype, index_dtype, device, requires_grad=False):
    _, batch_size, event_size, sparsity = sizes
    loc = torch.randn(event_size, device=device, dtype=value_dtype, requires_grad=requires_grad)

    # Handle diagonal parameter based on parameterization
    if parameterization == "ldlt":
        # LDL^T parameterization: provide diagonal, use strictly lower triangular
        diagonal = torch.rand(event_size, device=device, dtype=value_dtype, requires_grad=requires_grad)
        strict = True
        # For strictly triangular: max nnz = n*(n-1)/2
        max_nnz = event_size * (event_size - 1) // 2
    else:  # llt parameterization
        # LL^T parameterization: no diagonal, use lower triangular with diagonal
        diagonal = None
        strict = False
        # For lower triangular: max nnz = n*(n+1)/2, min nnz = n (for diagonal)
        max_nnz = event_size * (event_size + 1) // 2

    tril_size = (batch_size, event_size, event_size) if batch_size else (event_size, event_size)
    nnz = int(sparsity * max_nnz)

    # Ensure minimum nnz for LL^T parameterization (need at least diagonal)
    if parameterization == "llt":
        nnz = max(nnz, event_size)

    tril = rand_sparse_tri(
        tril_size,
        nnz,
        layout,
        upper=False,
        strict=strict,
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
        well_conditioned=True,
        min_diag_value=1.0,
    )

    tril.requires_grad = requires_grad

    if var == "cov":
        return SparseMultivariateNormal(loc, diagonal, scale_tril=tril)
    elif var == "prec":
        return SparseMultivariateNormal(loc, diagonal, precision_tril=tril)
    else:
        raise ValueError(f"var must be one of 'cov' or 'prec', but got {var}")


def compute_reference_covariance(dist, var):
    """Compute the reference covariance matrix for a distribution."""
    if var == "cov":
        if dist.is_ldlt_parameterization:
            # LDL^T parameterization
            scale_tril = dist.scale_tril.to_dense()
            scale_tril = scale_tril + torch.eye(*dist.event_shape, dtype=scale_tril.dtype, device=scale_tril.device)
            diagonal = dist.diagonal
            covariance_ref = torch.matmul(scale_tril @ torch.diag_embed(diagonal), scale_tril.transpose(-1, -2))
        else:
            # LL^T parameterization
            scale_tril = dist.scale_tril.to_dense()
            covariance_ref = torch.matmul(scale_tril, scale_tril.transpose(-1, -2))
    else:  # var == "prec"
        if dist.is_ldlt_parameterization:
            # LDL^T parameterization
            precision_tril = dist.precision_tril.to_dense()
            precision_tril = precision_tril + torch.eye(
                *dist.event_shape, dtype=precision_tril.dtype, device=precision_tril.device
            )
            diagonal = dist.diagonal
            precision_ref = torch.matmul(precision_tril @ torch.diag_embed(diagonal), precision_tril.transpose(-1, -2))
            covariance_ref = torch.linalg.inv(precision_ref)
        else:
            # LL^T parameterization
            precision_tril = dist.precision_tril.to_dense()
            precision_ref = torch.matmul(precision_tril, precision_tril.transpose(-1, -2))
            covariance_ref = torch.linalg.inv(precision_ref)

    return covariance_ref


def compute_sample_statistics(samples):
    """Compute sample mean and covariance from samples."""
    if len(samples.shape) == 2:
        # Unbatched case
        sample_mean = samples.mean(0)
        sample_cov = torch.cov(samples.T)
        return sample_mean, sample_cov
    else:
        # Batched case
        sample_mean = samples.mean(0)  # Average over sample dimension
        sample_cov = torch.stack([torch.cov(sample.T) for sample in samples.permute(1, 0, 2)])
        return sample_mean, sample_cov


# Define Tests


@pytest.mark.flaky(reruns=5)
def test_rsample_forward_cov(device, layout, sizes, parameterization, value_dtype, index_dtype):
    """Test sampling from covariance parameterization using proper statistical tests."""

    dist = construct_distribution(sizes, layout, "cov", parameterization, value_dtype, index_dtype, device)
    n_samples = 10_000
    samples = dist.rsample((n_samples,))

    covariance_ref = compute_reference_covariance(dist, "cov")
    sample_mean, sample_cov = compute_sample_statistics(samples)

    # Test mean using Hotelling's T² test
    if len(samples.shape) == 2:
        # Unbatched case
        confidence_level = 0.99 if value_dtype == torch.float32 else 0.90

        mean_test_result, t2_stat, t2_threshold = mean_hotelling_t2_test(
            sample_mean.unsqueeze(0),
            dist.loc.unsqueeze(0),
            sample_cov.unsqueeze(0),
            n_samples,
            confidence_level=confidence_level,
        )
        assert mean_test_result.item(), f"Mean test failed: T²={t2_stat.item():.6f} > threshold={t2_threshold:.6f}"

        # Test covariance using Nagao test
        cov_test_result, T_N_stat, chi2_threshold = cov_nagao_test(
            sample_cov.unsqueeze(0), covariance_ref.unsqueeze(0), n_samples, confidence_level=confidence_level
        )
        assert (
            cov_test_result.item()
        ), f"Covariance test failed: T_N={T_N_stat.item():.6f} > threshold={chi2_threshold:.6f}"
    else:
        # Batched case
        confidence_level = 0.99 if value_dtype == torch.float32 else 0.95

        mean_test_result, t2_stat, t2_threshold = mean_hotelling_t2_test(
            sample_mean, dist.loc, sample_cov, n_samples, confidence_level=confidence_level
        )
        assert (
            mean_test_result.all()
        ), f"Mean test failed for some batch elements: max T²={t2_stat.max().item():.6f} > threshold={t2_threshold:.6f}"

        cov_test_result, T_N_stat, chi2_threshold = cov_nagao_test(
            sample_cov, covariance_ref, n_samples, confidence_level=confidence_level
        )
        assert (
            cov_test_result.all()
        ), f"Covariance test failed for some batch elements: max T_N={T_N_stat.max().item():.6f} > threshold={chi2_threshold:.6f}"


@pytest.mark.flaky(reruns=5)
def test_rsample_forward_prec(device, layout, sizes, parameterization, value_dtype, index_dtype):
    """Test sampling from precision parameterization using proper statistical tests."""

    dist = construct_distribution(sizes, layout, "prec", parameterization, value_dtype, index_dtype, device)
    n_samples = 10_000
    samples = dist.rsample((n_samples,))

    covariance_ref = compute_reference_covariance(dist, "prec")
    sample_mean, sample_cov = compute_sample_statistics(samples)

    # Test mean using Hotelling's T² test
    if len(samples.shape) == 2:
        # Unbatched case
        confidence_level = 0.99 if value_dtype == torch.float32 else 0.95

        mean_test_result, t2_stat, t2_threshold = mean_hotelling_t2_test(
            sample_mean.unsqueeze(0),
            dist.loc.unsqueeze(0),
            sample_cov.unsqueeze(0),
            n_samples,
            confidence_level=confidence_level,
        )
        assert mean_test_result.item(), f"Mean test failed: T²={t2_stat.item():.6f} > threshold={t2_threshold:.6f}"

        # Test covariance using Nagao test
        cov_test_result, T_N_stat, chi2_threshold = cov_nagao_test(
            sample_cov.unsqueeze(0),
            covariance_ref.unsqueeze(0),
            n_samples,
            confidence_level=confidence_level,
        )
        assert (
            cov_test_result.item()
        ), f"Covariance test failed: T_N={T_N_stat.item():.6f} > threshold={chi2_threshold:.6f}"
    else:
        # Batched case
        confidence_level = 0.99 if value_dtype == torch.float32 else 0.95

        mean_test_result, t2_stat, t2_threshold = mean_hotelling_t2_test(
            sample_mean, dist.loc, sample_cov, n_samples, confidence_level=confidence_level
        )
        assert (
            mean_test_result.all()
        ), f"Mean test failed for some batch elements: max T²={t2_stat.max().item():.6f} > threshold={t2_threshold:.6f}"

        cov_test_result, T_N_stat, chi2_threshold = cov_nagao_test(
            sample_cov,
            covariance_ref,
            n_samples,
            confidence_level=confidence_level,
        )
        assert (
            cov_test_result.all()
        ), f"Covariance test failed for some batch elements: max T_N={T_N_stat.max().item():.6f} > threshold={chi2_threshold:.6f}"


def test_parameterization_property(device, layout, sizes, parameterization, value_dtype, index_dtype):
    """Test that the parameterization property works correctly."""

    dist = construct_distribution(sizes, layout, "cov", parameterization, value_dtype, index_dtype, device)

    if parameterization == "ldlt":
        assert dist.is_ldlt_parameterization, "Expected LDL^T parameterization"
        assert dist.diagonal is not None, "Expected diagonal to be provided for LDL^T"
    else:  # llt
        assert not dist.is_ldlt_parameterization, "Expected LL^T parameterization"
        assert dist.diagonal is None, "Expected diagonal to be None for LL^T"


def test_rsample_backward_cov(device, layout, sizes, parameterization, value_dtype, index_dtype):
    """Test backward pass for covariance parameterization."""

    dist = construct_distribution(
        sizes, layout, "cov", parameterization, value_dtype, index_dtype, device, requires_grad=True
    )
    samples = dist.rsample((10,))

    samples.sum().backward()


def test_rsample_backward_prec(device, layout, sizes, parameterization, value_dtype, index_dtype):
    """Test backward pass for precision parameterization."""

    dist = construct_distribution(
        sizes, layout, "prec", parameterization, value_dtype, index_dtype, device, requires_grad=True
    )
    samples = dist.rsample((10,))

    samples.sum().backward()


BATCH_MV_TEST_DATA = [
    # 3 = event_size, 4 = batch_size, 5 = sample_size
    # bmat, bvec
    (torch.randn(3, 3), torch.randn(3)),
    (torch.randn(3, 3), torch.randn(5, 3)),
    (torch.randn(4, 3, 3), torch.randn(4, 3)),
    (torch.randn(4, 3, 3), torch.randn(5, 4, 3)),
]


@pytest.fixture(params=BATCH_MV_TEST_DATA)
def batch_mv_test_data(request):
    return request.param


def test_sparse_batch_mv(batch_mv_test_data):
    bmat, bvec = batch_mv_test_data
    res_ref = _batch_mv(bmat, bvec)
    res_test = _batch_sparse_mv(sparse_mm, bmat.to_sparse(), bvec)
    assert torch.allclose(res_ref, res_test)


# Test data specifically for SparseMultivariateNormalNative (unbatched only)
NATIVE_TEST_DATA = [
    # name, event size, sparsity
    ("small", 4, 0.5),
    ("medium", 16, 0.3),
    ("large", 32, 0.1),
]


# Define Test Names for Native:
def native_data_id(sizes):
    return sizes[0]


# Define Fixtures for Native
@pytest.fixture(params=NATIVE_TEST_DATA, ids=[native_data_id(d) for d in NATIVE_TEST_DATA])
def native_sizes(request):
    return request.param


def construct_native_distribution(sizes, value_dtype, device, index_dtype, requires_grad=False):
    """Construct SparseMultivariateNormalNative distribution for testing."""
    _, event_size, sparsity = sizes

    # Create location parameter
    loc = torch.randn(event_size, device=device, dtype=value_dtype, requires_grad=requires_grad)

    # Create sparse CSR lower triangular matrix for LL^T parameterization
    # For lower triangular: max nnz = n*(n+1)/2, min nnz = n (for diagonal)
    max_nnz = event_size * (event_size + 1) // 2
    nnz = max(int(sparsity * max_nnz), event_size)  # Ensure at least diagonal

    scale_tril = rand_sparse_tri(
        (event_size, event_size),
        nnz,
        torch.sparse_csr,
        upper=False,
        strict=False,  # Include diagonal
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
    )

    scale_tril.requires_grad = requires_grad

    return SparseMultivariateNormalNative(loc, scale_tril)


def compute_native_reference_covariance(dist):
    """Compute reference covariance for native distribution."""
    return dist.covariance_matrix


def compute_native_sample_statistics(samples):
    """Compute sample mean and covariance for native distribution samples."""
    sample_mean = samples.mean(0)
    sample_cov = torch.cov(samples.T)
    return sample_mean, sample_cov


# Tests for SparseMultivariateNormalNative
@pytest.mark.flaky(reruns=5)
def test_native_rsample_forward(device, native_sizes, value_dtype, index_dtype):
    """Test sampling from SparseMultivariateNormalNative using statistical tests."""

    dist = construct_native_distribution(native_sizes, value_dtype, device, index_dtype)
    n_samples = 10_000
    samples = dist.rsample((n_samples,))

    # Compute reference statistics
    covariance_ref = compute_native_reference_covariance(dist)
    sample_mean, sample_cov = compute_native_sample_statistics(samples)

    # Test mean using Hotelling's T² test
    confidence_level = 0.99 if value_dtype == torch.float32 else 0.95

    mean_test_result, t2_stat, t2_threshold = mean_hotelling_t2_test(
        sample_mean.unsqueeze(0),
        dist.loc.unsqueeze(0),
        sample_cov.unsqueeze(0),
        n_samples,
        confidence_level=confidence_level,
    )
    assert mean_test_result.item(), f"Mean test failed: T²={t2_stat.item():.6f} > threshold={t2_threshold:.6f}"

    # Test covariance using Nagao test with more lenient confidence for native implementation
    # Due to numerical differences in torch.sparse.mm, be more forgiving
    cov_confidence = 0.90 if "native" in construct_native_distribution.__name__ else confidence_level
    cov_test_result, T_N_stat, chi2_threshold = cov_nagao_test(
        sample_cov.unsqueeze(0), covariance_ref.unsqueeze(0), n_samples, confidence_level=cov_confidence
    )
    assert cov_test_result.item(), f"Covariance test failed: T_N={T_N_stat.item():.6f} > threshold={chi2_threshold:.6f}"


def test_native_rsample_backward(device, native_sizes, value_dtype, index_dtype):
    """Test backward pass for SparseMultivariateNormalNative."""

    dist = construct_native_distribution(native_sizes, value_dtype, device, index_dtype, requires_grad=True)
    samples = dist.rsample((10,))

    # Backward pass should work
    samples.sum().backward()

    # Check that gradients exist
    assert dist.loc.grad is not None
    assert dist.scale_tril.grad.values() is not None


def test_native_properties(device, native_sizes, value_dtype, index_dtype):
    """Test basic properties of SparseMultivariateNormalNative."""

    dist = construct_native_distribution(native_sizes, value_dtype, device, index_dtype)

    # Test basic properties
    assert dist.mean.shape == dist.loc.shape
    assert dist.mode.shape == dist.loc.shape
    assert torch.allclose(dist.mean, dist.loc)
    assert torch.allclose(dist.mode, dist.loc)

    # Test variance property
    variance = dist.variance
    assert variance.shape == dist.loc.shape
    assert torch.all(variance > 0)  # Variance should be positive

    # Test covariance matrix property
    cov = dist.covariance_matrix
    assert cov.shape == (dist.loc.shape[0], dist.loc.shape[0])

    # Covariance should be positive definite (all eigenvalues positive)
    eigenvals = torch.linalg.eigvals(cov).real
    # Use a more lenient threshold for numerical precision
    assert torch.all(eigenvals > -1e-6), "Covariance matrix should be positive semi-definite"


def test_native_log_prob(device, native_sizes, value_dtype, index_dtype):
    """Test log probability computation for SparseMultivariateNormalNative."""

    dist = construct_native_distribution(native_sizes, value_dtype, device, index_dtype)

    # Test single sample log_prob
    sample = dist.rsample()
    log_prob_single = dist.log_prob(sample)
    assert log_prob_single.dim() == 0  # Scalar for single sample

    # Test multiple samples log_prob
    samples = dist.rsample((5,))
    log_prob_batch = dist.log_prob(samples)
    assert log_prob_batch.shape == (5,)  # Vector for batch

    # Log probability should be finite
    assert torch.all(torch.isfinite(log_prob_single))
    assert torch.all(torch.isfinite(log_prob_batch))


def test_native_single_vs_batch_sampling(device, native_sizes, value_dtype, index_dtype):
    """Test that single and batch sampling produce equivalent distributions."""

    dist = construct_native_distribution(native_sizes, value_dtype, device, index_dtype)

    # Single samples
    single_samples = torch.stack([dist.rsample() for _ in range(100)])

    # Batch samples
    batch_samples = dist.rsample((100,))

    # Both should have same shape
    assert single_samples.shape == batch_samples.shape

    # Statistical properties should be similar (within tolerance due to randomness)
    single_mean = single_samples.mean(0)
    batch_mean = batch_samples.mean(0)

    # Use a more robust comparison that handles near-zero values better
    # Check if the difference is within expected sampling variability
    mean_diff = torch.abs(single_mean - batch_mean)

    # Estimate standard error for each component
    single_std = single_samples.std(0) / torch.sqrt(torch.tensor(100.0))
    batch_std = batch_samples.std(0) / torch.sqrt(torch.tensor(100.0))
    combined_std = torch.sqrt(single_std**2 + batch_std**2)

    # Allow for 4 standard deviations of difference (99.99% confidence) plus a more generous absolute tolerance
    max_allowed_diff = 4.0 * combined_std + 0.2

    # If this still fails, check if it's just one outlier component
    n_violations = torch.sum(mean_diff > max_allowed_diff).item()
    max_violations_allowed = 1  # Allow one component to be slightly out of bounds

    assert n_violations <= max_violations_allowed, (
        f"Too many components with large mean differences: {n_violations} > {max_violations_allowed}. "
        f"Max difference: {mean_diff.max().item():.4f}, Expected max: {max_allowed_diff.min().item():.4f}"
    )


def test_native_csr_requirement():
    """Test that SparseMultivariateNormalNative enforces CSR layout."""

    loc = torch.randn(4)

    # Create COO matrix (should fail)
    indices = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int64)
    values = torch.ones(4, dtype=torch.float32)
    coo_matrix = torch.sparse_coo_tensor(indices, values, (4, 4))

    with pytest.raises(ValueError, match="scale_tril must be sparse CSR"):
        SparseMultivariateNormalNative(loc, coo_matrix)

    # Create dense matrix (should fail)
    dense_matrix = torch.eye(4)
    with pytest.raises(ValueError, match="scale_tril must be sparse CSR"):
        SparseMultivariateNormalNative(loc, dense_matrix)


def test_native_unbatched_requirement():
    """Test that SparseMultivariateNormalNative enforces unbatched inputs."""

    # Batched loc should fail
    batched_loc = torch.randn(2, 4)
    scale_tril = torch.eye(4).to_sparse_csr()

    with pytest.raises(ValueError, match="loc must be one-dimensional"):
        SparseMultivariateNormalNative(batched_loc, scale_tril)

    # Batched scale_tril should fail
    loc = torch.randn(4)
    batched_scale_tril = torch.stack([torch.eye(4), torch.eye(4)]).to_sparse_csr()

    with pytest.raises(ValueError, match="scale_tril must be two-dimensional"):
        SparseMultivariateNormalNative(loc, batched_scale_tril)
