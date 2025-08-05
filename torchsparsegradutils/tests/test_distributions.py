import torch
import pytest

from torch.distributions.multivariate_normal import _batch_mv
from torchsparsegradutils.distributions.sparse_multivariate_normal import _batch_sparse_mv
from torchsparsegradutils.distributions import SparseMultivariateNormal
from torchsparsegradutils.utils import rand_sparse_tri
from torchsparsegradutils import sparse_mm, sparse_triangular_solve
from .dist_stats_helpers import mean_hotelling_t2_test, cov_nagao_test

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
