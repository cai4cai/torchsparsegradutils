import torch
import pytest

from torch.distributions.multivariate_normal import _batch_mv
from torchsparsegradutils.distributions.sparse_multivariate_normal import _batch_sparse_mv
from torchsparsegradutils.distributions import SparseMultivariateNormal
from torchsparsegradutils.utils import rand_sparse_tri
from torchsparsegradutils import sparse_mm, sparse_triangular_solve

# Identify Testing Parameters
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

TEST_DATA = [
    # name, batch size, event size, spartsity
    ("unbat", None, 4, 0.5),
    ("bat", 4, 4, 0.5),
    ("bat2", 4, 64, 0.01),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
SPASRE_LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

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


# def dist_id(dist):
#     return dist.__name__


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


# @pytest.fixture(params=DISTRIBUTIONS, ids=[dist_id(d) for d in DISTRIBUTIONS])
# def distribution(request):
#     return request.param


# Convenience functions:


def construct_distribution(sizes, layout, var, value_dtype, index_dtype, device, requires_grad=False):
    _, batch_size, event_size, sparsity = sizes
    loc = torch.randn(event_size, device=device, dtype=value_dtype, requires_grad=requires_grad)
    diag = torch.rand(event_size, device=device, dtype=value_dtype, requires_grad=requires_grad)

    tril_size = (batch_size, event_size, event_size) if batch_size else (event_size, event_size)
    nnz = int(sparsity * event_size * (event_size + 1) / 2)

    tril = rand_sparse_tri(
        tril_size,
        nnz,
        layout,
        upper=False,
        strict=True,
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
    )

    tril.requires_grad = requires_grad

    if var == "cov":
        return SparseMultivariateNormal(loc, diag, scale_tril=tril)
    elif var == "prec":
        return SparseMultivariateNormal(loc, diag, precision_tril=tril)
    else:
        raise ValueError(f"tril must be one of 'cov' or 'prec', but got {tril}")


def check_covariance_within_tolerance(
    covariance_test, covariance_ref, absolute_tolerance=None, relative_tolerance=None, desired_threshold=0
):
    # Calculate absolute difference if absolute_tolerance is provided
    if absolute_tolerance is not None:
        abs_diff = torch.abs(covariance_test - covariance_ref)
        abs_within_tolerance = abs_diff <= absolute_tolerance
    else:
        abs_diff = None
        abs_within_tolerance = torch.ones_like(covariance_test, dtype=torch.bool)

    # Calculate relative difference if relative_tolerance is provided
    if relative_tolerance is not None:
        if abs_diff is None:
            abs_diff = torch.abs(covariance_test - covariance_ref)
        rel_diff = abs_diff / torch.abs(covariance_ref)
        rel_within_tolerance = rel_diff <= relative_tolerance
    else:
        rel_within_tolerance = torch.ones_like(covariance_test, dtype=torch.bool)

    # Determine values within both absolute and relative tolerance
    within_tolerance = abs_within_tolerance & rel_within_tolerance

    # Calculate percentage within tolerance
    percentage_within_tolerance = (within_tolerance.sum() / within_tolerance.numel()) * 100

    # Check if the percentage meets the desired threshold
    is_within_desired_threshold = percentage_within_tolerance >= desired_threshold

    print(f"Percentage within tolerance: {percentage_within_tolerance.item():.2f}%")
    print(f"Is within desired threshold? {is_within_desired_threshold}")

    assert is_within_desired_threshold


# Define Tests


@pytest.mark.flaky(reruns=5)  # probably not needed, but seems sensible for CI
def test_rsample_forward_cov(device, layout, sizes, value_dtype, index_dtype):
    if layout == torch.sparse_coo and index_dtype == torch.int32:
        pytest.skip("Sparse COO with int32 indices is not supported")

    dist = construct_distribution(sizes, layout, "cov", value_dtype, index_dtype, device)
    samples = dist.rsample((100000,))

    scale_tril = dist.scale_tril.to_dense()
    scale_tril = scale_tril + torch.eye(*dist.event_shape, dtype=scale_tril.dtype, device=scale_tril.device)  # L matrix
    diagonal = dist.diagonal  # D matrix
    # Compute covariance from LDL^T decomposition
    covariance_ref = torch.matmul(scale_tril @ torch.diag_embed(diagonal), scale_tril.transpose(-1, -2))

    assert torch.allclose(samples.mean(0), dist.loc, atol=0.1)

    if len(samples.shape) == 2:
        covariance_test = torch.cov(samples.T)
    else:
        covariance_test = torch.stack([torch.cov(sample.T) for sample in samples.permute(1, 0, 2)])

    check_covariance_within_tolerance(covariance_test, covariance_ref, absolute_tolerance=0.1, desired_threshold=99.0)


@pytest.mark.flaky(reruns=5)
# NOTE: This test often failes, hence the flaky and reruns
def test_rsample_forward_prec(device, layout, sizes, value_dtype, index_dtype):
    if layout == torch.sparse_coo and index_dtype == torch.int32:
        pytest.skip("Sparse COO with int32 indices is not supported")

    dist = construct_distribution(sizes, layout, "prec", value_dtype, index_dtype, device)
    samples = dist.rsample((100000,))

    precision_tril = dist.precision_tril.to_dense()
    precision_tril = precision_tril + torch.eye(
        *dist.event_shape, dtype=precision_tril.dtype, device=precision_tril.device
    )  # L matrix
    diagonal = dist.diagonal  # D matrix
    # Compute precision matrix from LDL^T decomposition
    precision_ref = torch.matmul(precision_tril @ torch.diag_embed(diagonal), precision_tril.transpose(-1, -2))
    # Compute covariance from precision
    covariance_ref = torch.linalg.inv(precision_ref)

    assert torch.allclose(samples.mean(0), dist.loc, atol=0.1)

    if len(samples.shape) == 2:
        covariance_test = torch.cov(samples.T)
    else:
        covariance_test = torch.stack([torch.cov(sample.T) for sample in samples.permute(1, 0, 2)])

    # NOTE: higher atol due to larger covariance values after inversion
    # NOTE: lower threshold due to numerical instability of inversion
    check_covariance_within_tolerance(covariance_test, covariance_ref, absolute_tolerance=1, desired_threshold=95.0)


def test_rsample_backward_cov(device, layout, sizes, value_dtype, index_dtype):
    if layout == torch.sparse_coo and index_dtype == torch.int32:
        pytest.skip("Sparse COO with int32 indices is not supported")

    dist = construct_distribution(sizes, layout, "cov", value_dtype, index_dtype, device, requires_grad=True)
    samples = dist.rsample((10,))

    samples.sum().backward()


def test_rsample_backward_prec(device, layout, sizes, value_dtype, index_dtype):
    if layout == torch.sparse_coo and index_dtype == torch.int32:
        pytest.skip("Sparse COO with int32 indices is not supported")

    dist = construct_distribution(sizes, layout, "prec", value_dtype, index_dtype, device, requires_grad=True)
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
