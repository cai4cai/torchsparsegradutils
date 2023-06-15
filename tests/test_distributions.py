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
    ("bat", 4, 16, 0.5),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
SPASRE_LAYOUTS = [torch.sparse_coo, torch.sparse_csr]
VARS = ["cov", "prec"]  # whether to encode lower triangular covariance or precision matrix for distribution

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


def var_id(var):
    return var


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


@pytest.fixture(params=VARS, ids=[var_id(v) for v in VARS])
def var(request):
    return request.param


# @pytest.fixture(params=DISTRIBUTIONS, ids=[dist_id(d) for d in DISTRIBUTIONS])
# def distribution(request):
#     return request.param


# Convenience functions:


def construct_distribution(sizes, layout, var, value_dtype, index_dtype, device):
    _, batch_size, event_size, sparsity = sizes
    loc = torch.randn(event_size, device=device, dtype=value_dtype)
    diag = torch.rand(event_size, device=device, dtype=value_dtype)

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
    if var == "cov":
        return SparseMultivariateNormal(loc, diag, scale_tril=tril)
    elif var == "prec":
        return SparseMultivariateNormal(loc, diag, precision_tril=tril)
    else:
        raise ValueError(f"tril must be one of 'cov' or 'prec', but got {tril}")


# Define Tests


def test_rsample_forward(device, layout, var, sizes, value_dtype, index_dtype):
    if layout == torch.sparse_coo and index_dtype == torch.int32:
        pytest.skip("Sparse COO with int32 indices is not supported")

    dist = construct_distribution(sizes, layout, var, value_dtype, index_dtype, device)
    samples = dist.rsample((10000,))

    if var == "cov":
        scale_tril = dist.scale_tril.to_dense()
        # scale_tril_with_diag = scale_tril + torch.diag_embed(dist.diagonal)
        covariance = torch.matmul(scale_tril, scale_tril.t()) + torch.diag_embed(dist.diagonal)
    else:
        precision_tril = dist.precision_tril.to_dense()
        covariance = torch.inverse(torch.matmul(precision_tril, precision_tril.transpose(-1, -2)))

    # TODO: getting closer but still not there
    assert torch.allclose(samples.mean(0), dist.loc, atol=0.1)
    assert torch.allclose(torch.cov(samples.T), covariance, atol=0.1)


# def test_rsample_backward(device, layout, var, sizes, value_dtype, index_dtype):
#     pass


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