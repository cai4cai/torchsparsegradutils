import warnings

import numpy as np
import pytest
import scipy.sparse as nsp
import torch

import torchsparsegradutils.cupy as tsgucupy
from torchsparsegradutils.utils import rand_sparse

# Device fixture
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]


def _id_device(d):
    return str(d)


def _id_dtype(dtype):
    return str(dtype).split(".")[-1]


@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param


@pytest.fixture(params=INDEX_DTYPES, ids=_id_dtype)
def index_dtype(request):
    return request.param


@pytest.fixture(params=VALUE_DTYPES, ids=_id_dtype)
def value_dtype(request):
    return request.param


# Optional imports for Cupy
if tsgucupy.have_cupy:
    import cupy as cp
    import cupyx.scipy.sparse as csp


# Helper to convert Cupy array to NumPy
def _c2n(x_cupy):
    return cp.asnumpy(x_cupy) if tsgucupy.have_cupy else np.asarray(x_cupy)


# I/O setup fixture
@pytest.fixture
def cupy_bindings_io(device, index_dtype, value_dtype):
    x_shape = (4, 6)
    nnz = 8
    # Use rand_sparse to create a sparse tensor with actual zero values
    x_t = rand_sparse(
        x_shape, nnz, torch.sparse_coo, indices_dtype=index_dtype, values_dtype=value_dtype, device=device
    )

    if tsgucupy.have_cupy and device.type == "cuda":
        xp, xsp = cp, csp
    else:
        xp, xsp = np, nsp

    # Convert to dense, then to numpy/cupy for creating sparse matrices
    x_dense = x_t.to_dense()
    x_c = xp.asarray(x_dense.cpu().numpy())

    return x_t, x_c, xsp


# Test torch -> CuPy/NumPy COO conversion and back
def test_t2c_and_c2t_coo(device, cupy_bindings_io):
    x_t, x_c, xsp = cupy_bindings_io
    # torch to COO sparse Cupy
    x_t_coo = x_t.to_sparse_coo()
    x_c_coo = tsgucupy.t2c_coo(x_t_coo)
    assert x_c_coo.shape == x_t_coo.shape
    assert np.allclose(_c2n(x_c_coo.todense()), x_t_coo.to_dense().cpu().numpy())
    # back to torch
    x_t2 = tsgucupy.c2t_coo(x_c_coo)
    assert x_t2.shape == x_t_coo.shape
    assert np.allclose(x_t2.to_dense().cpu().numpy(), x_t_coo.to_dense().cpu().numpy())


# Test NumPy/CuPy -> torch COO conversion and back
def test_c2t_and_t2c_coo(device, cupy_bindings_io):
    x_t, x_c, xsp = cupy_bindings_io
    x_c_coo = xsp.coo_matrix(x_c)
    x_t_coo = tsgucupy.c2t_coo(x_c_coo)
    assert x_t_coo.shape == x_c_coo.shape
    assert np.allclose(_c2n(x_c_coo.todense()), x_t_coo.to_dense().cpu().numpy())
    x_c2 = tsgucupy.t2c_coo(x_t_coo)
    assert x_c2.shape == x_c_coo.shape
    assert np.allclose(_c2n(x_c2.todense()), _c2n(x_c_coo.todense()))


# Test torch -> CuPy/NumPy CSR conversion and back
def test_t2c_and_c2t_csr(device, cupy_bindings_io):
    x_t, x_c, xsp = cupy_bindings_io
    x_t_csr = x_t.to_sparse_csr()
    x_c_csr = tsgucupy.t2c_csr(x_t_csr)
    assert x_c_csr.shape == x_t_csr.shape
    assert np.allclose(_c2n(x_c_csr.todense()), x_t_csr.to_dense().cpu().numpy())
    x_t2 = tsgucupy.c2t_csr(x_c_csr)
    assert x_t2.shape == x_t_csr.shape
    assert np.allclose(x_t2.to_dense().cpu().numpy(), x_t_csr.to_dense().cpu().numpy())


# Test NumPy/CuPy -> torch CSR conversion and back
def test_c2t_and_t2c_csr(device, cupy_bindings_io):
    x_t, x_c, xsp = cupy_bindings_io
    x_c_csr = xsp.csr_matrix(x_c)
    x_t_csr = tsgucupy.c2t_csr(x_c_csr)
    assert x_t_csr.shape == x_c_csr.shape
    assert np.allclose(_c2n(x_c_csr.todense()), x_t_csr.to_dense().cpu().numpy())
    x_c2 = tsgucupy.t2c_csr(x_t_csr)
    assert x_c2.shape == x_c_csr.shape
    assert np.allclose(_c2n(x_c2.todense()), _c2n(x_c_csr.todense()))
