import jax
import numpy as np
import pytest
import torch

import torchsparsegradutils as tsgu
import torchsparsegradutils.jax as tsgujax

# skip if JAX unavailable
pytest.importorskip("jax")
if not tsgujax.have_jax:
    pytest.skip("JAX bindings unavailable, skipping jax tests", allow_module_level=True)

import jax.numpy as jnp

# Device fixture
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))


def _id_device(d):
    return str(d)


@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param


@pytest.fixture
def jax_io_setup(device):
    # prepare torch and jax inputs
    device_j = jax.devices("cpu")[0] if device.type == "cpu" else jax.devices("gpu")[0]
    x_shape = (4, 2)
    x_t = torch.randn(x_shape, dtype=torch.float64, device=device)
    rng = np.random.default_rng()
    x_n = rng.standard_normal(x_shape, dtype=np.float64)
    x_j = jax.device_put(x_n, device=device_j)
    A_shape = (4, 4)
    A = torch.randn(A_shape, dtype=torch.float64, device=device).relu().to_sparse_csr()
    return x_t, x_j, A


# Test conversions between torch and jax
def test_t2j_and_roundtrip(device, jax_io_setup):
    x_t, x_j, A = jax_io_setup
    # torch to jax and back
    x_j1 = tsgujax.t2j(x_t)
    assert x_j1.shape == x_t.shape
    assert np.allclose(np.asarray(x_j1), x_t.cpu().numpy())
    x_t1 = tsgujax.j2t(x_j1)
    assert x_t1.shape == x_t.shape
    assert np.allclose(x_t1.cpu().numpy(), x_t.cpu().numpy())


def test_j2t_and_roundtrip(device, jax_io_setup):
    x_t, x_j, A = jax_io_setup
    # jax to torch and back
    x_t1 = tsgujax.j2t(x_j)
    assert x_t1.shape == x_j.shape
    assert np.allclose(np.asarray(x_j), x_t1.cpu().numpy())
    x_j1 = tsgujax.t2j(x_t1)
    assert x_j1.shape == x_j.shape
    assert np.allclose(np.asarray(x_j1), np.asarray(x_j))


def test_spmm_t4j(device, jax_io_setup):
    x_t, x_j, A = jax_io_setup
    # select dtype
    A_t = A if jax.config.jax_enable_x64 else A.to(torch.float32)
    A_j = tsgujax.spmm_t4j(A_t)
    Ax_j = A_j(x_j)
    x_t1 = tsgujax.j2t(x_j)
    Ax_t = A_t @ x_t1
    assert Ax_j.shape == Ax_t.shape
    assert np.allclose(np.asarray(Ax_j), Ax_t.cpu().numpy())


# Test COO conversion
def test_t2j_and_j2t_coo(device, jax_io_setup):
    x_t, x_j, A = jax_io_setup
    x_t_coo = x_t.to_sparse_coo()
    x_j_coo = tsgujax.t2j_coo(x_t_coo)
    assert x_j_coo.shape == x_t_coo.shape
    assert np.allclose(np.asarray(x_j_coo.todense()), x_t_coo.to_dense().cpu().numpy())
    x_t1_coo = tsgujax.j2t_coo(x_j_coo)
    assert x_t1_coo.shape == x_t_coo.shape
    assert np.allclose(x_t1_coo.to_dense().cpu().numpy(), x_t_coo.to_dense().cpu().numpy())


def test_j2t_and_t2j_coo(device, jax_io_setup):
    x_t, x_j, A = jax_io_setup
    x_j_coo = jax.experimental.sparse.COO.fromdense(x_j)
    x_t_coo = tsgujax.j2t_coo(x_j_coo)
    assert x_j_coo.shape == x_t_coo.shape
    assert np.allclose(np.asarray(x_j_coo.todense()), x_t_coo.to_dense().cpu().numpy())
    x_j1_coo = tsgujax.t2j_coo(x_t_coo)
    assert x_j1_coo.shape == x_j_coo.shape
    assert np.allclose(np.asarray(x_j1_coo.todense()), np.asarray(x_j_coo.todense()))


# Test CSR conversion
def test_t2j_and_j2t_csr(device, jax_io_setup):
    x_t, x_j, A = jax_io_setup
    x_t_csr = x_t.to_sparse_csr()
    x_j_csr = tsgujax.t2j_csr(x_t_csr)
    assert x_j_csr.shape == x_t_csr.shape
    assert np.allclose(np.asarray(x_j_csr.todense()), x_t_csr.to_dense().cpu().numpy())
    x_t1_csr = tsgujax.j2t_csr(x_j_csr)
    assert x_t1_csr.shape == x_t_csr.shape
    assert np.allclose(x_t1_csr.to_dense().cpu().numpy(), x_t_csr.to_dense().cpu().numpy())


def test_j2t_and_t2j_csr(device, jax_io_setup):
    x_t, x_j, A = jax_io_setup
    x_j_csr = jax.experimental.sparse.CSR.fromdense(x_j)
    x_t_csr = tsgujax.j2t_csr(x_j_csr)
    assert x_j_csr.shape == x_t_csr.shape
    assert np.allclose(np.asarray(x_j_csr.todense()), x_t_csr.to_dense().cpu().numpy())
    x_j1_csr = tsgujax.t2j_csr(x_t_csr)
    assert x_j1_csr.shape == x_j_csr.shape
    assert np.allclose(np.asarray(x_j1_csr.todense()), np.asarray(x_j_csr.todense()))
