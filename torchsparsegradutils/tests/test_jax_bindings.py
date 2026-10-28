import numpy as np
import pytest
import torch
from test_config import DEVICES

import torchsparsegradutils.jax as tsgujax

# skip if JAX unavailable
jax = pytest.importorskip("jax")
if not tsgujax.have_jax:
    pytest.skip("JAX bindings unavailable, skipping jax tests", allow_module_level=True)


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


@pytest.mark.parametrize("layout", [torch.sparse_coo, torch.sparse_csr])
@pytest.mark.parametrize("pattern", ["column", "slice", "transpose"])
def test_spmm_t4j_compat_preserves_converted_operand(device, layout, pattern, monkeypatch):
    import importlib

    bindings = importlib.import_module("torchsparsegradutils.jax.jax_bindings")
    jax_device = jax.devices("cpu" if device.type == "cpu" else "gpu")[0]
    data = jax.device_put(np.arange(1, 13, dtype=np.float32).reshape(3, 4), jax_device)
    x = data[:, 1:2] if pattern == "column" else data[:, ::2] if pattern == "slice" else data.T
    dense_a = torch.arange(1, 1 + 2 * x.shape[0], device=device, dtype=torch.float32).reshape(2, x.shape[0])
    a = dense_a.to_sparse_coo() if layout == torch.sparse_coo else dense_a.to_sparse_csr()
    converted, operands = [], []
    convert, native_mm = bindings.j2t, torch.sparse.mm

    def record_conversion(value):
        result = convert(value)
        converted.append(result)
        return result

    def record_mm(sparse, dense):
        operands.append(dense)
        return native_mm(sparse, dense)

    monkeypatch.setattr(bindings, "j2t", record_conversion)
    monkeypatch.setattr(torch.sparse, "mm", record_mm)
    result = bindings.spmm_t4j(a)(x)
    assert len(converted) == len(operands) == 1
    torch.testing.assert_close(convert(result), dense_a @ converted[0])
    # These real JAX conversions are contiguous and need no workaround copy.
    assert converted[0].is_contiguous()
    assert operands[0].data_ptr() == converted[0].data_ptr()
    assert operands[0].stride() == converted[0].stride()
