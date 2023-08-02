import torch
import unittest
import torchsparsegradutils as tsgu
import torchsparsegradutils.jax as tsgujax

if not tsgujax.have_jax:
    raise unittest.SkipTest("Importing optional jax-related module failed to find jax -> skipping jax-related tests.")

import numpy as np

import jax
import jax.numpy as jnp


class J2TIOTest(unittest.TestCase):
    """IO conversion tests between torch and jax"""

    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
            self.device_j = jax.devices("cpu")[0]
        self.x_shape = (4, 2)
        self.x_t = torch.randn(self.x_shape, dtype=torch.float64, device=self.device)
        rng = np.random.default_rng()
        x_n = rng.standard_normal(self.x_shape, dtype=np.float64)
        # Note that by default jax does not support 64bit precision
        # https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
        self.x_j = jax.device_put(x_n, device=self.device_j)
        self.A_shape = (4, 4)
        self.A = torch.randn(self.A_shape, dtype=torch.float64, device=self.device).relu().to_sparse_csr()

    def test_t2j(self):
        x_j = tsgujax.t2j(self.x_t)
        self.assertTrue(x_j.shape == self.x_t.shape)
        self.assertTrue(np.isclose(np.asarray(x_j), self.x_t.cpu().numpy()).all())
        x_t2j2t = tsgujax.j2t(x_j)
        self.assertTrue(x_t2j2t.shape == self.x_t.shape)
        self.assertTrue(np.isclose(x_t2j2t.cpu().numpy(), self.x_t.cpu().numpy()).all())

    def test_j2t(self):
        x_t = tsgujax.j2t(self.x_j)
        self.assertTrue(x_t.shape == self.x_j.shape)
        self.assertTrue(np.isclose(np.asarray(self.x_j), x_t.cpu().numpy()).all())
        x_j2t2j = tsgujax.t2j(x_t)
        self.assertTrue(x_j2t2j.shape == self.x_j.shape)
        self.assertTrue(np.isclose(np.asarray(x_j2t2j), np.asarray(self.x_j)).all())

    def test_spmm_t4j(self):
        if jax.config.jax_enable_x64:
            A_t = self.A
        else:
            A_t = self.A.to(torch.float32)
        A_j = tsgujax.spmm_t4j(A_t)
        Ax_j = A_j(self.x_j)
        x_t = tsgujax.j2t(self.x_j)
        Ax_t = A_t @ x_t
        self.assertTrue(Ax_j.shape == Ax_t.shape)
        self.assertTrue(np.isclose(np.asarray(Ax_j), Ax_t.cpu().numpy()).all())

    def test_t2j_coo(self):
        x_t_coo = self.x_t.to_sparse_coo()
        x_j_coo = tsgujax.t2j_coo(x_t_coo)
        self.assertTrue(x_j_coo.shape == x_t_coo.shape)
        self.assertTrue(np.isclose(np.asarray(x_j_coo.todense()), x_t_coo.to_dense().cpu().numpy()).all())
        x_t2j2t_coo = tsgujax.j2t_coo(x_j_coo)
        self.assertTrue(x_t2j2t_coo.shape == x_t_coo.shape)
        self.assertTrue(np.isclose(x_t2j2t_coo.to_dense().cpu().numpy(), x_t_coo.to_dense().cpu().numpy()).all())

    def test_j2t_coo(self):
        x_j_coo = jax.experimental.sparse.COO.fromdense(self.x_j)
        x_t_coo = tsgujax.j2t_coo(x_j_coo)
        self.assertTrue(x_j_coo.shape == x_t_coo.shape)
        self.assertTrue(np.isclose(np.asarray(x_j_coo.todense()), x_t_coo.to_dense().cpu().numpy()).all())
        x_j2t2j_coo = tsgujax.t2j_coo(x_t_coo)
        self.assertTrue(x_j2t2j_coo.shape == x_j_coo.shape)
        self.assertTrue(np.isclose(np.asarray(x_j2t2j_coo.todense()), np.asarray(x_j_coo.todense())).all())

    def test_t2j_csr(self):
        x_t_csr = self.x_t.to_sparse_csr()
        x_j_csr = tsgujax.t2j_csr(x_t_csr)
        self.assertTrue(x_j_csr.shape == x_t_csr.shape)
        self.assertTrue(np.isclose(np.asarray(x_j_csr.todense()), x_t_csr.to_dense().cpu().numpy()).all())
        x_t2j2t_csr = tsgujax.j2t_csr(x_j_csr)
        self.assertTrue(x_t2j2t_csr.shape == x_t_csr.shape)
        self.assertTrue(np.isclose(x_t2j2t_csr.to_dense().cpu().numpy(), x_t_csr.to_dense().cpu().numpy()).all())

    def test_j2t_csr(self):
        x_j_csr = jax.experimental.sparse.CSR.fromdense(self.x_j)
        x_t_csr = tsgujax.j2t_csr(x_j_csr)
        self.assertTrue(x_j_csr.shape == x_t_csr.shape)
        self.assertTrue(np.isclose(np.asarray(x_j_csr.todense()), x_t_csr.to_dense().cpu().numpy()).all())
        x_j2t2j_csr = tsgujax.t2j_csr(x_t_csr)
        self.assertTrue(x_j2t2j_csr.shape == x_j_csr.shape)
        self.assertTrue(np.isclose(np.asarray(x_j2t2j_csr.todense()), np.asarray(x_j_csr.todense())).all())


class J2TIOTestCUDA(J2TIOTest):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        self.device_j = jax.devices("gpu")[0]
        super().setUp()


if __name__ == "__main__":
    unittest.main()
