import torch
import unittest
import torchsparsegradutils as tsgu
import torchsparsegradutils.cupy as tsgucupy

import warnings

if tsgucupy.have_cupy:
    import cupy as cp
    import cupyx.scipy.sparse as csp
else:
    warnings.warn(
        "Importing optional cupy-related module failed to find cupy -> cupy-related tests running as numpy only."
    )

import numpy as np
import scipy.sparse as nsp


def _c2n(x_cupy):
    if tsgucupy.have_cupy:
        return cp.asnumpy(x_cupy)
    else:
        return np.asarray(x_cupy)


class C2TIOTest(unittest.TestCase):
    """IO conversion tests between torch and cupy"""

    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
            self.xp = np
            self.xsp = nsp

        self.x_shape = (4, 2)
        self.x_t = torch.randn(self.x_shape, dtype=torch.float64, device=self.device)
        rng = np.random.default_rng()
        x_n = rng.standard_normal(self.x_shape, dtype=np.float64)

        self.x_c = self.xp.asarray(x_n)
        self.A_shape = (4, 4)
        self.A = torch.randn(self.A_shape, dtype=torch.float64, device=self.device).relu().to_sparse_csr()

    def test_t2c_coo(self):
        x_t_coo = self.x_t.to_sparse_coo()
        x_c_coo = tsgucupy.t2c_coo(x_t_coo)
        self.assertTrue(x_c_coo.shape == x_t_coo.shape)
        self.assertTrue(np.isclose(_c2n(x_c_coo.todense()), x_t_coo.to_dense().cpu().numpy()).all())
        x_t2c2t_coo = tsgucupy.c2t_coo(x_c_coo)
        self.assertTrue(x_t2c2t_coo.shape == x_t_coo.shape)
        self.assertTrue(np.isclose(x_t2c2t_coo.to_dense().cpu().numpy(), x_t_coo.to_dense().cpu().numpy()).all())

    def test_c2t_coo(self):
        x_c_coo = self.xsp.coo_matrix(self.x_c)
        x_t_coo = tsgucupy.c2t_coo(x_c_coo)
        self.assertTrue(x_c_coo.shape == x_t_coo.shape)
        self.assertTrue(np.isclose(_c2n(x_c_coo.todense()), x_t_coo.to_dense().cpu().numpy()).all())
        x_c2t2c_coo = tsgucupy.t2c_coo(x_t_coo)
        self.assertTrue(x_c2t2c_coo.shape == x_c_coo.shape)
        self.assertTrue(np.isclose(_c2n(x_c2t2c_coo.todense()), _c2n(x_c_coo.todense())).all())

    def test_t2c_csr(self):
        x_t_csr = self.x_t.to_sparse_csr()
        x_c_csr = tsgucupy.t2c_csr(x_t_csr)
        self.assertTrue(x_c_csr.shape == x_t_csr.shape)
        self.assertTrue(np.isclose(_c2n(x_c_csr.todense()), x_t_csr.to_dense().cpu().numpy()).all())
        x_t2c2t_csr = tsgucupy.c2t_csr(x_c_csr)
        self.assertTrue(x_t2c2t_csr.shape == x_t_csr.shape)
        self.assertTrue(np.isclose(x_t2c2t_csr.to_dense().cpu().numpy(), x_t_csr.to_dense().cpu().numpy()).all())

    def test_c2t_csr(self):
        x_c_csr = self.xsp.csr_matrix(self.x_c)
        x_t_csr = tsgucupy.c2t_csr(x_c_csr)
        self.assertTrue(x_c_csr.shape == x_t_csr.shape)
        self.assertTrue(np.isclose(_c2n(x_c_csr.todense()), x_t_csr.to_dense().cpu().numpy()).all())
        x_c2t2c_csr = tsgucupy.t2c_csr(x_t_csr)
        self.assertTrue(x_c2t2c_csr.shape == x_c_csr.shape)
        self.assertTrue(np.isclose(_c2n(x_c2t2c_csr.todense()), _c2n(x_c_csr.todense())).all())


class C2TIOTestCUDA(C2TIOTest):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        self.xp = cp
        self.xsp = csp
        super().setUp()


if __name__ == "__main__":
    unittest.main()
