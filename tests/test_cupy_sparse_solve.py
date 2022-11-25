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


class SparseSolveTestC4T(unittest.TestCase):
    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
            self.xp = np
            self.xsp = nsp

        self.RTOL = 1e-3

        self.A_shape = (4, 4)
        self.A = torch.randn(self.A_shape, dtype=torch.float64, device=self.device)
        self.A = self.A + self.A.t()
        self.A_csr = self.A.to_sparse_csr()
        self.B_shape = (4, 2)
        self.B = torch.randn(self.B_shape, dtype=torch.float64, device=self.device)

        self.x_ref = torch.linalg.solve(self.A, self.B)

    def test_solver_c4t(self):
        x = tsgucupy.sparse_solve_c4t(self.A_csr.to(torch.float32), self.B.to(torch.float32))
        self.assertTrue(torch.isclose(x, self.x_ref.to(torch.float32), rtol=self.RTOL).all())

    def test_solver_gradient_c4t(self):
        # Sparse solver:
        As1 = self.A_csr.detach().to(torch.float32).clone()
        As1.requires_grad = True
        Bd1 = self.B.detach().to(torch.float32).clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = tsgucupy.sparse_solve_c4t(As1, Bd1)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.A.detach().to(torch.float32).clone()
        Ad2.requires_grad = True
        Bd2 = self.B.detach().to(torch.float32).clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve(Ad2, Bd2)
        loss_torch = x2.sum()
        loss_torch.backward()

        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())


class SparseSolveTestC4TCUDA(SparseSolveTestC4T):
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
