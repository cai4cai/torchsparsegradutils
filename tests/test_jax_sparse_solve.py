import torch
import unittest
import torchsparsegradutils as tsgu
import torchsparsegradutils.jax as tsgujax

if not tsgujax.have_jax:
    raise unittest.SkipTest("Importing optional jax-related module failed to find jax -> skipping jax-related tests.")

import numpy as np

import jax
import jax.numpy as jnp


class SparseSolveTestJ4T(unittest.TestCase):
    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")

        self.RTOL = 1e-2

        self.A_shape = (4, 4)
        self.A = torch.randn(self.A_shape, dtype=torch.float64, device=self.device)
        self.A = self.A + self.A.t()
        self.A_csr = self.A.to_sparse_csr()
        self.B_shape = (4, 2)
        self.B = torch.randn(self.B_shape, dtype=torch.float64, device=self.device)

        self.x_ref = torch.linalg.solve(self.A, self.B)

    def test_solver_j4t(self):
        x = tsgujax.sparse_solve_j4t(self.A_csr.to(torch.float32), self.B.to(torch.float32))
        self.assertTrue(torch.isclose(x, self.x_ref.to(torch.float32), rtol=self.RTOL).all())

    def test_solver_gradient_j4t(self):
        # Sparse solver:
        As1 = self.A_csr.detach().to(torch.float32).clone()
        As1.requires_grad = True
        Bd1 = self.B.detach().to(torch.float32).clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = tsgujax.sparse_solve_j4t(As1, Bd1)
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


class SparseSolveTestJ4TCUDA(SparseSolveTestJ4T):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        super().setUp()


if __name__ == "__main__":
    unittest.main()
