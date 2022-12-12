import torch
import unittest

from torchsparsegradutils import sparse_generic_lstsq


class SparseGenericLstsqTest(unittest.TestCase):
    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")

        self.RTOL = 1e-2

        self.A_shape = (7, 4)
        self.A = torch.randn(self.A_shape, dtype=torch.float64, device=self.device)
        self.A_csr = self.A.to_sparse_csr()
        self.B_shape = (7, 1)
        self.B = torch.randn(self.B_shape, dtype=torch.float64, device=self.device)

        self.x_ref = torch.linalg.lstsq(self.A, self.B).solution

    def test_generic_lstsq_default(self):
        x = sparse_generic_lstsq(self.A_csr, self.B)
        # print("x",x)
        # print("self.x_ref",self.x_ref)
        self.assertTrue(torch.isclose(x, self.x_ref, rtol=self.RTOL).all())

    def test_generic_lstsq_gradient_default(self):
        # Sparse lstsq:
        As1 = self.A_csr.detach().clone()
        As1.requires_grad = True
        Bd1 = self.B.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = sparse_generic_lstsq(As1, Bd1)
        loss = x.sum()
        loss.backward()

        # torch dense lstsq:
        Ad2 = self.A.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.B.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.lstsq(Ad2, Bd2).solution
        loss_torch = x2.sum()
        loss_torch.backward()

        # print("x",x)
        # print("x2",x2)

        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

        # print("Bd1.grad",Bd1.grad)
        # print("Bd2.grad",Bd2.grad)
        # print("As1.grad.to_dense()",As1.grad.to_dense())
        # print("Ad2.grad",Ad2.grad)

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())


class SparseGenericLstsqTestCUDA(SparseGenericLstsqTest):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        super().setUp()


if __name__ == "__main__":
    unittest.main()
