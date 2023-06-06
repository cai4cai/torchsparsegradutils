import torch
import unittest
from parameterized import parameterized_class, parameterized
from random import randrange
from torchsparsegradutils import sparse_mm, sparse_bmm
from torchsparsegradutils.utils.random_sparse import generate_random_sparse_csr_matrix, generate_random_sparse_coo_matrix


def gencoordinates(nr, nc, ni, device="cuda"):
    """Used to genererate ni random unique coordinates for sparse matrix with size [nr, nc]"""
    coordinates = set()
    while True:
        r, c = randrange(nr), randrange(nc)
        coordinates.add((r, c))
        if len(coordinates) == ni:
            return torch.stack([torch.tensor(co) for co in coordinates], dim=-1).to(device)


class SparseMatMulTest(unittest.TestCase):
    """Test Sparse x Dense matrix multiplication with back propagation for COO and CSR matrices"""

    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
        self.A_shape = (8, 16)
        self.B_shape = (self.A_shape[1], 10)
        self.A_nnz = 32
        self.A_idx = gencoordinates(*self.A_shape, self.A_nnz, device=self.device)
        self.A_val = torch.randn(self.A_nnz, dtype=torch.float64, device=self.device)
        self.As_coo = torch.sparse_coo_tensor(self.A_idx, self.A_val, self.A_shape, requires_grad=True).coalesce()
        self.As_csr = self.As_coo.to_sparse_csr()
        self.Ad = self.As_coo.to_dense()

        self.Bd = torch.randn(*self.B_shape, dtype=torch.float64, requires_grad=True, device=self.device)
        self.matmul = sparse_mm

    def test_matmul_forward_coo(self):
        x = self.matmul(self.As_coo, self.Bd)
        self.assertIsInstance(x, torch.Tensor)

    def test_matmul_forward_csr(self):
        x = self.matmul(self.As_csr, self.Bd)
        self.assertIsInstance(x, torch.Tensor)

    def test_matmul_gradient_coo(self):
        # Sparse matmul:
        As1 = self.As_coo.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.matmul(As1, Bd1)
        loss = x.sum()
        loss.backward()

        # torch dense matmul:
        Ad2 = self.Ad.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x_torch = Ad2 @ Bd2
        loss_torch = x_torch.sum()
        loss_torch.backward()

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask]).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad).all())

    def test_matmul_gradient_csr(self):
        # Sparse solver:
        As1 = self.As_csr.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.matmul(As1, Bd1)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.Ad.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x_torch = Ad2 @ Bd2
        loss_torch = x_torch.sum()
        loss_torch.backward()
        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask]).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad).all())


class SparseMatMulTestCUDA(SparseMatMulTest):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        super().setUp()
        
        
@parameterized_class(
    (
        "name",
        "device",
    ),
    [
        ("CPU", torch.device("cpu")),
        (
            "CUDA",
            torch.device("cuda"),
        ),
    ],
)
class TestBatchedSparseMatMul(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
    
    
    @parameterized.expand([
        ("1", (4, 8, 16), (4, 16, 10), 32),
    ])
    def test_batched_sparse_matmul_coo(self, _, A_shape, B_shape, A_nnz):
        A = generate_random_sparse_coo_matrix(A_shape, A_nnz, indices_dtype=torch.int64, values_dtype=torch.float64, device=self.device)
        B = torch.randn(*B_shape, dtype=torch.float64, device=self.device)
        Ad = A.to_dense()
        res_sparse = sparse_bmm(A, B)
        res_dense = torch.bmm(Ad, B)
        self.assertTrue(torch.allclose(res_sparse, res_dense))


if __name__ == "__main__":
    unittest.main()
