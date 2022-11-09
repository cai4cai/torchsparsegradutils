import torch
import unittest
from random import randrange
from torchsparsegradutils import sparse_triangular_solve


def gencoordinates_square_tri(n, ni, upper=True, device="cuda"):
    """Used to genererate ni random unique off-diagonal coordinates
    for upper or lower triangular sparse matrix with size [n, n]"""
    coordinates = set()
    while True:
        r, c = randrange(n), randrange(n)
        if (r < c and upper) or (r > c and not upper):
            coordinates.add((r, c))
        if len(coordinates) == ni:
            return torch.stack([torch.tensor(co) for co in coordinates], dim=-1).to(device)


def gencoordinates_square_diag(n, device="cuda"):
    """Generate diagonal indices for square matrix with size [n, n]"""
    d_idx = torch.arange(n, device=device)
    d_idx = torch.stack([d_idx, d_idx], dim=0).to(device)
    return d_idx


class SparseLinearSolveTest(unittest.TestCase):
    """Test Triangular Linear Sparse Solver for COO and CSR formatted sparse matrices"""

    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
        self.RTOL = 1e-3
        self.A_shape = (16, 16)  # square matrix
        self.B_shape = (self.A_shape[1], 10)
        self.A_nnz = 32  # excluding diagonal of A which will be + A_size nnz
        self.A_idx = torch.cat(
            [
                gencoordinates_square_diag(self.A_shape[0], self.device),
                gencoordinates_square_tri(self.A_shape[0], self.A_nnz, device=self.device),
            ],
            dim=1,
        )
        self.A_val = torch.cat(
            [
                torch.rand(self.A_shape[0], dtype=torch.float64, device=self.device),  # [0, 1) to avoid 0 on diag
                torch.randn(self.A_nnz, dtype=torch.float64, device=self.device),
            ]
        )

        self.As_coo_triu = torch.sparse_coo_tensor(self.A_idx, self.A_val, self.A_shape, requires_grad=True).coalesce()
        self.As_coo_tril = self.As_coo_triu.t()
        self.As_csr_triu = self.As_coo_triu.to_sparse_csr()
        self.As_csr_tril = self.As_coo_tril.to_sparse_csr()

        self.Ad_triu = self.As_coo_triu.to_dense()
        self.Ad_tril = self.As_coo_tril.to_dense()

        self.Bd = torch.randn(16, 4, dtype=torch.float64, device=self.device)
        self.solve = sparse_triangular_solve

    def test_solver_result_coo_triu(self):
        x = self.solve(self.As_coo_triu, self.Bd, upper=True)
        x2 = torch.linalg.solve_triangular(self.Ad_triu, self.Bd, upper=True)
        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

    def test_solver_result_coo_tril(self):
        x = self.solve(self.As_coo_tril, self.Bd, upper=False)
        x2 = torch.linalg.solve_triangular(self.Ad_tril, self.Bd, upper=False)
        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

    def test_solver_result_csr_triu(self):
        x = self.solve(self.As_csr_triu, self.Bd, upper=True)
        x2 = torch.linalg.solve_triangular(self.Ad_triu, self.Bd, upper=True)
        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

    def test_solver_result_csr_tril(self):
        x = self.solve(self.As_csr_tril, self.Bd, upper=False)
        x2 = torch.linalg.solve_triangular(self.Ad_tril, self.Bd, upper=False)
        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

    def test_solver_gradient_coo_triu(self):
        # Sparse solver:
        As1 = self.As_coo_triu.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.solve(As1, Bd1, upper=True)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.Ad_triu.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve_triangular(Ad2, Bd2, upper=True)
        loss_torch = x2.sum()
        loss_torch.backward()

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())

    def test_solver_gradient_coo_tril(self):
        # Sparse solver:
        As1 = self.As_coo_tril.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.solve(As1, Bd1, upper=False)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.Ad_tril.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve_triangular(Ad2, Bd2, upper=False)
        loss_torch = x2.sum()
        loss_torch.backward()

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())

    def test_solver_gradient_csr_triu(self):
        # Sparse solver:
        As1 = self.As_csr_triu.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.solve(As1, Bd1, upper=True)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.Ad_triu.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve_triangular(Ad2, Bd2, upper=True)
        loss_torch = x2.sum()
        loss_torch.backward()

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())

    def test_solver_gradient_csr_tril(self):
        # Sparse solver:
        As1 = self.As_csr_tril.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.solve(As1, Bd1, upper=False)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.Ad_tril.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve_triangular(Ad2, Bd2, upper=False)
        loss_torch = x2.sum()
        loss_torch.backward()

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())


class SparseLinearSolveTestCUDA(SparseLinearSolveTest):
    """Override superclass setUp to run on CPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("Skipping SparseMatMulTestCUDA since CUDA is not available")
        self.device = torch.device("cuda")
        super().setUp()


if __name__ == "__main__":
    unittest.main()
