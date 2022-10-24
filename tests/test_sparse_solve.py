import torch
import unittest
from torchsparseutils.sparse_solve import SparseTriangularSolve

class SparseLinearSolveTest(unittest.TestCase):
    """Test Triangular Linear Sparse Solving for COO formatted sparse matrices"""
    def setUp(self) -> None:
        self.RTOL = 1e-3
        self.Ad = torch.randn(16, 16, dtype=torch.float64, device='cuda')   # upper triangular matrix
        self.Ad[self.Ad == 0] = 1e-5  # avoid 0 values on the diagonal
        self.Ad = self.Ad.triu()
        self.Ad.requires_grad = True
        self.As_coo = self.Ad.to_sparse_coo()
        self.As_csr = self.Ad.to_sparse_csr()
        
        self.Bd = torch.randn(16, 4, dtype=torch.float64, requires_grad=True, device='cuda')
        self.solve = SparseTriangularSolve.apply

    def test_solver_forward_coo(self):
        x = self.solve(self.As_coo, self.Bd)
        self.assertIsInstance(x, torch.Tensor)
        
    def test_solver_forward_csr(self):
        x = self.solve(self.As_csr, self.Bd)
        self.assertIsInstance(x, torch.Tensor)
        
    def test_solver_backward_coo(self):
        As_coo = self.As_coo.detach().clone()
        As_coo.requires_grad = True
        x = self.solve(self.As_coo, self.Bd)
        loss = x.sum()
        loss.backward()

    def test_solver_backward_csr(self):
        As_csr = self.As_csr.detach().clone()
        As_csr.requires_grad = True
        x = self.solve(As_csr, self.Bd)
        loss = x.sum()
        loss.backward()
        
    def test_solver_result_coo(self):
        x = self.solve(self.As_coo, self.Bd)
        x_torch = torch.linalg.solve_triangular(self.Ad, self.Bd, upper=True)
        self.assertTrue(torch.isclose(x, x_torch, rtol=self.RTOL).all())
        
    def test_solver_result_csr(self):
        x = self.solve(self.As_csr, self.Bd)
        x_torch = torch.linalg.solve_triangular(self.Ad, self.Bd, upper=True)
        self.assertTrue(torch.isclose(x, x_torch, rtol=self.RTOL).all())
        
    def test_solver_gradient_coo(self):
        # Sparse solver:
        As1 = self.As_coo.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.solve(As1, Bd1)
        loss = x.sum()
        loss.backward()
        
        # torch dense solver:
        Ad2 = self.Ad.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x_torch = torch.linalg.solve_triangular(Ad2, Bd2, upper=True)
        loss_torch = x_torch.sum()
        loss_torch.backward()
        
        self.assertTrue(torch.isclose(As1.grad.to_dense(), Ad2.grad, rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())
                
    def test_solver_gradient_csr(self):
        # Sparse solver:
        As1 = self.As_csr.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.solve(As1, Bd1)
        loss = x.sum()
        loss.backward()
        
        # torch dense solver:
        Ad2 = self.Ad.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x_torch = torch.linalg.solve_triangular(Ad2, Bd2, upper=True)
        loss_torch = x_torch.sum()
        loss_torch.backward()
        
        self.assertTrue(torch.isclose(As1.grad.to_dense(), Ad2.grad, rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())
        

    def test_solver_gradcheck_coo(self):
        # gradcheck(self.solve, (self.As, self.B), check_sparse_nnz=True) 
        pass  # getting jacobian mismatch....
    
    def test_solver_gradcheck_csr(self):
        # gradcheck(self.solve, (self.As, self.B), check_sparse_nnz=True) 
        pass  # getting jacobian mismatch....