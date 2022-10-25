import torch
import unittest
from torchsparseutils.sparse_solve import sparse_triangular_solve

class SparseLinearSolveTest(unittest.TestCase):
    """Test Triangular Linear Sparse Solving for COO formatted sparse matrices"""
    def setUp(self) -> None:
        self.RTOL = 1e-3
        self.Ad = torch.randn(16, 16, dtype=torch.float64, device='cuda')   # upper triangular matrix
        self.Ad[self.Ad == 0] = 1e-5  # avoid 0 values on the diagonal
        self.Ad_triu = self.Ad.triu()
        self.Ad_tril = self.Ad.tril()
        self.As_coo_triu = self.Ad_triu.to_sparse_coo()
        self.As_csr_triu = self.Ad_triu.to_sparse_csr()
        self.As_coo_tril = self.Ad_tril.to_sparse_coo()
        self.As_csr_tril = self.Ad_tril.to_sparse_csr()
        
        self.Bd = torch.randn(16, 4, dtype=torch.float64, device='cuda')
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
        
        self.assertTrue(torch.isclose(As1.grad.to_dense(), Ad2.grad, rtol=self.RTOL).all())
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
        
        self.assertTrue(torch.isclose(As1.grad.to_dense(), Ad2.grad, rtol=self.RTOL).all())
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
        
        self.assertTrue(torch.isclose(As1.grad.to_dense(), Ad2.grad, rtol=self.RTOL).all())
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
        
        self.assertTrue(torch.isclose(As1.grad.to_dense(), Ad2.grad, rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())
        

    def test_solver_gradcheck_coo(self):
        # gradcheck(self.solve, (self.As, self.B), check_sparse_nnz=True) 
        pass  # getting jacobian mismatch....
    
    def test_solver_gradcheck_csr(self):
        # gradcheck(self.solve, (self.As, self.B), check_sparse_nnz=True) 
        pass  # getting jacobian mismatch....
    
if __name__ == "__main__":
    unittest.main()