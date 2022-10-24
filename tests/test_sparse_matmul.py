import torch
import unittest
from torchsparseutils.sparse_matmul import SparseMatMul

class SparseMatMulTest(unittest.TestCase):
    """Test Sparse COO x Dense matrix multiplication with back propagation"""
    def setUp(self) -> None:
        self.Ad = torch.randn(16, 16, dtype=torch.float64, device='cuda', requires_grad=True)
        self.As_coo = self.Ad.to_sparse_coo()
        self.As_csr = self.Ad.to_sparse_csr()
        self.Bd = torch.randn(16, 4, dtype=torch.float64, requires_grad=True, device='cuda')
        self.matmul = SparseMatMul.apply
        
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
        
        self.assertTrue(torch.isclose(As1.grad.to_dense(), Ad2.grad).all())
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
        
        self.assertTrue(torch.isclose(As1.grad.to_dense(), Ad2.grad).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad).all())