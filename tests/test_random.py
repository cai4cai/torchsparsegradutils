import unittest
from parameterized import parameterized, parameterized_class
import torch
from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix, 
    generate_random_sparse_csr_matrix,
    )


@parameterized_class(('name', 'device',), [
    ("CPU", torch.device("cpu")),
    ("CUDA", torch.device("cuda"),),
])
class TestGenRandomCOO(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        
    # error handling:
    def test_too_few_dims(self):
        with self.assertRaises(ValueError):
            generate_random_sparse_coo_matrix(torch.Size([16]), 6)
                
    def test_too_many_dims(self):
        with self.assertRaises(ValueError):
            generate_random_sparse_coo_matrix(torch.Size([4, 4, 8, 8]), 32)

    def test_too_many_nnz(self):
        with self.assertRaises(ValueError):
            generate_random_sparse_coo_matrix(torch.Size([4, 4]), 17)
    
    # basic properties:        
    def test_device(self):
        A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, device=self.device)
        self.assertEqual(A.device.type, self.device.type)
    
    @parameterized.expand([
        ("int32", torch.int32),
        ("int64", torch.int64),
    ])
    def test_indices_dtype(self, _, indices_dtype):
        A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype, device=self.device)
        A = A.coalesce()
        self.assertEqual(A.indices().dtype, torch.int64)
        # NOTE: int32 is not supported for COO indices, it will be converted to int64
    
    @parameterized.expand([
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("float64", torch.float64),
    ])
    def test_values_dtype(self, _, values_dtype):
        A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, values_dtype=values_dtype, device=self.device)
        A = A.coalesce()
        self.assertEqual(A.values().dtype, values_dtype)
        
    @parameterized.expand([
        ("4x4", torch.Size([4, 4]), 12),
        ("2x4x4", torch.Size([2, 4, 4]), 12),
        ("8x16", torch.Size([8, 16]), 32),
        ("4x8x16", torch.Size([4, 8, 16]), 32),
    ])
    def test_size(self, _, size, nnz):
        A = generate_random_sparse_coo_matrix(size, nnz, device=self.device)
        A = A.coalesce()
        self.assertEqual(A.size(), size)
    
    @parameterized.expand([
        ("4x4", torch.Size([4, 4]), 12),
        ("2x4x4", torch.Size([2, 4, 4]), 12),
        ("8x16", torch.Size([8, 16]), 32),
        ("4x8x16", torch.Size([4, 8, 16]), 32),
    ])
    def test_nnz(self, _, size, nnz):
        A = generate_random_sparse_coo_matrix(size, nnz, device=self.device)
        A = A.coalesce()
        self.assertEqual(A._nnz(), nnz*((1,)+size)[-3])
            
        
