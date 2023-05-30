import unittest
from parameterized import parameterized_class
import torch
from torchsparsegradutils.utils.random_sparse import (
    generate_sparse_coo_matrix_indices, 
    generate_sparse_csr_matrix_indices,
    generate_sparse_coo_matrix_indices_strictly_triangular,
    generate_sparse_csr_matrix_indices_strictly_triangular,
    )


@parameterized_class(('size', 'nnz', 'dtype'), [
    (torch.Size([   4,  4]), 12, torch.int64),
    (torch.Size([2, 4,  4]), 12, torch.int64),
    (torch.Size([   8, 16]), 32, torch.int64),
    (torch.Size([4, 8, 16]), 32, torch.int64),
    (torch.Size([4, 8, 16]),  2, torch.int64),
    # NOTE: int32 is not supported for COO indices
])
class TestGenIndicesCOO(unittest.TestCase):
    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
        
        self.indices = generate_sparse_coo_matrix_indices(self.size, self.nnz, dtype=self.dtype, device=self.device)
        
    # error handling:
    def test_too_few_dims(self):
        with self.assertRaises(ValueError):
            generate_sparse_coo_matrix_indices(torch.Size([1]), self.nnz, dtype=self.dtype, device=self.device)
                
    def test_too_many_dims(self):
        with self.assertRaises(ValueError):
            if len(self.size) == 2:
                generate_sparse_coo_matrix_indices((1, 1) + self.size, self.nnz, dtype=self.dtype, device=self.device)
            elif len(self.size) == 3:
                generate_sparse_coo_matrix_indices((1,) + self.size, self.nnz, dtype=self.dtype, device=self.device)
            
    def test_too_many_nnz(self):
        nnz = self.size[-2:].numel() + 1
        with self.assertRaises(ValueError):
            generate_sparse_coo_matrix_indices(self.size, nnz, dtype=self.dtype, device=self.device)
    
    # basic properties:        
    def test_gencoords_device(self):
        self.assertEqual(self.indices.device, self.device)
        
    def test_gencoords_dtype(self):
        self.assertEqual(self.indices.dtype, self.dtype)
        
    # specific properties:
    def test_shape(self):
        if len(self.size) == 2:
            self.assertEqual(self.indices.shape, torch.Size([2, self.nnz]))
        elif len(self.size) == 3:
            self.assertEqual(self.indices.shape, torch.Size([3, self.nnz*self.size[0]]))
        
    def test_unique(self):
        if len(self.size) == 2:
            self.assertEqual(len(set([self.indices[:, i] for i in range(self.indices.shape[-1])])), self.nnz)
        elif len(self.size) == 3:
            self.assertEqual(len(set([self.indices[1:, :] for i in range(self.indices.shape[-1]//self.size[0])])), self.nnz)
            
    def test_range(self):
        if len(self.size) == 2:
            self.assertTrue((self.indices.t() < torch.tensor([self.size[-2], self.size[-1]])).all())
        elif len(self.size) == 3:
            self.assertTrue((self.indices[1:, :].t() < torch.tensor([self.size[-2], self.size[-1]])).all())
            
    def test_indices(self):
        if len(self.size) == 2:
            dummy_values = torch.ones(self.nnz, dtype=torch.float32, device=self.device)
        elif len(self.size) == 3:
            dummy_values = torch.ones(self.nnz*self.size[0], dtype=torch.float32, device=self.device)
            
        try:  
            torch._validate_sparse_coo_tensor_args(self.indices, dummy_values, self.size)
        except RuntimeError as e:
                self.fail(f"Error: {e}")
            
    def test_indices_int32(self):
        if len(self.size) == 2:
            dummy_values = torch.ones(self.nnz, dtype=torch.float32, device=self.device)
        elif len(self.size) == 3:
            dummy_values = torch.ones(self.nnz*self.size[0], dtype=torch.float32, device=self.device)
        self.assertRaises(RuntimeError, torch._validate_sparse_coo_tensor_args, self.indices.to(torch.int32), dummy_values, self.size)
        
        
@parameterized_class(('size', 'nnz', 'dtype'), [
    (torch.Size([   4,  4]), 12, torch.int64),
    (torch.Size([2, 4,  4]), 12, torch.int64),
    (torch.Size([   8, 16]), 32, torch.int64),
    (torch.Size([4, 8, 16]), 32, torch.int64),
    (torch.Size([4, 8, 16]), 32, torch.int32),  # int32 works with CSR
    (torch.Size([4, 8, 16]),  2, torch.int64),
])        
class TestGenIndicesCSR(unittest.TestCase):
    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
        
        self.crow_indices, self.col_indices = generate_sparse_csr_matrix_indices(self.size, self.nnz, dtype=self.dtype, device=self.device)
        
    # error handling:
    def test_too_few_dims(self):
        with self.assertRaises(ValueError):
            generate_sparse_csr_matrix_indices(torch.Size([1]), self.nnz, dtype=self.dtype, device=self.device)
                
    def test_too_many_dims(self):
        with self.assertRaises(ValueError):
            if len(self.size) == 2:
                generate_sparse_csr_matrix_indices((1, 1) + self.size, self.nnz, dtype=self.dtype, device=self.device)
            elif len(self.size) == 3:
                generate_sparse_csr_matrix_indices((1,) + self.size, self.nnz, dtype=self.dtype, device=self.device)
            
    def test_too_many_nnz(self):
        nnz = self.size[-2:].numel() + 1
        with self.assertRaises(ValueError):
            generate_sparse_csr_matrix_indices(self.size, nnz, dtype=self.dtype, device=self.device)
    
    # basic properties:        
    def test_gencoords_device(self):
        self.assertEqual(self.crow_indices.device, self.device)
        self.assertEqual(self.col_indices.device, self.device)
        
    def test_gencoords_dtype(self):
        self.assertEqual(self.crow_indices.dtype, self.dtype)
        self.assertEqual(self.col_indices.device, self.device)
        
    # specific properties:
    def test_shape(self):
       if len(self.size) == 2:
            self.assertEqual(self.crow_indices.shape, torch.Size([self.size[-2] + 1]))
            self.assertEqual(self.col_indices.shape, torch.Size([self.nnz]))
       elif len(self.size) == 3:
            self.assertEqual(self.crow_indices.shape, torch.Size([self.size[0], self.size[-2] + 1]))
            self.assertEqual(self.col_indices.shape, torch.Size([self.size[0], self.nnz]))
            
    def test_unique(self):
        self.skipTest("Cannot test CSR matrices for uniqueness. In other words they cannot be coalesced.")
        
    def test_range(self):
        if len(self.size) == 2:
            self.assertEqual(self.crow_indices[0], 0)
            self.assertEqual(self.crow_indices[-1], self.nnz)
            self.assertTrue((self.col_indices < torch.tensor([self.size[-1]])).all())
        elif len(self.size) == 3:
            self.assertEqual(self.crow_indices[0, 0], 0)
            self.assertEqual(self.crow_indices[0, -1], self.nnz)
            self.assertTrue((self.col_indices[0] < torch.tensor([self.size[-1]])).all())
            
    def test_indices(self):
        if len(self.size) == 2:
            dummy_values = torch.ones(self.nnz, dtype=torch.float32, device=self.device)
        elif len(self.size) == 3:
            dummy_values = torch.ones(self.size[0], self.nnz, dtype=torch.float32, device=self.device)
            
        try:  
            torch._validate_sparse_csr_tensor_args(self.crow_indices, self.col_indices, dummy_values, self.size)
        except RuntimeError as e:
                self.fail(f"Error: {e}")

    
# @parameterized_class(('size', 'nnz', 'layout', 'dtype'), [
#     (torch.Size([   12, 12]), 32, torch.sparse_coo, torch.int64),
#     (torch.Size([4, 12, 12]), 32, torch.sparse_coo, torch.int64),
#     # NOTE: int32 is not supported for COO indices
    
#     (torch.Size([   12, 12]), 32, torch.sparse_csr, torch.int64),
#     (torch.Size([4, 12, 12]), 32, torch.sparse_csr, torch.int64),
#     (torch.Size([4, 12, 12]), 32, torch.sparse_csr, torch.int32),
# ])
# class TestGenCoordinatesTril(TestGenCoordinates):
#     def setUp(self) -> None:
#         super().setUp()    
#         # self.size = torch.Size([4, 12, 12])
#         # self.nnz = 32  # nnz per batch element
#         self.dtype = torch.int64
        
#         self.coo_coords_unbatched = gencoordinates_square_strictly_tri(self.size[-2:], self.nnz, layout=torch.sparse_coo,
#                                                    dtype=self.dtype, device=self.device)
        
#         self.coo_coords_batched = gencoordinates_square_strictly_tri(self.size, self.nnz, layout=torch.sparse_coo, 
#                                                  dtype=self.dtype, device=self.device)
        
#         self.csr_crow_indices_unbatched, self.csr_col_indices_unbatched = gencoordinates_square_strictly_tri(self.size[-2:], self.nnz, layout=torch.sparse_csr,
#                                                     dtype=self.dtype, device=self.device)
        
        
#         self.csr_crow_indices_batched, self.csr_col_indices_batched = gencoordinates_square_strictly_tri(self.size, self.nnz, layout=torch.sparse_csr,
#                                                     dtype=self.dtype, device=self.device)