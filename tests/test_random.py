import unittest
import torch
from torchsparsegradutils.utils.random_sparse import gencoordinates, gencoordinates_square_strictly_tri

class TestGenCoordinates(unittest.TestCase):
    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
            
        self.size = torch.Size([4, 8, 16])
        self.nnz = 32  # nnz per batch element
        self.dtype = torch.int64
        
        self.coo_coords_unbatched = gencoordinates(self.size[-2:], self.nnz, layout=torch.sparse_coo,
                                                   dtype=self.dtype, device=self.device)
        
        self.coo_coords_batched = gencoordinates(self.size, self.nnz, layout=torch.sparse_coo, 
                                                 dtype=self.dtype, device=self.device)
        
        self.csr_crow_indices_unbatched, self.csr_col_indices_unbatched = gencoordinates(self.size[-2:], self.nnz, layout=torch.sparse_csr,
                                                    dtype=self.dtype, device=self.device)
        
        
        self.csr_crow_indices_batched, self.csr_col_indices_batched = gencoordinates(self.size, self.nnz, layout=torch.sparse_csr,
                                                    dtype=self.dtype, device=self.device)
    
    # error handling:    
    def test_incorrect_shape(self):
        with self.assertRaises(ValueError):
            gencoordinates((1,) + self.size, self.nnz, layout=torch.sparse_coo, dtype=self.dtype, device=self.device)
            
    def test_incorrect_layout(self):
        with self.assertRaises(ValueError):
            gencoordinates(self.size, self.nnz, layout=torch.sparse_bsr, dtype=self.dtype, device=self.device)
            
    def test_too_many_nnz(self):
        nnz = self.size[-2:].numel() + 1
        with self.assertRaises(ValueError):
            gencoordinates(self.size, nnz + 1, layout=torch.sparse_coo, dtype=self.dtype, device=self.device)
    
    # unmbacthed COO:    
    def test_gencoords_coo_unbatched_shape(self):
        self.assertEqual(self.coo_coords_unbatched.shape, torch.Size([2, self.nnz]))
        
    def test_gencoords_coo_unbatched_unique(self):
        self.assertEqual(len(set([self.coo_coords_unbatched[:, i] for i in range(self.coo_coords_unbatched.shape[-1])])), self.nnz)
        
    def test_gencoords_coo_unbatched_range(self):
        print(self.coo_coords_unbatched)
        self.assertTrue((self.coo_coords_unbatched.t() < torch.tensor([self.size[-2], self.size[-1]])).all())
        
    def test_gencoords_coo_unbatched_device(self):
        self.assertEqual(self.coo_coords_unbatched.device, self.device)
        
    def test_gencoords_coo_unbatched_dtype(self):
        self.assertEqual(self.coo_coords_unbatched.dtype, self.dtype)
        
    def test_gencoords_coo_unbatched_coords(self):
        dummy_values = torch.ones(self.nnz, dtype=torch.float32, device=self.device)
        try:  
            torch._validate_sparse_coo_tensor_args(self.coo_coords_unbatched, dummy_values, self.size[-2:])
        except RuntimeError as e:
            self.fail(f"Error: {e}")
            
    def test_gencoords_coo_unbatched_coords_int32_dtype(self):
        dummy_values = torch.ones(self.nnz, dtype=torch.float32, device=self.device)
        self.assertRaises(RuntimeError, torch._validate_sparse_coo_tensor_args, self.coo_coords_unbatched.to(torch.int32), dummy_values, self.size[-2:])

    # batched COO:    
    def test_gencoords_coo_batched_shape(self):
        self.assertEqual(self.coo_coords_batched.shape, torch.Size([3, self.nnz*self.size[0]]))
        
    def test_gencoords_coo_batched_device(self):
        self.assertEqual(self.coo_coords_batched.device, self.device)
        
    def test_gencoords_coo_batched_dtype(self):
        self.assertEqual(self.coo_coords_batched.dtype, self.dtype)
        
    def test_gencoords_coo_batched_coords(self):
        dummy_values = torch.ones(self.nnz*self.size[0], dtype=torch.float32, device=self.device)
        try:  
            torch._validate_sparse_coo_tensor_args(self.coo_coords_batched, dummy_values, self.size)
        except RuntimeError as e:
            self.fail(f"Error: {e}")
             
    def test_gencoords_coo_batched_coords_int32_dtype(self):
        dummy_values = torch.ones(self.nnz*self.size[0], dtype=torch.float32, device=self.device)
        self.assertRaises(RuntimeError, torch._validate_sparse_coo_tensor_args, self.coo_coords_batched.to(torch.int32), dummy_values, self.size)
            
    # unbatched CSR:
    def test_gencoords_csr_unbatched_shape(self):
        self.assertEqual(self.csr_crow_indices_unbatched.shape, torch.Size([self.size[-2] + 1]))
        self.assertEqual(self.csr_col_indices_unbatched.shape, torch.Size([self.nnz]))
        
    def test_gencoords_csr_unbatched_device(self):
        self.assertEqual(self.csr_crow_indices_unbatched.device, self.device)
        self.assertEqual(self.csr_col_indices_unbatched.device, self.device)
        
    def test_gencoords_csr_unbatched_dtype(self):
        self.assertEqual(self.csr_crow_indices_unbatched.dtype, self.dtype)
        self.assertEqual(self.csr_col_indices_unbatched.dtype, self.dtype)
        
    def test_gencoords_csr_unbatched_coords(self):
        dummy_values = torch.ones(self.nnz, dtype=torch.float32, device=self.device)
        try:  
            torch._validate_sparse_csr_tensor_args(self.csr_crow_indices_unbatched, self.csr_col_indices_unbatched, dummy_values, self.size[-2:])
        except RuntimeError as e:
            self.fail(f"Error: {e}")
            
    def test_gencoords_csr_unbatched_coords(self):
        dummy_values = torch.ones(self.nnz, dtype=torch.float32, device=self.device)
        try:  
            torch._validate_sparse_csr_tensor_args(self.csr_crow_indices_unbatched, self.csr_col_indices_unbatched, dummy_values, self.size[-2:])
        except RuntimeError as e:
            self.fail(f"Error: {e}")
    
    # batched CSR:
    def test_gencoords_csr_batched_shape(self):
        self.assertEqual(self.csr_crow_indices_batched.shape, torch.Size([self.size[0], self.size[-2] + 1]))
        self.assertEqual(self.csr_col_indices_batched.shape, torch.Size([self.size[0], self.nnz]))
        
    def test_gencoords_csr_batched_device(self):
        self.assertEqual(self.csr_crow_indices_batched.device, self.device)
        self.assertEqual(self.csr_col_indices_batched.device, self.device)
        
    def test_gencoords_csr_batched_dtype(self):
        self.assertEqual(self.csr_crow_indices_batched.dtype, self.dtype)
        self.assertEqual(self.csr_col_indices_batched.dtype, self.dtype)
        
    def test_gencoords_csr_batched_coords(self):
        dummy_values = torch.ones(self.nnz, dtype=torch.float32, device=self.device).repeat(self.size[0], 1)
        try:  
            torch._validate_sparse_csr_tensor_args(self.csr_crow_indices_batched, self.csr_col_indices_batched, dummy_values, self.size)
        except RuntimeError as e:
            self.fail(f"Error: {e}")
    

class TestGenCoordinatesTril(TestGenCoordinates):
    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
            
        self.size = torch.Size([4, 12, 12])
        self.nnz = 32  # nnz per batch element
        self.dtype = torch.int64
        
        self.coo_coords_unbatched = gencoordinates_square_strictly_tri(self.size[-2:], self.nnz, layout=torch.sparse_coo,
                                                   dtype=self.dtype, device=self.device)
        
        self.coo_coords_batched = gencoordinates_square_strictly_tri(self.size, self.nnz, layout=torch.sparse_coo, 
                                                 dtype=self.dtype, device=self.device)
        
        self.csr_crow_indices_unbatched, self.csr_col_indices_unbatched = gencoordinates_square_strictly_tri(self.size[-2:], self.nnz, layout=torch.sparse_csr,
                                                    dtype=self.dtype, device=self.device)
        
        
        self.csr_crow_indices_batched, self.csr_col_indices_batched = gencoordinates_square_strictly_tri(self.size, self.nnz, layout=torch.sparse_csr,
                                                    dtype=self.dtype, device=self.device)