import unittest
import torch
from torchsparsegradutils.utils.random_sparse import _gencoordinates_2d, gencoordinates

class TestGenCoordinates(unittest.TestCase):
    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
            
        self.size = torch.Size([4, 8, 16])
        self.nnz = 32
        self.dtype = torch.int64
        
        self.coo_coords_unbatched = _gencoordinates_2d(self.size[-2], self.size[-1], 
                                                self.nnz, dtype=self.dtype, device=self.device)
        
        self.coo_coords_batched = gencoordinates(self.size, self.nnz, layout=torch.sparse_coo, 
                                                 dtype=self.dtype, device=self.device)
        
    def test_gen_2d_shape(self):
        self.assertEqual(self.coo_coords_unbatched.shape, torch.Size([2, self.nnz]))
        
    def test_gen_2d_unique(self):
        self.assertEqual(len(set([self.coo_coords_unbatched[:, i] for i in range(self.coo_coords_unbatched.shape[-1])])), self.nnz)
        
    def test_gen_2d_range(self):
        self.assertTrue((self.coo_coords_unbatched.t() < torch.tensor([self.size[-2], self.size[-1]])).all())
        
    def test_gen_2d_device(self):
        self.assertEqual(self.coo_coords_unbatched.device, self.device)
        
    def test_gen_2d_dtype(self):
        self.assertEqual(self.coo_coords_unbatched.dtype, self.dtype)
        
    def test_gencoordinates_coo_batched_shape(self):
        self.assertEqual(self.coo_coords_batched.shape, torch.Size([3, self.nnz*self.size[0]]))
        
    def test_gencoordinate_coo_batched_device(self):
        self.assertEqual(self.coo_coords_batched.device, self.device)
        
    def test_gencoordinate_coo_batched_dtype(self):
        self.assertEqual(self.coo_coords_batched.dtype, self.dtype)