import torch
import unittest
from parameterized import parameterized_class, parameterized
from torchsparsegradutils.utils.random_sparse import generate_random_sparse_coo_matrix, generate_random_sparse_strictly_triangular_coo_matrix
from torchsparsegradutils.utils.utils import compress_row_indices, demcompress_crow_indices

from torchsparsegradutils.utils.utils import (
    compress_row_indices,
    demcompress_crow_indices,
)
@parameterized_class(('name', 'device',), [
    ("CPU", torch.device("cpu")),
    ("CUDA", torch.device("cuda"),),
])
class TestRowIndicesCompressionDecompression(unittest.TestCase):
    def setUp(self) -> None:
        self.A_coo = generate_random_sparse_coo_matrix(torch.Size([8, 8]), 12, device=self.device)
        self.A_coo_tril = generate_random_sparse_strictly_triangular_coo_matrix(torch.Size([8, 8]), 12, device=self.device)
        self.A_csr = self.A_coo.to_sparse_csr()
        self.A_csr_tril = self.A_coo_tril.to_sparse_csr()
        
    # row compression cannot be done without sorting the col indices and applying the same sort change to the values
    
    # TODO: these unit tests need to build a CSR from the compressed and then convert back
    # as they do not check either the values or the column indices
    
    # TODO: let's also check the other way around, i.e. CSR to COO
    # I could just use to .to_sparse_coo() and .to_sparse_csr() methods
    # however, that would force conversion to int64, which won't be ideal for my use case
    # Having a way to convert COO - CSR indices, strictly in int32 is useful and will prevent the memory errors I have been having
    # Would also be nice to be able to do this batched.
    
    def test_compress_row_indices(self):
        row_idx, col_idx = self.A_coo.indices()
        crow_idx = compress_row_indices(row_idx, self.A_coo.shape[0])
        self.assertTrue(torch.allclose(crow_idx, self.A_csr.crow_indices()))
        
    def test_demcompress_crow_indices(self):
        crow_idx = self.A_csr.crow_indices()
        row_idx = demcompress_crow_indices(crow_idx, self.A_coo.shape[0])
        self.assertTrue(torch.allclose(row_idx, self.A_coo.indices()[0]))
        
    def test_compress_row_indices_tril(self):
        row_idx, col_idx = self.A_coo_tril.indices()
        crow_idx = compress_row_indices(row_idx, self.A_coo_tril.shape[0])
        self.assertTrue(torch.allclose(crow_idx, self.A_csr_tril.crow_indices()))
        
    def test_demcompress_crow_indices_tril(self):
        crow_idx = self.A_csr_tril.crow_indices()
        row_idx = demcompress_crow_indices(crow_idx, self.A_coo_tril.shape[0])
        self.assertTrue(torch.allclose(row_idx, self.A_coo_tril.indices()[0]))
        
    @parameterized.expand([
        ("int32", torch.int32),
        ("int64", torch.int64),
    ])
    def test_indices_dtype(self, _, indices_dtype):
        num_rows = 4
        row_idx  = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=indices_dtype, device=self.device)
        crow_idx = compress_row_indices(row_idx, num_rows)
        self.assertEqual(crow_idx.dtype, indices_dtype)
        
    def test_device(self):
        num_rows = 4
        row_idx  = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=torch.int32, device=self.device)
        crow_idx = compress_row_indices(row_idx, num_rows)
        self.assertEqual(crow_idx.device.type, self.device.type)