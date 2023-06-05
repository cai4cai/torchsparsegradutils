import torch
import unittest
from parameterized import parameterized_class, parameterized
from torchsparsegradutils.utils.random_sparse import generate_random_sparse_coo_matrix, generate_random_sparse_strictly_triangular_coo_matrix
from torchsparsegradutils.utils.utils import _compress_row_indices, demcompress_crow_indices, convert_coo_to_csr

from torchsparsegradutils.utils.utils import (
    _compress_row_indices,
    demcompress_crow_indices,
    _sort_coo_indices,
)


@parameterized_class(('name', 'device',), [
    ("CPU", torch.device("cpu")),
    ("CUDA", torch.device("cuda"),),
])
class TestSortCOOIndices(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")

    def test_unbatched_sort(self):
        nr, nc = 4, 4
        indices = torch.randperm(nr * nc, device=self.device)
        indices = torch.stack([indices // nc, indices % nc])

        values = torch.arange(16, device=self.device)
        sorted_indices_coalesced = torch.sparse_coo_tensor(indices, values).coalesce().indices()
        coalesce_permutation = torch.sparse_coo_tensor(indices, values).coalesce().values()
        sorted_indices, permutation = _sort_coo_indices(indices)
        self.assertTrue(torch.equal(sorted_indices_coalesced, sorted_indices))
        self.assertTrue(torch.equal(coalesce_permutation, permutation))

    def test_batched_sort(self):
        nr, nc = 4, 4
        batch_size = 3
        indices = torch.randperm(nr * nc, device=self.device)
        indices = torch.stack([indices // nc, indices % nc])
        sparse_indices = torch.cat([indices] * batch_size, dim=-1)
        batch_indices = torch.arange(batch_size, device=self.device).repeat(16).unsqueeze(0)
        batched_sparse_indices = torch.cat([batch_indices, sparse_indices])

        values = torch.arange(nr * nc * batch_size, device=self.device)
        sorted_indices_coalesced = torch.sparse_coo_tensor(batched_sparse_indices, values).coalesce().indices()
        coalesce_permutation = torch.sparse_coo_tensor(batched_sparse_indices, values).coalesce().values()

        sorted_indices, permutation = _sort_coo_indices(batched_sparse_indices)
        self.assertTrue(torch.equal(sorted_indices_coalesced, sorted_indices))
        self.assertTrue(torch.equal(coalesce_permutation, permutation))


@parameterized_class(('name', 'device',), [
    ("CPU", torch.device("cpu")),
    ("CUDA", torch.device("cuda"),),
])
class TestCOOtoCSR(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
    
    
    def batched_coo_to_csr(self, A_coo):
        """      
        Converts batched sparse COO matrix A with shape [B, N, M] 
        to batched sparse CSR matrix with shape [B, N, M]
        
        Inneficient implementation as the COO tensors only support int64 indices.
        Meaning that int32 indices cannot be maintained if a CSR matrix is created via to_sparse_csr() from COO.
        """
        A_crow_indices_list = []
        A_row_indices_list = []
        A_values_list = []
        
        size = A_coo.size()
        
        for a_coo in A_coo:
            a_csr = a_coo.detach().to_sparse_csr()  # detach to prevent grad on indices
            A_crow_indices_list.append(a_csr.crow_indices())
            A_row_indices_list.append(a_csr.col_indices())
            A_values_list.append(a_csr.values())
            
        A_crow_indices = torch.stack(A_crow_indices_list, dim=0)
        A_row_indices = torch.stack(A_row_indices_list, dim=0)
        A_values = torch.stack(A_values_list, dim=0)
    
        return torch.sparse_csr_tensor(A_crow_indices, A_row_indices, A_values, size=size)
    
    
    @parameterized.expand([
        ("4x4_12n", torch.Size([4, 4]), 12),
        ("8x16_32n", torch.Size([8, 16]), 32),
    ])
    def test_compress_row_indices(self, _, size, nnz):
        A_coo = generate_random_sparse_coo_matrix(size, nnz, device=self.device)
        A_csr = A_coo.to_sparse_csr()
        A_csr_crow_indices = A_csr.crow_indices()
        crow_indices = _compress_row_indices(A_coo.indices()[0], A_coo.size()[0])
        self.assertTrue(torch.equal(A_csr_crow_indices, crow_indices))
    
    
    @parameterized.expand([
        ("4x4_12n", torch.Size([4, 4]), 12),
        ("8x16_32n", torch.Size([8, 16]), 32),
    ])
    def test_coo_to_csr_indices(self, _, size, nnz):
        A_coo = generate_random_sparse_coo_matrix(size, nnz, device=self.device)
        A_csr = convert_coo_to_csr(A_coo)
        if len(size) == 2:
            A_csr_2 = A_coo.to_sparse_csr()
        elif len(size) == 3:
            A_csr_2 = self.batched_coo_to_csr(A_coo)
        else:
            raise ValueError(f"Size {size} not supported")
        
        self.assertTrue(torch.equal(A_csr.crow_indices(), A_csr_2.crow_indices()))
        self.assertTrue(torch.equal(A_csr.col_indices(), A_csr_2.col_indices()))
        
        
    @parameterized.expand([
        ("4x4_12n", torch.Size([4, 4]), 12),
        ("8x16_32n", torch.Size([8, 16]), 32),
    ])
    def test_coo_to_csr_values(self, _, size, nnz):
        A_coo = generate_random_sparse_coo_matrix(size, nnz, device=self.device)
        A_csr = convert_coo_to_csr(A_coo)
        if len(size) == 2:
            A_csr_2 = A_coo.to_sparse_csr()
        elif len(size) == 3:
            A_csr_2 = self.batched_coo_to_csr(A_coo)
        else:
            raise ValueError(f"Size {size} not supported")
        
        self.assertTrue(torch.equal(A_csr.values(), A_csr_2.values()))
        
