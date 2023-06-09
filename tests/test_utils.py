import torch
import unittest
from unittest.mock import Mock

from parameterized import parameterized_class, parameterized
from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix,
    generate_random_sparse_csr_matrix,
)

from torchsparsegradutils.utils.utils import (
    _compress_row_indices,
    _demcompress_crow_indices,
    _sort_coo_indices,
    convert_coo_to_csr,
    sparse_block_diag,
)

# https://pytorch.org/docs/stable/generated/torch.sparse.check_sparse_tensor_invariants.html
torch.sparse.check_sparse_tensor_invariants.enable()


@parameterized_class(
    (
        "name",
        "device",
    ),
    [
        ("CPU", torch.device("cpu")),
        (
            "CUDA",
            torch.device("cuda"),
        ),
    ],
)
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


@parameterized_class(
    (
        "name",
        "device",
    ),
    [
        ("CPU", torch.device("cpu")),
        (
            "CUDA",
            torch.device("cuda"),
        ),
    ],
)
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

    @parameterized.expand(
        [
            ("4x4_12n", torch.Size([4, 4]), 12),
            ("8x16_32n", torch.Size([8, 16]), 32),
        ]
    )
    def test_compress_row_indices(self, _, size, nnz):
        A_coo = generate_random_sparse_coo_matrix(size, nnz, device=self.device)
        A_csr = A_coo.to_sparse_csr()
        A_csr_crow_indices = A_csr.crow_indices()
        crow_indices = _compress_row_indices(A_coo.indices()[0], A_coo.size()[0])
        self.assertTrue(torch.equal(A_csr_crow_indices, crow_indices))

    @parameterized.expand(
        [
            ("4x4_12n", torch.Size([4, 4]), 12),
            ("8x16_32n", torch.Size([8, 16]), 32),
            ("4x4x4_12n", torch.Size([4, 4, 4]), 12),
            ("6x8x14_32n", torch.Size([6, 8, 14]), 32),
        ]
    )
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

    @parameterized.expand(
        [
            ("4x4_12n", torch.Size([4, 4]), 12),
            ("8x16_32n", torch.Size([8, 16]), 32),
            ("4x4x4_12n", torch.Size([4, 4, 4]), 12),
            ("6x8x14_32n", torch.Size([6, 8, 14]), 32),
        ]
    )
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


@parameterized_class(
    (
        "name",
        "device",
    ),
    [
        ("CPU", torch.device("cpu")),
        (
            "CUDA",
            torch.device("cuda"),
        ),
    ],
)
class TestCSRtoCOO(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")

    @parameterized.expand(
        [
            ("4x4_12n", torch.Size([4, 4]), 12),
            ("8x16_32n", torch.Size([8, 16]), 32),
        ]
    )
    def test_compress_row_indices(self, _, size, nnz):
        A_csr = generate_random_sparse_csr_matrix(size, nnz, device=self.device)
        A_coo = A_csr.to_sparse_coo()
        A_coo_row_indices = A_coo.indices()[0]
        row_indices = _demcompress_crow_indices(A_csr.crow_indices(), A_coo.size()[0])
        self.assertTrue(torch.equal(A_coo_row_indices, row_indices))


@parameterized_class(
    (
        "name",
        "device",
    ),
    [
        ("CPU", torch.device("cpu")),
        (
            "CUDA",
            torch.device("cuda"),
        ),
    ],
)
class TestSparseBlockDiag(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
            
        
    @parameterized.expand(
        [
            ("4x4x4_12n", torch.Size([4, 4, 4]), 12),
            ("6x8x14_32n", torch.Size([6, 8, 14]), 32),
        ]
    )
    def test_sparse_block_diag_coo(self, _, size, nnz):
        A_coo = generate_random_sparse_coo_matrix(size, nnz, device=self.device)
        A_d = A_coo.to_dense()
        A_coo_block_diag = sparse_block_diag(*A_coo)
        Ad_block_diag = torch.block_diag(*A_d)
        self.assertTrue(torch.equal(A_coo_block_diag.to_dense(), Ad_block_diag))

    @parameterized.expand(
        [
            ("4x4x4_12n", torch.Size([4, 4, 4]), 12),
            ("6x8x14_32n", torch.Size([6, 8, 14]), 32),
        ]
    )
    def test_sparse_block_diag_csr(self, _, size, nnz):
        A_csr = generate_random_sparse_csr_matrix(size, nnz, device=self.device)
        A_d = A_csr.to_dense()
        A_csr_block_diag = sparse_block_diag(*A_csr)
        Ad_block_diag = torch.block_diag(*A_d)
        self.assertTrue(torch.equal(A_csr_block_diag.to_dense(), Ad_block_diag))
        # TODO: sparse_block_diag changes (in place) the indices of the input CSR tensor

    def test_no_arguments(self):
        with self.assertRaises(ValueError):
            sparse_block_diag()

    def test_incorrect_tensor_layout(self):
        coo_tensor = Mock(spec=torch.Tensor)
        coo_tensor.layout = torch.sparse_coo
        csr_tensor = Mock(spec=torch.Tensor)
        csr_tensor.layout = torch.sparse_csr
        with self.assertRaises(ValueError):
            sparse_block_diag(coo_tensor, csr_tensor)

    def test_incorrect_sparse_dim(self):
        coo_tensor = Mock(spec=torch.Tensor)
        coo_tensor.layout = torch.sparse_coo
        coo_tensor.sparse_dim.return_value = 1
        with self.assertRaises(ValueError):
            sparse_block_diag(coo_tensor)

    def test_incorrect_dense_dim(self):
        coo_tensor = Mock(spec=torch.Tensor)
        coo_tensor.layout = torch.sparse_coo
        coo_tensor.dense_dim.return_value = 1
        with self.assertRaises(ValueError):
            sparse_block_diag(coo_tensor)

    def test_incorrect_input_type(self):
        with self.assertRaises(TypeError):
            sparse_block_diag("Not a list or tuple")

    def test_incorrect_tensor_type_in_list(self):
        tensor1 = torch.randn(5, 5).to_sparse().to(device=self.device)
        tensor2 = "Not a tensor"
        with self.assertRaises(TypeError):
            sparse_block_diag(tensor1, tensor2)

    def test_different_shapes_coo(self):
        tensor1 = torch.randn(5, 5).to_sparse_coo().to(device=self.device)
        tensor2 = torch.randn(3, 3).to_sparse_coo().to(device=self.device)
        tensor3 = torch.randn(2, 2).to_sparse_coo().to(device=self.device)
        result = sparse_block_diag(tensor1, tensor2, tensor3)
        self.assertEqual(result.shape, torch.Size([10, 10]))

    def test_different_shapes_csr(self):
        tensor1 = torch.randn(5, 5).to_sparse_csr().to(device=self.device)
        tensor2 = torch.randn(3, 3).to_sparse_csr().to(device=self.device)
        tensor3 = torch.randn(2, 2).to_sparse_csr().to(device=self.device)
        result = sparse_block_diag(tensor1, tensor2, tensor3)
        self.assertEqual(result.shape, torch.Size([10, 10]))

    def test_too_many_tensor_dimensions(self):
        tensor1 = torch.randn(5, 5, 5).to_sparse().to(device=self.device)
        with self.assertRaises(ValueError):
            sparse_block_diag(tensor1)

    def test_empty_tensor_coo(self):
        tensor1 = torch.sparse_coo_tensor(torch.empty([2, 0]), torch.empty([0])).to(device=self.device)
        result = sparse_block_diag(tensor1)
        self.assertEqual(result.shape, torch.Size([0, 0]))

    # TODO: these commented tests are failing
    # def test_empty_tensor_coo_mix(self):
    #     tensor1 = torch.empty((0, 0)).to_sparse_coo().to(device=self.device)
    #     tensor2 = torch.randn(3, 3).to_sparse_coo().to(device=self.device)
    #     result = sparse_block_diag(tensor1, tensor2)
    #     expected = tensor2.to_dense()
    #     self.assertTrue(torch.equal(result.to_dense(), expected))

    # def test_empty_tensor_csr(self):
    #     tensor1 = torch.sparse_csr_tensor(torch.empty([2, 0]), torch.empty([0])).to(device=self.device)
    #     result = sparse_block_diag(tensor1)
    #     self.assertEqual(result.shape, torch.Size([0, 0]))

    # def test_empty_tensor_csr_mix(self):
    #     tensor1 = torch.empty((0, 0)).to_sparse_csr().to(device=self.device)
    #     tensor2 = torch.randn(3, 3).to_sparse_csr().to(device=self.device)
    #     result = sparse_block_diag(tensor1, tensor2)
    #     expected = tensor2.to_dense()
    #     self.assertTrue(torch.equal(result.to_dense(), expected))

    def test_single_tensor_coo(self):
        tensor1 = torch.randn(5, 5).to_sparse_coo().to(device=self.device)
        result = sparse_block_diag(tensor1)
        self.assertTrue(torch.equal(result.to_dense(), tensor1.to_dense()))

    def test_single_tensor_csr(self):
        tensor1 = torch.randn(5, 5).to_sparse_csr().to(device=self.device)
        result = sparse_block_diag(tensor1)
        self.assertTrue(torch.equal(result.to_dense(), tensor1.to_dense()))

    def test_zero_tensor_coo(self):
        tensor1 = torch.zeros(5, 5).to_sparse().to(device=self.device)
        tensor2 = torch.zeros(3, 3).to_sparse().to(device=self.device)
        result = sparse_block_diag(tensor1, tensor2)
        self.assertEqual(result.shape, torch.Size([8, 8]))
        self.assertTrue((result.to_dense() == 0).all())

    def test_zero_tensor_csr(self):
        tensor1 = torch.zeros(5, 5).to_sparse_csr().to(device=self.device)
        tensor2 = torch.zeros(3, 3).to_sparse_csr().to(device=self.device)
        result = sparse_block_diag(tensor1, tensor2)
        expected = torch.zeros(8, 8).to(device=self.device)
        self.assertTrue(torch.equal(result.to_dense(), expected))

    def test_non_square_tensor_coo(self):
        tensor1 = torch.randn(5, 7).to_sparse().to(device=self.device)
        tensor2 = torch.randn(3, 2).to_sparse().to(device=self.device)
        result = sparse_block_diag(tensor1, tensor2)
        self.assertEqual(result.shape, torch.Size([8, 9]))

    def test_non_square_tensor_csr(self):
        tensor1 = torch.randn(5, 7).to_sparse_csr().to(device=self.device)
        tensor2 = torch.randn(3, 2).to_sparse_csr().to(device=self.device)
        result = sparse_block_diag(tensor1, tensor2)
        self.assertEqual(result.shape, torch.Size([8, 9]))
