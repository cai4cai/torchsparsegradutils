import torch
import unittest
import pytest
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
    sparse_block_diag_split,
    stack_csr,
    sparse_eye,
)

if torch.__version__ >= (2,):
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
class TestStackCSR(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")

    @parameterized.expand(
        [
            ("4x4_12n_d0", torch.Size([4, 4]), 12, 0),
            ("8x16_32n_d0", torch.Size([8, 16]), 32, 0),
            ("4x4_12n_d-1", torch.Size([4, 4]), 12, -1),
            ("8x16_32n_d-1", torch.Size([8, 16]), 32, -1),
        ]
    )
    def test_stack_csr(self, _, size, nnz, dim):
        csr_list = [generate_random_sparse_csr_matrix(size, nnz) for _ in range(3)]
        dense_list = [csr.to_dense() for csr in csr_list]
        csr_stacked = stack_csr(csr_list)
        dense_stacked = torch.stack(dense_list)
        self.assertTrue(torch.equal(csr_stacked.to_dense(), dense_stacked))


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
            ("1x4x4_12n", torch.Size([1, 4, 4]), 12),
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
            ("1x4x4_12n", torch.Size([1, 4, 4]), 12),
            ("4x4x4_12n", torch.Size([4, 4, 4]), 12),
            ("6x8x14_32n", torch.Size([6, 8, 14]), 32),
        ]
    )
    def test_sparse_block_diag_coo_backward(self, _, size, nnz):
        A_coo = generate_random_sparse_coo_matrix(size, nnz, device=self.device)
        A_d = A_coo.detach().clone().to_dense()

        A_coo.requires_grad_(True)
        A_d.requires_grad_(True)

        A_coo_block_diag = sparse_block_diag(*A_coo)
        A_d_block_diag = torch.block_diag(*A_d)

        torch.sparse.sum(A_coo_block_diag).backward()
        A_d_block_diag.sum().backward()

        nz_mask = A_coo.grad.to_dense() != 0.0

        self.assertTrue(torch.allclose(A_coo.grad.to_dense()[nz_mask], A_d.grad[nz_mask]))

    @parameterized.expand(
        [
            ("1x4x4_12n", torch.Size([1, 4, 4]), 12),
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

    @parameterized.expand(
        [
            ("1x4x4_12n", torch.Size([1, 4, 4]), 12),  # passes for this case
            ("4x4x4_12n", torch.Size([4, 4, 4]), 12),
            ("6x8x14_32n", torch.Size([6, 8, 14]), 32),
        ]
    )
    def test_sparse_block_diag_csr_backward_dense_grad(self, _, size, nnz):
        self.skipTest(
            reason="This test is not passing due to a BUG in PyTorch, which tries to differential CSR indices and results in: RuntimeError: isDifferentiableType(variable.scalar_type())"
        )
        A_csr = generate_random_sparse_csr_matrix(size, nnz, device=self.device)
        A_d = A_csr.detach().clone().to_dense()

        A_csr.requires_grad_(True)
        A_d.requires_grad_(True)

        A_csr_block_diag = sparse_block_diag(*A_csr)
        A_d_block_diag = torch.block_diag(*A_d)

        torch.sparse.sum(A_csr_block_diag).backward()
        A_d_block_diag.sum().backward()

        nz_mask = A_csr.grad.to_dense() != 0.0

        self.assertTrue(torch.allclose(A_csr.grad.to_dense()[nz_mask], A_d.grad[nz_mask]))

    @parameterized.expand(
        [
            ("1x4x4_12n", torch.Size([1, 4, 4]), 12),  # Fails with is_conitous error
            ("4x4x4_12n", torch.Size([4, 4, 4]), 12),  # Fails with differentiably error
            ("6x8x14_32n", torch.Size([6, 8, 14]), 32),
        ]
    )
    def test_sparse_block_diag_csr_backward(self, _, size, nnz):
        self.skipTest(
            reason="This test is not passing due to a BUG in PyTorch, which tries to differential CSR indices and results in: RuntimeError: isDifferentiableType(variable.scalar_type())"
        )
        A_csr = generate_random_sparse_csr_matrix(size, nnz, device=self.device)
        A_d = A_csr.detach().clone().to_dense()

        A_csr.requires_grad_(True)
        A_d.requires_grad_(True)

        A_csr_block_diag = sparse_block_diag(*A_csr)
        A_d_block_diag = torch.block_diag(*A_d)

        # generate a sparse CSR tensor of the same sparsity pattern as A_csr_block_diag, but all values are unique:
        grad_output = torch.sparse_csr_tensor(
            crow_indices=A_csr_block_diag.crow_indices(),
            col_indices=A_csr_block_diag.col_indices(),
            values=torch.arange(A_csr_block_diag._nnz(), dtype=torch.float),
            size=A_csr_block_diag.shape,
        ).to(self.device)

        # set the gradient manually
        A_csr_block_diag.backward(grad_output)
        A_d_block_diag.backward(grad_output.to_dense())

        nz_mask = A_csr.grad.to_dense() != 0.0

        self.assertTrue(torch.allclose(A_csr.grad.to_dense()[nz_mask], A_d.grad[nz_mask]))

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
class TestSparseBlockDiaSplit(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")

    @parameterized.expand(
        [
            ("1x4x4_12n", torch.Size([1, 4, 4]), 12),
            ("4x4x4_12n", torch.Size([4, 4, 4]), 12),
            ("6x8x14_32n", torch.Size([6, 8, 14]), 32),
        ]
    )
    def test_coo(self, _, shape, nnz):
        A_coo = generate_random_sparse_coo_matrix(shape, nnz, device=self.device)
        A_coo_block_diag = sparse_block_diag(*A_coo)
        shapes = shape[0] * (shape[-2:],)
        A_coo_block_diag_split = sparse_block_diag_split(A_coo_block_diag, *shapes)
        for i, A in enumerate(A_coo):
            self.assertTrue(torch.equal(A.to_dense(), A_coo_block_diag_split[i].to_dense()))

    @parameterized.expand(
        [
            ("1x4x4_12n", torch.Size([1, 4, 4]), 12),
            ("4x4x4_12n", torch.Size([4, 4, 4]), 12),
            ("6x8x14_32n", torch.Size([6, 8, 14]), 32),
        ]
    )
    def test_csr(self, _, shape, nnz):
        A_csr = generate_random_sparse_csr_matrix(shape, nnz, device=self.device)
        A_csr_block_diag = sparse_block_diag(*A_csr)
        shapes = shape[0] * (shape[-2:],)
        A_csr_block_diag_split = sparse_block_diag_split(A_csr_block_diag, *shapes)
        for i, A in enumerate(A_csr):
            self.assertTrue(torch.equal(A.to_dense(), A_csr_block_diag_split[i].to_dense()))


# Sparse eye tests, using pytest framework:

# Identify Testing Parameters
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

TEST_DATA = [
    # name  shape,
    ("unbat", (4, 4)),
    ("unbat", (8, 8)),
    ("bat", (2, 4, 4)),
    ("bat", (4, 8, 8)),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]
LAYOUTS = [torch.sparse_coo, torch.sparse_csr]


# Define Test Names:
def data_id(shapes):
    return shapes[0]


def device_id(device):
    return str(device)


def dtype_id(dtype):
    return str(dtype).split(".")[-1]


def layout_id(layout):
    return str(layout).split(".")[-1].split("_")[-1].upper()


# Define Fixtures


@pytest.fixture(params=TEST_DATA, ids=[data_id(d) for d in TEST_DATA])
def shapes(request):
    return request.param


@pytest.fixture(params=VALUE_DTYPES, ids=[dtype_id(d) for d in VALUE_DTYPES])
def value_dtype(request):
    return request.param


@pytest.fixture(params=INDEX_DTYPES, ids=[dtype_id(d) for d in INDEX_DTYPES])
def index_dtype(request):
    return request.param


@pytest.fixture(params=DEVICES, ids=[device_id(d) for d in DEVICES])
def device(request):
    return request.param


@pytest.fixture(params=LAYOUTS, ids=[layout_id(lay) for lay in LAYOUTS])
def layout(request):
    return request.param


# Define Tests:


def test_sparse_eye(shapes, layout, value_dtype, index_dtype, device):
    if index_dtype == torch.int32 and layout is torch.sparse_coo:
        pytest.skip("Skipping test as sparse COO tensors with int32 indices are not supported")

    name, size = shapes

    # TODO: it would be good to unify the terminology of indices_dtype and index_dtype
    Id = sparse_eye(size, layout=layout, values_dtype=value_dtype, indices_dtype=index_dtype, device=device)

    assert Id.shape == size
    assert Id.layout == layout
    assert Id.device.type == device.type
    assert Id.values().dtype == value_dtype

    if layout == torch.sparse_coo:
        assert Id.indices().dtype == index_dtype
    elif layout == torch.sparse_csr:
        assert Id.crow_indices().dtype == index_dtype
        assert Id.col_indices().dtype == index_dtype
    else:
        raise ValueError("layout not supported")

    if len(size) == 2:
        Id_torch = torch.eye(size[-1], dtype=value_dtype, device=device)
        assert torch.equal(Id.to_dense(), Id_torch)


# def test_sparse_eye_coo():
#     n = 5
#     I = sparse_eye(n, layout=torch.sparse_coo)
#     assert I.shape == (n, n)
#     assert I.is_sparse
#     assert I.dtype == torch.float64
#     assert I.is_cuda == False
#     assert I.requires_grad == False

# def test_sparse_eye_csr():
#     n = 5
#     I = sparse_eye(n, layout=torch.sparse_csr_tensor)
#     assert I.shape == (n, n)
#     assert I.is_sparse
#     assert I.dtype == torch.float64
#     assert I.is_cuda == False
#     assert I.requires_grad == False

# def test_sparse_eye_values_dtype():
#     n = 5
#     with pytest.raises(ValueError):
#         sparse_eye(n, values_dtype=torch.int32)

# def test_sparse_eye_indices_dtype_coo():
#     n = 5
#     with pytest.raises(ValueError):
#         sparse_eye(n, indices_dtype=torch.int32, layout=torch.sparse_coo_tensor)

# def test_sparse_eye_indices_dtype_csr():
#     n = 5
#     with pytest.raises(ValueError):
#         sparse_eye(n, indices_dtype=torch.float32, layout=torch.sparse_csr_tensor)

# def test_sparse_eye_device():
#     n = 5
#     if torch.cuda.is_available():
#         I = sparse_eye(n, device=torch.device('cuda'))
#         assert I.is_cuda == True

# def test_sparse_eye_grad():
#     n = 5
#     I = sparse_eye(n, requires_grad=True)
#     assert I.requires_grad == True

# def test_sparse_eye_coo_vs_dense():
#     n = 5
#     I_sparse = sparse_eye(n, layout=torch.sparse_coo_tensor).to_dense()
#     I_dense = torch.eye(n, dtype=torch.float64)
#     assert torch.allclose(I_sparse, I_dense)

# def test_sparse_eye_csr_vs_dense():
#     n = 5
#     I_sparse = sparse_eye(n, layout=torch.sparse_csr_tensor).to_dense()
#     I_dense = torch.eye(n, dtype=torch.float64)
#     assert torch.allclose(I_sparse, I_dense)

# def test_sparse_eye_coo_vs_dense_device():
#     n = 5
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#         I_sparse = sparse_eye(n, layout=torch.sparse_coo_tensor, device=device).to_dense()
#         I_dense = torch.eye(n, dtype=torch.float64, device=device)
#         assert torch.allclose(I_sparse, I_dense)

# def test_sparse_eye_csr_vs_dense_device():
#     n = 5
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#         I_sparse = sparse_eye(n, layout=torch.sparse_csr_tensor, device=device).to_dense()
#         I_dense = torch.eye(n, dtype=torch.float64, device=device)
#         assert torch.allclose(I_sparse, I_dense)
