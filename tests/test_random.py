import unittest
from parameterized import parameterized, parameterized_class
import torch
from torchsparsegradutils.utils.random_sparse import (
    generate_random_sparse_coo_matrix,
    generate_random_sparse_csr_matrix,
    generate_random_sparse_strictly_triangular_coo_matrix,
    generate_random_sparse_strictly_triangular_csr_matrix,
)

if torch.__version__ >= (2,):
    # https://pytorch.org/docs/stable/generated/torch.sparse.check_sparse_tensor_invariants.html#torch.sparse.check_sparse_tensor_invariants
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

    @parameterized.expand(
        [
            ("int08", torch.int8),
            ("int16", torch.int16),
            ("int32", torch.int32),
        ]
    )
    def test_incompatible_indices_dtype(self, _, indices_dtype):
        with self.assertRaises(ValueError):
            generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype)

    # basic properties:
    def test_device(self):
        A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, device=self.device)
        self.assertEqual(A.device.type, self.device.type)

    @parameterized.expand(
        [
            ("int64", torch.int64),
        ]
    )  # NOTE: only torch.int64 is supported for COO indices, all other dtypes are casted to torch.int64
    def test_indices_dtype(self, _, indices_dtype):
        A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype, device=self.device)
        self.assertEqual(A.indices().dtype, indices_dtype)

    @parameterized.expand(
        [
            ("float16", torch.float16),
            ("float32", torch.float32),
            ("float64", torch.float64),
        ]
    )
    def test_values_dtype(self, _, values_dtype):
        A = generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, values_dtype=values_dtype, device=self.device)
        self.assertEqual(A.values().dtype, values_dtype)

    @parameterized.expand(
        [
            ("4x4", torch.Size([4, 4]), 12),
            ("2x4x4", torch.Size([2, 4, 4]), 12),
            ("8x16", torch.Size([8, 16]), 32),
            ("4x8x16", torch.Size([4, 8, 16]), 32),
        ]
    )
    def test_size(self, _, size, nnz):
        A = generate_random_sparse_coo_matrix(size, nnz, device=self.device)
        self.assertEqual(A.size(), size)

    @parameterized.expand(
        [
            ("4x4", torch.Size([4, 4]), 12),
            ("2x4x4", torch.Size([2, 4, 4]), 12),
            ("8x16", torch.Size([8, 16]), 32),
            ("4x8x16", torch.Size([4, 8, 16]), 32),
        ]
    )
    def test_nnz(self, _, size, nnz):
        # NOTE: COO tensors return ._nnz() over all batch elements
        A = generate_random_sparse_coo_matrix(size, nnz, device=self.device)
        self.assertEqual(A._nnz(), nnz * ((1,) + size)[-3])


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
class TestGenRandomCSR(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")

    # error handling:
    def test_too_few_dims(self):
        with self.assertRaises(ValueError):
            generate_random_sparse_csr_matrix(torch.Size([16]), 6)

    def test_too_many_dims(self):
        with self.assertRaises(ValueError):
            generate_random_sparse_csr_matrix(torch.Size([4, 4, 8, 8]), 32)

    def test_too_many_nnz(self):
        with self.assertRaises(ValueError):
            generate_random_sparse_csr_matrix(torch.Size([4, 4]), 17)

    @parameterized.expand(
        [
            ("int08", torch.int8),
            ("int16", torch.int16),
        ]
    )
    def test_incompatible_indices_dtype(self, _, indices_dtype):
        with self.assertRaises(ValueError):
            generate_random_sparse_coo_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype)

    # basic properties:
    def test_device(self):
        A = generate_random_sparse_csr_matrix(torch.Size([4, 4]), 12, device=self.device)
        self.assertEqual(A.device.type, self.device.type)

    @parameterized.expand(
        [
            ("int32", torch.int32),
            ("int64", torch.int64),
        ]
    )  # NOTE: Only int32 and int64 are supported for CSR indices
    def test_indices_dtype(self, _, indices_dtype):
        A = generate_random_sparse_csr_matrix(torch.Size([4, 4]), 12, indices_dtype=indices_dtype, device=self.device)
        self.assertEqual(A.crow_indices().dtype, indices_dtype)
        self.assertEqual(A.col_indices().dtype, indices_dtype)

    @parameterized.expand(
        [
            ("float16", torch.float16),
            ("float32", torch.float32),
            ("float64", torch.float64),
        ]
    )
    def test_values_dtype(self, _, values_dtype):
        A = generate_random_sparse_csr_matrix(torch.Size([4, 4]), 12, values_dtype=values_dtype, device=self.device)
        self.assertEqual(A.values().dtype, values_dtype)

    @parameterized.expand(
        [
            ("4x4", torch.Size([4, 4]), 12),
            ("2x4x4", torch.Size([2, 4, 4]), 12),
            ("8x16", torch.Size([8, 16]), 32),
            ("4x8x16", torch.Size([4, 8, 16]), 32),
        ]
    )
    def test_size(self, _, size, nnz):
        A = generate_random_sparse_csr_matrix(size, nnz, device=self.device)
        self.assertEqual(A.size(), size)

    @parameterized.expand(
        [
            ("4x4", torch.Size([4, 4]), 12),
            ("2x4x4", torch.Size([2, 4, 4]), 12),
            ("8x16", torch.Size([8, 16]), 32),
            ("4x8x16", torch.Size([4, 8, 16]), 32),
        ]
    )
    def test_nnz(self, _, size, nnz):
        A = generate_random_sparse_csr_matrix(size, nnz, device=self.device)
        # NOTE: CSR tensors return ._nnz() per batch element
        self.assertEqual(A._nnz(), nnz)


# Strictly triangular tests:


@parameterized_class(
    (
        "name",
        "device",
    ),
    [
        ("Lower_CPU", torch.device("cpu")),
        ("CUDA", torch.device("cuda")),
    ],
)
class TestGenRandomStrictlyTriCOO(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")

    # error handling:
    def test_too_few_dims(self):
        with self.assertRaises(ValueError):
            generate_random_sparse_strictly_triangular_coo_matrix(torch.Size([16]), 6)

    def test_too_many_dims(self):
        with self.assertRaises(ValueError):
            generate_random_sparse_strictly_triangular_coo_matrix(torch.Size([4, 4, 8, 8]), 12)

    def test_too_many_nnz(self):
        with self.assertRaises(ValueError):
            limit = 4 * (4 - 1) // 2
            generate_random_sparse_strictly_triangular_coo_matrix(torch.Size([4, 4]), limit + 1)

    @parameterized.expand(
        [
            ("int08", torch.int8),
            ("int16", torch.int16),
            ("int32", torch.int32),
        ]
    )
    def test_incompatible_indices_dtype(self, _, indices_dtype):
        with self.assertRaises(ValueError):
            generate_random_sparse_strictly_triangular_coo_matrix(torch.Size([4, 4]), 5, indices_dtype=indices_dtype)

    # basic properties:
    def test_device(self):
        A = generate_random_sparse_strictly_triangular_coo_matrix(torch.Size([4, 4]), 5, device=self.device)
        self.assertEqual(A.device.type, self.device.type)

    @parameterized.expand(
        [
            ("int64", torch.int64),
        ]
    )  # NOTE: only torch.int64 is supported for COO indices, all other dtypes are casted to torch.int64
    def test_indices_dtype(self, _, indices_dtype):
        A = generate_random_sparse_strictly_triangular_coo_matrix(
            torch.Size([4, 4]), 5, indices_dtype=indices_dtype, device=self.device
        )
        self.assertEqual(A.indices().dtype, indices_dtype)

    @parameterized.expand(
        [
            ("float16", torch.float16),
            ("float32", torch.float32),
            ("float64", torch.float64),
        ]
    )
    def test_values_dtype(self, _, values_dtype):
        A = generate_random_sparse_strictly_triangular_coo_matrix(
            torch.Size([4, 4]), 5, values_dtype=values_dtype, device=self.device
        )
        self.assertEqual(A.values().dtype, values_dtype)

    # specific properties:
    @parameterized.expand(
        [
            ("4x4", torch.Size([4, 4]), 4),
            ("2x4x4", torch.Size([2, 4, 4]), 4),
            ("8x8", torch.Size([8, 8]), 28),
            ("4x8x8", torch.Size([4, 8, 8]), 28),
        ]
    )
    def test_size_upper(self, _, size, nnz):
        A = generate_random_sparse_strictly_triangular_coo_matrix(size, nnz, upper=True, device=self.device)
        self.assertEqual(A.size(), size)

    @parameterized.expand(
        [
            ("4x4", torch.Size([4, 4]), 4),
            ("2x4x4", torch.Size([2, 4, 4]), 4),
            ("8x8", torch.Size([8, 8]), 28),
            ("4x8x8", torch.Size([4, 8, 8]), 28),
        ]
    )
    def test_size_lower(self, _, size, nnz):
        A = generate_random_sparse_strictly_triangular_coo_matrix(size, nnz, upper=False, device=self.device)
        self.assertEqual(A.size(), size)

    @parameterized.expand(
        [
            ("4x4_12nnz", torch.Size([4, 4]), 5),
            ("2x4x4_12nnz", torch.Size([2, 4, 4]), 4),
            ("8x8_22nnz", torch.Size([8, 8]), 22),
            ("4x8x8_28nnz", torch.Size([4, 8, 8]), 28),
        ]
    )
    def test_nnz_upper(self, _, size, nnz):
        # NOTE: COO tensors return ._nnz() over all batch elements
        A = generate_random_sparse_strictly_triangular_coo_matrix(size, nnz, upper=True, device=self.device)
        self.assertEqual(A._nnz(), nnz * ((1,) + size)[-3])

    @parameterized.expand(
        [
            ("4x4_12nnz", torch.Size([4, 4]), 5),
            ("2x4x4_12nnz", torch.Size([2, 4, 4]), 4),
            ("8x8_22nnz", torch.Size([8, 8]), 22),
            ("4x8x8_28nnz", torch.Size([4, 8, 8]), 28),
        ]
    )
    def test_nnz_lower(self, _, size, nnz):
        # NOTE: COO tensors return ._nnz() over all batch elements
        A = generate_random_sparse_strictly_triangular_coo_matrix(size, nnz, upper=False, device=self.device)
        self.assertEqual(A._nnz(), nnz * ((1,) + size)[-3])

    def test_is_strictly_lower(self):
        A = generate_random_sparse_strictly_triangular_coo_matrix(
            torch.Size([4, 4]), 5, upper=False, device=self.device
        )
        Ad = A.to_dense()
        self.assertTrue(torch.equal(Ad, Ad.tril(-1)))

    def test_is_strictly_upper(self):
        A = generate_random_sparse_strictly_triangular_coo_matrix(
            torch.Size([4, 4]), 5, upper=False, device=self.device
        )
        Ad = A.to_dense()
        self.assertTrue(torch.equal(Ad, Ad.tril(1)))


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
class TestGenRandomStrictlyTriCSR(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available() and self.device == torch.device("cuda"):
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")

    # error handling:
    def test_too_few_dims(self):
        with self.assertRaises(ValueError):
            generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([16]), 6)

    def test_too_many_dims(self):
        with self.assertRaises(ValueError):
            generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4, 8, 8]), 12)

    def test_too_many_nnz(self):
        with self.assertRaises(ValueError):
            limit = 4 * (4 - 1) // 2
            generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4]), limit + 1)

    @parameterized.expand(
        [
            ("int08", torch.int8),
            ("int16", torch.int16),
        ]
    )
    def test_incompatible_indices_dtype(self, _, indices_dtype):
        with self.assertRaises(ValueError):
            generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4]), 5, indices_dtype=indices_dtype)

    # basic properties:
    def test_device(self):
        A = generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4]), 5, device=self.device)
        self.assertEqual(A.device.type, self.device.type)

    @parameterized.expand(
        [
            ("int32", torch.int32),
            ("int64", torch.int64),
        ]
    )
    def test_indices_dtype(self, _, indices_dtype):
        A = generate_random_sparse_strictly_triangular_csr_matrix(
            torch.Size([4, 4]), 5, indices_dtype=indices_dtype, device=self.device
        )
        self.assertEqual(A.crow_indices().dtype, indices_dtype)
        self.assertEqual(A.col_indices().dtype, indices_dtype)

    @parameterized.expand(
        [
            ("float16", torch.float16),
            ("float32", torch.float32),
            ("float64", torch.float64),
        ]
    )
    def test_values_dtype(self, _, values_dtype):
        A = generate_random_sparse_strictly_triangular_csr_matrix(
            torch.Size([4, 4]), 5, values_dtype=values_dtype, device=self.device
        )
        self.assertEqual(A.values().dtype, values_dtype)

    # specific properties:
    @parameterized.expand(
        [
            ("4x4", torch.Size([4, 4]), 4),
            ("2x4x4", torch.Size([2, 4, 4]), 4),
            ("8x8", torch.Size([8, 8]), 28),
            ("4x8x8", torch.Size([4, 8, 8]), 28),
        ]
    )
    def test_size_upper(self, _, size, nnz):
        A = generate_random_sparse_strictly_triangular_csr_matrix(size, nnz, upper=True, device=self.device)
        self.assertEqual(A.size(), size)

    @parameterized.expand(
        [
            ("4x4", torch.Size([4, 4]), 4),
            ("2x4x4", torch.Size([2, 4, 4]), 4),
            ("8x8", torch.Size([8, 8]), 28),
            ("4x8x8", torch.Size([4, 8, 8]), 28),
        ]
    )
    def test_size_lower(self, _, size, nnz):
        A = generate_random_sparse_strictly_triangular_csr_matrix(size, nnz, upper=False, device=self.device)
        self.assertEqual(A.size(), size)

    @parameterized.expand(
        [
            ("4x4_12nnz", torch.Size([4, 4]), 5),
            ("2x4x4_12nnz", torch.Size([2, 4, 4]), 4),
            ("8x8_22nnz", torch.Size([8, 8]), 22),
            ("4x8x8_28nnz", torch.Size([4, 8, 8]), 28),
        ]
    )
    def test_nnz_upper(self, _, size, nnz):
        A = generate_random_sparse_strictly_triangular_csr_matrix(size, nnz, upper=True, device=self.device)
        # NOTE: CSR tensors return ._nnz() per batch element
        self.assertEqual(A._nnz(), nnz)

    @parameterized.expand(
        [
            ("4x4_12nnz", torch.Size([4, 4]), 5),
            ("2x4x4_12nnz", torch.Size([2, 4, 4]), 4),
            ("8x8_22nnz", torch.Size([8, 8]), 22),
            ("4x8x8_28nnz", torch.Size([4, 8, 8]), 28),
        ]
    )
    def test_nnz_lower(self, _, size, nnz):
        A = generate_random_sparse_strictly_triangular_csr_matrix(size, nnz, upper=False, device=self.device)
        # NOTE: CSR tensors return ._nnz() per batch element
        self.assertEqual(A._nnz(), nnz)

    def test_is_strictly_upper(self):
        A = generate_random_sparse_strictly_triangular_csr_matrix(torch.Size([4, 4]), 5, upper=True, device=self.device)
        Ad = A.to_dense()
        self.assertTrue(torch.equal(Ad, Ad.triu(1)))

    def test_is_strictly_lower(self):
        A = generate_random_sparse_strictly_triangular_csr_matrix(
            torch.Size([4, 4]), 5, upper=False, device=self.device
        )
        Ad = A.to_dense()
        print(Ad)
        self.assertTrue(torch.equal(Ad, Ad.tril(-1)))
