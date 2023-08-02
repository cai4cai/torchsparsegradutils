import torch
import pytest
import unittest
from random import randrange
from sys import platform
from torchsparsegradutils import sparse_triangular_solve, sparse_generic_solve
from torchsparsegradutils.utils import linear_cg, minres, rand_sparse, rand_sparse_tri


# Identify Testing Parameters
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

# Pytest test currently just implemented for unit triangular solve
TEST_DATA = [
    # name  A_shape, B_shape, A_nnz
    ("unbat", (4, 4), (4, 2), 4),
    ("unbat", (12, 12), (12, 6), 32),
    ("bat", (2, 4, 4), (2, 4, 2), 4),
    ("bat", (4, 12, 12), (4, 12, 6), 32),
]

INDEX_DTYPES = [torch.int32, torch.int64]
VALUE_DTYPES = [torch.float32, torch.float64]

UPPER = [True, False]
UNITRIANGULAR = [
    True,
]  # just unit triangular solve for now
TRANSPOSE = [True, False]

ATOL = 1e-6  # relaxed tolerance to allow for float32
RTOL = 1e-4


# Define Test Names:
def data_id(shapes):
    return shapes[0]


def device_id(device):
    return str(device)


def dtype_id(dtype):
    return str(dtype).split(".")[-1]


def upper_id(upper):
    return "upp" if upper else "low"


def unitriangular_id(unitriangular):
    return "unit" if unitriangular else "nonunit"


def transpose_id(transpose):
    return "t" if transpose else ""


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


@pytest.fixture(params=UPPER, ids=[upper_id(d) for d in UPPER])
def upper(request):
    return request.param


@pytest.fixture(params=UNITRIANGULAR, ids=[unitriangular_id(d) for d in UNITRIANGULAR])
def unitriangular(request):
    return request.param


@pytest.fixture(params=TRANSPOSE, ids=[transpose_id(d) for d in TRANSPOSE])
def transpose(request):
    return request.param


# Define Tests


def forward_routine(layout, device, value_dtype, index_dtype, shapes, upper, unitriangular, transpose):
    if index_dtype == torch.int32 and layout is torch.sparse_coo:
        pytest.skip("Skipping test as sparse COO tensors with int32 indices are not supported")
    if platform == "win32" and device == torch.device("cpu"):
        pytest.skip("Skipping triangular solve CPU tests as solver not implemented for Windows OS")

    _, A_shape, B_shape, A_nnz = shapes
    A = rand_sparse_tri(
        A_shape,
        A_nnz,
        layout,
        upper=upper,
        strict=unitriangular,
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
    )
    B = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Ad = A.to_dense()

    res_ref = torch.triangular_solve(B, Ad, upper=upper, unitriangular=unitriangular, transpose=transpose).solution
    res_test = sparse_triangular_solve(A, B, upper=upper, unitriangular=unitriangular, transpose=transpose)

    assert torch.allclose(res_test, res_ref, atol=ATOL, rtol=RTOL)


def backward_routine(layout, device, value_dtype, index_dtype, shapes, upper, unitriangular, transpose):
    if index_dtype == torch.int32 and layout is torch.sparse_coo:
        pytest.skip("Skipping test as sparse COO tensors with int32 indices are not supported")
    if platform == "win32" and device == torch.device("cpu"):
        pytest.skip("Skipping triangular solve CPU tests as solver not implemented for Windows OS")

    _, A_shape, B_shape, A_nnz = shapes

    As1 = rand_sparse_tri(
        A_shape,
        A_nnz,
        layout,
        upper=upper,
        strict=unitriangular,
        indices_dtype=index_dtype,
        values_dtype=value_dtype,
        device=device,
    )

    Ad2 = As1.to_dense().detach().clone()  # detach and clone to create seperate graph
    Ad3 = Ad2.detach().clone()

    Bd1 = torch.rand(*B_shape, dtype=value_dtype, device=device)
    Bd2 = Bd1.detach().clone()
    Bd3 = Bd1.detach().clone()

    As1.requires_grad_()
    Ad2.requires_grad_()
    Ad3.requires_grad_()
    Bd1.requires_grad_()
    Bd2.requires_grad_()
    Bd3.requires_grad_()

    res_ref = torch.triangular_solve(Bd2, Ad2, upper=upper, unitriangular=unitriangular, transpose=transpose).solution
    res_test = sparse_triangular_solve(As1, Bd1, upper=upper, unitriangular=unitriangular, transpose=transpose)

    # Let's add another test to make sure that the transpose argument is working as epexcted:
    if transpose:
        res_test2 = torch.linalg.solve_triangular(
            Ad3.transpose(-2, -1), Bd3, upper=not upper, unitriangular=unitriangular
        )
    else:
        res_test2 = torch.linalg.solve_triangular(Ad3, Bd3, upper=upper, unitriangular=unitriangular)

    # Generate random gradients for the backward pass
    grad_output = torch.rand_like(res_test, dtype=value_dtype, device=device)

    res_ref.backward(grad_output)
    res_test.backward(grad_output)
    res_test2.backward(grad_output)

    nz_mask = As1.grad.to_dense() != 0.0

    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], atol=ATOL, rtol=RTOL)
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad3.grad[nz_mask], atol=ATOL, rtol=RTOL)

    assert torch.allclose(Bd1.grad, Bd2.grad, atol=ATOL, rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd3.grad, atol=ATOL, rtol=RTOL)


def test_forward_result_coo(device, value_dtype, index_dtype, shapes, upper, unitriangular, transpose):
    forward_routine(
        torch.sparse_coo,
        device,
        value_dtype,
        index_dtype,
        shapes,
        upper,
        unitriangular,
        transpose,
    )


def test_forward_result_csr(device, value_dtype, index_dtype, shapes, upper, unitriangular, transpose):
    forward_routine(
        torch.sparse_csr,
        device,
        value_dtype,
        index_dtype,
        shapes,
        upper,
        unitriangular,
        transpose,
    )


def test_backward_result_coo(device, value_dtype, index_dtype, shapes, upper, unitriangular, transpose):
    backward_routine(
        torch.sparse_coo,
        device,
        value_dtype,
        index_dtype,
        shapes,
        upper,
        unitriangular,
        transpose,
    )


def test_backward_result_csr(device, value_dtype, index_dtype, shapes, upper, unitriangular, transpose):
    backward_routine(
        torch.sparse_csr,
        device,
        value_dtype,
        index_dtype,
        shapes,
        upper,
        unitriangular,
        transpose,
    )


##### Older test, TODO: needs restructuring: #####


def gencoordinates_square_tri(n, ni, upper=True, device="cuda"):
    """Used to genererate ni random unique off-diagonal coordinates
    for upper or lower triangular sparse matrix with size [n, n]"""
    coordinates = set()
    while True:
        r, c = randrange(n), randrange(n)
        if (r < c and upper) or (r > c and not upper):
            coordinates.add((r, c))
        if len(coordinates) == ni:
            return torch.stack([torch.tensor(co) for co in coordinates], dim=-1).to(device)


def gencoordinates_square_diag(n, device="cuda"):
    """Generate diagonal indices for square matrix with size [n, n]"""
    d_idx = torch.arange(n, device=device)
    d_idx = torch.stack([d_idx, d_idx], dim=0).to(device)
    return d_idx


class SparseTriangularSolveTest(unittest.TestCase):
    """Test Triangular Linear Sparse Solver for COO and CSR formatted sparse matrices"""

    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
            if platform == "win32":
                self.skipTest(f"Skipping {self.__class__.__name__} CPU test as solver not implemented for Windows OS")
        self.RTOL = 1e-3
        self.unitriangular = False
        self.A_shape = (16, 16)  # square matrix
        self.B_shape = (self.A_shape[1], 10)
        self.A_nnz = 32  # excluding diagonal of A which will be + A_size nnz
        self.A_idx = torch.cat(
            [
                gencoordinates_square_diag(self.A_shape[0], self.device),
                gencoordinates_square_tri(self.A_shape[0], self.A_nnz, device=self.device),
            ],
            dim=1,
        )
        self.A_val = torch.cat(
            [
                torch.rand(self.A_shape[0], dtype=torch.float64, device=self.device),  # [0, 1) to avoid 0 on diag
                torch.randn(self.A_nnz, dtype=torch.float64, device=self.device),
            ]
        )

        self.As_coo_triu = torch.sparse_coo_tensor(self.A_idx, self.A_val, self.A_shape, requires_grad=True).coalesce()
        self.As_coo_tril = self.As_coo_triu.t()
        self.As_csr_triu = self.As_coo_triu.to_sparse_csr()
        self.As_csr_tril = self.As_coo_tril.to_sparse_csr()

        self.Ad_triu = self.As_coo_triu.to_dense()
        self.Ad_tril = self.As_coo_tril.to_dense()

        self.Bd = torch.randn(16, 4, dtype=torch.float64, device=self.device)
        self.solve = sparse_triangular_solve

    def test_solver_result_coo_triu(self):
        x = self.solve(self.As_coo_triu, self.Bd, upper=True, unitriangular=self.unitriangular)
        x2 = torch.linalg.solve_triangular(self.Ad_triu, self.Bd, upper=True, unitriangular=self.unitriangular)
        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

    def test_solver_result_coo_tril(self):
        x = self.solve(self.As_coo_tril, self.Bd, upper=False, unitriangular=self.unitriangular)
        x2 = torch.linalg.solve_triangular(self.Ad_tril, self.Bd, upper=False, unitriangular=self.unitriangular)
        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

    def test_solver_result_csr_triu(self):
        x = self.solve(self.As_csr_triu, self.Bd, upper=True, unitriangular=self.unitriangular)
        x2 = torch.linalg.solve_triangular(self.Ad_triu, self.Bd, upper=True, unitriangular=self.unitriangular)
        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

    def test_solver_result_csr_tril(self):
        x = self.solve(self.As_csr_tril, self.Bd, upper=False, unitriangular=self.unitriangular)
        x2 = torch.linalg.solve_triangular(self.Ad_tril, self.Bd, upper=False, unitriangular=self.unitriangular)
        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

    def test_solver_gradient_coo_triu(self):
        # Sparse solver:
        As1 = self.As_coo_triu.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.solve(As1, Bd1, upper=True, unitriangular=self.unitriangular)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.Ad_triu.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve_triangular(Ad2, Bd2, upper=True, unitriangular=self.unitriangular)
        loss_torch = x2.sum()
        loss_torch.backward()

        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())

    def test_solver_gradient_coo_tril(self):
        # Sparse solver:
        As1 = self.As_coo_tril.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.solve(As1, Bd1, upper=False, unitriangular=self.unitriangular)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.Ad_tril.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve_triangular(Ad2, Bd2, upper=False, unitriangular=self.unitriangular)
        loss_torch = x2.sum()
        loss_torch.backward()

        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())

    def test_solver_gradient_csr_triu(self):
        # Sparse solver:
        As1 = self.As_csr_triu.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.solve(As1, Bd1, upper=True, unitriangular=self.unitriangular)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.Ad_triu.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve_triangular(Ad2, Bd2, upper=True, unitriangular=self.unitriangular)
        loss_torch = x2.sum()
        loss_torch.backward()

        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())

    def test_solver_gradient_csr_tril(self):
        # Sparse solver:
        As1 = self.As_csr_tril.detach().clone()
        As1.requires_grad = True
        Bd1 = self.Bd.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = self.solve(As1, Bd1, upper=False, unitriangular=self.unitriangular)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.Ad_tril.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.Bd.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve_triangular(Ad2, Bd2, upper=False, unitriangular=self.unitriangular)
        loss_torch = x2.sum()
        loss_torch.backward()

        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())

    def test_solver_non_triangular_error(self):
        """Test to check that solver throws an ValueError if a diagonal is specified in input
        but unitriangular=True in solver arguments"""
        if self.unitriangular is False:  # statement to stop test running in SparseUnitTriangularSolveTest
            As1 = self.As_csr_triu.detach().clone()
            As1.requires_grad = True
            Bd1 = self.Bd.detach().clone()
            Bd1.requires_grad = True
            As1.retain_grad()
            Bd1.retain_grad()
            x = self.solve(As1, Bd1, upper=True, unitriangular=True)
            loss = x.sum()
            with self.assertRaises(ValueError):
                loss.backward()


class SparseTriangularSolveTestCUDA(SparseTriangularSolveTest):
    """Override superclass setUp to run on CPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        super().setUp()


class SparseUnitTriangularSolveTest(SparseTriangularSolveTest):
    """Test Triangular Linear Sparse Solver for COO and CSR formatted sparse matrices
    for unit triangular case, where unit diagonal is implicit in solver, provided unitriangular = True"""

    def setUp(self) -> None:
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
            if platform == "win32":
                self.skipTest(f"Skipping {self.__class__.__name__} CPU test as solver not implemented for Windows OS")
        self.RTOL = 1e-3
        self.unitriangular = True
        self.A_shape = (16, 16)  # square matrix
        self.B_shape = (self.A_shape[1], 10)
        self.A_nnz = 32
        self.A_idx = gencoordinates_square_tri(self.A_shape[0], self.A_nnz, device=self.device)
        self.A_val = torch.randn(self.A_nnz, dtype=torch.float64, device=self.device)

        self.As_coo_triu = torch.sparse_coo_tensor(self.A_idx, self.A_val, self.A_shape, requires_grad=True).coalesce()
        self.As_coo_tril = self.As_coo_triu.t()
        self.As_csr_triu = self.As_coo_triu.to_sparse_csr()
        self.As_csr_tril = self.As_coo_tril.to_sparse_csr()

        self.Ad_triu = self.As_coo_triu.to_dense()
        self.Ad_tril = self.As_coo_tril.to_dense()

        self.Bd = torch.randn(16, 4, dtype=torch.float64, device=self.device)
        self.solve = sparse_triangular_solve


class SparseUnitTriangularSolveTestCUDA(SparseUnitTriangularSolveTest):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        super().setUp()


class SparseGenericSolveTest(unittest.TestCase):
    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")

        self.RTOL = 1e-3

        self.A_shape = (4, 4)
        self.A = torch.randn(self.A_shape, dtype=torch.float64, device=self.device)
        self.A = self.A + self.A.t() + 10.0 * torch.eye(4, device=self.device)
        self.A_csr = self.A.to_sparse_csr()
        self.B_shape = (4, 2)
        self.B = torch.randn(self.B_shape, dtype=torch.float64, device=self.device)

        self.x_ref = torch.linalg.solve(self.A, self.B)

    def test_generic_solver_default(self):
        x = sparse_generic_solve(self.A_csr, self.B)
        self.assertTrue(torch.isclose(x, self.x_ref, rtol=self.RTOL).all())

    def test_generic_solver_gradient_default(self):
        # Sparse solver:
        As1 = self.A_csr.detach().clone()
        As1.requires_grad = True
        Bd1 = self.B.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = sparse_generic_solve(As1, Bd1)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.A.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.B.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve(Ad2, Bd2)
        loss_torch = x2.sum()
        loss_torch.backward()

        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())

    def test_generic_solver_cg(self):
        x = sparse_generic_solve(self.A_csr, self.B, solve=linear_cg, transpose_solve=linear_cg)
        self.assertTrue(torch.isclose(x, self.x_ref, rtol=self.RTOL).all())

    def test_generic_solver_gradient_cg(self):
        # Sparse solver:
        As1 = self.A_csr.detach().clone()
        As1.requires_grad = True
        Bd1 = self.B.detach().clone()
        Bd1.requires_grad = True
        As1.retain_grad()
        Bd1.retain_grad()
        x = sparse_generic_solve(As1, Bd1, solve=linear_cg, transpose_solve=linear_cg)
        loss = x.sum()
        loss.backward()

        # torch dense solver:
        Ad2 = self.A.detach().clone()
        Ad2.requires_grad = True
        Bd2 = self.B.detach().clone()
        Bd2.requires_grad = True
        Ad2.retain_grad()
        Bd2.retain_grad()
        x2 = torch.linalg.solve(Ad2, Bd2)
        loss_torch = x2.sum()
        loss_torch.backward()

        self.assertTrue(torch.isclose(x, x2, rtol=self.RTOL).all())

        nz_mask = As1.grad.to_dense() != 0.0
        self.assertTrue(torch.isclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=self.RTOL).all())
        self.assertTrue(torch.isclose(Bd1.grad, Bd2.grad, rtol=self.RTOL).all())


class SparseGenericSolveTestCUDA(SparseGenericSolveTest):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        super().setUp()


if __name__ == "__main__":
    unittest.main()
