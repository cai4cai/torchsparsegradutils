import pytest
import torch

from torchsparsegradutils import sparse_generic_lstsq
from torchsparsegradutils.utils.random_sparse import rand_sparse

# Device fixture
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))


def _id_device(d):
    return str(d)


@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param


# Tolerance for all tests
RTOL = 1e-2


# Test generic least-squares solve with single RHS
def test_generic_lstsq_default(device):
    A_shape = (7, 4)
    B_shape = (7, 1)
    dtype = torch.float64

    A = torch.randn(A_shape, dtype=dtype, device=device)
    A_csr = A.to_sparse_csr()
    B = torch.randn(B_shape, dtype=dtype, device=device)

    x_ref = torch.linalg.lstsq(A, B).solution
    x = sparse_generic_lstsq(A_csr, B)

    assert torch.allclose(x, x_ref, rtol=RTOL)


# Test generic least-squares solve with single RHS (1D)
def test_generic_lstsq_single_rhs_1d(device):
    A_shape = (7, 4)
    B_shape = (7,)  # 1D vector
    dtype = torch.float64

    A = torch.randn(A_shape, dtype=dtype, device=device)
    A_csr = A.to_sparse_csr()
    B = torch.randn(B_shape, dtype=dtype, device=device)

    x_ref = torch.linalg.lstsq(A, B).solution
    x = sparse_generic_lstsq(A_csr, B)

    assert torch.allclose(x, x_ref, rtol=RTOL)
    assert x.shape == x_ref.shape


# Test generic least-squares solve with multiple RHS
def test_generic_lstsq_multiple_rhs(device):
    A_shape = (7, 4)
    B_shape = (7, 3)  # Multiple RHS
    dtype = torch.float64

    A = torch.randn(A_shape, dtype=dtype, device=device)
    A_csr = A.to_sparse_csr()
    B = torch.randn(B_shape, dtype=dtype, device=device)

    x_ref = torch.linalg.lstsq(A, B).solution
    x = sparse_generic_lstsq(A_csr, B)

    assert torch.allclose(x, x_ref, rtol=RTOL)
    assert x.shape == x_ref.shape
    assert x.shape == (4, 3)  # Should be (n_features, n_rhs)


# Test generic least-squares solve with COO format
def test_generic_lstsq_coo_format(device):
    A_shape = (7, 4)
    B_shape = (7, 2)
    dtype = torch.float64
    nnz = 12  # Truly sparse

    # Create truly sparse COO matrix using your random_sparse module
    A_coo = rand_sparse(A_shape, nnz, layout=torch.sparse_coo, values_dtype=dtype, device=device, well_conditioned=True)
    B = torch.randn(B_shape, dtype=dtype, device=device)

    x_ref = torch.linalg.lstsq(A_coo.to_dense(), B).solution
    x = sparse_generic_lstsq(A_coo, B)

    assert torch.allclose(x, x_ref, rtol=RTOL)
    assert x.shape == x_ref.shape


# Test gradient correctness with COO format
def test_generic_lstsq_gradient_coo_format(device):
    A_shape = (6, 3)
    B_shape = (6, 2)
    dtype = torch.float64
    nnz = 9  # Truly sparse

    # Create truly sparse COO matrix using your random_sparse module
    A_coo = rand_sparse(A_shape, nnz, layout=torch.sparse_coo, values_dtype=dtype, device=device, well_conditioned=True)
    B = torch.randn(B_shape, dtype=dtype, device=device)

    # Sparse least-squares
    As1 = A_coo.detach().clone()
    As1.requires_grad_()
    Bd1 = B.detach().clone()
    Bd1.requires_grad_()
    As1.retain_grad()
    Bd1.retain_grad()

    x = sparse_generic_lstsq(As1, Bd1)
    loss = x.sum()
    loss.backward()

    # Dense reference
    Ad2 = A_coo.to_dense().detach().clone()
    Ad2.requires_grad_()
    Bd2 = B.detach().clone()
    Bd2.requires_grad_()
    Ad2.retain_grad()
    Bd2.retain_grad()

    x2 = torch.linalg.lstsq(Ad2, Bd2).solution
    loss2 = x2.sum()
    loss2.backward()

    # Check gradients exist
    assert As1.grad is not None
    assert Bd1.grad is not None
    assert Ad2.grad is not None
    assert Bd2.grad is not None

    # Check sparsity preservation - COO tensors should return True for is_sparse
    assert As1.grad.is_sparse

    # Compare gradients at non-zero locations
    nz_mask = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, rtol=RTOL)


# Test gradient correctness with single RHS
def test_generic_lstsq_gradient_default(device):
    A_shape = (7, 4)
    B_shape = (7, 1)
    dtype = torch.float64
    nnz = 10  # Truly sparse with only 10 non-zeros

    # Create truly sparse matrix using your random_sparse module
    A_csr = rand_sparse(A_shape, nnz, layout=torch.sparse_csr, values_dtype=dtype, device=device, well_conditioned=True)
    B = torch.randn(B_shape, dtype=dtype, device=device)

    # Sparse least-squares
    As1 = A_csr.detach().clone()
    As1.requires_grad_()
    Bd1 = B.detach().clone()
    Bd1.requires_grad_()
    As1.retain_grad()
    Bd1.retain_grad()

    x = sparse_generic_lstsq(As1, Bd1)
    loss = x.sum()
    loss.backward()

    # Dense reference
    Ad2 = A_csr.to_dense().detach().clone()
    Ad2.requires_grad_()
    Bd2 = B.detach().clone()
    Bd2.requires_grad_()
    Ad2.retain_grad()
    Bd2.retain_grad()

    x2 = torch.linalg.lstsq(Ad2, Bd2).solution
    loss2 = x2.sum()
    loss2.backward()

    # Check gradients exist
    assert As1.grad is not None
    assert Bd1.grad is not None
    assert Ad2.grad is not None
    assert Bd2.grad is not None

    # Check sparsity preservation - CSR tensors should have sparse_csr layout
    assert As1.grad.layout == torch.sparse_csr

    # Compare gradients at non-zero locations
    nz_mask = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, rtol=RTOL)


# Test gradient correctness with multiple RHS
def test_generic_lstsq_gradient_multiple_rhs(device):
    A_shape = (6, 3)
    B_shape = (6, 4)  # Multiple RHS
    dtype = torch.float64
    nnz = 8  # Truly sparse with only 8 non-zeros

    # Create truly sparse matrix using your random_sparse module
    A_csr = rand_sparse(A_shape, nnz, layout=torch.sparse_csr, values_dtype=dtype, device=device, well_conditioned=True)
    B = torch.randn(B_shape, dtype=dtype, device=device)

    # Sparse least-squares
    As1 = A_csr.detach().clone()
    As1.requires_grad_()
    Bd1 = B.detach().clone()
    Bd1.requires_grad_()
    As1.retain_grad()
    Bd1.retain_grad()

    x = sparse_generic_lstsq(As1, Bd1)
    loss = x.sum()
    loss.backward()

    # Dense reference
    Ad2 = A_csr.to_dense().detach().clone()
    Ad2.requires_grad_()
    Bd2 = B.detach().clone()
    Bd2.requires_grad_()
    Ad2.retain_grad()
    Bd2.retain_grad()

    x2 = torch.linalg.lstsq(Ad2, Bd2).solution
    loss2 = x2.sum()
    loss2.backward()

    # Check gradients exist
    assert As1.grad is not None
    assert Bd1.grad is not None
    assert Ad2.grad is not None
    assert Bd2.grad is not None

    # Check sparsity preservation - CSR tensors should have sparse_csr layout
    assert As1.grad.layout == torch.sparse_csr

    # Compare gradients at non-zero locations
    nz_mask = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, rtol=RTOL)
