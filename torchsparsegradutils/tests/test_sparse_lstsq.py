import pytest
import torch
from test_config import DEVICES, Tolerances

from torchsparsegradutils import sparse_generic_lstsq
from torchsparsegradutils.utils.random_sparse import rand_sparse


def _id_device(d):
    return str(d)


@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param


# Least squares solver tolerances for float64
RTOL = Tolerances.lstsq(torch.float64)


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


# Dense (strided) A
@pytest.mark.parametrize("B_shape", [(7,), (7, 1), (7, 3)], ids=["vector_1d", "vector_2d", "multi_rhs"])
def test_generic_lstsq_dense(device, B_shape):
    A_shape = (7, 4)
    dtype = torch.float64
    torch.manual_seed(0)

    A = torch.randn(A_shape, dtype=dtype, device=device)
    B = torch.randn(B_shape, dtype=dtype, device=device)

    A1 = A.clone().requires_grad_()
    B1 = B.clone().requires_grad_()
    A2 = A.clone().requires_grad_()
    B2 = B.clone().requires_grad_()

    x = sparse_generic_lstsq(A1, B1)
    x_ref = torch.linalg.lstsq(A2, B2.unsqueeze(-1) if B.dim() == 1 else B2).solution
    if B.dim() == 1:
        x_ref = x_ref.squeeze(-1)

    assert x.shape == x_ref.shape
    assert torch.allclose(x, x_ref, rtol=RTOL)

    grad_output = torch.randn_like(x)
    x.backward(grad_output)
    x_ref.backward(grad_output)

    assert A1.grad.layout == torch.strided
    assert torch.allclose(A1.grad, A2.grad, rtol=RTOL, atol=RTOL)
    assert torch.allclose(B1.grad, B2.grad, rtol=RTOL, atol=RTOL)


def test_generic_lstsq_unsupported_layout_raises():
    A = torch.randn(5, 3).to_sparse_csc()
    with pytest.raises(TypeError, match="Unsupported layout"):
        sparse_generic_lstsq(A, torch.randn(5))


@pytest.mark.parametrize("A_layout", [torch.sparse_csr, torch.strided], ids=["csr", "dense"])
def test_generic_lstsq_backward_B_only_skips_A_grad(device, A_layout):
    dtype = torch.float64
    torch.manual_seed(0)
    A_dense = torch.randn(7, 4, dtype=dtype, device=device)
    A = A_dense if A_layout == torch.strided else A_dense.to_sparse_csr()
    B = torch.randn(7, 2, dtype=dtype, device=device, requires_grad=True)
    B_ref = B.detach().clone().requires_grad_()

    calls = []

    def counting_lstsq(AA, BB):
        calls.append(1)
        return torch.linalg.lstsq(AA.to_dense(), BB).solution

    sparse_generic_lstsq(A, B, lstsq=counting_lstsq).sum().backward()
    torch.linalg.lstsq(A_dense, B_ref).solution.sum().backward()

    assert len(calls) == 1  # forward only; the A^+ gradB solve is skipped
    assert A.grad is None
    assert torch.allclose(B.grad, B_ref.grad, rtol=RTOL, atol=RTOL)


@pytest.mark.parametrize("A_layout", [torch.sparse_csr, torch.strided], ids=["csr", "dense"])
def test_generic_lstsq_wide_A_requires_grad_raises(device, A_layout):
    A = torch.randn(3, 5, dtype=torch.float64, device=device)
    if A_layout != torch.strided:
        A = A.to_sparse_csr()
    A.requires_grad_()
    with pytest.raises(ValueError, match="tall"):
        sparse_generic_lstsq(A, torch.randn(3, dtype=torch.float64, device=device))


@pytest.mark.parametrize("B_shape", [(3,), (3, 2)], ids=["vector_1d", "multi_rhs"])
@pytest.mark.parametrize("A_layout", [torch.sparse_csr, torch.strided], ids=["csr", "dense"])
def test_generic_lstsq_wide_A_B_only_gradient(device, A_layout, B_shape):
    dtype = torch.float64
    torch.manual_seed(0)
    A_dense = torch.randn(3, 5, dtype=dtype, device=device)
    A = A_dense if A_layout == torch.strided else A_dense.to_sparse_csr()
    B = torch.randn(B_shape, dtype=dtype, device=device, requires_grad=True)
    B_ref = B.detach().clone().requires_grad_()

    # Minimum-norm solution, as returned by the default LSMR
    x = sparse_generic_lstsq(A, B)
    x_ref = torch.linalg.pinv(A_dense) @ B_ref
    assert torch.allclose(x, x_ref, rtol=RTOL, atol=RTOL)

    grad_output = torch.randn_like(x)
    x.backward(grad_output)
    x_ref.backward(grad_output)
    assert torch.allclose(B.grad, B_ref.grad, rtol=RTOL, atol=RTOL)
