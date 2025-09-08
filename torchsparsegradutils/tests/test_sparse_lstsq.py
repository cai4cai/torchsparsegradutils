import pytest
import torch

from torchsparsegradutils import sparse_generic_lstsq

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


# Test generic least-squares solve
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


# Test gradient correctness
def test_generic_lstsq_gradient_default(device):
    A_shape = (7, 4)
    B_shape = (7, 1)
    dtype = torch.float64

    A = torch.randn(A_shape, dtype=dtype, device=device)
    A_csr = A.to_sparse_csr()
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
    Ad2 = A.detach().clone()
    Ad2.requires_grad_()
    Bd2 = B.detach().clone()
    Bd2.requires_grad_()
    Ad2.retain_grad()
    Bd2.retain_grad()

    x2 = torch.linalg.lstsq(Ad2, Bd2).solution
    loss2 = x2.sum()
    loss2.backward()

    nz_mask = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz_mask], Ad2.grad[nz_mask], rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, rtol=RTOL)
