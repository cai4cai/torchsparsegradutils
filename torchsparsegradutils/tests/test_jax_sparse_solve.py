import pytest
import torch
import torchsparsegradutils.jax as tsgujax

# Skip entire module if JAX isn't available
pytest.importorskip("jax")
if not tsgujax.have_jax:
    pytest.skip("JAX bindings unavailable, skipping jax tests", allow_module_level=True)


# Device fixture
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))


def _id_device(d):
    return str(d)


@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param


# Relative tolerance for comparisons
def _rtol():
    return 1e-2


# Test forward sparse_j4t solve


def test_solver_j4t(device):
    RTOL = _rtol()
    A_shape = (4, 4)
    A = torch.randn(A_shape, dtype=torch.float64, device=device)
    A = A + A.t()
    A_csr = A.to_sparse_csr()
    B_shape = (4, 2)
    B = torch.randn(B_shape, dtype=torch.float64, device=device)
    x_ref = torch.linalg.solve(A, B)

    x = tsgujax.sparse_solve_j4t(A_csr.to(torch.float32), B.to(torch.float32))
    assert torch.allclose(x, x_ref.to(torch.float32), rtol=RTOL)


# Test backward gradient of sparse_j4t solve


def test_solver_gradient_j4t(device):
    RTOL = _rtol()
    # Setup data
    A_shape = (4, 4)
    A = torch.randn(A_shape, dtype=torch.float64, device=device)
    A = A + A.t()
    A_csr = A.to_sparse_csr().to(torch.float32)
    B_shape = (4, 2)
    B = torch.randn(B_shape, dtype=torch.float64, device=device).to(torch.float32)
    # Sparse solver
    As1 = A_csr.clone().detach()
    As1.requires_grad_(True)
    Bd1 = B.clone().detach()
    Bd1.requires_grad_(True)
    x = tsgujax.sparse_solve_j4t(As1, Bd1)
    x.sum().backward()
    # Dense reference
    Ad2 = A.to(torch.float32).clone().detach()
    Ad2.requires_grad_(True)
    Bd2 = B.clone().detach()
    Bd2.requires_grad_(True)
    x2 = torch.linalg.solve(Ad2, Bd2)
    x2.sum().backward()
    # Compare outputs
    assert torch.allclose(x, x2, rtol=RTOL)
    # Compare gradients
    nz = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz], Ad2.grad[nz], rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, rtol=RTOL)
