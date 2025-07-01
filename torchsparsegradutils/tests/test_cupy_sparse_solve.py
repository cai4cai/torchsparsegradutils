import torch
import torchsparsegradutils as tsgu
import torchsparsegradutils.cupy as tsgucupy
import warnings
import pytest

if tsgucupy.have_cupy:
    import cupy as cp
    import cupyx.scipy.sparse as csp
else:
    warnings.warn(
        "Importing optional cupy-related module failed to find cupy -> cupy-related tests running as numpy only."
    )

import numpy as np
import scipy.sparse as nsp

# warn if cupy missing
if not tsgucupy.have_cupy:
    warnings.warn(
        "Importing optional cupy-related module failed to find cupy -> cupy-related tests running as numpy only."
    )

# Device fixture
DEVICES = [torch.device('cpu')]
if torch.cuda.is_available():
    DEVICES.append(torch.device('cuda:0'))

def _id_device(d):   return str(d)

@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param

# relative tolerance for comparisons
RTOL = 1e-3

def _setup(device):
    # common setup
    A = torch.randn((4, 4), dtype=torch.float64, device=device)
    A = A + A.t()
    A_csr = A.to_sparse_csr()
    B = torch.randn((4, 2), dtype=torch.float64, device=device)
    x_ref = torch.linalg.solve(A, B)
    return A_csr, B, x_ref

def test_solver_c4t(device):
    A_csr, B, x_ref = _setup(device)
    x = tsgucupy.sparse_solve_c4t(A_csr.to(torch.float32), B.to(torch.float32))
    assert torch.allclose(x, x_ref.to(torch.float32), rtol=RTOL)

def test_solver_gradient_c4t(device):
    A_csr, B, _ = _setup(device)
    # sparse solver gradient
    As1 = A_csr.to(torch.float32).detach().clone(); As1.requires_grad_(True)
    Bd1 = B.to(torch.float32).detach().clone(); Bd1.requires_grad_(True)
    x = tsgucupy.sparse_solve_c4t(As1, Bd1)
    x.sum().backward()
    # dense reference gradient
    A = As1.to_dense().detach().clone(); A.requires_grad_(True)
    Bd2 = Bd1.detach().clone(); Bd2.requires_grad_(True)
    x2 = torch.linalg.solve(A, Bd2)
    x2.sum().backward()
    # compare results
    assert torch.allclose(x, x2, rtol=RTOL)
    nz = As1.grad.to_dense() != 0.0
    assert torch.allclose(As1.grad.to_dense()[nz], A.grad[nz], rtol=RTOL)
    assert torch.allclose(Bd1.grad, Bd2.grad, rtol=RTOL)
