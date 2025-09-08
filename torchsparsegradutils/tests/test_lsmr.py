"""
Code imported from scipy

Copyright (C) 2010 David Fong and Michael Saunders
Distributed under the same license as SciPy

Testing Code for LSMR.

03 Jun 2010: First version release with lsmr.py

David Chin-lung Fong            clfong@stanford.edu
Institute for Computational and Mathematical Engineering
Stanford University

Michael Saunders                saunders@stanford.edu
Systems Optimization Laboratory
Dept of MS&E, Stanford University.

"""

import random

import pytest
import torch

from torchsparsegradutils.utils import lsmr

# Device fixture
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda:0"))


def _id_device(d):
    return str(d)


@pytest.fixture(params=DEVICES, ids=_id_device)
def device(request):
    return request.param


# Helper to setup SPD test problem


def _gettestproblem(dtype, device):
    n = 35
    G = torch.eye(n, dtype=dtype, device=device)
    for _ in range(5):
        gg = torch.randn(n, dtype=dtype, device=device)
        G += torch.outer(gg, gg)
    b = torch.randn(n, dtype=dtype, device=device)
    return G, b


# Build lower bidiagonal sparse COO matrix


def lowerBidiagonalMatrix(m, n, dtype, device):
    if m <= n:
        row = torch.hstack((torch.arange(m, device=device), torch.arange(1, m, device=device)))
        col = torch.hstack((torch.arange(m, device=device), torch.arange(m - 1, device=device)))
        idx = torch.vstack((row, col))
        data = torch.hstack(
            (torch.arange(1, m + 1, dtype=dtype, device=device), torch.arange(1, m, dtype=dtype, device=device))
        )
    else:
        row = torch.hstack((torch.arange(n, device=device), torch.arange(1, n + 1, device=device)))
        col = torch.hstack((torch.arange(n, device=device), torch.arange(n, device=device)))
        idx = torch.vstack((row, col))
        data = torch.hstack(
            (torch.arange(1, n + 1, dtype=dtype, device=device), torch.arange(1, n + 1, dtype=dtype, device=device))
        )
    return torch.sparse_coo_tensor(idx, data, size=(m, n))


# Verbose test harness from original


def lsmrtest(m, n, damp, dtype, device):
    A = lowerBidiagonalMatrix(m, n, dtype, device)
    xtrue = torch.arange(n, 0, -1, dtype=dtype, device=device)
    Afun = A.matmul
    b = Afun(xtrue)
    atol = 1e-7
    btol = 1e-7
    conlim = 1e10
    itnlim = 10 * n
    x = lsmr(A, b, damp=damp, atol=atol, btol=btol, conlim=conlim, maxiter=itnlim)[0]
    assert torch.allclose(Afun(x), b, atol=1e-3, rtol=1e-4)


# --- Tests for real-valued systems ---


def test_identity_case1(device):
    n = 10
    A = torch.eye(n, dtype=torch.float64, device=device)
    xtrue = torch.zeros((n, 1), dtype=torch.float64, device=device)
    b = A.matmul(xtrue)
    x = lsmr(A, b)[0]
    assert torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4)


def test_identity_case2(device):
    n = 10
    A = torch.eye(n, dtype=torch.float64, device=device)
    xtrue = torch.ones((n, 1), dtype=torch.float64, device=device)
    b = A.matmul(xtrue)
    x = lsmr(A, b)[0]
    assert torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4)


def test_identity_case3(device):
    n = 10
    A = torch.eye(n, dtype=torch.float64, device=device)
    xtrue = torch.arange(n, 0, -1, dtype=torch.float64, device=device).unsqueeze(0)
    b = A.matmul(xtrue.unsqueeze(-1)).squeeze(-1)
    x = lsmr(A, b)[0]
    assert torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4)


def test_bidiagonalA(device):
    n = 10
    m = 20
    A = lowerBidiagonalMatrix(m, n, torch.float64, device)
    xtrue = torch.arange(n, 0, -1, dtype=torch.float64, device=device)
    b = A.matmul(xtrue)
    x = lsmr(A, b)[0]
    assert torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4)


def test_scalarB(device):
    A = torch.tensor([[1.0, 2.0]], dtype=torch.float64, device=device)
    b = torch.tensor([3.0], dtype=torch.float64, device=device)
    x = lsmr(A, b)[0]
    assert torch.allclose(A.matmul(x), b, atol=1e-3, rtol=1e-4)


# --- Tests for complex systems ---


@pytest.mark.parametrize("cdtype", [torch.complex128])
def test_complexX(device, cdtype):
    n = 10
    A = torch.eye(n, dtype=cdtype, device=device)
    xtrue = torch.arange(n, 0, -1, dtype=torch.float64, device=device).to(dtype=cdtype) * (1 + 1j)
    b = A.matmul(xtrue)
    x = lsmr(A, b)[0]
    assert torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("cdtype", [torch.complex128])
def test_complexX0(device, cdtype):
    n = 10
    A = 4 * torch.eye(n, dtype=cdtype, device=device) + torch.ones((n, n), dtype=cdtype, device=device)
    xtrue = torch.arange(n, 0, -1, dtype=torch.float64, device=device).to(dtype=cdtype)
    b = A.matmul(xtrue)
    x0 = torch.zeros(n, dtype=cdtype, device=device)
    x = lsmr(A, b, x0=x0)[0]
    assert torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("cdtype", [torch.complex128])
def test_complexA(device, cdtype):
    n = 10
    A = 4 * torch.eye(n, dtype=cdtype, device=device) + 1j * torch.ones((n, n), dtype=cdtype, device=device)
    xtrue = torch.arange(n, 0, -1, dtype=torch.float64, device=device).to(dtype=cdtype)
    b = A.matmul(xtrue)
    x = lsmr(A, b)[0]
    assert torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("cdtype", [torch.complex128])
def test_complexB(device, cdtype):
    n = 10
    A = 4 * torch.eye(n, dtype=cdtype, device=device) + torch.ones((n, n), dtype=cdtype, device=device)
    xtrue = torch.arange(n, 0, -1, dtype=torch.float64, device=device).to(dtype=cdtype) * (1 + 1j)
    b = A.matmul(xtrue)
    x = lsmr(A, b)[0]
    assert torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4)


def test_columnB(device):
    n = 10
    A = torch.eye(n, dtype=torch.float64, device=device)
    b = torch.ones((n, 1), dtype=torch.float64, device=device)
    x = lsmr(A, b)[0]
    assert torch.allclose(A.matmul(x), b, atol=1e-3, rtol=1e-4)


# --- Test initialization and warm-start ---


def test_initialization(device):
    dtype = torch.float64
    G, b = _gettestproblem(dtype, device)
    x_ref = lsmr(G, b)[0]
    assert torch.allclose(G @ x_ref, b, atol=1e-3, rtol=1e-4)
    x0 = torch.zeros_like(b)
    x = lsmr(G, b, x0=x0)[0]
    assert torch.allclose(x, x_ref, atol=1e-3, rtol=1e-4)
    x0_ws = lsmr(G, b, maxiter=1)[0]
    x = lsmr(G, b, x0=x0_ws)[0]
    assert torch.allclose(G @ x, b, atol=1e-3, rtol=1e-4)


# --- Verbose run ---


def test_verbose(device):
    lsmrtest(20, 10, 0, torch.float64, device)


# --- Tests for return values and state ---


def test_unchanged_x0(device):
    dtype = torch.float64
    n = 10
    m = 20
    A = lowerBidiagonalMatrix(m, n, dtype, device)
    xtrue = torch.arange(n, 0, -1, dtype=dtype, device=device)
    Afun = A.matmul
    b = Afun(xtrue)
    x0 = torch.ones(n, dtype=dtype, device=device)
    x0_copy = x0.clone()
    _ = lsmr(A, b, x0=x0)
    assert torch.allclose(x0, x0_copy, atol=1e-3, rtol=1e-4)


def test_normr(device):
    dtype = torch.float64
    n = 10
    m = 20
    A = lowerBidiagonalMatrix(m, n, dtype, device)
    xtrue = torch.arange(n, 0, -1, dtype=dtype, device=device)
    Afun = A.matmul
    b = Afun(xtrue)
    x = lsmr(A, b)[0]
    assert torch.allclose(Afun(x), b, atol=1e-3, rtol=1e-4)


def test_normar(device):
    dtype = torch.float64
    n = 10
    m = 20
    A = lowerBidiagonalMatrix(m, n, dtype, device)
    xtrue = torch.arange(n, 0, -1, dtype=dtype, device=device)
    Afun = A.matmul
    Arfun = A.T.matmul
    b = Afun(xtrue)
    x = lsmr(A, b)[0]
    residual = b - Afun(x)
    assert torch.allclose(Arfun(residual), torch.zeros_like(x), atol=1e-3, rtol=1e-4)
