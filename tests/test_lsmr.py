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

import torch
import unittest

from torchsparsegradutils.utils import lsmr

# from numpy import array, arange, eye, zeros, ones, sqrt, transpose, hstack
# from numpy.linalg import norm
# from numpy.testing import assert_allclose
# import pytest
# from scipy.sparse import coo_matrix
# from scipy.sparse.linalg._interface import aslinearoperator
# from scipy.sparse.linalg import lsmr
# from .test_lsqr import G, b


def _gettestproblem(dtype, device):
    # Set up a test problem
    n = 35
    G = torch.eye(n, dtype=dtype, device=device)

    for jj in range(5):
        gg = torch.randn(n, dtype=dtype, device=device)
        hh = torch.outer(gg, gg)
        G += hh
        # G += (hh + hh.reshape(-1,1)) * 0.5
        # G += torch.randn(n, dtype=dtype, device=device) * torch.randn(n, dtype=dtype, device=device)

    b = torch.randn(n, dtype=dtype, device=device)

    return G, b


class TestLSMR(unittest.TestCase):
    def setUp(self):
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
        self.n = 10
        self.m = 10
        self.dtype = torch.float64
        self.cdtype = torch.complex128

    def assertCompatibleSystem(self, A, xtrue):
        Afun = A.matmul
        b = Afun(xtrue)
        x = lsmr(A, b)[0]
        # print("A",A)
        # print("b",b)
        # print("x",x)
        # print("xtrue",xtrue)
        self.assertTrue(torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4))

    def testIdentityACase1(self):
        A = torch.eye(self.n, dtype=self.dtype, device=self.device)
        xtrue = torch.zeros((self.n, 1), dtype=self.dtype, device=self.device)
        self.assertCompatibleSystem(A, xtrue)

    def testIdentityACase2(self):
        A = torch.eye(self.n, dtype=self.dtype, device=self.device)
        xtrue = torch.ones((self.n, 1), dtype=self.dtype, device=self.device)
        self.assertCompatibleSystem(A, xtrue)

    def testIdentityACase3(self):
        A = torch.eye(self.n, dtype=self.dtype, device=self.device)
        xtrue = torch.t(torch.arange(self.n, 0, -1, dtype=self.dtype, device=self.device))
        self.assertCompatibleSystem(A, xtrue)

    def testBidiagonalA(self):
        A = lowerBidiagonalMatrix(20, self.n, self.dtype, self.device)
        xtrue = torch.t(torch.arange(self.n, 0, -1, dtype=self.dtype, device=self.device))
        self.assertCompatibleSystem(A, xtrue)

    def testScalarB(self):
        A = torch.tensor([[1.0, 2.0]], dtype=self.dtype, device=self.device)
        b = torch.tensor([3.0], dtype=self.dtype, device=self.device)
        x = lsmr(A, b)[0]
        self.assertTrue(torch.allclose(A.matmul(x), b, atol=1e-3, rtol=1e-4))

    def testComplexX(self):
        A = torch.eye(self.n, dtype=self.cdtype, device=self.device)
        xtrue = torch.t(
            torch.arange(self.n, 0, -1, dtype=self.dtype, device=self.device).to(dtype=self.cdtype) * (1 + 1j)
        )
        self.assertCompatibleSystem(A, xtrue)

    def testComplexX0(self):
        A = 4 * torch.eye(self.n, dtype=self.cdtype, device=self.device) + torch.ones(
            (self.n, self.n), dtype=self.cdtype, device=self.device
        )
        xtrue = torch.t(torch.arange(self.n, 0, -1, dtype=self.dtype, device=self.device).to(dtype=self.cdtype))
        b = A.matmul(xtrue)
        x0 = torch.zeros(self.n, dtype=self.cdtype, device=self.device)
        x = lsmr(A, b, x0=x0)[0]
        self.assertTrue(torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4))

    def testComplexA(self):
        A = 4 * torch.eye(self.n, dtype=self.cdtype, device=self.device) + 1j * torch.ones(
            (self.n, self.n), dtype=self.cdtype, device=self.device
        )
        xtrue = torch.t(torch.arange(self.n, 0, -1, dtype=self.dtype, device=self.device).to(dtype=self.cdtype))
        self.assertCompatibleSystem(A, xtrue)

    def testComplexB(self):
        A = 4 * torch.eye(self.n, dtype=self.cdtype, device=self.device) + torch.ones(
            (self.n, self.n), dtype=self.cdtype, device=self.device
        )
        xtrue = torch.t(
            torch.arange(self.n, 0, -1, dtype=self.dtype, device=self.device).to(dtype=self.cdtype) * (1 + 1j)
        )
        b = A.matmul(xtrue)
        x = lsmr(A, b)[0]
        self.assertTrue(torch.allclose(x, xtrue, atol=1e-3, rtol=1e-4))

    def testColumnB(self):
        A = torch.eye(self.n, dtype=self.dtype, device=self.device)
        b = torch.ones((self.n, 1), dtype=self.dtype, device=self.device)
        x = lsmr(A, b)[0]
        self.assertTrue(torch.allclose(A.matmul(x), b, atol=1e-3, rtol=1e-4))

    def testInitialization(self):
        G, b = _gettestproblem(self.dtype, self.device)
        # Test that the default setting is not modified
        # x_ref, _, itn_ref, normr_ref, *_ = lsmr(G, b)
        x_ref = lsmr(G, b)[0]
        self.assertTrue(torch.allclose(G @ x_ref, b, atol=1e-3, rtol=1e-4))

        # Test passing zeros yields similiar result
        x0 = torch.zeros(b.shape, dtype=self.dtype, device=self.device)
        x = lsmr(G, b, x0=x0)[0]
        self.assertTrue(torch.allclose(x, x_ref, atol=1e-3, rtol=1e-4))

        # Test warm-start with single iteration
        x0 = lsmr(G, b, maxiter=1)[0]

        # x, _, itn, normr, *_ = lsmr(G, b, x0=x0)
        x = lsmr(G, b, x0=x0)[0]
        self.assertTrue(torch.allclose(G @ x, b, atol=1e-3, rtol=1e-4))

        # NOTE(gh-12139): This doesn't always converge to the same value as
        # ref because error estimates will be slightly different when calculated
        # from zeros vs x0 as a result only compare norm and itn (not x).

        # x generally converges 1 iteration faster because it started at x0.
        # itn == itn_ref means that lsmr(x0) took an extra iteration see above.
        # -1 is technically possible but is rare (1 in 100000) so it's more
        # likely to be an error elsewhere.
        # assert itn - itn_ref in (0, 1)

        # If an extra iteration is performed normr may be 0, while normr_ref
        # may be much larger.
        # assert normr < normr_ref * (1 + 1e-6)

    def testVerbose(self):
        lsmrtest(20, 10, 0, self.dtype, self.device)


class TestLSMRCUDA(TestLSMR):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        super().setUp()


class TestLSMRReturns(unittest.TestCase):
    def setUp(self):
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.n = 10
        self.A = lowerBidiagonalMatrix(20, self.n, self.dtype, self.device)
        self.xtrue = torch.t(torch.arange(self.n, 0, -1, dtype=self.dtype, device=self.device))
        self.Afun = self.A.matmul
        self.Arfun = self.A.T.matmul
        self.b = self.Afun(self.xtrue)
        self.x0 = torch.ones(self.n, dtype=self.dtype, device=self.device)
        self.x00 = self.x0.clone()
        self.returnValues = lsmr(self.A, self.b)
        self.returnValuesX0 = lsmr(self.A, self.b, x0=self.x0)

    def test_unchanged_x0(self):
        # x, istop, itn, normr, normar, normA, condA, normx = self.returnValuesX0
        # x = self.returnValuesX0[0]  # variable unused
        self.assertTrue(torch.allclose(self.x00, self.x0, atol=1e-3, rtol=1e-4))

    def testNormr(self):
        # x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        x = self.returnValues[0]
        self.assertTrue(torch.allclose(self.Afun(x), self.b, atol=1e-3, rtol=1e-4))

    def testNormar(self):
        # x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
        x = self.returnValuesX0[0]
        self.assertTrue(
            torch.allclose(
                self.Arfun(self.b - self.Afun(x)),
                torch.zeros(self.n, dtype=self.dtype, device=self.device),
                atol=1e-3,
                rtol=1e-4,
            )
        )

    # def testNormx(self):
    #    x, istop, itn, normr, normar, normA, condA, normx = self.returnValues
    #    assert norm(x) == pytest.approx(normx)


class TestLSMRReturnsCUDA(TestLSMRReturns):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        super().setUp()


def lowerBidiagonalMatrix(m, n, dtype, device):
    # This is a simple example for testing LSMR.
    # It uses the leading m*n submatrix from
    # A = [ 1
    #       1 2
    #         2 3
    #           3 4
    #             ...
    #               n ]
    # suitably padded by zeros.
    #
    # 04 Jun 2010: First version for distribution with lsmr.py
    if m <= n:
        row = torch.hstack((torch.arange(m, dtype=int, device=device), torch.arange(1, m, dtype=int, device=device)))
        col = torch.hstack((torch.arange(m, dtype=int, device=device), torch.arange(m - 1, dtype=int, device=device)))
        idx = torch.vstack((row, col))
        data = torch.hstack(
            (torch.arange(1, m + 1, dtype=dtype, device=device), torch.arange(1, m, dtype=dtype, device=device))
        )
        return torch.sparse_coo_tensor(idx, data, size=(m, n))
    else:
        row = torch.hstack(
            (torch.arange(n, dtype=int, device=device), torch.arange(1, n + 1, dtype=int, device=device))
        )
        col = torch.hstack((torch.arange(n, dtype=int, device=device), torch.arange(n, dtype=int, device=device)))
        idx = torch.vstack((row, col))
        data = torch.hstack(
            (torch.arange(1, n + 1, dtype=dtype, device=device), torch.arange(1, n + 1, dtype=dtype, device=device))
        )
        return torch.sparse_coo_tensor(idx, data, size=(m, n))


def lsmrtest(m, n, damp, dtype, device):
    """Verbose testing of lsmr"""

    A = lowerBidiagonalMatrix(m, n, dtype, device)
    xtrue = torch.arange(n, 0, -1, dtype=dtype, device=device)
    Afun = A.matmul

    b = Afun(xtrue)

    atol = 1.0e-7
    btol = 1.0e-7
    conlim = 1.0e10
    itnlim = 10 * n
    # show = 1  # variable unused

    # x, istop, itn, normr, normar, norma, conda, normx \
    #  = lsmr(A, b, damp, atol, btol, conlim, itnlim, show)
    x = lsmr(A, b, damp=damp, atol=atol, btol=btol, conlim=conlim, maxiter=itnlim)[0]

    j1 = min(n, 5)
    j2 = max(n - 4, 1)
    print(" ")
    print("First elements of x:")
    str = ["%10.4f" % (xi) for xi in x[0:j1]]
    print("".join(str))
    print(" ")
    print("Last  elements of x:")
    str = ["%10.4f" % (xi) for xi in x[j2 - 1 :]]
    print("".join(str))

    r = b - Afun(x)
    r2 = torch.sqrt(torch.norm(r) ** 2 + (damp * torch.norm(x)) ** 2)
    print(" ")
    # str = 'normr (est.)  %17.10e' % (normr)
    str2 = "normr (true)  %17.10e" % (r2)
    # print(str)
    print(str2)
    print(" ")


if __name__ == "__main__":
    # lsmrtest(20,10,0)
    unittest.main()
