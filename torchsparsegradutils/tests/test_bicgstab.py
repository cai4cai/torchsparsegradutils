import torch
import unittest

from torchsparsegradutils.utils import bicgstab

import numpy as np
import scipy.sparse.linalg


class BICGSTABTest(unittest.TestCase):
    """Test bicgstab implementation"""

    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")

    def test_bicgstab(self):
        size = 100
        matrix_dense = torch.randn(size, size, dtype=torch.float64, device=self.device) + 10 * torch.eye(
            size, device=self.device
        )
        matrix_sparse = matrix_dense.to_sparse_csr()

        rhs = torch.randn(size, dtype=torch.float64, device=self.device)

        actual = torch.linalg.solve(matrix_dense, rhs)
        # print("actual",actual)

        # mn = matrix_dense.cpu().numpy()
        # rn = rhs.cpu().numpy()
        # solves = scipy.sparse.linalg.bicgstab(mn,rn)
        # print("np solves",solves)

        solves = bicgstab(matrix_dense, rhs=rhs)
        # print("our solves",solves)
        self.assertTrue(torch.allclose(solves, actual, atol=1e-3, rtol=1e-4))

        solves = bicgstab(matrix_dense.matmul, rhs=rhs)
        self.assertTrue(torch.allclose(solves, actual, atol=1e-3, rtol=1e-4))

        solves = bicgstab(matrix_sparse.matmul, rhs=rhs)
        self.assertTrue(torch.allclose(solves, actual, atol=1e-3, rtol=1e-4))

        solves = bicgstab(matrix_sparse, rhs=rhs)
        self.assertTrue(torch.allclose(solves, actual, atol=1e-3, rtol=1e-4))


class BICGSTABTestCUDA(BICGSTABTest):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        super().setUp()


if __name__ == "__main__":
    unittest.main()
