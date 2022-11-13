import torch
import unittest
import torchsparsegradutils as tsgu
import torchsparsegradutils.jax as tsgujax

if not tsgujax.have_jax:
    raise unittest.SkipTest("Importing optional jax-related module failed to find jax -> skipping jax-related tests.")

import numpy as np

import jax
import jax.numpy as jnp


class J2TIOTest(unittest.TestCase):
    """IO conversion tests between torch and jax"""

    def setUp(self) -> None:
        # The device can be specialised by a daughter class
        if not hasattr(self, "device"):
            self.device = torch.device("cpu")
            self.device_j = jax.devices("cpu")[0]
        self.x_shape = (4, 4)
        self.x_t = torch.randn(self.x_shape, dtype=torch.float64, device=self.device)
        self.x_j = jax.device_put(np.random.randn(self.x_shape[0], self.x_shape[1]), device=self.device_j)
        # print(self.x_j.device_buffer.device())

    def test_t2j(self):
        x_j = tsgujax.t2j(self.x_t)
        self.assertTrue(x_j.shape == self.x_t.shape)
        self.assertTrue(np.isclose(np.asarray(x_j), self.x_t.numpy()).all())
        x_t2j2t = tsgujax.j2t(x_j)
        self.assertTrue(x_t2j2t.shape == self.x_t.shape)
        self.assertTrue(np.isclose(x_t2j2t.numpy(), self.x_t.numpy()).all())

    def test_j2t(self):
        x_t = tsgujax.j2t(self.x_j)
        self.assertTrue(x_t.shape == self.x_j.shape)
        self.assertTrue(np.isclose(np.asarray(self.x_j), x_t.numpy()).all())
        x_j2t2j = tsgujax.t2j(x_t)
        self.assertTrue(x_j2t2j.shape == self.x_j.shape)
        self.assertTrue(np.isclose(np.asarray(x_j2t2j), np.asarray(self.x_j)).all())


class J2TIOTestCUDA(J2TIOTest):
    """Override superclass setUp to run on GPU"""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest(f"Skipping {self.__class__.__name__} since CUDA is not available")
        self.device = torch.device("cuda")
        self.device_j = jax.devices("gpu")[0]
        super().setUp()


if __name__ == "__main__":
    unittest.main()
