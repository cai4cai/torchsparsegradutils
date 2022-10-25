import unittest

# import torch

import torchsparsegradutils

# # set deterministic seed
# torch.manual_seed(15)


class TestDummyInterface(unittest.TestCase):
    def test_dummy_shape(self):
        torchsparsegradutils.dummy_interface()

if __name__ == "__main__":
    unittest.main()