import unittest

# import torch

import torchsparseutils

# # set deterministic seed
# torch.manual_seed(15)


class TestDummyInterface(unittest.TestCase):
    def test_dummy_shape(self):
        torchsparseutils.dummy_interface()

if __name__ == "__main__":
    unittest.main()