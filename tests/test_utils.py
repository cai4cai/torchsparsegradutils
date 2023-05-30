import torch
import unittest
from parameterized import parameterized_class

from torchsparsegradutils.utils.utils import (
    compress_row_indices,
    demcompress_crow_indices,
)

class TestRowIndicesCompressionDecompression(unittest.TestCase):
    def setUp(self) -> None:
        pass

