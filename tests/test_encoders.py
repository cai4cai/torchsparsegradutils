import pytest
import torch

from torchsparsegradutils.encoders.pairwise_voxel_encoder import _trim_3d, _calc_pariwise_coo_indices

# Test the _trim function, required for removing pairwise relationships beyond the volume boundary


@pytest.fixture
def tensor_3d():
    return torch.arange(27).reshape(3, 3, 3)  # a 3D tensor


@pytest.mark.parametrize(
    "offsets, expected_output_slice",
    [
        ((0, 0, 1), (slice(None, None), slice(None, None), slice(1, None))),
        ((0, 0, -1), (slice(None, None), slice(None, None), slice(None, -1))),
        ((0, 0, -2), (slice(None, None), slice(None, None), slice(None, -2))),
        ((1, -2, 1), (slice(1, None), slice(None, -2), slice(1, None))),
        ((1, -2), None),  # this should raise a ValueError
    ],
)
def test_trim(tensor_3d, offsets, expected_output_slice):
    if expected_output_slice is None:
        with pytest.raises(ValueError):
            output = _trim_3d(tensor_3d, offsets)
    else:
        output = _trim_3d(tensor_3d, offsets)
        expected_output = tensor_3d[expected_output_slice]
        assert torch.all(output == expected_output)
