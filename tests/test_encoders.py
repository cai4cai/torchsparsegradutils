import pytest
import torch
import json
import os

from torchsparsegradutils.encoders.pairwise_voxel_encoder import _trim_3d, _calc_pariwise_coo_indices, NEIGHBOURS
from torchsparsegradutils.utils.utils import _sort_coo_indices

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


# Test the _calc_pariwise_coo_indices function, required for calculating the pairwise relationships


with open("tests/test_params/pariwise_coo_indices.json") as f:  # load test cases from file to avoid a massive mess
    print(os.getcwd())
    pairwise_coo_indices_test_cases = json.load(f)

params = [tuple(tc.values())[1:] for tc in pairwise_coo_indices_test_cases]  # Skip the first value, which is 'id'
ids = [tc["id"] for tc in pairwise_coo_indices_test_cases]  # Extract 'id' separately


@pytest.mark.parametrize(
    "nneigh, vshape, batch_size, nchannels, upper, diag, channel_relation, dtype, device, expected_idx",
    params,
    ids=ids,
)
def test_pariwise_coo_indices(
    nneigh, vshape, batch_size, nchannels, upper, diag, channel_relation, dtype, device, expected_idx
):
    neighbours = NEIGHBOURS[:nneigh] if nneigh > 1 else list((NEIGHBOURS[0],))
    vshape = tuple(vshape)
    expected_idx = torch.tensor(expected_idx, dtype=getattr(torch, dtype), device=device)
    idx = _calc_pariwise_coo_indices(neighbours, vshape, batch_size, nchannels, diag, upper, dtype=dtype, device=device)
    expected_idx, _ = _sort_coo_indices(expected_idx)

    if upper:
        assert (idx[0] <= idx[1]).all()
    else:
        assert (idx[0] >= idx[1]).all()

    assert torch.equal(idx, expected_idx)
    assert idx.dtype == expected_idx.dtype
    assert idx.device.type == expected_idx.device.type
