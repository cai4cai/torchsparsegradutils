import pytest
import torch
import json
import yaml
import os
from pathlib import Path
from ast import literal_eval

from functools import reduce
from operator import mul

from torchsparsegradutils.encoders.pairwise_voxel_encoder import (
    _trim_nd,
    _gen_coords,
    _gen_offsets,
    calc_pariwise_coo_indices,
    PairwiseVoxelEncoder,
)
from torchsparsegradutils.utils.utils import _sort_coo_indices

if torch.__version__ >= (2,):
    # https://pytorch.org/docs/stable/generated/torch.sparse.check_sparse_tensor_invariants.html
    torch.sparse.check_sparse_tensor_invariants.enable()

# Overall testing parameters:
DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))

INDICES_DTYPES = [torch.int32, torch.int64]
VALUES_DTYPES = [torch.float32, torch.float64]
SPASRE_LAYOUTS = [torch.sparse_coo, torch.sparse_csr]

RADII = [1, 1.5, 2]
VOLUME_SHAPES = [(3, 5, 5, 5), (5, 7, 7, 7)]
BATCH_SIZES = [None, 1, 4]
DIAG = [True, False]
UPPER = [None, True, False]
CHANNEL_VOXEL_RELATION = ["indep", "intra", "inter"]

# Define id functions


def upper_id(upper):
    if upper is None:
        return "full"
    elif upper:
        return "upper"
    else:
        return "lower"


# Define testing fixtures


@pytest.fixture(params=RADII, ids=lambda x: f"r{x}")
def radius(request):
    return request.param


@pytest.fixture(params=VOLUME_SHAPES, ids=lambda x: f"v{x}")
def volume_shape(request):
    return request.param


@pytest.fixture(params=DIAG, ids=lambda x: "diag" if x else "")
def diag(request):
    return request.param


@pytest.fixture(params=BATCH_SIZES, ids=lambda x: f"b{x}" if x else "b0")
def batch_size(request):
    return request.param


@pytest.fixture(params=UPPER, ids=upper_id)
def upper(request):
    return request.param


@pytest.fixture(params=CHANNEL_VOXEL_RELATION, ids=lambda x: x)
def channel_voxel_relation(request):
    return request.param


@pytest.fixture(params=DEVICES, ids=lambda x: x.type)
def device(request):
    return request.param


@pytest.fixture(params=INDICES_DTYPES, ids=lambda x: str(x).split(".")[-1])
def indices_dtype(request):
    return request.param


@pytest.fixture(params=VALUES_DTYPES, ids=lambda x: str(x).split(".")[-1])
def values_dtype(request):
    return request.param


@pytest.fixture(params=SPASRE_LAYOUTS, ids=lambda x: str(x).split(".")[-1].split("_")[-1].upper())
def layout(request):
    return request.param


# Test the _trim function, required for removing pairwise relationships beyond the volume boundary


# Fixture for tensors of different dimensions
@pytest.fixture
def tensor_nd(request):
    dim = request.param
    size = 3  # Size of each dimension
    tensor = torch.arange(size**dim).reshape(*([size] * dim))  # Create an nD tensor
    return tensor


@pytest.mark.parametrize(
    "tensor_nd, offsets, expected_output_slice",
    [
        (3, (0, 0, 1), (slice(None, None), slice(None, None), slice(1, None))),
        (3, (0, 0, -1), (slice(None, None), slice(None, None), slice(None, -1))),
        (3, (0, 0, -2), (slice(None, None), slice(None, None), slice(None, -2))),
        (3, (1, -2, 1), (slice(1, None), slice(None, -2), slice(1, None))),
        (3, (1, -2), None),  # this should raise a ValueError
        (2, (1, -1), (slice(1, None), slice(None, -1))),
        (2, (2, 2), (slice(2, None), slice(2, None))),
        (4, (0, 0, 1, -1), (slice(None, None), slice(None, None), slice(1, None), slice(None, -1))),
        (4, (1, -1, 1, -1), (slice(1, None), slice(None, -1), slice(1, None), slice(None, -1))),
    ],
    indirect=["tensor_nd"],
    ids=[
        "3D_tensor_trim_last_dim_positive_offset",
        "3D_tensor_trim_last_dim_negative_offset",
        "3D_tensor_trim_last_dim_larger_negative_offset",
        "3D_tensor_mixed_offsets",
        "3D_tensor_mismatched_offsets_and_dim_count",
        "2D_tensor_positive_and_negative_offsets",
        "2D_tensor_same_positive_offsets",
        "4D_tensor_mixed_offsets_first_dim_zero",
        "4D_tensor_mixed_offsets_first_dim_positive",
    ],
)
def test_trim(tensor_nd, offsets, expected_output_slice):
    if expected_output_slice is None:
        with pytest.raises(ValueError):
            output = _trim_nd(tensor_nd, offsets)
    else:
        output = _trim_nd(tensor_nd, offsets)
        expected_output = tensor_nd[expected_output_slice]
        assert torch.all(output == expected_output)


# Test neighbourgood coordinate generation:

# Get the absolute path to the directory of the current module:
current_dir = Path(os.path.abspath(os.path.dirname(__file__)))

# Construct the path to the yaml file:
yaml_file = current_dir / "test_params" / "xyz_coords.yaml"

with open(yaml_file) as f:  # load test cases from file
    coord_test_cases = yaml.safe_load(f)

params = [tuple(tc.values())[1:] for tc in coord_test_cases]  # Skip the first value, which is 'id'
ids = [tc["id"] for tc in coord_test_cases]  # Extract 'id' separately


@pytest.mark.parametrize(
    "radius, expected_coords",
    params,
    ids=ids,
)
def test_gen_coords(radius, expected_coords):
    coords = _gen_coords(radius)
    expected_coords = [tuple(e) for e in expected_coords]
    assert set(coords) == set(expected_coords)


# Test neighbourgood offset generation:

yaml_file = current_dir / "test_params" / "czyx_shifts.yaml"

with open(yaml_file) as f:  # load test cases from file
    shift_test_cases = yaml.safe_load(f)

params = [tuple(tc.values())[1:] for tc in shift_test_cases]  # Skip the first value, which is 'id'
ids = [tc["id"] for tc in shift_test_cases]  # Extract 'id' separately


@pytest.mark.parametrize(
    "radius, upper, num_channels, channel_voxel_relation, expected_shifts",
    params,
    ids=ids,
)
def test_gen_offsets(radius, upper, num_channels, channel_voxel_relation, expected_shifts):
    offsets = _gen_offsets(radius, upper, num_channels, channel_voxel_relation)
    expected_shifts = [tuple(e) for e in expected_shifts]
    assert offsets == expected_shifts


# Test the calc_pariwise_coo_indices function:


# Test invalid inputs:
@pytest.mark.parametrize(
    "radius, volume_shape, dtype, channel_voxel_relation",
    [
        (0.5, (1, 1, 1, 1), torch.int64, "indep"),  # radius less than 1
        (1, (1, 1, 1), torch.int64, "indep"),  # volume_shape length not 4
        (1, (1, 0, 1, 1), torch.int64, "indep"),  # volume_shape contains non-positive integer
        (1, (1, 1, 1, 1), torch.int64, "invalid"),  # invalid channel_voxel_relation
        (1, (1, 1, 1, 1), torch.int64, "intra"),  # single channel non 'indep' relation
    ],
)
def test_pariwise_coo_indices_invalid_inputs(radius, volume_shape, dtype, channel_voxel_relation):
    with pytest.raises(ValueError):
        calc_pariwise_coo_indices(radius, volume_shape, dtype=dtype, channel_voxel_relation=channel_voxel_relation)


# Test dtype and devices:
def test_pairwise_coo_indices_dtype_and_device(indices_dtype, device):
    radius = 1
    volume_shape = (2, 2, 2, 2)
    indices_dict = calc_pariwise_coo_indices(
        radius, volume_shape, diag=True, upper=False, dtype=indices_dtype, device=device
    )
    for k, indices in indices_dict.items():
        assert indices.dtype == indices_dtype
        assert indices.device.type == device.type


# Test upper flag:
def test_pairwise_coo_indices_upper_lower(radius, volume_shape, diag, upper, channel_voxel_relation):
    indices_dict = calc_pariwise_coo_indices(radius, volume_shape, diag, upper, channel_voxel_relation)
    for k, indices in indices_dict.items():
        if upper is not None:
            if upper and not diag:
                assert (indices[0] < indices[1]).all()
            elif not upper and not diag:
                assert (indices[0] > indices[1]).all()
            elif upper and diag:
                assert (indices[0] <= indices[1]).all()
            elif not upper and diag:
                assert (indices[0] >= indices[1]).all()


# Test all the indices are unique:
def test_pairwise_coo_indices_unique(radius, volume_shape, diag, upper, channel_voxel_relation):
    indices_dict = calc_pariwise_coo_indices(radius, volume_shape, diag, upper, channel_voxel_relation)
    all_indices = []
    total_indices_count = 0
    for k, indices in indices_dict.items():
        # Convert the indices to tuples and add to the list
        indices_tuples = [tuple(x) for x in indices.t().tolist()]
        all_indices.extend(indices_tuples)
        total_indices_count += indices.size(1)

    assert len(set(all_indices)) == total_indices_count


# Test the indices generated are as expected for a simple (3, 2, 2, 2) volume:

yaml_file = current_dir / "test_params" / "pairwise_coo_indices.yaml"

with open(yaml_file) as f:  # load test cases from file to avoid a massive mess
    data = yaml.safe_load(f)

test_cases = data["test_cases"]
params = [tuple(tc.values())[1:] for tc in test_cases]  # Skip the first value, which is 'id'
ids = [tc["id"] for tc in test_cases]  # Extract 'id' separately


@pytest.mark.parametrize(
    "radius, volume_shape, diag, upper, channel_relation, expected_indices",
    params,
    ids=ids,
)
def test_pariwise_coo_indices(radius, volume_shape, diag, upper, channel_relation, expected_indices):
    volume_shape = tuple(volume_shape)

    # Convert string keys to tuples and values to tensors:
    expected_indices_dict = {
        literal_eval(k): torch.tensor(v).reshape(2, -1) for k, v in expected_indices.items()
    }  # reshape just required to make empty tensors have shape (2, 0)

    indices_dict = calc_pariwise_coo_indices(radius, volume_shape, diag, upper, channel_relation)

    assert indices_dict.keys() == expected_indices_dict.keys()

    for k, indices in indices_dict.items():
        expected_indices = expected_indices_dict[k]

        # Sort to ensure that the indices are in the same order
        indices, _ = _sort_coo_indices(indices)
        expected_indices, _ = _sort_coo_indices(expected_indices)

        try:
            assert torch.equal(indices, expected_indices)

        except AssertionError as e:
            raise AssertionError(f"Assertion failed for key {k}: {e}")


# Test the PairwiseVoxelEncoder class:


# Test just the initialisation of the class:
def test_PVE_init():
    pve = PairwiseVoxelEncoder(1.5, (1, 2, 3, 4))

    assert pve.radius == 1.5
    assert pve.volume_shape == (1, 2, 3, 4)
    assert pve.diag is False
    assert pve.upper is None
    assert pve.channel_voxel_relation == "indep"
    assert pve.layout == torch.sparse_coo
    assert pve.indices_dtype == torch.int64
    assert pve.device == torch.device("cpu")


# Test invalid inputs to pairwise voxel encoder:


@pytest.mark.parametrize(
    "radius,volume_shape,layout,indices_dtype,expected_error",
    [
        (1.0, (4, 5, 5, 0), torch.sparse_coo, torch.int64, ValueError),  # volume shape with a dimension of 0
        (1.0, (4, 5, -5, 5), torch.sparse_coo, torch.int64, ValueError),  # volume shape with a negative dimension
        (1.0, (4, 5, "5", 5), torch.sparse_coo, torch.int64, ValueError),  # volume shape with a string dimension
        (1.0, (4, 5, 5), torch.sparse_coo, torch.int64, ValueError),  # volume shape with too few dimensions
        (1.0, (4, 5, 5, 5, 5, 5), torch.sparse_coo, torch.int64, ValueError),  # volume shape with too many dimensions
        (1.0, (4, 5, 5, 5), torch.sparse_coo, torch.float32, ValueError),  # indices_dtype is a float
        (1.0, (4, 5, 5, 5), torch.sparse_coo, torch.bool, ValueError),  # indices_dtype is bool
        (1.0, (4, 5, 5, 5), torch.sparse_coo, torch.int32, ValueError),  # int32 is not allowed with COO layout
    ],
)
def test_PVE_init_invalid_inputs(radius, volume_shape, layout, indices_dtype, expected_error):
    with pytest.raises(expected_error):
        PairwiseVoxelEncoder(
            radius=radius,
            volume_shape=volume_shape,
            layout=layout,
            indices_dtype=indices_dtype,
        )


@pytest.mark.parametrize(
    "values,expected_error",
    [
        (torch.randn((4, 5, 5, 5)), ValueError),  # too few dimensions
        (torch.randn((4, 5, 6, 5, 5, 5, 5)), ValueError),  # too many dimensions
        (torch.randn((4, 6, 5, 4, 5, 5)), ValueError),  # Last four dimensions do not match the volume shape
        (
            torch.randn((4, 7, 5, 5, 5, 5)),
            ValueError,
        ),  # The shape of values at index -5 does not match the number of offsets
        (
            torch.randn((4, 6, 5, 5, 5, 5), dtype=torch.float16),
            ValueError,
        ),  # The data type of values is not either torch.float32 or torch.float64
    ],
)
def test_PVE_call_invalid_inputs(values, expected_error):
    encoder = PairwiseVoxelEncoder(
        radius=1.0,
        volume_shape=(5, 5, 5, 5),
    )

    with pytest.raises(expected_error):
        encoder(values)


def test_PVE_dtype_device(batch_size, layout, indices_dtype, values_dtype, device):
    if (indices_dtype == torch.int32) and (layout == torch.sparse_coo):
        pytest.skip("Sparse COO with int32 indices is not supported")

    encoder = PairwiseVoxelEncoder(
        radius=1.0,
        volume_shape=(5, 5, 5, 5),
        layout=layout,
        indices_dtype=indices_dtype,
        device=device,
    )

    num_offsets = len(encoder.offsets)

    values_shape = (batch_size, num_offsets, 5, 5, 5, 5) if batch_size else (num_offsets, 5, 5, 5, 5)
    values = torch.randn(values_shape, dtype=values_dtype, device=device)

    sparse_matrix = encoder(values)

    assert sparse_matrix.dtype == values_dtype
    assert sparse_matrix.device.type == device.type
    assert sparse_matrix.values().dtype == values_dtype
    assert sparse_matrix.values().device.type == device.type

    if layout == torch.sparse_coo:
        assert sparse_matrix.indices().dtype == indices_dtype
        assert sparse_matrix.indices().device.type == device.type

    elif layout == torch.sparse_csr:
        assert sparse_matrix.crow_indices().dtype == indices_dtype
        assert sparse_matrix.crow_indices().device.type == device.type
        assert sparse_matrix.col_indices().dtype == indices_dtype
        assert sparse_matrix.col_indices().device.type == device.type


def test_PVE_to_device(layout, device):
    encoder = PairwiseVoxelEncoder(
        radius=1.0,
        volume_shape=(5, 5, 5, 5),
        layout=layout,
        indices_dtype=torch.int64,
        device=torch.device("cpu"),
    )
    encoder.to(device)

    if layout == torch.sparse_coo:
        assert encoder.indices.device.type == device.type

    elif layout == torch.sparse_csr:
        assert encoder.crow_indices.device.type == device.type
        assert encoder.col_indices.device.type == device.type
        assert encoder.csr_permutation.device.type == device.type


# Test based on expected indices in pairwise_coo_indices.yaml
@pytest.mark.parametrize(
    "radius, volume_shape, diag, upper, channel_relation, expected_indices",
    params,
    ids=ids,
)
def test_PVE_values(radius, volume_shape, diag, upper, channel_relation, expected_indices, batch_size, layout):
    indices_dtype = torch.int64
    values_dtype = torch.float64
    device = torch.device("cpu")

    expected_indices_dict = {
        literal_eval(k): torch.tensor(v, dtype=indices_dtype, device=device).reshape(2, -1)
        for k, v in expected_indices.items()
    }  # reshape just required to make empty tensors have shape (2, 0)
    volume_shape = tuple(volume_shape)

    pve = PairwiseVoxelEncoder(radius, volume_shape, diag, upper, channel_relation, layout, indices_dtype, device)
    num_offsets = len(pve.offsets)

    # create tensor with different values for each offset
    if batch_size is None:
        values_list = [
            torch.full((1, *volume_shape), i, dtype=values_dtype, device=device) for i in range(1, num_offsets + 1)
        ]
        values = torch.cat(values_list, dim=0)
    else:
        values_list = [
            torch.full((1, *volume_shape), i, dtype=values_dtype, device=device)
            for i in range(1, num_offsets * batch_size + 1)
        ]
        values = torch.cat(values_list, dim=0).reshape(batch_size, num_offsets, *volume_shape)

    sparse_matrix = pve(values)

    # Check all values are in the correct place based on the expected indices

    if batch_size is None:
        for i, expected_indices in enumerate(expected_indices_dict.values(), start=1):
            # Not the most efficient, but it works
            for idx in expected_indices.t():
                assert sparse_matrix[idx[0], idx[1]] == i
    else:
        i = 1
        for b in range(batch_size):
            for expected_indices in expected_indices_dict.values():
                for idx in expected_indices.t():
                    assert sparse_matrix[b, idx[0], idx[1]] == i
                i += 1


def test_PVE_COO_vs_CSR():
    volume_shape = (5, 5, 5, 5)
    pve_coo = PairwiseVoxelEncoder(1, volume_shape, layout=torch.sparse_coo)
    pve_csr = PairwiseVoxelEncoder(1, volume_shape, layout=torch.sparse_csr)
    values = torch.randn((6, 5, 5, 5, 5))
    matrix_coo = pve_coo(values)
    matrix_csr = pve_csr(values)
    assert torch.allclose(matrix_coo.to_dense(), matrix_csr.to_dense())


def test_PVE_size(volume_shape, layout, batch_size):
    pve = PairwiseVoxelEncoder(1, volume_shape, layout=layout)

    volume_numel = reduce(mul, volume_shape)
    if batch_size is None:
        expected_size = torch.Size((volume_numel, volume_numel))
        values = torch.randn((6, *volume_shape))
    else:
        expected_size = torch.Size((batch_size, volume_numel, volume_numel))
        values = torch.randn((batch_size, 6, *volume_shape))

    matrix = pve(values)
    assert matrix.size() == expected_size


def test_PVE_diagonal_consistency(radius, volume_shape, layout, batch_size, channel_voxel_relation):
    # NOTE: this test fails is the volume size is too small relative to the radius, not sure why
    # eg: (3, 2, 2, 2) fails for radius 1.5 and 2.0
    upper = False
    diag = True
    indices_dtype = torch.int64
    values_dtype = torch.float64
    device = torch.device("cpu")

    pve = PairwiseVoxelEncoder(radius, volume_shape, diag, upper, channel_voxel_relation, layout, indices_dtype, device)
    num_offsets = len(pve.offsets)

    # create tensor with different values for each offset
    if batch_size is None:
        values_list = [
            torch.full((1, *volume_shape), i, dtype=values_dtype, device=device) for i in range(1, num_offsets + 1)
        ]
        values = torch.cat(values_list, dim=0)
    else:
        values_list = [
            torch.full((1, *volume_shape), i, dtype=values_dtype, device=device)
            for i in range(1, num_offsets * batch_size + 1)
        ]
        values = torch.cat(values_list, dim=0).reshape(batch_size, num_offsets, *volume_shape)

    sparse_matrix = pve(values)
    dense_matrix = sparse_matrix.to_dense()

    s = dense_matrix.size()[-1]

    for i in range(-s + 1, s):
        diag = dense_matrix.diagonal(i, dim1=-2, dim2=-1)
        if batch_size is None:
            assert len(diag[diag != 0].unique()) < 2
        else:
            for d in diag:
                assert len(d[d != 0].unique()) < 2


def test_PVE_row_consistency(radius, volume_shape, upper, layout, batch_size, channel_voxel_relation):
    diag = True
    indices_dtype = torch.int64
    values_dtype = torch.float64
    device = torch.device("cpu")

    pve = PairwiseVoxelEncoder(radius, volume_shape, diag, upper, channel_voxel_relation, layout, indices_dtype, device)
    num_offsets = len(pve.offsets)

    # create tensor with different values for each voxel, repeated for each offset

    values = torch.arange(1, reduce(mul, volume_shape) + 1, dtype=values_dtype, device=device).reshape(volume_shape)
    values = values.repeat(num_offsets, 1, 1, 1, 1)

    if batch_size is not None:
        values = values.repeat(batch_size, 1, 1, 1, 1, 1)

    sparse_matrix = pve(values)
    dense_matrix = sparse_matrix.to_dense()

    # Check if the unique non-zero value in each row is equal to the row index
    if batch_size is None:
        is_valid = all((torch.unique(row[row.nonzero()]) == idx).all() for idx, row in enumerate(dense_matrix, start=1))
    else:
        is_valid = all(
            all((torch.unique(row[row.nonzero()]) == idx).all() for idx, row in enumerate(dm, start=1))
            for dm in dense_matrix
        )
    assert is_valid


@pytest.mark.skip(reason="Unmark to create plots of the pairwise relationships")
def test_pariwise_coo_indices_visually():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.colors as mcolors

    volume_shape = (3, 3, 3, 3)
    diag = False
    upper = False
    dtype = torch.int64
    device = torch.device("cpu")

    # Define the parameters to test
    radius_list = [1, 2]
    channel_voxel_relation_list = ["indep", "intra", "inter"]

    # Get the tab20 colormap
    cmap_tab10 = plt.cm.get_cmap("tab10", 10)

    # Create a new colormap from the existing colormap
    cmaplist = [cmap_tab10(i) for i in range(cmap_tab10.N)] * 10

    # Insert white color at the start
    cmaplist = [
        (1.0, 1.0, 1.0, 1.0),
    ] + cmaplist

    # Create the new colormap
    cmap_tab20_white = mcolors.LinearSegmentedColormap.from_list("Custom cmap", cmaplist, len(cmaplist))

    cmap = cmap_tab20_white

    # Iterate over the radii
    for radius in radius_list:
        # Create a color dictionary for maximum shifts
        indices_dict_max = calc_pariwise_coo_indices(
            radius=radius,
            volume_shape=volume_shape,
            diag=diag,
            upper=upper,
            channel_voxel_relation="inter",
            dtype=dtype,
            device=device,
        )

        color_dict = {shift: color for color, (shift, _) in enumerate(indices_dict_max.items(), start=1)}
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create subplots for each channel_voxel_relation

        # Iterate over the channel_voxel_relations
        for j, channel_voxel_relation in enumerate(channel_voxel_relation_list):
            # Create the encoder and get the indices dictionary
            indices_dict = calc_pariwise_coo_indices(
                radius=radius,
                volume_shape=volume_shape,
                diag=diag,
                upper=upper,
                channel_voxel_relation=channel_voxel_relation,
                dtype=dtype,
                device=device,
            )

            # Create output matrix and fill it with the unique numbers corresponding to each offset
            size = reduce(mul, volume_shape)
            output_matrix = np.zeros((size, size), dtype=int)

            # For each offset, add a unique number to the corresponding indices in the volume
            for shift, indices in indices_dict.items():
                color = color_dict[shift]  # Get the color from the color_dict
                row_indices, col_indices = indices.cpu().numpy()  # convert indices to numpy arrays
                output_matrix[row_indices, col_indices] += color  # Assign color as a unique number

            # Plot the output matrix
            axs[j].imshow(output_matrix, cmap=cmap, vmin=0, vmax=len(cmaplist))
            axs[j].set_title(f"channel_relation={channel_voxel_relation}")

        # Adjust spacing between subplots and save figure
        plt.tight_layout()
        plt.savefig(f"sparse_encodings_radius_{radius}.png")

        # Create separate figure for the legend
        legend_elements = [
            Patch(facecolor=cmap(i), edgecolor=cmap(i), label=str(shift)) for shift, i in color_dict.items()
        ][:20]
        fig_legend = plt.figure(figsize=(3, 8))
        plt.legend(handles=legend_elements, loc="center")
        plt.axis("off")
        fig_legend.savefig(f"legend_radius_{radius}.png")
