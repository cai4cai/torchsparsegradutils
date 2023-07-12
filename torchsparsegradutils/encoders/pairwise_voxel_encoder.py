from functools import reduce
from operator import mul
import torch
import warnings
import numpy
from itertools import product

from typing import Iterable, List, Tuple, Union, Any, Optional

from torchsparsegradutils.utils.utils import _sort_coo_indices

Shape = Union[List[int], Tuple[int, ...], torch.Size]

# Lower triangular neighbours - can be made upper triangular by *-1: tuple(map(lambda x: -x, offsets))
NEIGHBOURS = [
    #  z  y  x
    (0, 0, 1),  # 1
    (0, 1, 0),  # 2
    (1, 0, 0),  # 3
    (0, 1, 1),  # 4
    (1, 1, 0),  # 5
    (1, 0, 1),  # 6
    (0, 1, -1),  # 7
    (1, -1, 0),  # 8
    (1, 0, -1),  # 9
    (1, 1, 1),  # 10
    (1, -1, 1),  # 11
    (1, 1, -1),  # 12
    (1, -1, -1),  # 13 symmetric 3D Moore Neighbourhood
    (0, 0, 2),  # 14
    (0, 2, 0),  # 15
    (2, 0, 0),  # 16
]


def _trim_3d(x: torch.Tensor, offsets: Tuple[int, int, int]) -> torch.Tensor:
    """
    Trim a 3 dimensional pytorch tensor along each axis with the extent specified by offsets.

    Positive offsets trim from the beginning of the tensor, negative offsets trim from the end,
    and zero leaves the tensor unchanged along that axis. This function raises a ValueError if
    the dimensionality of the tensor does not match the number of offsets.

    Args:
        x (torch.Tensor): The input tensor to be trimmed.
        offsets (Tuple[int, ...]): A tuple of integers specifying the extent of the trim in each axis.

    Returns:
        torch.Tensor: The trimmed tensor.

    Examples:
        For a 3D tensor x:
        >>> _trim_3d(x, (0, 0, 1))
        returns x[0: ,0:, 1: ]

        >>> _trim_3d(x, (0, 0, -1))
        returns x[0:, 0:, :-1]

        >>> _trim_3d(x, (5, -6, 3))
        returns x[5:, :-6, 3:]

    Notes:
        This function is equivalent to:

        slices = tuple()
        for off in offsets:
            start = None if off<0 else off
            end = None if off>-1 else off
            slices += (slice(start, end))

        return x[slices]
    """
    if x.ndim != len(offsets):
        raise ValueError("Dimensionality of tensor x and number of offsets must much")

    return x[tuple(map(lambda i: slice(None if i < 0 else i, None if i > -1 else i), offsets))]


def _gen_coords(radius: float) -> List[Tuple[int, int, int]]:
    """
    Generate a list of tuples representing 3D coordinates.

    The coordinates generated are all points (x, y, z) where each of x, y, z is a non-zero integer and
    the point lies within a sphere of the provided radius around the origin (0, 0, 0).
    The range of x, y, z is from -radius to radius (inclusive).

    Args:
        radius (float): The radius of the sphere within which the coordinates are generated.
        It defines the range of the x, y, z coordinates.

    Returns:
        List[Tuple[int, int, int]]: A list of tuples. Each tuple represents a point (x, y, z)
        within the defined spherical volume.
    """
    coords = [
        (x, y, z)
        for x, y, z in product(range(-radius, radius + 1), repeat=3)
        if x**2 + y**2 + z**2 <= radius**2
    ]
    coords.remove((0, 0, 0))
    return coords


def _calc_pariwise_coo_indices(
    radius: float,
    volume_shape: Tuple[int, int, int],
    batch_size: int = 1,
    num_channels: int = 1,
    diag: bool = False,
    upper: bool = False,
    channel_voxel_relation: str = "indep",
    dtype: torch.dtype = torch.int64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    # TODO: update docstring
    """
    if channel_voxel_relation not in ["indep", "intra", "inter"]:
        raise ValueError("channel_voxel_relation must be one of 'indep', 'intra', or 'inter'")

    if num_channels == 1 and channel_voxel_relation != "indep":
        raise ValueError("channel_voxel_relation must be 'indep' if num_channels = 1")

    # Generate neighbours:
    neighbours = _gen_coords(radius)

    volume_numel = reduce(mul, volume_shape)

    idx = torch.arange(num_channels * volume_numel, device=device).reshape(
        (num_channels,) + volume_shape
    )  # create numbered array

    indices = []

    if diag is True:
        indices.append(torch.stack([idx.flatten(), idx.flatten()], dim=0))

    for offsets in neighbours:
        # First do independent channel relationships - ie. just spatial neighbours:
        col_idx = idx.roll((0,) + offsets, (0, 1, 2, 3))  # shift based offset
        # Trim off neighbours that wrap around boundaries of volume, then flatten
        row_idx = _trim_3d(idx, (0,) + offsets).flatten()
        col_idx = _trim_3d(col_idx, (0,) + offsets).flatten()

        if (row_idx < col_idx).all():  # all are upper triangular
            upper_shift = True  # Does this spatial shift correspond to the upper triangular
        elif not (row_idx < col_idx).any():  # all are lower triangular
            upper_shift = False
        else:
            raise ValueError("Spatial shift is neither upper or lower triangular -- something has gone wrong")

        # If the shift is upper and we want upper, or the shift is lower and we want lower
        # or if upper is None, then we want both upper and lower
        # Then append the indices and carry on to intra and/or inter channel relationships if needed
        # else continue to the next shift:
        if upper == upper_shift or upper is None:
            indices.append(torch.stack([row_idx, col_idx], dim=0))
        else:
            continue

        # inter-channel relationships:

        if channel_voxel_relation == "inter":
            for c in range(1, num_channels):
                col_idx = idx.roll((c,) + offsets, (0, 1, 2, 3))  # shift based offset

                row_idx = _trim_3d(idx, (0,) + offsets).flatten()
                col_idx = _trim_3d(col_idx, (0,) + offsets).flatten()

                if upper is not None:
                    select = row_idx < col_idx if upper else row_idx > col_idx
                else:
                    select = torch.ones_like(row_idx, dtype=torch.bool)

                indices.append(torch.stack([row_idx[select], col_idx[select]], dim=0))

    # Add intra-channel relationships, which are spatial shift independent:
    if channel_voxel_relation != "indep":
        for c in range(1, num_channels):
            row_idx = idx.flatten()
            col_idx = idx.roll((c, 0, 0, 0), (0, 0, 0, 0)).flatten()

            if upper is not None:
                select = row_idx < col_idx if upper else row_idx > col_idx
            else:
                select = torch.ones_like(row_idx, dtype=torch.bool)

            indices.append(torch.stack([row_idx[select], col_idx[select]], dim=0))

    indices = torch.cat(indices, dim=1)
    indices, _ = _sort_coo_indices(indices)

    return indices


class PairwiseVoxelEncoder:
    pass


# class COOSparseEncoder:  # TODO: should this also inherit from nn.Module?
#     def __init__(self, num_neighbours, diag=True, upper=False, device="cuda"):
#         """
#         Sparse encoder in COO format
#         TODO: add more content here
#         """

#         if num_neighbours > 16:
#             raise ValueError("Encoder can only support up to 16 neighbours")

#         self.neighbours = NEIGHBOURS[:num_neighbours]

#         if diag is True:
#             self.neighbours.insert(0, (0, 0, 0))

#         self.diag = diag
#         self.triu = upper
#         self.device = device

#         # Set when self._event_shape is set
#         self._event_shape = None
#         self._event_size = None
#         self.indices = None

#     @property
#     def event_shape(self):
#         return self._event_shape

#     @event_shape.setter
#     def event_shape(self, s):
#         if self.event_shape != s:
#             if len(s) != 4:
#                 raise ValueError(f"Expected shape with size [C, H, D, W] but got shape {s}")

#             self._shape = s
#             self._event_shape = self._shape
#             self._event_size = reduce(mul, self._event_shape)
#             self.indices = self._calc_indices()

#     def _calc_values(self, values):
#         if values.shape[0] != len(self.neighbours):
#             raise ValueError(
#                 f"Number of values ({values.shape[-1]}) must match number of neighbours ({len(self.neighbours)})"
#             )  # TODO: what about diag?

#         values_out = []
#         for offsets, val in zip(self.neighbours, values):
#             val = trim(val, (0,) + offsets)
#             values_out.append(val.flatten())

#         return torch.cat(values_out)

#     def __call__(self, values):
#         self.event_shape = values.shape[1:]
#         return torch.sparse_coo_tensor(self.indices, self._calc_values(values), size=(self._event_size,) * 2)


# class CSRSparseEncoder(COOSparseEncoder):
#     def __init__(self, num_neighbours, diag=True, upper=False, device="cuda"):
#         """
#         Sparse encoder in CSR format
#         """
#         super().__init__(num_neighbours, diag, upper, device)

#         self.crow_indices = None
#         self.col_indices = None

#     def _calc_indices(self):
#         indices_coo = COOSparseEncoder._calc_indices(self)
#         values_coo = torch.arange(indices_coo.shape[-1], dtype=torch.float32, device=self.device)

#         A_coo = torch.sparse_coo_tensor(indices_coo, values_coo, size=(self._event_size, self._event_size))
#         A_csr = A_coo.to_sparse_csr()

#         # deleting coo and clearing cuda cache saves memory:
#         del A_coo
#         if self.device == "cuda":
#             torch.cuda.empty_cache()

#         self.crow_indices = A_csr.crow_indices()
#         self.col_indices = A_csr.col_indices()

#         values_csr = A_csr.values()
#         _, val_permutation = values_csr.sort()
#         inverse_val_permutation = torch.empty_like(val_permutation)
#         inverse_val_permutation[val_permutation] = torch.arange(val_permutation.numel(), device=self.device)
#         self._csr_val_permutation = inverse_val_permutation

#     def _calc_values(self, values):
#         values_coo = COOSparseEncoder._calc_values(self, values)
#         # return values_coo[self._csr_val_permutation]    #### THIS IS INCREDIBLY SLOW NOW
#         return values_coo.index_select(0, self._csr_val_permutation)

#     def __call__(self, values):
#         self.event_shape = values.shape[1:]
#         return torch.sparse_csr_tensor(
#             self.crow_indices,
#             self.col_indices,
#             self._calc_values(values),
#             size=(self._event_size,) * 2,
#         )
