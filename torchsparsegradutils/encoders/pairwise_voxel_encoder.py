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
# TODO: Generate these neighbours automatically, using conditions for lower/upper or full


def _generate_neighbours(range_: int, num_neighbours: int, upper=None):
    """
    # TODO: This is currently not implemented correctly

    This is a means to generate symmetric 3D Von Neumann or Moore neighbourhoods.
    Symmetric from the perspective of if you labelled each element in the 3D volume with a number,
    the relationship between that element and its neighbour would occupy either a lower or upper triangular matrix
    when flattened and representing a covariance/correspondence matrix.

    Parameters
    ----------
    range_ : int
        The maximum distance in each direction (x, y, z) to consider neighbours.
    num_neighbours : int
        The number of neighbours to generate.
    upper : bool, optional
        If True, only neighbours encoding the upper triangular portion of the covariance matrix are considered.
        If False, only neighbours encoding the lower triangular portion of the covariance matrix are considered.
        If None (default), neighbours encoding the whole matrix are considered.

    Returns
    -------
    list of tuple
        The list of neighbours. Each neighbour is represented as a tuple of three integers (x, y, z).
    """

    neighbours = list(product(range(-range_, range_ + 1), repeat=3))

    # Filter tuples based on the 'upper' flag
    if upper is True:
        neighbours = [x for x in neighbours if x[0] >= 0]
    elif upper is False:
        neighbours = [x for x in neighbours if x[0] < 0]
    elif upper is None:
        pass
    else:
        raise ValueError("Invalid value for upper: {}. Valid options are True, False or None".format(upper))

    # Sort tuples based on the sum of their absolute values
    neighbours = sorted(neighbours, key=lambda x: sum(abs(val) for val in x))

    # Limit to the requested number of neighbours
    try:
        neighbours = neighbours[:num_neighbours]
    except IndexError:
        raise ValueError(
            "Invalid value for num_neighbours: {}. Must be less than or equal to {}".format(
                num_neighbours, len(neighbours)
            )
        )

    return neighbours


def _trim_3d(x: torch.Tensor, offsets: Tuple[int, int, int]) -> torch.Tensor:
    """
    Trim a 3 dimensional pytorch tensor along each axis with the extent specified by offsets.

    Positive offsets trim from the beginning of the tensor, negative offsets trim from the end,
    and zero leaves the tensor unchanged along that axis. This function raises a ValueError if
    the dimensionality of the tensor does not match the number of offsets.

    Parameters
    ----------
    x : torch.Tensor
        The input tensor to be trimmed.
    offsets : Tuple[int, ...]
        A tuple of integers specifying the extent of the trim in each axis.

    Returns
    -------
    torch.Tensor
        The trimmed tensor.

    Examples
    --------
    For a 3D tensor x:
    >>> _trim(x, (0, 0, 1))
    returns x[0: ,0:, 1: ]

    >>> _trim(x, (0, 0, -1))
    returns x[0:, 0:, :-1]

    >>> _trim(x, (0, 0, -2))
    returns x[0:, 0:, :-2]

    >>> _trim(x, (5, -6, 3))
    returns x[5:, :-6, 3:]

    Notes
    -----
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


def _calc_pariwise_coo_indices(
    neighbourhood_size: int,
    num_neighbours: int,
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
    TODO: update doc string
    Generates 2-dimensional COO (Coordinate Format) indices for pairwise relationships described
    by neighbours in a 3D volume of shape specified by `volume_shape`, with optional batch
    and channel dimensions.

    The indices generated by this function can be used to represent a sparse matrix representation
    of pairwise relationships in a 3D volume, which is particularly useful in the case of covariance matrices.

    The 3D volume is assumed to be in the (batch, channel, z, y, x) format, where batch and channel
    dimensions are optional. However, if a channel dimension is provided, a batch dimension must also
    be provided and set to 1 if not needed.

    The indices correspond to the flattened volume, with a range from 0 to (C*H*D*W) - 1,
    where C, H, D, and W represent the number of channels, height, depth, and width of the volume,
    respectively. This range is also known as the "event_size" in the context of PyTorch distributions.

    If `diag` is set to True, then diagonal elements are included in the pairwise relationships,
    essentially describing the self-relationship of each element. This is useful for including variance
    values in a covariance matrix.

    If an unbatched volume shape is provided, then the COO indices have shape [2, C*H*D*W],
    describing the row and column indices. For batched volumes, the COO indices will have shape
    [3, C*H*D*W], describing the batch, row, and column indices.

    The function also provides the option to model relationships between the voxels of different channels
    using the `channel_voxel_relation` parameter. This can be one of 'indep' (default),
    'intra_inter_channel', or 'inter_inter_channel'.
    Where 'intra' encodes relationships between the same voxel across different channels,
    and, 'inter' encodes relationships between different voxels across different channels.

    if upper = True, then the neighbour (0, 0, 1) corresponds to the relationship between a given voxel
    and the voxel to its right, ie + 1 in dim 0

    if upper = True, then the neighbour (0, 1, 0) corresponds to the relationship between a given voxel
    and the voxel below, ie + 1 in dim 1

    if upper = True, then the neighbour (1, 0, 0) corresponds to the relationship between a given voxel
    and the voxel behind, ie + 1 in dim 2

    Return unsorted indices

    Parameters
    ----------
    neighbours : List[Tuple[int, int, int]]
        List of tuples specifying the relative offsets of neighbouring voxels for pairwise relationships.
    volume_shape : Tuple[int, int, int]
        The shape of the volume to generate indices for. If batched, also include the batch size.
    num_channels : int, optional
        The number of channels in the volume, by default 1.
    batch_size : int, optional
        The number of volumes in the batch, by default 1.
    channel_voxel_relation : str, optional
        Specifies the type of channel relationship to model, can be 'indep', 'intra',
        or 'inter'. By default 'indep'.
    diag : bool, optional
        If True, includes diagonal elements in the pairwise relationships. By default False.
    upper : bool, optional
        If True, only upper triangle of the matrix is computed. By default False.
    dtype : torch.dtype, optional
        The data type of the indices tensor, can be torch.int32 or torch.int64, by default torch.int64.
    device : Optional[torch.device], optional
        The device to place the indices tensor on, by default None, which places it on the CPU.

    Returns
    -------
    torch.Tensor
        The COO indices tensor representing the pairwise relationships in the 3D volume.
    """
    # TODO: remember that this is triangular only
    # TODO: create loop that creates shifts based on all permutations and then just keeps indices that are upper/lower
    # TODO: Indices need to be returned in a predictable order, so sort cannot be used at the end here..
    
    # TODO: add some more checks  - neighbourhood_size and volume size must be compatible
    if channel_voxel_relation not in ["indep", "intra", "inter"]:
        raise ValueError(
            "channel_voxel_relation must be one of 'indep', 'intra', or 'inter'"
        )

    if num_channels == 1 and channel_voxel_relation != "indep":
        # TODO: change to warning
        raise ValueError("channel_voxel_relation must be 'indep' if num_channels = 1")  

    # TODO: do we need a list here or can we do it as a generator?
    neighbours = list(product(range(-neighbourhood_size, neighbourhood_size+1), repeat=3))
    neighbours = sorted(neighbours, key=lambda x: sum(abs(val) for val in x))

    if diag is False:
        neighbours.remove((0, 0, 0))

    volume_numel = reduce(mul, volume_shape)

    idx = torch.arange(num_channels * volume_numel, device=device).reshape(
        (num_channels,) + volume_shape
    )  # create numbered array

    indices = []

    for offsets in neighbours:
        pass
    
    for offsets in neighbours:
        if upper is True:
            offsets = tuple(map(lambda x: -x, offsets))

        col_idx = idx.roll((0,) + offsets, (0, 1, 2, 3))  # shift based offset
        
        # Trim off neighbours that wrap around boundaries of volume, then flatten
        row_idx = _trim_3d(idx, (0,) + offsets).flatten()
        col_idx = _trim_3d(col_idx, (0,) + offsets).flatten()

        indices.append(torch.stack([row_idx, col_idx], dim=0))

    if channel_voxel_relation != "indep":
        for c in range(1, num_channels):
            
            row_idx = idx.flatten()
            col_idx = idx.roll((c, 0, 0, 0), (0, 0, 0, 0)).flatten()
            
            if upper is not None:
                select = row_idx < col_idx if upper else row_idx > col_idx
            else:
                select = torch.ones_like(row_idx, dtype=torch.bool)
                
            # TODO: can we use index select here?
            
            indices.append(torch.stack([row_idx[select], col_idx[select]], dim=0))
            
            # Alternative method:
            # indices.append(
            #     torch.stack([idx.flatten()[c * volume_numel :], idx.flatten()[: -c * volume_numel]], dim=0)
            # )

    if channel_voxel_relation == "inter":
        for offsets in neighbours:
            if upper is True:
                offsets = tuple(map(lambda x: -x, offsets))
                
            for c in range(1, num_channels):
                col_idx = idx.roll((c,) + offsets, (0, 1, 2, 3))  # shift based offset
                
                row_idx = _trim_3d(idx, (0,) + offsets).flatten()
                col_idx = _trim_3d(col_idx, (0,) + offsets).flatten()
                
                if upper is not None:
                    select = row_idx < col_idx if upper else row_idx > col_idx
                else:
                    select = torch.ones_like(row_idx, dtype=torch.bool)
                
                indices.append(torch.stack([row_idx[select], col_idx[select]], dim=0))
            
            

    indices = torch.cat(indices, dim=1)

    # indices = indices.flip(dims=(0,)) if upper is True else indices

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
