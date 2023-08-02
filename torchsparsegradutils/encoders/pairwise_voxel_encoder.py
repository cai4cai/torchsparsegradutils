from functools import reduce
from operator import mul
import torch
import warnings
import numpy
from itertools import product, chain
from math import floor, ceil

from typing import Iterable, List, Tuple, Union, Any, Optional, Set, Dict

from torchsparsegradutils.utils.utils import _sort_coo_indices
from torchsparsegradutils.utils import convert_coo_to_csr_indices_values

Shape = Union[List[int], Tuple[int, ...], torch.Size]


def _trim_nd(x: torch.Tensor, offsets: Tuple[int, ...]) -> torch.Tensor:
    """
    Trim the tensor along each axis based on the provided offsets.

    Positive offsets trim from the beginning of the tensor, negative offsets trim
    from the end, and zero leaves the tensor unchanged along that axis. A ValueError
    is raised if the dimensionality of the tensor does not match the number of offsets.

    Args:
        x (torch.Tensor): The input tensor to be trimmed.
        offsets (Tuple[int, ...]): A tuple of integers specifying the extent of the
        trim in each axis.

    Returns:
        torch.Tensor: The trimmed tensor.

    Examples:
        For a 3D tensor x:
        >>> _trim_nd(x, (0, 0, 1))    # returns x[0: ,0:, 1: ]
        >>> _trim_nd(x, (0, 0, -1))   # returns x[0:, 0:, :-1]
        >>> _trim_nd(x, (5, -6, 3))   # returns x[5:, :-6, 3:]

        For a 2D tensor x:
        >>> _trim_nd(x, (0, -1))      # returns x[0:, :-1]

        For a 4D tensor x:
        >>> _trim_nd(x, (0, 0, -1, 5)) # returns x[0:, 0:, :-1, 5:]

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


def _gen_coords(radius: float) -> Set[Tuple[int, int, int]]:
    """
    Generate a set of tuples representing 3D coordinates within a sphere.

    The generated coordinates are all points (x, y, z) where each of x, y, z is a
    non-zero integer, and the point lies within a sphere of the given radius
    around the origin (0, 0, 0). The range for each of x, y, z is from
    floor(-radius) to ceil(radius) inclusive.

    The size of the resulting set depends on the radius:
    - radius < 1 will result in an empty set.
    - 1 <= radius < sqrt(2) will result in a set of 6 points.
    - sqrt(2) <= radius < sqrt(3) will result in a set of 18 points.
    - sqrt(3) < radius < 2 will result in a set of 26 points.
    - 2 <= radius < sqrt(5) will result in a set of 32 points.

    Args:
        radius (float): The radius of the sphere within which the coordinates are
        generated. Defines the range of the x, y, z coordinates. Can be either
        an integer or a floating point value.

    Returns:
        Set[Tuple[int, int, int]]: A set of tuples. Each tuple represents a point
        (x, y, z) within the defined spherical volume.
    """
    coords = set(
        (x, y, z)
        for x, y, z in product(range(floor(-radius), ceil(radius) + 1), repeat=3)
        if x**2 + y**2 + z**2 <= radius**2 and (x, y, z) != (0, 0, 0)
    )
    return coords


def _gen_offsets(
    radius: float, upper: bool = None, num_channels: int = 1, channel_voxel_relation: str = "indep"
) -> List[Tuple[int, int, int, int]]:
    """
    Generate a list of tuples representing offsets in 4D space, where the first element
    of the tuple represents the channel offset, and the remaining three elements represent
    the spatial offset.

    The spatial coordinate (z, y, x) lies within a sphere of the given radius around the
    origin (0, 0, 0), and the range of z, y, x is from -radius to radius (inclusive).

    The channel offset coordinate c is determined by the channel_voxel_relation argument:
        If 'indep', no offset is applied to the channel coordinate (c=0).
        If 'intra', the offset corresponds to the intra-voxel offset between channels.
        If 'inter', includes both intra-voxel and inter-voxel offsets between channels.

    Args:
        radius (float): Defines the range of z, y, x coordinates.
        upper (bool): If False, only tuples whose first non-zero value is positive are kept.
                      If True, only tuples whose first non-zero value is negative are kept.
                      If None, all tuples are kept except for (0, 0, 0, 0).
        num_channels (int): Number of channels of the 3D volume. This affects the range of
                            channel offset when 'channel_voxel_relation' is not 'indep'.
        channel_voxel_relation (str, optional): Specifies the type of channel relationship.
                                                 Can be 'indep', 'intra', or 'inter'. Default is 'indep'.

    Returns:
        List[Tuple[int, int, int, int]]: A list of 4D offset tuples (c, z, y, x).
    """

    def first_non_zero_positive(coord):
        for i in coord:
            if i != 0:
                return i > 0
        return False

    def first_non_zero_negative(coord):
        for i in coord:
            if i != 0:
                return i < 0
        return False

    # Generate spatial offset coordinates:
    coords = _gen_coords(radius)

    # Add channel offsets based on channel_voxel_relation:

    offsets = [(0,) + s for s in coords]  # channel independent offsets

    if channel_voxel_relation != "indep":  # add intra-voxel inter-channel offsets
        channels = list(chain(range(-num_channels + 1, 0), range(1, num_channels)))
        offsets += [(c, 0, 0, 0) for c in channels]

    if channel_voxel_relation == "inter":  # add inter-voxel inter-channel offsets
        offsets += [(c,) + s for c in channels for s in coords]

    if upper is False:  # lower triangular filter
        offsets = [s for s in offsets if first_non_zero_positive(s)]
    elif upper is True:  # upper triangular filter
        offsets = [s for s in offsets if first_non_zero_negative(s)]

    # Offsets are sorted first by radius then lexigraphically by absolute value of each element
    # If the absolute values are equal, the positive element is considered bigger than the positive element.
    # This is done for reproducibility and also logical ordering of the offsets.
    offsets = sorted(
        offsets,
        key=lambda x: (sum([i**2 for i in (10 * x[0],) + x[1:]]), tuple(map(abs, x)), tuple(y >= 0 for y in x)),
    )

    return offsets


def calc_pariwise_coo_indices(
    radius: float,
    volume_shape: Tuple[int, int, int, int],
    diag: bool = False,
    upper: bool = None,
    channel_voxel_relation: str = "indep",
    dtype: torch.dtype = torch.int64,
    device: torch.device = torch.device("cpu"),
) -> Dict[Tuple[int, int, int, int], torch.Tensor]:
    """
    Calculate pairwise coordinate indices (in COO format) of spatial offsets in a 3D
    neighbourhood specified by a radius around each voxel in a volume. The volume's
    shape describes a 4D tensor of shape (C, H, D, W), where C is the number of channels,
    and H, D, W are the spatial dimensions.

    The offsets used to generate these pairwise indices are returned as the keys of the
    returned dictionary. These offsets relate to coordinates (c, z, y, x) around each voxel
    where each of c, z, y, x is a non-zero integer, where z, y, x corresponds to H, D, W:


    Args:
        radius (float): The maximum distance from the origin of each voxel within which the
                        spatial coordinates are generated.
                        Must be a positive number not less than 1.
        volume_shape (Tuple[int, int, int, int]): The shape of the volume. The first
                                                   element represents the number of channels,
                                                   and the remaining elements represent the
                                                   spatial shape in z, y, x order.
        diag (bool, optional): If True, the diagonal indices (offset of (0,0,0,0)) are
                               calculated. If False, only non-diagonal offsets are
                               calculated. Default is False.
        upper (Optional[bool], optional): Determines indices generation for an upper or
                                          lower triangular matrix or a full matrix. If None,
                                          indices relating to both upper and lower triangular
                                          matrices are generated. If False, only indices
                                          relating to a lower triangular matrix are
                                          generated. If True, only indices relating to an
                                          upper triangular matrix are generated.
        channel_voxel_relation (str, optional): Specifies the type of channel relationship
                                                to model, can be 'indep', 'intra', or
                                                'inter'. Default is 'indep'.
        dtype (torch.dtype, optional): The data type of the output indices.
                                       Must be a valid torch.dtype. Default is torch.int64.
        device (torch.device, optional): Device assigned to generated indices tensor.
                                         Defaults to torch.device("cpu").

    Returns:
        Dict[Tuple[int, int, int, int], torch.Tensor]: A dictionary where each key is a 4D
                                                       offset, and the corresponding value is
                                                       a tensor of pairwise indices of the
                                                       coordinates with this offset.

    Raises:
        ValueError: If input arguments are out of specification.
    """

    if radius < 1:
        raise ValueError("`radius` must be a positive number larger or equal to 1")

    if not (len(volume_shape) == 4 and all(isinstance(dim, int) and dim > 0 for dim in volume_shape)):
        raise ValueError("`volume_shape` must be a 4D tuple of positive integers")

    if channel_voxel_relation not in ["indep", "intra", "inter"]:
        raise ValueError("`channel_voxel_relation` must be one of 'indep', 'intra', or 'inter'")

    if volume_shape[0] == 1 and channel_voxel_relation != "indep":
        raise ValueError("`channel_voxel_relation` must be 'indep' if num_channels = 1")

    device = torch.device(device) if device is not None else None

    # Generate offsets:
    offsets = _gen_offsets(radius, upper, volume_shape[0], channel_voxel_relation)

    idx = torch.arange(reduce(mul, volume_shape), device=device, dtype=dtype).reshape(
        volume_shape
    )  # create numbered array

    indices = {}

    if diag is True:
        indices[(0, 0, 0, 0)] = torch.stack([idx.flatten(), idx.flatten()], dim=0)

    for offset in offsets:
        col_idx = idx.roll(offset, (0, 1, 2, 3))

        row_idx = _trim_nd(idx, offset).flatten()
        col_idx = _trim_nd(col_idx, offset).flatten()

        indices[offset] = torch.stack([row_idx, col_idx], dim=0)

    return indices


class PairwiseVoxelEncoder:
    """
    A class for encoding pairwise spatial local neighbourhoods and channel based voxel relations
    into a sparse tensor of either COO or CSR format.

    The indices of pairwise relationships are determined and chached during initialization for a
    single batch element and are based on the radius, volume shape, and channel-to-voxel relationship.
    The bigger the neighbourhood radius, and the higher the order of the channel-to-voxel
    relationship, the more offsets that are considered and more nnz elements in the resultant
    sparse tensor.
    Additionally, diagonal entries can be included with the diag flag.
    The output matrix can be restricted to upper or lower triangular with the upper flag, if
    symmetric relationships are assumed, such as distance or correlation.

    The sparse tensor is returned in the `__call__` method, which takes a tensor of values
    with shape [(B), N, C, H, D, W] and returns a sparse tensor of shape [(B), S, S]
    where B is an optional batch dimension, and S is C*H*D*W.
    N is the numnber of spatial offsets being considered, governed by radius and
    channel_voxel_relation, and can be deduced from the offsets attribute, which contains
    an ordered list of 4D offsets relating to (c, z, y, x) coordinates around each voxel.
    The order of these offsets is fixed, and should match the same order of the values
    that are passed to the `__call__` method, along the second dimension (N).
    Only the indices for a single batch element are stored, and the indices are repeated
    for each batch element required as specified by the input tensor to the `__call__` method.
    Some of the values will be trimmed from the 3D volume edges, as values are not allowed
    to wrap around the edges of the volume of spatial volume.


    Args:
        radius (float): The maximum distance from the origin within which the spatial
                        coordinates are generated.
        volume_shape (Tuple[int, ...]): A tuple of 4 integers representing the volume
                                        shape of (C, H, D, W), where C is the number of
                                        channels and H, D, W are the spatial dimension sizes.
        diag (bool, optional): If True, the diagonal indices (offset of (0,0,0,0)) are
                               calculated. If False, only non-diagonal offsets are
                               calculated. Default is False.
        upper (Optional[bool], optional): Determines indices generation for an upper or
                                          lower triangular matrix or a full matrix. If None,
                                          indices relating to both upper and lower triangular
                                          matrices are generated. If False, only indices
                                          relating to a lower triangular matrix are
                                          generated. If True, only indices relating to an
                                          upper triangular matrix are generated.
        channel_voxel_relation (str, optional): Specifies the type of channel relationship
                                                to model, can be 'indep', 'intra', or
                                                'inter'. Default is 'indep'.
        layout (torch.layout, optional): Determines the layout of the output tensor.
                                         Options are torch.sparse_coo (default) or torch.sparse_csr.
        indices_dtype (torch.dtype, optional): The data type of the output indices.
                                               Must be either torch.int32 or torch.int64.
                                               Default is torch.int64.
        device (torch.device, optional): Device assigned to generate sparse tensor.
                                         Defaults to torch.device("cpu").


    Attributes:
        volume_numel (int): Total number of elements in the volume.
        offsets (List[Tuple[int, int, int, int]]): List of 4D offsets used to calculate indices.
        indices (torch.Tensor): Tensor of pairwise indices in COO format.
        crow_indices (torch.Tensor): Tensor of row indices in CSR format.
        col_indices (torch.Tensor): Tensor of column indices in CSR format.
        csr_permutation (torch.Tensor): Permutation used to convert COO to CSR format.
    """

    def __init__(
        self,
        radius: float,
        volume_shape: Tuple[int, int, int, int],
        diag: bool = False,
        upper: bool = None,
        channel_voxel_relation: str = "indep",
        layout=torch.sparse_coo,
        indices_dtype: torch.dtype = torch.int64,
        device: torch.device = torch.device("cpu"),
    ):
        if not ((len(volume_shape) == 4) and all(isinstance(dim, int) and dim > 0 for dim in volume_shape)):
            raise ValueError("`volume_shape` must be a 4D tuple of positive integers, representing [C, H, D, W]")

        if layout == torch.sparse_coo and indices_dtype not in [torch.int64]:
            raise ValueError("`indices_dtype` must be torch.int64 for torch.sparse_coo")
        elif layout == torch.sparse_csr and indices_dtype not in [torch.int64, torch.int32]:
            raise ValueError("`indices_dtype` must be torch.int64 or torch.int32 for torch.sparse_csr")

        self.radius = radius
        self.volume_shape = volume_shape
        self.diag = diag
        self.upper = upper
        self.channel_voxel_relation = channel_voxel_relation
        self.layout = layout
        self.indices_dtype = indices_dtype
        self.device = device

        self.volume_numel = reduce(mul, volume_shape)

        indices_coo_dict = calc_pariwise_coo_indices(
            radius, volume_shape[-4:], diag, upper, channel_voxel_relation, indices_dtype, device
        )
        self.offsets = list(indices_coo_dict.keys())  # dictionary keys are ordered as of Python 3.7
        indices_coo = torch.cat([indices_coo_dict[offset] for offset in indices_coo_dict], dim=1)

        if layout == torch.sparse_coo:
            self.indices = indices_coo
            self.csr_permutation = None

        elif layout == torch.sparse_csr:
            self.crow_indices, self.col_indices, self.csr_permutation = convert_coo_to_csr_indices_values(
                indices_coo, num_rows=self.volume_numel, values=None
            )
        else:
            raise ValueError("layout must be either torch.sparse_coo or torch.sparse_csr")

    def _calc_values(self, values: torch.Tensor) -> torch.Tensor:
        """
        Calculate the values for the sparse tensor based on the input values and offsets.

        Args:
            values (torch.Tensor): Input tensor of values with shape (N, C, H, D, W).

        Returns:
            torch.Tensor: Flattened tensor of values for sparse tensor.
        """
        values_out = []
        for offset, val in zip(self.offsets, values):
            val = _trim_nd(val, offset)
            values_out.append(val.flatten())

        return torch.cat(values_out)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """
        Return a sparse COO or CSR tensor endoded with the pairwise indices calculated
        during class initialization, and the values passed into this function call.
        An optional batch dimension is supported, cached indices will repeated for each batch
        element, and the values of each batch will be encoded in the same order for each
        batch element.

        The output tensor will contain fewer values than the input tensor, as values at the
        boundaries of the input volume are trimmed to ensure that the output tensor doesn't
        contain any relationships that wrap around the spatial volume described by this encoder.

        Args:
            values (torch.Tensor): Input tensor of values with shape [(B), N, C, H, D, W]
                                   where B is an optional batch dimension, N is the number of offsets
                                   being modelled, described by self.offets, and CxHxDxW is the shape
                                   of the spatial volume, where C is the number of channels, and H, D, W
                                   are the height, depth, and width of the volume respectively.

        Returns:
            torch.Tensor: Output tensor in either COO or CSR format, with shape [(B), S, S]
            where is C*H*D*W.

        Raises:
            ValueError: If the shape of 'values' is not a 5D or 6D tensor.
            ValueError: If the last four dimensions of 'values' do not match the volume shape.
            ValueError: If the shape of 'values' at index -5 does not match the number of offsets.
            ValueError: If the data type of 'values' is not either torch.float32 or torch.float64.

        """
        if not len(values.shape) in [5, 6]:
            raise ValueError("values must be either 5D or 6D tensor [(B), N, C, H, D, W]")

        if values.shape[-4:] != self.volume_shape:
            raise ValueError(f"Shape of values {values.shape[-4:]} must match volume shape {self.volume_shape}")

        if values.shape[-5] != len(self.offsets):
            raise ValueError(
                f"Shape of values at index -5 ({values.shape[-5]}) must match number of offsets ({len(self.offsets)})"
            )

        if values.dtype not in [torch.float32, torch.float64]:
            raise ValueError("values must be either torch.float32 or torch.float64 for sparse tensors")

        batched = len(values.shape) == 6

        # Calculate values:
        if not batched:
            size = (self.volume_numel, self.volume_numel)
            values = self._calc_values(values)

        else:
            batch_size = values.shape[0]
            size = (batch_size, self.volume_numel, self.volume_numel)
            batched_values = []
            for batch in values:
                batched_values.append(self._calc_values(batch))
            values = torch.stack(batched_values)

        # Create sparse COO tensor:
        if self.layout == torch.sparse_coo:
            if not batched:
                indices = self.indices
            else:
                sparse_dim_indices = self.indices.repeat(1, batch_size)
                batch_dim_indices = (
                    torch.arange(batch_size, dtype=self.indices.dtype, device=self.device)
                    .repeat_interleave(self.indices.shape[-1])
                    .unsqueeze(0)
                )
                indices = torch.cat([batch_dim_indices, sparse_dim_indices])
                values = values.flatten()

            return torch.sparse_coo_tensor(
                indices,
                values,
                size=size,
                dtype=values.dtype,
                device=values.device,
            ).coalesce()

        # Create sparse CSR tensor:
        elif self.layout == torch.sparse_csr:
            values = values.index_select(dim=-1, index=self.csr_permutation)
            if not batched:
                crow_indices = self.crow_indices
                col_indices = self.col_indices
            else:
                crow_indices = self.crow_indices.repeat(batch_size, 1)
                col_indices = self.col_indices.repeat(batch_size, 1)

            return torch.sparse_csr_tensor(
                crow_indices,
                col_indices,
                values,
                size=size,
                dtype=values.dtype,
                device=values.device,
            )
