import warnings
import torch
from typing import Tuple

# Deprecation warning for the entire module
warnings.warn(
    "The 'pairwise_voxel_encoder' module is deprecated and will be removed in a future version. "
    "Please use 'pairwise_encoder' instead, which provides the same functionality with support for arbitrary N-dimensional spatial relationships.",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from the new module for backward compatibility
from .pairwise_encoder import (
    _trim_nd,
    _gen_coords_nd,
    _gen_coords,
    _gen_offsets_nd,
    _gen_offsets,
    calc_pairwise_coo_indices_nd,
    calc_pariwise_coo_indices,
    PairwiseEncoder,
    Shape,
)


class PairwiseVoxelEncoder(PairwiseEncoder):
    """
    A class for encoding pairwise spatial 3D neighbourhoods and channel based relations
    into a sparse tensor of either COO or CSR format.

    **DEPRECATED**: Use PairwiseEncoder instead, which supports arbitrary N-dimensional spatial relationships.

    This class provides the same functionality as before but is now deprecated in favor of
    the more general PairwiseEncoder class. PairwiseVoxelEncoder is restricted to 4D tensors
    with shape (C, H, D, W), while PairwiseEncoder supports arbitrary N-dimensional spatial relationships.

    Args:
        radius (float): The maximum distance from the origin within which the spatial
                        coordinates are generated.
        volume_shape (Tuple[int, int, int, int]): A tuple of integers representing the volume
                                                   shape of (C, H, D, W), where C is the number of
                                                   channels, and H, D, W are the spatial dimensions.
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
        device (torch.device, optional): Device assigned to store sparse tensor indices
                                         at initialisation.
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
        # Issue deprecation warning
        warnings.warn(
            "PairwiseVoxelEncoder is deprecated and will be removed in a future version. "
            "Use PairwiseEncoder instead, which supports arbitrary N-dimensional spatial relationships.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Validate input for backward compatibility
        if not ((len(volume_shape) == 4) and all(isinstance(dim, int) and dim > 0 for dim in volume_shape)):
            raise ValueError("`volume_shape` must be a 4D tuple of positive integers, representing [C, H, D, W]")

        # Call parent constructor
        super().__init__(
            radius=radius,
            volume_shape=volume_shape,
            diag=diag,
            upper=upper,
            channel_voxel_relation=channel_voxel_relation,
            layout=layout,
            indices_dtype=indices_dtype,
            device=device,
        )
