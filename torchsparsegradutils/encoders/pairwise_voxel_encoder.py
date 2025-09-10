import warnings
from typing import Tuple

import torch

# Deprecation warning for the entire module
warnings.warn(
    "The 'pairwise_voxel_encoder' module is deprecated and will be removed in a future version. "
    "Please use 'pairwise_encoder' instead, which provides the same functionality with support for arbitrary N-dimensional spatial relationships.",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from the new module for backward compatibility
from .pairwise_encoder import (
    PairwiseEncoder,
    _gen_coords,
    _gen_coords_nd,
    _gen_offsets,
    _gen_offsets_nd,
    _trim_nd,
    calc_pairwise_coo_indices_nd,
    calc_pariwise_coo_indices,
)


class PairwiseVoxelEncoder(PairwiseEncoder):
    r"""3D voxel specialization of :class:`PairwiseEncoder` (deprecated).

    .. deprecated:: 0.x
       Use :class:`PairwiseEncoder` which supports arbitrary N-D spatial relationships.

    Encodes pairwise 3D spatial neighborhoods and channel relations into sparse
    COO or CSR tensors for volumes ``(C, H, D, W)``. Maintained only for backward
    compatibility; internally delegates to the generic encoder.

    Parameters
    ----------
    radius : float
        Spatial radius (neighborhood extent).
    volume_shape : Tuple[int, int, int, int]
        ``(C, H, D, W)`` with all positive integers.
    diag : bool, optional
        Include diagonal offsets. Default ``False``.
    upper : bool or None, optional
        Triangular selection over offsets (see generic encoder). Default ``None``.
    channel_voxel_relation : {"indep","intra","inter"}, optional
        Channel relation mode. Default ``"indep"``.
    layout : torch.layout, optional
        Output layout (``sparse_coo`` default or ``sparse_csr``).
    indices_dtype : torch.dtype, optional
        Index dtype (``int32`` or ``int64``; default ``int64``).
    device : torch.device, optional
        Storage device (default CPU).

    Attributes
    ----------
    volume_numel : int
        Total sites ``C * H * D * W``.
    offsets : list[tuple[int,int,int,int]]
        Ordered offset tuples.
    indices / crow_indices / col_indices / csr_permutation : torch.Tensor
        Sparse structure (depending on layout).

    Warnings
    --------
    DeprecationWarning
        Issued upon instantiation.

    Examples
    --------
    Deprecated usage:
    >>> from torchsparsegradutils.encoders import PairwiseVoxelEncoder
    >>> import warnings
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', DeprecationWarning)
    ...     enc = PairwiseVoxelEncoder(radius=1.5, volume_shape=(2, 8, 8, 8))
    >>> enc.indices.shape[0]
    2

    Preferred replacement:
    >>> from torchsparsegradutils.encoders import PairwiseEncoder
    >>> new = PairwiseEncoder(radius=1.5, volume_shape=(2, 8, 8, 8), channel_voxel_relation='indep')
    >>> new.indices.shape[0]
    2

    See Also
    --------
    PairwiseEncoder : General N-D implementation.
    """

    def __init__(
        self,
        radius: float,
        volume_shape: Tuple[int, int, int, int],
        diag: bool = False,
        upper: bool | None = None,
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
