# Import from the new pairwise_encoder module (recommended)
from .pairwise_encoder import PairwiseEncoder, calc_pairwise_coo_indices_nd

# Import from the deprecated pairwise_voxel_encoder module for backward compatibility
from .pairwise_voxel_encoder import PairwiseVoxelEncoder, calc_pariwise_coo_indices

__all__ = [
    # New recommended classes and functions
    "PairwiseEncoder",
    "calc_pairwise_coo_indices_nd",
    # Deprecated but maintained for backward compatibility
    "PairwiseVoxelEncoder",
    "calc_pariwise_coo_indices",
]
