# Import from the new pairwise_encoder module (recommended)
from .pairwise_encoder import PairwiseEncoder, calc_pairwise_coo_indices_nd


def __getattr__(name):
    """Lazy-load deprecated aliases to avoid import-time warnings."""
    if name == "PairwiseVoxelEncoder":
        # Import without warning here - the class itself will warn when instantiated
        from .pairwise_voxel_encoder import PairwiseVoxelEncoder

        return PairwiseVoxelEncoder
    elif name == "calc_pairwise_coo_indices":
        import warnings

        warnings.warn(
            "calc_pairwise_coo_indices is deprecated and will be removed in a future version. "
            "Use calc_pairwise_coo_indices_nd instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .pairwise_encoder import calc_pairwise_coo_indices

        return calc_pairwise_coo_indices
    elif name == "calc_pariwise_coo_indices":
        import warnings

        warnings.warn(
            "calc_pariwise_coo_indices (typo) is deprecated and will be removed in a future version. "
            "Use calc_pairwise_coo_indices_nd instead. "
            "Note: This function name contains a typo; use calc_pairwise_coo_indices for the corrected spelling.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .pairwise_encoder import calc_pariwise_coo_indices

        return calc_pariwise_coo_indices
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # New recommended classes and functions
    "PairwiseEncoder",
    "calc_pairwise_coo_indices_nd",
]
