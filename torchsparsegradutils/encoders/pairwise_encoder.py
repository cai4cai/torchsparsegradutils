import warnings
from functools import reduce
from itertools import chain, product
from math import ceil, floor
from operator import mul
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy
import torch

from torchsparsegradutils.utils import convert_coo_to_csr_indices_values
from torchsparsegradutils.utils.utils import _sort_coo_indices


def _trim_nd(x: torch.Tensor, offsets: Tuple[int, ...]) -> torch.Tensor:
    r"""Trim a tensor along each axis according to per-dimension offsets.

    Positive offsets drop elements from the **start** of a dimension
    (keep ``offset:``); negative offsets drop elements from the **end**
    (keep ``:offset``). A zero offset leaves that dimension unchanged.
    The number of offsets must match ``x.ndim``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary shape.
    offsets : Tuple[int, ...]
        Tuple of integer offsets (one per dimension of ``x``). For an entry ``k``:
        * ``k > 0``  → keep ``x[k:]`` along that axis
        * ``k == 0`` → keep the whole axis (``x[:]``)
        * ``k < 0``  → keep ``x[:k]`` (drop ``|k|`` elements from the end)

    Returns
    -------
    torch.Tensor
        A **view** of ``x`` trimmed according to ``offsets`` (device, dtype and
        strides are preserved, subject to standard PyTorch slicing semantics).

    Raises
    ------
    ValueError
        If ``len(offsets) != x.ndim``.

    Notes
    -----
    Equivalent slice construction (for demonstration):

    >>> import torch
    >>> x = torch.arange(6)
    >>> offsets = (2,)
    >>> slices = tuple(slice(None if off < 0 else off, None if off > -1 else off) for off in offsets)
    >>> y = x[slices]

    Slicing returns a view when possible—no data copy is performed.

    Examples
    --------
    1D:
    >>> x = torch.arange(6)          # tensor([0, 1, 2, 3, 4, 5])
    >>> _trim_nd(x, (2,))            # keep from index 2 onward
    tensor([2, 3, 4, 5])
    >>> _trim_nd(x, (-2,))           # drop last 2
    tensor([0, 1, 2, 3])

    2D:
    >>> x = torch.arange(12).view(3, 4)
    >>> x
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    >>> _trim_nd(x, (1, 0))          # drop first row
    tensor([[ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    >>> _trim_nd(x, (0, -1))         # drop last column
    tensor([[ 0,  1,  2],
            [ 4,  5,  6],
            [ 8,  9, 10]])
    """
    if x.ndim != len(offsets):
        raise ValueError(f"Number of dimensions in tensor ({x.ndim}) does not match number of offsets ({len(offsets)})")

    return x[tuple(map(lambda i: slice(None if i < 0 else i, None if i > -1 else i), offsets))]


def _gen_coords_nd(radius: float, spatial_dims: int) -> Set[Tuple[int, ...]]:
    r"""Generate integer lattice coordinates inside an :math:`N`-D :math:`\ell_2` ball.

    Returns all integer points :math:`x \in \mathbb{Z}^d` such that
    :math:`\|x\|_2 \le r` (``r = radius``), excluding the origin
    :math:`(0,\dots,0)`. Points are enumerated from the hypercube
    :math:`[\lfloor-r\rfloor,\lceil r\rceil]^d` and filtered by the
    Euclidean norm test.

    Parameters
    ----------
    radius : float
        Radius of the hypersphere (may be non-integer). If ``radius < 0`` the
        result is the empty set.
    spatial_dims : int
        Number of spatial dimensions ``d``.

    Returns
    -------
    Set[Tuple[int, ...]]
        Integer coordinate tuples inside the closed ball of radius ``radius``
        (origin excluded). Order is unspecified.

    Raises
    ------
    ValueError
        If ``spatial_dims <= 0``.

    Notes
    -----
    * Only the all-zero vector is excluded; individual components may be zero.
    * Runtime / output size scale like :math:`O((2\lceil r\rceil+1)^d)`.
    * For large ``radius`` or ``spatial_dims`` consider streaming instead of
      materializing the full set.

    Examples
    --------
    1D (interval on integers):
    >>> _gen_coords_nd(2.0, 1) == {(-2,), (-1,), (1,), (2,)}
    True

    2D (disk of radius 1.5):
    >>> pts = _gen_coords_nd(1.5, 2)
    >>> sorted(pts)  # doctest: +ELLIPSIS
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    3D (ball of radius 1):
    >>> sorted(_gen_coords_nd(1.0, 3))  # doctest: +ELLIPSIS
    [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
    """
    if spatial_dims <= 0:
        raise ValueError("spatial_dims must be a positive integer")

    range_vals = range(floor(-radius), ceil(radius) + 1)
    coords = set(
        coord
        for coord in product(range_vals, repeat=spatial_dims)
        if sum(x**2 for x in coord) <= radius**2 and coord != tuple(0 for _ in range(spatial_dims))
    )
    return coords


def _gen_coords(radius: float) -> Set[Tuple[int, int, int]]:
    r"""Integer lattice points inside a 3D :math:`\ell_2` ball (deprecated wrapper).

    .. deprecated:: 0.x
       Use :func:`_gen_coords_nd` with ``spatial_dims=3`` instead:
       ``_gen_coords_nd(radius, 3)``. This function remains for backward
       compatibility and forwards directly.

    Returns all integer points :math:`(x,y,z) \in \mathbb{Z}^3` with
    :math:`\sqrt{x^2 + y^2 + z^2} \le r`, excluding the origin
    :math:`(0,0,0)`.

    Parameters
    ----------
    radius : float
        Sphere radius (if ``radius < 0`` the result is empty).

    Returns
    -------
    Set[Tuple[int, int, int]]
        Integer triples inside the closed 3D ball (origin excluded).

    Notes
    -----
    Enumeration over :math:`[\lfloor-r\rfloor, \lceil r\rceil]^3` filtered by the
    norm test. Complexity :math:`O((2\lceil r\rceil+1)^3)`.

    Cardinality (reference):

    * ``r < 1`` → 0 points
    * ``1 \le r < \sqrt{2}`` → 6 (axis neighbors)
    * ``\sqrt{2} \le r < \sqrt{3}`` → 18 (adds edge neighbors)
    * ``\sqrt{3} \le r < 2`` → 26 (adds corner neighbors)
    * ``2 \le r < \sqrt{5}`` → 32 (adds distance-2 axis neighbors)

    See Also
    --------
    _gen_coords_nd : Preferred N-D implementation.

    Examples
    --------
    >>> pts = _gen_coords(1.0)
    >>> sorted(pts)  # doctest: +ELLIPSIS
    [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (1, 0, 0)]
    >>> _gen_coords(1.0) == _gen_coords_nd(1.0, 3)
    True
    """
    # Cast for type checker: underlying returns Set[Tuple[int, ...]] but here we constrain to 3D.
    return set(tuple(c) for c in _gen_coords_nd(radius, 3))  # type: ignore[return-value]


def _gen_offsets_nd(
    radius: float,
    spatial_dims: int,
    upper: bool | None = None,
    num_channels: int = 1,
    channel_voxel_relation: str = "indep",
) -> list[tuple[int, ...]]:
    r"""Generate :math:`(1+N)`-D channel+voxel offset tuples inside an :math:`N`-D ball.

    Returns a **sorted** list of tuples ``(c, s1, ..., sN)`` where ``c`` is the channel
    offset and ``(s1..sN)`` a spatial offset with :math:`\sum_i s_i^2 \le r^2` (``r = radius``),
    excluding the all-zero tuple. Depending on ``channel_voxel_relation`` the set is
    augmented with pure channel offsets and/or combined channel+spatial offsets.

    Sign filtering (argument ``upper``) keeps offsets based on the *first non-zero* entry
    in the full tuple ``(c, s1, ..., sN)``:

    * ``upper is False`` → keep those whose first non-zero is positive
    * ``upper is True``  → keep those whose first non-zero is negative
    * ``upper is None``  → keep all (except the all-zero)

    Ordering key (stable, deterministic):

    1. Squared radius in augmented space where the channel component is scaled by 10
    2. Lexicographic order of absolute values ``(|c|, |s1|, ..., |sN|)``
    3. Sign preference (non-negative entries ordered after negative ones on ties)

    Parameters
    ----------
    radius : float
        Spatial neighborhood radius (may be non-integer).
    spatial_dims : int
        Number of spatial dims ``N``.
    upper : bool or None, optional
        Sign-selection filter (see above). Default ``None``.
    num_channels : int, optional
        Number of channels (affects channel offsets). Default ``1``.
    channel_voxel_relation : {'indep', 'intra', 'inter'}, optional
        * ``'indep'`` – only spatial offsets ``(0, s1..sN)``
        * ``'intra'`` – plus intra-voxel channel offsets ``(c, 0, ..., 0)``
        * ``'inter'`` – plus intra offsets and inter-voxel ``(c, s1..sN)``

    Returns
    -------
    list[tuple[int, ...]]
        List of offset tuples of length ``1 + spatial_dims`` (no all-zero tuple).

    Raises
    ------
    ValueError
        If ``spatial_dims <= 0`` (from :func:`_gen_coords_nd`).

    Notes
    -----
    * Spatial offsets from :func:`_gen_coords_nd` never include the zero vector.
    * Channel component is scaled by 10 in the radius used for ordering to keep
      channel steps ranked above small spatial ties.

    See Also
    --------
    _gen_coords_nd : Enumerate spatial coordinates within radius.

    Examples
    --------
    2D, channel independent:
    >>> _gen_offsets_nd(1.5, spatial_dims=2, upper=None, num_channels=1, channel_voxel_relation='indep')  # doctest: +ELLIPSIS
    [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), ...]

    Add intra-voxel channel offsets (two channels):
    >>> _gen_offsets_nd(1.0, 2, num_channels=2, channel_voxel_relation='intra')  # doctest: +ELLIPSIS
    [(0, 0, -1), (0, 0, 1), (0, -1, 0), (0, 1, 0), (1, 0, 0)]

    Inter-voxel channel+spatial combinations:
    >>> offs = _gen_offsets_nd(1.0, 2, num_channels=2, channel_voxel_relation='inter')
    >>> any(o[0] == 1 and o[1:] != (0, 0) for o in offs)
    True

    Restrict to ``upper=False``:
    >>> _gen_offsets_nd(1.0, 1, upper=False, num_channels=1, channel_voxel_relation='indep')
    [(0, 1)]
    """

    def first_non_zero_positive(coord):
        for c in coord:
            if c != 0:
                return c > 0
        return False

    def first_non_zero_negative(coord):
        for c in coord:
            if c != 0:
                return c < 0
        return False

    # Generate spatial offset coordinates:
    coords = _gen_coords_nd(radius, spatial_dims)

    # Add channel offsets based on channel_voxel_relation:
    offsets = [(0,) + s for s in coords]  # channel independent offsets

    if channel_voxel_relation != "indep":
        # Add intra-voxel channel offsets (no spatial offset):
        for c in range(1, num_channels):
            offsets.append((c,) + tuple(0 for _ in range(spatial_dims)))

    if channel_voxel_relation == "inter":
        # Add inter-voxel channel offsets (both spatial and channel offsets):
        for c in range(1, num_channels):
            offsets.extend([(c,) + s for s in coords])

    if upper is False:
        offsets = [offset for offset in offsets if first_non_zero_positive(offset)]
    elif upper is True:
        offsets = [offset for offset in offsets if first_non_zero_negative(offset)]

    # Offsets are sorted first by radius then lexigraphically by absolute value of each element
    # If the absolute values are equal, the positive element is considered bigger than the negative element.
    # This is done for reproducibility and also logical ordering of the offsets.
    offsets = sorted(
        offsets,
        key=lambda x: (sum([i**2 for i in (10 * x[0],) + x[1:]]), tuple(map(abs, x)), tuple(y >= 0 for y in x)),
    )

    return offsets


def _gen_offsets(
    radius: float,
    upper: bool | None = None,
    num_channels: int = 1,
    channel_voxel_relation: str = "indep",
) -> list[tuple[int, int, int, int]]:
    r"""Generate 4D channel+spatial offsets in a 3D spherical neighborhood (deprecated).

    .. deprecated:: 0.x
       Use :func:`_gen_offsets_nd(radius, 3, upper, num_channels, channel_voxel_relation)`.

    Produces sorted tuples ``(c, z, y, x)`` where ``(z,y,x)`` satisfy
    :math:`z^2 + y^2 + x^2 \le r^2` and channel offsets are added according to
    ``channel_voxel_relation``.

    Parameters
    ----------
    radius : float
        Spatial radius ``r``.
    upper : bool or None, optional
        Sign-selection filter (first non-zero criterion). Default ``None``.
    num_channels : int, optional
        Number of channels. Default ``1``.
    channel_voxel_relation : {'indep', 'intra', 'inter'}, optional
        Channel/spatial relation mode.

    Returns
    -------
    list[tuple[int, int, int, int]]
        4D offset tuples (without the all-zero tuple).

    Notes
    -----
    Equivalent to calling :func:`_gen_offsets_nd` with ``spatial_dims=3``.

    See Also
    --------
    _gen_offsets_nd : N-D generalization.
    _gen_coords_nd : Underlying spatial coordinate generator.

    Examples
    --------
    Channel-independent (only spatial):
    >>> _gen_offsets(1.5, upper=None, num_channels=1, channel_voxel_relation='indep')  # doctest: +ELLIPSIS
    [(0, 0, 0, -1), (0, 0, 0, 1), (0, 0, -1, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 1, 0, 0), ...]

    Intra-voxel channel offsets:
    >>> _gen_offsets(1.0, num_channels=2, channel_voxel_relation='intra')  # doctest: +ELLIPSIS
    [(0, 0, 0, -1), (0, 0, 0, 1), (0, 0, -1, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0)]

    Inter-voxel combinations:
    >>> offs = _gen_offsets(1.0, num_channels=2, channel_voxel_relation='inter')
    >>> any(o[0] == 1 and o[1:] != (0, 0, 0) for o in offs)
    True
    """
    offs = _gen_offsets_nd(radius, 3, upper, num_channels, channel_voxel_relation)
    return [tuple(o) for o in offs]  # type: ignore[return-value]


def calc_pairwise_coo_indices_nd(
    radius: float,
    volume_shape: Tuple[int, ...],
    diag: bool = False,
    upper: bool | None = None,
    channel_voxel_relation: str = "indep",
    dtype: torch.dtype = torch.int64,
    device: torch.device | None = torch.device("cpu"),
) -> Dict[Tuple[int, ...], torch.Tensor]:
    r"""Compute per-offset COO linear index pairs for an :math:`(C,*S)` volume.

    For a volume ``(C, *spatial_dims)`` and spatial radius ``r``, return a dictionary
    mapping each offset tuple ``(c, *spatial_offset)`` to a ``(2, M)`` tensor of linear
    index pairs ``[[i...],[j...]]`` such that the second row is the first row shifted
    by the offset (within bounds). Linearization follows row-major order
    (``torch.arange(prod(volume_shape)).reshape(volume_shape).flatten()``).

    Offsets come from :func:`_gen_offsets_nd` (sorted), which enumerates spatial offsets
    with :math:`\|o_{spatial}\|_2 \le r` and augments them with channel offsets
    according to ``channel_voxel_relation``.

    Parameters
    ----------
    radius : float
        Neighborhood radius (``>= 1``).
    volume_shape : tuple[int, ...]
        Shape ``(C, *spatial_dims)`` with at least one spatial dimension.
    diag : bool, optional
        Include diagonal key ``(0,...,0)`` mapping to ``(i,i)`` pairs. Default ``False``.
    upper : bool or None, optional
        Forwarded sign filter (see :func:`_gen_offsets_nd`). Default ``None``.
    channel_voxel_relation : {'indep','intra','inter'}, optional
        Channel relation mode. Default ``'indep'``.
    dtype : torch.dtype, optional
        Integer dtype of output index tensors (default ``torch.int64``).
    device : torch.device, optional
        Target device (default CPU).

    Returns
    -------
    dict[tuple[int, ...], torch.Tensor]
        Mapping from offset tuple to a ``(2, M_o)`` tensor of linear index pairs.

    Raises
    ------
    ValueError
        If arguments are inconsistent (e.g. ``radius < 1``).

    Notes
    -----
    Each non-zero offset ``o`` yields pairs by trimming the index lattice twice with
    :func:`_trim_nd`: once by ``o`` and once by ``-o``. Only valid in-bounds pairs
    are produced (no padding). Sorting matches :func:`_gen_offsets_nd`.

    See Also
    --------
    _gen_offsets_nd : Generate (sorted) offsets.
    _trim_nd : Bounds-aware slicing used for forming pairs.

    Examples
    --------
    2D single channel:
    >>> idxs = calc_pairwise_coo_indices_nd(
    ...     radius=1.0,
    ...     volume_shape=(1, 3, 3), # (C,H,W)
    ...     diag=True,
    ...     upper=None,
    ...     channel_voxel_relation='indep',
    ... )
    >>> sorted(list(idxs.keys()))[:3]  # doctest: +ELLIPSIS
    [(0, -1, 0), (0, 0, -1), (0, 0, 0)]
    >>> z = (0, 0, 0)
    >>> idxs[z].shape
    torch.Size([2, 9])

    3D, inter-channel:
    >>> idxs3d = calc_pairwise_coo_indices_nd(
    ...     radius=1.0,
    ...     volume_shape=(2, 3, 3, 3),
    ...     channel_voxel_relation='inter',
    ... )
    >>> any(o[0] == 1 and o[1:] != (0, 0, 0) for o in idxs3d.keys())
    True
    """

    if radius < 1:
        raise ValueError("radius must be >= 1")

    if not (len(volume_shape) >= 2 and all(isinstance(dim, int) and dim > 0 for dim in volume_shape)):
        raise ValueError("volume_shape must be a tuple of at least 2 positive integers")

    if channel_voxel_relation not in ["indep", "intra", "inter"]:
        raise ValueError("channel_voxel_relation must be 'indep', 'intra', or 'inter'")

    if volume_shape[0] == 1 and channel_voxel_relation != "indep":
        raise ValueError("channel_voxel_relation must be 'indep' when number of channels is 1")

    device = torch.device(device) if device is not None else None

    spatial_dims = len(volume_shape) - 1

    # Generate offsets:
    offsets = _gen_offsets_nd(radius, spatial_dims, upper, volume_shape[0], channel_voxel_relation)

    idx = torch.arange(reduce(mul, volume_shape), device=device, dtype=dtype).reshape(
        volume_shape
    )  # create numbered array

    indices = {}

    if diag is True:
        zero_offset = tuple(0 for _ in range(len(volume_shape)))
        indices[zero_offset] = torch.stack([idx.flatten(), idx.flatten()])

    for offset in offsets:
        # Compute trimmed indices:
        x1_idx = _trim_nd(idx, offset)
        x2_idx = _trim_nd(idx, tuple([-o for o in offset]))

        # Stack into indices list:
        indices[offset] = torch.stack([x1_idx.flatten(), x2_idx.flatten()])

    return indices


def calc_pariwise_coo_indices(
    radius: float,
    volume_shape: Tuple[int, int, int, int],
    diag: bool = False,
    upper: bool | None = None,
    channel_voxel_relation: str = "indep",
    dtype: torch.dtype = torch.int64,
    device: torch.device = torch.device("cpu"),
) -> Dict[Tuple[int, int, int, int], torch.Tensor]:
    r"""3D wrapper for :func:`calc_pairwise_coo_indices_nd` (deprecated).

    .. deprecated:: 0.x
       Use :func:`calc_pairwise_coo_indices_nd`.

    Parameters
    ----------
    radius : float
        Spatial radius (``>=1``).
    volume_shape : tuple[int,int,int,int]
        ``(C,H,D,W)``.
    diag : bool, optional
        Include diagonal. Default ``False``.
    upper : bool or None, optional
        Sign-selection (see N-D version). Default ``None``.
    channel_voxel_relation : {'indep','intra','inter'}, optional
        Channel relation mode. Default ``'indep'``.
    dtype : torch.dtype, optional
        Index dtype (default ``torch.int64``).
    device : torch.device, optional
        Target device (default CPU).

    Returns
    -------
    dict[tuple[int,int,int,int], torch.Tensor]
        Per-offset COO index pairs.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    # Validate 4D shape for backward compatibility
    if not (len(volume_shape) == 4 and all(isinstance(dim, int) and dim > 0 for dim in volume_shape)):
        raise ValueError("volume_shape must be a 4D tuple of positive integers for backward compatibility")

    out = calc_pairwise_coo_indices_nd(radius, volume_shape, diag, upper, channel_voxel_relation, dtype, device)
    # Narrow key type for static checker (all offsets have length 4 here)
    return {tuple(k): v for k, v in out.items()}  # type: ignore[return-value]


class PairwiseEncoder(torch.nn.Module):
    r"""Encode pairwise spatial–channel neighborhoods as sparse tensors.

    Precomputes a mapping from local neighborhoods (within spatial radius ``r``)
    and optional channel interactions to global sparse matrix indices over the
    linearized volume. Useful for graph-like layers, covariance assembly, sparse
    attention and any operator exploiting local geometric structure.

    Parameters
    ----------
    radius : float
        Spatial neighborhood radius ``r``.
    volume_shape : tuple[int, ...]
        ``(C,*spatial_dims)`` (at least one spatial dimension).
    diag : bool, optional
        Include diagonal (self-edges). Default ``False``.
    upper : bool or None, optional
        Triangular selection on offset set (first non-zero criterion). ``None`` keeps all.
    channel_voxel_relation : {'indep','intra','inter'}, optional
        Channel interaction model. Default ``'indep'``.
    layout : torch.layout, optional
        ``torch.sparse_coo`` (default) or ``torch.sparse_csr``.
    indices_dtype : torch.dtype, optional
        Integer dtype for indices (``int32`` or ``int64``). Default ``int64``.
    device : torch.device, optional
        Device to store cached indices (default CPU).

    Attributes
    ----------
    volume_numel : int
        ``C * prod(spatial_dims)``.
    spatial_dims : int
        ``len(volume_shape) - 1``.
    offsets : list[tuple[int,...]]
        Ordered offsets (optionally with diagonal key first if ``diag``).
    indices : torch.Tensor
        (COO) ``(2, nnz_total)`` tensor of linear index pairs.
    crow_indices, col_indices, csr_permutation : torch.Tensor
        (CSR) components & permutation to reorder values into CSR order.

    Notes
    -----
    * Input to :meth:`__call__` must have shape ``[(B), N, C, *S]`` where
      ``N == len(self.offsets)``. Batch dimension optional.
    * Edge handling uses trimming (no wrap, no padding).
    * CSR values are internally reordered via ``self.csr_permutation``.
    * Complexity scales with number of offsets times valid pairs (≈ :math:`O(r^2)` in 2D, :math:`O(r^3)` in 3D).

    See Also
    --------
    calc_pairwise_coo_indices_nd : Build per-offset COO indices.
    convert_coo_to_csr_indices_values : COO→CSR conversion + permutation.
    _gen_offsets_nd : Construct ordered offset set.
    _trim_nd : Bounds-aware slicing for forming value blocks.

    Examples
    --------
    Basic 2D:
    >>> from torchsparsegradutils.encoders import PairwiseEncoder
    >>> encoder = PairwiseEncoder(
    ...     radius=1.5,
    ...     volume_shape=(3, 8, 8),
    ...     diag=True,
    ...     channel_voxel_relation='indep'
    ... )
    >>> encoder.volume_numel
    192
    >>> len(encoder.offsets)  # doctest: +SKIP
    13

    Create sparse tensor from values:
    >>> values = torch.randn(len(encoder.offsets), 3, 8, 8)
    >>> sp = encoder(values)
    >>> sp.shape
    torch.Size([192, 192])
    >>> sp.is_sparse
    True

    Batched:
    >>> values_b = torch.randn(4, len(encoder.offsets), 3, 8, 8)
    >>> sp_b = encoder(values_b)
    >>> sp_b.shape
    torch.Size([4, 192, 192])

    3D with inter-channel relations:
    >>> encoder3d = PairwiseEncoder(
    ...     radius=2.0,
    ...     volume_shape=(5, 16, 16, 16),
    ...     channel_voxel_relation='inter',
    ...     layout=torch.sparse_csr,
    ... )

    Upper-triangular (symmetric use-case):
    >>> sym = PairwiseEncoder(
    ...     radius=1.0,
    ...     volume_shape=(1, 10, 10),
    ...     upper=True,
    ...     diag=True,
    ... )
    >>> v = torch.randn(len(sym.offsets), 1, 10, 10)
    >>> _ = sym(v)
    """

    def __init__(
        self,
        radius: float,
        volume_shape: Tuple[int, ...],
        diag: bool = False,
        upper: bool | None = None,
        channel_voxel_relation: str = "indep",
        layout=torch.sparse_coo,
        indices_dtype: torch.dtype = torch.int64,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        if not ((len(volume_shape) >= 2) and all(isinstance(dim, int) and dim > 0 for dim in volume_shape)):
            raise ValueError(
                "`volume_shape` must be a tuple of at least 2 positive integers, representing [C, *spatial_dims]"
            )

        if indices_dtype not in [torch.int64, torch.int32]:
            raise ValueError("`indices_dtype` must be torch.int64 or torch.int32 for torch.sparse_coo")

        self.radius = radius
        self.volume_shape = volume_shape
        self.diag = diag
        self.upper = upper
        self.channel_voxel_relation = channel_voxel_relation
        self.layout = layout
        self.indices_dtype = indices_dtype

        self.volume_numel = reduce(mul, volume_shape)
        self.spatial_dims = len(volume_shape) - 1

        indices_coo_dict = calc_pairwise_coo_indices_nd(
            radius, volume_shape, diag, upper, channel_voxel_relation, indices_dtype, device
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

    def _apply(self, fn, recurse=True):
        # Applying the function to the desired attributes
        # This has been implemented to allow using the .to() method
        for attr in ["indices", "csr_permutation", "crow_indices", "col_indices"]:
            tensor = getattr(self, attr, None)
            if tensor is not None:
                setattr(self, attr, fn(tensor))

        return self

    @property
    def device(self):
        if self.layout == torch.sparse_coo:
            return self.indices.device
        elif self.layout == torch.sparse_csr:
            return self.crow_indices.device

    def _calc_values(self, values: torch.Tensor) -> torch.Tensor:
        r"""Assemble flattened value vector for one (unbatched) call.

        Parameters
        ----------
        values : torch.Tensor
            Tensor of shape ``(N, C, *spatial_dims)`` where ``N == len(self.offsets)``.

        Returns
        -------
        torch.Tensor
            Flattened concatenation of trimmed per-offset blocks (order matches ``self.indices`` or CSR permutation input order).
        """
        values_out = []
        for offset, val in zip(self.offsets, values):
            trimmed_val = _trim_nd(val, offset).flatten()
            values_out.append(trimmed_val)

        return torch.cat(values_out)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        r"""Construct sparse tensor (COO or CSR) from per-offset value blocks.

        Parameters
        ----------
        values : torch.Tensor
            Shape ``[(B), N, C, *spatial_dims]`` with optional batch ``B`` and
            ``N == len(self.offsets)``.

        Returns
        -------
        torch.Tensor
            Sparse tensor of shape ``[(B), S, S]`` where ``S = C * prod(spatial_dims)``.

        Raises
        ------
        ValueError
            If shape, dtype or offset count are inconsistent.
        """
        expected_spatial_dims = len(self.volume_shape) - 1
        expected_full_dims = expected_spatial_dims + 2  # C + spatial_dims

        if len(values.shape) < expected_full_dims or len(values.shape) > expected_full_dims + 1:
            raise ValueError(
                f"values must have {expected_full_dims} dimensions (N, C, *spatial_dims) "
                f"or {expected_full_dims + 1} dimensions (B, N, C, *spatial_dims)"
            )

        # Check spatial dimensions match
        spatial_shape_in_values = values.shape[-expected_spatial_dims:]
        expected_spatial_shape = self.volume_shape[-expected_spatial_dims:]

        if spatial_shape_in_values != expected_spatial_shape:
            raise ValueError(
                f"Spatial dimensions do not match: expected {expected_spatial_shape}, " f"got {spatial_shape_in_values}"
            )

        # Check number of offsets
        offset_dim_idx = -expected_full_dims
        if values.shape[offset_dim_idx] != len(self.offsets):
            raise ValueError(
                f"Shape of values at index {offset_dim_idx} ({values.shape[offset_dim_idx]}) "
                f"must match number of offsets ({len(self.offsets)})"
            )

        if values.dtype not in [torch.float32, torch.float64]:
            raise ValueError("values must be either torch.float32 or torch.float64 for sparse tensors")

        batched = len(values.shape) == expected_full_dims + 1
        batch_size: int | None = None

        if batched:
            batch_size = values.shape[0]

        # Calculate values
        if batched:
            assert batch_size is not None  # for type checker
            size_batched: tuple[int, int, int] = (batch_size, self.volume_numel, self.volume_numel)
            size_any = size_batched  # unified name
            processed = [self._calc_values(batch) for batch in values]  # type: ignore[assignment]
            values = torch.stack(processed)
        else:
            size_unbatched: tuple[int, int] = (self.volume_numel, self.volume_numel)
            size_any = size_unbatched
            values = self._calc_values(values)

        if self.layout == torch.sparse_coo:
            if batched:
                assert batch_size is not None
                sparse_dim_indices = self.indices.repeat(1, batch_size)
                batch_dim_indices = (
                    torch.arange(batch_size, dtype=self.indices.dtype, device=self.indices.device)
                    .repeat_interleave(self.indices.shape[-1])
                    .unsqueeze(0)
                )
                indices = torch.cat([batch_dim_indices, sparse_dim_indices])
                values = values.flatten()
            else:
                indices = self.indices
            return torch.sparse_coo_tensor(
                indices, values, size=size_any, dtype=values.dtype, device=values.device
            ).coalesce()

        if self.layout == torch.sparse_csr:
            if self.csr_permutation is None:
                raise RuntimeError("csr_permutation is None; expected a permutation tensor when layout is sparse_csr.")
            values = values.index_select(dim=-1, index=self.csr_permutation)
            if batched:
                assert batch_size is not None
                crow_indices = self.crow_indices.repeat(batch_size, 1)
                col_indices = self.col_indices.repeat(batch_size, 1)
            else:
                crow_indices = self.crow_indices
                col_indices = self.col_indices
            return torch.sparse_csr_tensor(
                crow_indices, col_indices, values, size=size_any, dtype=values.dtype, device=values.device
            )

        raise RuntimeError("Unsupported sparse layout")
