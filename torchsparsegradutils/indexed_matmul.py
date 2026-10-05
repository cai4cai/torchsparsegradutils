from typing import Literal

import torch

try:
    import dgl.ops as dglops

    dgl_installed = True
except ImportError:
    dgl_installed = False


Strategy = Literal["auto", "dgl", "pad", "expand"]
_STRATEGIES = ("auto", "dgl", "pad", "expand")


_DGL_DEVICE_TYPES = ("cpu", "cuda")


def _use_dgl(strategy: str, device: torch.device) -> bool:
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}, expected one of {_STRATEGIES}")
    if strategy == "dgl" and not dgl_installed:
        raise ImportError("strategy='dgl' requires DGL to be installed")
    # "auto" only picks DGL on devices it supports (e.g. not MPS)
    return strategy == "dgl" or (strategy == "auto" and dgl_installed and device.type in _DGL_DEVICE_TYPES)


def _prefer_pad(N: int, R: int, L: int, D1: int, D2: int) -> bool:
    # Compare the extra memory of each pure PyTorch variant:
    # "pad" allocates (R, L, D1) and (R, L, D2) blocks, "expand" gathers an (N, D1, D2) copy of b
    return R * L * (D1 + D2) < N * D1 * D2


def _segment_mm_pad(a: torch.Tensor, b: torch.Tensor, seglen_a: torch.Tensor, L: int) -> torch.Tensor:
    # Scatter each segment of a into a zero-padded (R, L, D1) block and run a single dense bmm
    R = b.shape[0]
    starts = torch.cumsum(seglen_a, dim=0) - seglen_a
    seg = torch.repeat_interleave(torch.arange(R, device=a.device), seglen_a, output_size=a.shape[0])
    pos = torch.arange(a.shape[0], device=a.device) - starts[seg]
    a_pad = a.new_zeros((R, L, a.shape[1])).index_put((seg, pos), a)
    return torch.bmm(a_pad, b)[seg, pos]


def _gather_mm_expand(a: torch.Tensor, b: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
    # Materialise one (D1, D2) matrix per row and run a batched matrix-vector product
    return torch.bmm(a.unsqueeze(1), b[idx_b]).squeeze(1)


def segment_mm(a: torch.Tensor, b: torch.Tensor, seglen_a: torch.Tensor, strategy: Strategy = "auto") -> torch.Tensor:
    r"""
    Segmented matrix multiplication with variable-length segments.

    Performs matrix multiplication between contiguous segments of ``a`` and the
    corresponding matrices in ``b``. If ``seglen_a == [10, 5, 0, 3]``, the
    operator computes::

        a[0:10] @ b[0], a[10:15] @ b[1],
        a[15:15] @ b[2], a[15:18] @ b[3]

    Parameters
    ----------
    a : torch.Tensor, shape ``(N, D1)``
        Left operand containing the concatenation of all segments.
    b : torch.Tensor, shape ``(R, D1, D2)``
        Right operand containing one ``(D1, D2)`` matrix per segment.
    seglen_a : torch.Tensor, shape ``(R,)``, integer dtype
        Length of each segment in ``a``. ``seglen_a.sum()`` must equal ``N``.
    strategy : {"auto", "dgl", "pad", "expand"}, optional
        Implementation to use (default ``"auto"``). See Notes.

    Returns
    -------
    torch.Tensor, shape ``(N, D2)``
        Concatenation of all segment results in original order.

    Raises
    ------
    ValueError
        If input ranks or sizes are incompatible, or ``strategy`` is unknown.
    ImportError
        If ``strategy="dgl"`` and DGL is not installed.

    Notes
    -----
    Available strategies:

    - ``"dgl"``: :func:`dgl.ops.segment_mm` [1c]_ (typically fastest).
    - ``"pad"``: scatters ``a`` into a zero-padded ``(R, max(seglen_a), D1)`` block
      and runs a single :func:`torch.bmm`. Extra memory is
      :math:`O(R \cdot \max(\text{seglen}_a) \cdot (D_1 + D_2))`, which grows when
      segment lengths are unbalanced. Requires one device-to-host synchronisation.
    - ``"expand"``: gathers one copy of ``b`` per row of ``a`` and runs a batched
      matrix-vector product. Extra memory is :math:`O(N \cdot D_1 \cdot D_2)`.
      Does not synchronise with the host.
    - ``"auto"``: uses ``"dgl"`` if DGL is installed and supports the device of
      ``a`` (CPU or CUDA), otherwise whichever of ``"pad"`` and ``"expand"``
      needs less extra memory.

    See Also
    --------
    gather_mm : Per-row indexed matrix multiplication.

    References
    ----------
    .. [1c] DGL ``segment_mm`` documentation:
           https://www.dgl.ai/dgl_docs/generated/dgl.ops.segment_mm.html

    Examples
    --------
    >>> import torch
    >>> # N = 18, D1 = 4, D2 = 2
    >>> a = torch.randn(18, 4)
    >>> b = torch.randn(3, 4, 2)
    >>> seglen_a = torch.tensor([10, 5, 3])
    >>> out = segment_mm(a, b, seglen_a)
    >>> out.shape
    torch.Size([18, 2])

    Zero-length segment::

        >>> seglen_a = torch.tensor([10, 5, 0, 3])
        >>> b = torch.randn(4, 4, 2)
        >>> segment_mm(a, b, seglen_a).shape
        torch.Size([18, 2])
    """
    if _use_dgl(strategy, a.device):
        return dglops.segment_mm(a, b, seglen_a)

    if not a.dim() == 2 or not b.dim() == 3 or not seglen_a.dim() == 1:
        raise ValueError("Input tensors have unexpected dimensions")

    N, _ = a.shape
    R, D1, D2 = b.shape

    # Sanity check sizes
    if not a.shape[1] == D1 or not seglen_a.shape[0] == R:
        raise ValueError("Incompatible size for inputs")

    L = int(seglen_a.max()) if strategy != "expand" and R > 0 else 0
    if strategy == "pad" or (strategy == "auto" and _prefer_pad(N, R, L, D1, D2)):
        return _segment_mm_pad(a, b, seglen_a, L)

    idx_b = torch.repeat_interleave(torch.arange(R, device=a.device), seglen_a, output_size=N)
    return _gather_mm_expand(a, b, idx_b)


def gather_mm(a: torch.Tensor, b: torch.Tensor, idx_b: torch.Tensor, strategy: Strategy = "auto") -> torch.Tensor:
    r"""
    Per-row indexed matrix multiplication.

    For each row ``i`` in ``a`` this computes ``a[i] @ b[idx_b[i]]`` and stacks
    the results into the output.

    Parameters
    ----------
    a : torch.Tensor, shape ``(N, D1)``
        Left operand with one row per output.
    b : torch.Tensor, shape ``(R, D1, D2)``
        Bank of transformation matrices.
    idx_b : torch.Tensor, shape ``(N,)``, integer dtype
        Indices selecting which matrix in ``b`` to use for each row. Values
        must satisfy ``0 <= idx_b[i] < R``.
    strategy : {"auto", "dgl", "pad", "expand"}, optional
        Implementation to use (default ``"auto"``). See Notes.

    Returns
    -------
    torch.Tensor, shape ``(N, D2)``
        Row-wise results where ``out[i] = a[i] @ b[idx_b[i]]``.

    Raises
    ------
    ValueError
        If inputs are not tensors, ranks are incorrect, sizes are incompatible,
        or ``strategy`` is unknown.
    ImportError
        If ``strategy="dgl"`` and DGL is not installed.

    Notes
    -----
    Available strategies:

    - ``"dgl"``: :func:`dgl.ops.gather_mm` [1b]_ (typically fastest).
    - ``"pad"``: sorts the rows of ``a`` by ``idx_b`` and runs the ``"pad"``
      strategy of :func:`segment_mm`. Extra memory is
      :math:`O(R \cdot c_{\max} \cdot (D_1 + D_2))` where :math:`c_{\max}` is the
      largest number of rows sharing the same index. Requires one device-to-host
      synchronisation.
    - ``"expand"``: gathers ``b[idx_b]`` and runs a batched matrix-vector product.
      Extra memory is :math:`O(N \cdot D_1 \cdot D_2)`. Does not synchronise with
      the host.
    - ``"auto"``: uses ``"dgl"`` if DGL is installed and supports the device of
      ``a`` (CPU or CUDA), otherwise whichever of ``"pad"`` and ``"expand"``
      needs less extra memory.

    See Also
    --------
    segment_mm : Segmented matrix multiplication over contiguous chunks.

    References
    ----------
    .. [1b] DGL ``gather_mm`` documentation:
           https://www.dgl.ai/dgl_docs/generated/dgl.ops.gather_mm.html

    Examples
    --------
    >>> import torch
    >>> # N = 5, D1 = 3, D2 = 2, R = 3
    >>> a = torch.randn(5, 3)
    >>> b = torch.randn(3, 3, 2)
    >>> idx_b = torch.tensor([0, 1, 0, 2, 1])
    >>> out = gather_mm(a, b, idx_b)
    >>> out.shape
    torch.Size([5, 2])

    All rows using the same matrix::

        >>> torch.allclose(gather_mm(a, b, torch.zeros(5, dtype=torch.long)), a @ b[0])
        True

    Mixed indexing example::

        >>> # Different transformation for each row
        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        >>> b = torch.tensor([[[1.0, 0.0], [0.0, 1.0]],  # Identity
        ...                   [[2.0, 0.0], [0.0, 2.0]]])  # 2x scale
        >>> idx_b = torch.tensor([0, 1])  # Use identity, then 2x scale
        >>> result = gather_mm(a, b, idx_b)
        >>> result
        tensor([[1., 2.],
                [6., 8.]])
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not isinstance(idx_b, torch.Tensor):
        raise ValueError("Inputs should be instances of torch.Tensor")

    if _use_dgl(strategy, a.device):
        return dglops.gather_mm(a, b, idx_b)

    if not a.dim() == 2 or not b.dim() == 3 or not idx_b.dim() == 1:
        raise ValueError("Input tensors have unexpected dimensions")

    N = idx_b.shape[0]
    R, D1, D2 = b.shape

    # Sanity check sizes
    if not a.shape[0] == N or not a.shape[1] == D1:
        raise ValueError("Incompatible size for inputs")

    if strategy == "expand":
        return _gather_mm_expand(a, b, idx_b)

    seglen = torch.bincount(idx_b, minlength=R)
    L = int(seglen.max()) if R > 0 else 0
    if strategy == "pad" or _prefer_pad(N, R, L, D1, D2):
        # Group rows by index so that the "pad" segment_mm can be used, then undo the permutation
        perm = torch.argsort(idx_b, stable=True)
        ab = a.new_empty((N, D2))
        ab[perm] = _segment_mm_pad(a[perm], b, seglen, L)
        return ab

    return _gather_mm_expand(a, b, idx_b)
