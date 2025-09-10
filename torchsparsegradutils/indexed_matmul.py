import torch

try:
    import dgl.ops as dglops

    dgl_installed = True
except ImportError:
    dgl_installed = False


def segment_mm(a: torch.Tensor, b: torch.Tensor, seglen_a: torch.Tensor) -> torch.Tensor:
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

    Returns
    -------
    torch.Tensor, shape ``(N, D2)``
        Concatenation of all segment results in original order.

    Raises
    ------
    NotImplementedError
        If the fallback path is used on a PyTorch version lacking nested
        tensor matmul support (requires PyTorch >= 2.4).
    ValueError
        If input ranks or sizes are incompatible.

    Notes
    -----
    If DGL is available, this uses :func:`dgl.ops.segment_mm` [1c]_ (typically faster).
    Otherwise it falls back to a PyTorch nested-tensor implementation.

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
    if torch.__version__ < (2, 4):
        raise NotImplementedError("PyTorch version is too old for nested tensors")

    if dgl_installed:
        # DGL is probably more computationally efficient
        # See https://github.com/pytorch/pytorch/issues/136747
        return dglops.segment_mm(a, b, seglen_a)

    if not a.dim() == 2 or not b.dim() == 3 or not seglen_a.dim() == 1:
        raise ValueError("Input tensors have unexpected dimensions")

    N, _ = a.shape
    R, D1, D2 = b.shape

    # Sanity check sizes
    if not a.shape[1] == D1 or not seglen_a.shape[0] == R:
        raise ValueError("Incompatible size for inputs")

    segidx_a = torch.cumsum(seglen_a[:-1], dim=0).cpu()

    # Ideally the conversions below to nested tensor would be handled natively
    nested_a = torch.nested.as_nested_tensor(torch.tensor_split(a, segidx_a, dim=0))
    nested_b = torch.nested.as_nested_tensor(torch.split(b, 1, dim=0)).reshape((R, D1, D2))

    # The actual gather matmul computation
    nested_ab = torch.matmul(nested_a, nested_b)

    # Convert back to tensors, again ideally this would be handled natively
    ab = torch.cat(nested_ab.unbind(), dim=0)
    return ab


def gather_mm(a: torch.Tensor, b: torch.Tensor, idx_b: torch.Tensor) -> torch.Tensor:
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

    Returns
    -------
    torch.Tensor, shape ``(N, D2)``
        Row-wise results where ``out[i] = a[i] @ b[idx_b[i]]``.

    Raises
    ------
    NotImplementedError
        If the fallback path is used on a PyTorch version lacking nested
        tensor matmul support (requires PyTorch >= 2.4).
    ValueError
        If inputs are not tensors, ranks are incorrect, or sizes are incompatible.

    Notes
    -----
    If DGL is available, this uses :func:`dgl.ops.gather_mm` [1b]_. Otherwise it uses
    a dependency-free PyTorch nested-tensor fallback.

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
    if torch.__version__ < (2, 4):
        raise NotImplementedError("PyTorch version is too old for nested tensors")

    if dgl_installed:
        # DGL is more computationally efficient
        # See https://github.com/pytorch/pytorch/issues/136747
        return dglops.gather_mm(a, b, idx_b)

    # Dependency free fallback
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not isinstance(idx_b, torch.Tensor):
        raise ValueError("Inputs should be instances of torch.Tensor")

    if not a.dim() == 2 or not b.dim() == 3 or not idx_b.dim() == 1:
        raise ValueError("Input tensors have unexpected dimensions")

    N = idx_b.shape[0]
    R, D1, D2 = b.shape

    # Sanity check sizes
    if not a.shape[0] == N or not a.shape[1] == D1:
        raise ValueError("Incompatible size for inputs")

    torchdevice = a.device
    src_idx = torch.arange(N, device=torchdevice)

    # Ideally the conversions below to nested tensor would be handled without for loops and without copy
    nested_a = torch.nested.as_nested_tensor([a[idx_b == i, :] for i in range(R)])
    src_idx_reshuffled = torch.cat([src_idx[idx_b == i] for i in range(R)])
    nested_b = torch.nested.as_nested_tensor(torch.split(b, 1, dim=0)).reshape((R, D1, D2))

    # The actual gather matmul computation
    nested_ab = torch.matmul(nested_a, nested_b)

    # Convert back to tensors, again, ideally this would be handled natively with no copy
    ab_segmented = torch.cat(nested_ab.unbind(), dim=0)
    ab = torch.empty((N, D2), device=torchdevice)
    ab[src_idx_reshuffled] = ab_segmented
    return ab
