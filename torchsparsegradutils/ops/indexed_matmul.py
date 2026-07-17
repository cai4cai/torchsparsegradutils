import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# tsgu::grouped_gemm — op schema, fake kernel, autograd registration, and the
# public ``segment_mm``/``gather_mm`` wrappers that dispatch to it.
# spec/commit.md Phase 3 commit 18; routing verbatim from spec/map.md "Kernel
# routing": both ``segment_mm`` and ``gather_mm`` route their forward *and*
# both gradients through this single op ("transposed operands"). Schema
# takes plain dense tensors only (architecture.md §2); per architecture.md
# §6 this op bypasses BatchedCSR entirely — nothing sparse touches it. As of
# commit 18 the wrappers below call the op directly (their ``_legacy_*``
# nested-tensor bodies are deleted in the same commit, per the kernel-commit
# template T4) and the CUDA implementation is registered
# (cuda/csrc/kernels/grouped_gemm/).
# ---------------------------------------------------------------------------


@torch.library.custom_op("tsgu::grouped_gemm", mutates_args=())
def grouped_gemm(a: Tensor, b: Tensor, idx: Tensor, num_groups: int, reduce: bool) -> Tensor:
    r"""Grouped (indexed) matrix multiplication — the single kernel behind
    both ``segment_mm`` and ``gather_mm`` (map.md routing: forward and both
    grads of each route through this op; DGL-exact semantics; bypasses
    BatchedCSR entirely per architecture.md §6 — "nothing sparse touches
    these ops").

    Two modes, selected by ``reduce``:

    - ``reduce=False`` (**gather** mode — the forward pass of both public
      ops, and each op's ``gradA``): for every row ``i`` of ``a``, computes
      ``a[i] @ b[idx[i]]``. ``segment_mm(a, b, seglen_a)`` is this mode with
      ``idx = repeat_interleave(arange(R), seglen_a)`` built host-side (its
      contiguous segments are a special case of arbitrary per-row
      grouping); ``gather_mm(a, b, idx_b)`` is this mode directly with
      ``idx = idx_b``. Each op's ``gradA`` reuses this same mode with ``a``
      = the upstream gradient and ``b`` = the forward's ``b`` operand
      transposed per group (map.md: "transposed operands"). ``idx`` may be
      arbitrary (any order, repeats allowed) in this mode.
    - ``reduce=True`` (**scatter-reduce** mode — each op's ``gradB``): for
      every group ``k``, computes
      ``sum_{i : idx[i] == k} outer(a[i], b[i])`` — the adjoint of gather
      mode's linear map in ``b``. Both ``a`` and ``b`` are row-indexed by
      ``idx`` here (unlike gather mode, where ``b`` is indexed by group);
      this is what map.md's routing table means by "transposed operands"
      for ``gradB``: the same op, with the roles of the per-row and
      per-group operand swapped between input and output. **Contract:**
      this mode requires ``idx`` sorted non-decreasing — the kernel
      accumulates each group's rows in index order (deterministic ordered
      accumulation, no atomics) and *trusts* that ordering: it recovers
      each group's row range by binary search, so out-of-order rows would
      be silently misassigned (verifying sortedness on device would cost a
      sync — kernels.md shared rules). Callers with arbitrary ``idx``
      (this module's ``_grouped_gemm_backward``) sort rows with a stable
      ``argsort`` first; ``segment_mm``-built ``idx`` is sorted by
      construction.

    Parameters
    ----------
    a : Tensor, shape ``(N, D1)``
        Per-row left operand (both modes).
    b : Tensor, shape ``(num_groups, D1, D2)`` in gather mode, or
        ``(N, D2)`` in scatter mode
        Per-group right operand (gather mode) or per-row second operand
        (scatter mode).
    idx : Tensor, shape ``(N,)``, integer dtype
        Group index per row of ``a`` (and, in scatter mode, of ``b``), in
        ``[0, num_groups)``. Must be sorted non-decreasing when
        ``reduce=True`` (see above); arbitrary when ``reduce=False``.
    num_groups : int
        Number of groups (``R`` in map.md's ``segment_mm``/``gather_mm``
        signatures).
    reduce : bool
        Selects gather mode (``False``) or scatter-reduce mode (``True``).

    Returns
    -------
    Tensor
        Gather mode: shape ``(N, D2)``, ``D2 = b.shape[-1]``. Scatter mode:
        shape ``(num_groups, D1, D2)``, ``D1 = a.shape[-1]``,
        ``D2 = b.shape[-1]``.
    """
    raise NotImplementedError(
        "tsgu::grouped_gemm is CUDA-only (architecture.md §4): install a compatible torchsparsegradutils_cuda "
        "backend and pass CUDA tensors."
    )


@grouped_gemm.register_fake
def _grouped_gemm_fake(a: Tensor, b: Tensor, idx: Tensor, num_groups: int, reduce: bool) -> Tensor:
    # Value-independent: shapes derive only from a/b's own shapes and
    # num_groups/reduce, never from idx's contents.
    if reduce:
        return a.new_empty(num_groups, a.shape[-1], b.shape[-1])
    return a.new_empty(a.shape[0], b.shape[-1])


def _grouped_gemm_setup_context(ctx, inputs, output):
    a, b, idx, num_groups, reduce = inputs
    ctx.save_for_backward(a, b, idx)
    ctx.num_groups = num_groups
    ctx.reduce = reduce
    ctx.a_requires_grad = a.requires_grad
    ctx.b_requires_grad = b.requires_grad


def _grouped_gemm_backward(ctx, grad_output):
    a, b, idx = ctx.saved_tensors

    if ctx.reduce:
        # This invocation computed a gradB (scatter-reduce mode); map.md's
        # routing never differentiates *through* that computation itself
        # (it's only ever used as a gradient leaf), so no further autograd
        # is defined here.
        raise NotImplementedError(
            "tsgu::grouped_gemm(reduce=True) is only used as a gradient leaf (map.md routing); "
            "it is not itself differentiated."
        )

    grad_a = None
    if ctx.a_requires_grad:
        # gradA (map.md "transposed operands"): gather mode with b's groups
        # transposed. Gather mode accepts arbitrary idx — no sort needed.
        grad_a = torch.ops.tsgu.grouped_gemm(grad_output, b.mT, idx, ctx.num_groups, False)

    grad_b = None
    if ctx.b_requires_grad:
        # gradB (map.md "transposed operands"): scatter-reduce mode. The
        # kernel's reduce=True contract requires idx sorted non-decreasing
        # (deterministic ordered accumulation, no atomics — see the op
        # docstring), and this forward's idx is arbitrary for gather_mm, so
        # sort rows first. Stable argsort keeps within-group row order equal
        # to input order, and sorting unconditionally is branch-free — an
        # "is it already sorted?" check would cost a device sync, whereas
        # re-sorting segment_mm's already-sorted idx is cheap.
        perm = torch.argsort(idx, stable=True)
        grad_b = torch.ops.tsgu.grouped_gemm(a[perm], grad_output[perm], idx[perm], ctx.num_groups, True)

    return grad_a, grad_b, None, None, None


grouped_gemm.register_autograd(_grouped_gemm_backward, setup_context=_grouped_gemm_setup_context)


def _integer_dtype(t: torch.Tensor) -> bool:
    return not (t.dtype.is_floating_point or t.dtype.is_complex or t.dtype == torch.bool)


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
    ValueError
        If input ranks, dtypes, or sizes are incompatible (map.md invariant
        7: raise, never silently accept).
    NotImplementedError
        If no CUDA ``torchsparsegradutils_cuda`` backend is available —
        ``tsgu::grouped_gemm`` is CUDA-only (architecture.md §4).

    Notes
    -----
    Dispatches to ``tsgu::grouped_gemm`` (spec/commit.md Phase 3 commit 18)
    with ``idx = repeat_interleave(arange(R), seglen_a)`` — DGL-exact
    ``segment_mm`` semantics [1c]_ (``a[off:off+len] @ b[i]``, map.md
    contract), without the DGL dependency.

    See Also
    --------
    gather_mm : Per-row indexed matrix multiplication.

    References
    ----------
    .. [1c] DGL ``segment_mm`` documentation:
           https://www.dgl.ai/dgl_docs/generated/dgl.ops.segment_mm.html

    Examples
    --------
    ``tsgu::grouped_gemm`` (spec/commit.md Phase 3 commit 18) is CUDA-only
    (architecture.md §4) -- these examples run on CUDA tensors.

    >>> import torch
    >>> # N = 18, D1 = 4, D2 = 2
    >>> a = torch.randn(18, 4).cuda()
    >>> b = torch.randn(3, 4, 2).cuda()
    >>> seglen_a = torch.tensor([10, 5, 3])
    >>> out = segment_mm(a, b, seglen_a)
    >>> out.shape
    torch.Size([18, 2])

    Zero-length segment::

        >>> seglen_a = torch.tensor([10, 5, 0, 3])
        >>> b = torch.randn(4, 4, 2).cuda()
        >>> segment_mm(a, b, seglen_a).shape
        torch.Size([18, 2])
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not isinstance(seglen_a, torch.Tensor):
        raise ValueError("a, b and seglen_a must all be instances of torch.Tensor")
    if a.dim() != 2:
        raise ValueError(f"a must be a matrix with shape (N, D1); got shape {tuple(a.shape)}")
    if b.dim() != 3:
        raise ValueError(f"b must be a batched matrix with shape (R, D1, D2); got shape {tuple(b.shape)}")
    if seglen_a.dim() != 1:
        raise ValueError(f"seglen_a must be a vector with shape (R,); got shape {tuple(seglen_a.shape)}")
    if not _integer_dtype(seglen_a):
        raise ValueError(f"seglen_a must be a vector with shape (R,) of integer dtype; got dtype {seglen_a.dtype}")
    if b.shape[1] != a.shape[1]:
        raise ValueError(
            "a and b must share the inner dimension D1: a has shape (N, D1) and b has shape (R, D1, D2); "
            f"got a with shape {tuple(a.shape)} and b with shape {tuple(b.shape)}"
        )
    R = b.shape[0]
    if seglen_a.shape[0] != R:
        raise ValueError(
            f"seglen_a must be a vector with shape (R,) with R = b.shape[0] = {R}; got shape {tuple(seglen_a.shape)}"
        )

    N = a.shape[0]
    idx = torch.repeat_interleave(torch.arange(R, device=a.device), seglen_a.to(device=a.device))
    if idx.numel() != N:
        # DGL-exact semantics (map.md contract): segment lengths partition
        # a's rows exactly.
        raise ValueError(f"seglen_a must sum to N = a.shape[0] = {N}; got sum {idx.numel()}")

    return torch.ops.tsgu.grouped_gemm(a, b, idx, R, False)


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
        must satisfy ``0 <= idx_b[i] < R``. Any order, repeats allowed.

    Returns
    -------
    torch.Tensor, shape ``(N, D2)``
        Row-wise results where ``out[i] = a[i] @ b[idx_b[i]]``.

    Raises
    ------
    ValueError
        If inputs are not tensors, or ranks, dtypes, or sizes are
        incompatible (map.md invariant 7: raise, never silently accept).
    NotImplementedError
        If no CUDA ``torchsparsegradutils_cuda`` backend is available —
        ``tsgu::grouped_gemm`` is CUDA-only (architecture.md §4).

    Notes
    -----
    Dispatches to ``tsgu::grouped_gemm`` (spec/commit.md Phase 3 commit 18)
    with ``idx = idx_b`` directly — DGL-exact ``gather_mm`` semantics [1b]_
    (map.md contract), without the DGL dependency.

    See Also
    --------
    segment_mm : Segmented matrix multiplication over contiguous chunks.

    References
    ----------
    .. [1b] DGL ``gather_mm`` documentation:
           https://www.dgl.ai/dgl_docs/generated/dgl.ops.gather_mm.html

    Examples
    --------
    ``tsgu::grouped_gemm`` (spec/commit.md Phase 3 commit 18) is CUDA-only
    (architecture.md §4) -- these examples run on CUDA tensors.

    >>> import torch
    >>> # N = 5, D1 = 3, D2 = 2, R = 3
    >>> a = torch.randn(5, 3).cuda()
    >>> b = torch.randn(3, 3, 2).cuda()
    >>> idx_b = torch.tensor([0, 1, 0, 2, 1]).cuda()
    >>> out = gather_mm(a, b, idx_b)
    >>> out.shape
    torch.Size([5, 2])

    All rows using the same matrix::

        >>> torch.allclose(gather_mm(a, b, torch.zeros(5, dtype=torch.long, device="cuda")), a @ b[0])
        True

    Mixed indexing example::

        >>> # Different transformation for each row
        >>> a = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).cuda()  # (2, 2)
        >>> b = torch.tensor([[[1.0, 0.0], [0.0, 1.0]],  # Identity
        ...                   [[2.0, 0.0], [0.0, 2.0]]]).cuda()  # 2x scale
        >>> idx_b = torch.tensor([0, 1]).cuda()  # Use identity, then 2x scale
        >>> result = gather_mm(a, b, idx_b)
        >>> torch.allclose(result, torch.tensor([[1.0, 2.0], [6.0, 8.0]]).cuda())
        True
    """
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or not isinstance(idx_b, torch.Tensor):
        raise ValueError("a, b and idx_b must all be instances of torch.Tensor")
    if a.dim() != 2:
        raise ValueError(f"a must be a matrix with shape (N, D1); got shape {tuple(a.shape)}")
    if b.dim() != 3:
        raise ValueError(f"b must be a batched matrix with shape (R, D1, D2); got shape {tuple(b.shape)}")
    if idx_b.dim() != 1:
        raise ValueError(f"idx_b must be a vector with shape (N,); got shape {tuple(idx_b.shape)}")
    if not _integer_dtype(idx_b):
        raise ValueError(f"idx_b must be a vector with shape (N,) of integer dtype; got dtype {idx_b.dtype}")
    if idx_b.shape[0] != a.shape[0]:
        raise ValueError(
            f"idx_b must be a vector with shape (N,) with N = a.shape[0] = {a.shape[0]}; got shape {tuple(idx_b.shape)}"
        )
    if b.shape[1] != a.shape[1]:
        raise ValueError(
            "a and b must share the inner dimension D1: a has shape (N, D1) and b has shape (R, D1, D2); "
            f"got a with shape {tuple(a.shape)} and b with shape {tuple(b.shape)}"
        )

    return torch.ops.tsgu.grouped_gemm(a, b, idx_b, b.shape[0], False)
