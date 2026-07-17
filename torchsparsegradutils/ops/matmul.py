import torch
from torch import Tensor

from torchsparsegradutils._batched import BatchedCSR

# ---------------------------------------------------------------------------
# tsgu::spmm / tsgu::sddmm — op schemas, fake kernels, autograd registration.
# spec/commit.md Phase 1 #9; routing verbatim from spec/map.md "Kernel
# routing". Schemas take plain dense tensors only (architecture.md §2):
# BatchedCSR's `values`/`rowptr`/`col` fields (naming.md §2 kernel short
# names) plus shape ints `B`/`n`/`m` and a dense operand — never a
# torch.sparse_* tensor or a BatchedCSR object crosses this boundary.
#
# spec/commit.md Phase 3 commit 15: `sparse_mm` dispatches to `tsgu::spmm`
# (forward and, on the cached CSC, gradB) + `tsgu::sddmm` (gradA) below --
# its `_legacy_sparse_mm`/`SparseMatMul` bodies are deleted in this commit.
# Both CUDA implementations are registered (cuda/csrc/kernels/spmm/spmm.cu,
# cuda/csrc/kernels/sddmm/sddmm.cu, commits 14-15).
# ---------------------------------------------------------------------------


@torch.library.custom_op("tsgu::spmm", mutates_args=())
def spmm(vals: Tensor, rowptr: Tensor, col: Tensor, dense: Tensor, B: int, n: int, m: int) -> Tensor:
    r"""Batched sparse-CSR @ dense matmul (map.md: ``sparse_mm`` forward;
    also serves its ``gradB`` on the cached transpose — a CSC-shaped call
    with ``n``/``m`` swapped, ``BatchedCSR.transposed`` architecture.md §3 —
    and ``spmv``/iterative-solver matvecs with ``p = 1``, no separate op per
    map.md's kernel-routing note: "``spmv`` = ``spmm`` with ``p = 1``").

    Computes, for each batch item ``b``, the dense product of the sparse
    matrix ``A[b]`` (logical shape ``(n_rows, n_cols)``) against
    ``dense[b]``.

    Parameters
    ----------
    vals : Tensor, shape ``(nse_total,)``
        Stored values of the batched sparse matrix (naming.md §2 ``vals``).
    rowptr : Tensor, shape ``(B * n + 1,)``
        Absolute CSR pointer over folded rows ``row_global = b * n_rows + r``
        (naming.md §2 ``rowptr``).
    col : Tensor, shape ``(nse_total,)``
        Local column indices in ``[0, n_cols)`` (naming.md §2 ``col``).
    dense : Tensor, shape ``(B, n_cols, p)``
        Dense right operand, one ``(n_cols, p)`` matrix per batch item.
    B, n, m : int
        ``batch_size``, ``n_rows``, ``n_cols`` of the sparse matrix
        (naming.md §2 shape-int mapping).

    Returns
    -------
    Tensor, shape ``(B, n, p)``
        Dense product; ``p = dense.shape[-1]``.
    """
    raise NotImplementedError(
        "tsgu::spmm is CUDA-only (architecture.md §4): install a compatible torchsparsegradutils_cuda "
        "backend and pass CUDA tensors."
    )


@spmm.register_fake
def _spmm_fake(vals: Tensor, rowptr: Tensor, col: Tensor, dense: Tensor, B: int, n: int, m: int) -> Tensor:
    # Value-independent (architecture.md §2): the output shape derives only
    # from the shape ints and dense's own shape, never from index/value
    # contents of vals/rowptr/col.
    p = dense.shape[-1]
    return dense.new_empty(B, n, p)


@torch.library.custom_op("tsgu::sddmm", mutates_args=())
def sddmm(rowptr: Tensor, col: Tensor, g: Tensor, mat: Tensor, B: int, n: int, m: int, negate: bool) -> Tensor:
    r"""Sampled dense-dense matmul at a fixed sparsity pattern (map.md Family
    1, kernels.md Family 1) — the shared sparse-gradient backward of
    ``sparse_mm``, ``sparse_triangular_solve``, ``sparse_generic_solve``,
    ``sparse_generic_lstsq``.

    For every specified entry ``(row, col)`` of the pattern given by
    ``rowptr``/``col``, computes the row-wise inner product
    ``dot(g[b, row, :], mat[b, col, :])`` — negated when ``negate=True``,
    the fused negate epilogue map.md's routing table calls for
    ``sparse_triangular_solve``'s ``gradA`` (kernels.md: "negate-and-scale
    folded in for the solve backwards"). No dense materialisation: output
    values are aligned 1:1 with ``col`` — the output reuses the pattern's own
    index arrays (zero index allocation, kernels.md Family 1).

    Parameters
    ----------
    rowptr : Tensor, shape ``(B * n + 1,)``
        Absolute CSR pointer over folded rows of the sparsity pattern being
        sampled (naming.md §2 ``rowptr``) — the pattern's own stored values
        are not read here, only its structure.
    col : Tensor, shape ``(nse_total,)``
        Local column indices of the pattern (naming.md §2 ``col``).
    g : Tensor, shape ``(B, n, p)``
        Dense left operand (typically the upstream gradient).
    mat : Tensor, shape ``(B, m, p)``
        Dense right operand.
    B, n, m : int
        ``batch_size``, ``n_rows``, ``n_cols`` of the sampled pattern.
    negate : bool
        Fused negate epilogue (map.md routing: ``sparse_triangular_solve``'s
        ``gradA``); ``False`` for ``sparse_mm``'s plain ``gradA``.

    Returns
    -------
    Tensor, shape ``(nse_total,)``
        New stored values aligned with ``col`` / the pattern's ``rowptr``.
    """
    raise NotImplementedError(
        "tsgu::sddmm is CUDA-only (architecture.md §4): install a compatible torchsparsegradutils_cuda "
        "backend and pass CUDA tensors."
    )


@sddmm.register_fake
def _sddmm_fake(rowptr: Tensor, col: Tensor, g: Tensor, mat: Tensor, B: int, n: int, m: int, negate: bool) -> Tensor:
    # Value-independent: shape derives only from col's length.
    return g.new_empty(col.shape[0])


# tsgu::sddmm autograd (spec/commit.md Phase 3 commit 17): the solve ops'
# backwards route gradA through sddmm, and their higher-order-gradient
# guarantee (testing.md: "gradgradcheck ... solve ops — see PR #85") means
# the backward itself must be differentiable. Both adjoints of the sampled
# product are plain SpMMs at the sampled pattern (values = upstream grad):
# grad_g = spmm(±grad_out @ pattern, mat); grad_mat = the same on the
# transposed pattern — no new kernel, exactly map.md's routing vocabulary.


def _sddmm_setup_context(ctx, inputs, output):
    rowptr, col, g, mat, B, n, m, negate = inputs
    ctx.save_for_backward(rowptr, col, g, mat)
    ctx.B, ctx.n, ctx.m, ctx.negate = B, n, m, negate
    ctx.g_requires_grad = g.requires_grad
    ctx.mat_requires_grad = mat.requires_grad


def _sddmm_backward(ctx, grad_output):
    rowptr, col, g, mat = ctx.saved_tensors
    B, n, m = ctx.B, ctx.n, ctx.m
    # grad_output can arrive as an expanded stride-0 view (e.g. out.sum()'s
    # ones-grad). It is passed below as tsgu::spmm's `vals`, which the
    # launcher reads through a flat pointer without a contiguity guard (only
    # its dense operand is guarded) — a stride-0 vals silently reads garbage
    # past the one-element storage. Materialise it first; torch.neg already
    # produces a contiguous result on the negate path.
    signed_grad = torch.neg(grad_output) if ctx.negate else grad_output.contiguous()

    grad_g = None
    if ctx.g_requires_grad:
        grad_g = torch.ops.tsgu.spmm(signed_grad, rowptr, col, mat, B, n, m)

    grad_mat = None
    if ctx.mat_requires_grad:
        csc = BatchedCSR(values=signed_grad, rowptr=rowptr, col=col, shape=(B, n, m)).transposed
        grad_mat = torch.ops.tsgu.spmm(csc.values, csc.colptr, csc.row, g, B, m, n)

    return None, None, grad_g, grad_mat, None, None, None, None


sddmm.register_autograd(_sddmm_backward, setup_context=_sddmm_setup_context)


def _spmm_setup_context(ctx, inputs, output):
    vals, rowptr, col, dense, B, n, m = inputs
    ctx.save_for_backward(vals, rowptr, col, dense)
    ctx.B, ctx.n, ctx.m = B, n, m
    ctx.vals_requires_grad = vals.requires_grad
    ctx.dense_requires_grad = dense.requires_grad


def _spmm_backward(ctx, grad_output):
    vals, rowptr, col, dense = ctx.saved_tensors
    B, n, m = ctx.B, ctx.n, ctx.m

    grad_vals = None
    if ctx.vals_requires_grad:
        # gradA (map.md): tsgu::sddmm at A's own pattern, no negate epilogue.
        grad_vals = torch.ops.tsgu.sddmm(rowptr, col, grad_output, dense, B, n, m, False)

    grad_dense = None
    if ctx.dense_requires_grad:
        # gradB (map.md): tsgu::spmm on A^T. Unlike tsgu::spsm (which has its
        # own `transpose` flag), spmm has none, so gradB genuinely needs the
        # transposed pattern — built here via BatchedCSR.transposed, the
        # architecture.md §3 lazy member this is for ("transposed BatchedCSC
        # (gradB, ...)"). n/m swap roles under the transpose.
        csc = BatchedCSR(values=vals, rowptr=rowptr, col=col, shape=(B, n, m)).transposed
        grad_dense = torch.ops.tsgu.spmm(csc.values, csc.colptr, csc.row, grad_output, B, m, n)

    return grad_vals, None, None, grad_dense, None, None, None


spmm.register_autograd(_spmm_backward, setup_context=_spmm_setup_context)


def sparse_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    r"""Sparse–dense matrix multiplication with memory-efficient gradients.

     Computes :math:`\mathbf{C} = \mathbf{A}\,\mathbf{B}` where
     :math:`\mathbf{A} \in \mathbb{R}^{n\times m}` is sparse (COO/CSR),
     :math:`\mathbf{B} \in \mathbb{R}^{m\times p}` is dense, and
     :math:`\mathbf{C} \in \mathbb{R}^{n\times p}`. Gradients preserve the sparsity pattern
     of :math:`\mathbf{A}`. Supports unbatched 2D ``(n,m) @ (m,p)`` and batched 3D inputs by
     block–diagonalising the batch of sparse matrices and concatenating dense matrices along
     the batch dimension.

     Let the upstream gradient be :math:`\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{C}} \in \mathbb{R}^{n\times p}`.
     The gradients are:

     Gradient with respect to B (dense):

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{B}} \;=\; \mathbf{A}^{\top} \, \mathbf{G}.

     Gradient with respect to A (sparse): For a dense view one has

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{A}} \;=\; \mathbf{G}\, \mathbf{B}^{\top},

     but we evaluate only the entries at the nonzeros of :math:`\mathbf{A}`. Equivalently,
     for a nonzero entry :math:`\mathbf{A}_{ij}` the contribution is

     .. math::
         \bigg[\frac{\partial \mathcal{L}}{\partial \mathbf{A}}\bigg]_{ij}
         \;=\; \sum_{k=1}^{p} \mathbf{G}_{ik} \, \mathbf{B}_{jk}
         \;=\; \mathbf{G}_{i,:} \,\cdot\, \mathbf{B}_{j,:},

     where the dot denotes a row-wise inner product across the :math:`p` right-hand sides.

    Parameters
    ----------
    A : torch.Tensor, sparse COO or CSR, shape ``(n, m)`` or ``(b, n, m)``
        Left operand. For batched input, all batch items must share ``(n, m)``. All tensors
        must be on the same device.
    B : torch.Tensor, dense (strided), shape ``(m, p)`` or ``(b, m, p)``
        Right operand. Must have the same number of dimensions as ``A`` and
        matching batch size / inner dimension ``m``.

    Returns
    -------
    torch.Tensor
        Dense result of shape ``(n, p)`` or ``(b, n, p)``.

    Raises
    ------
    ValueError
        If ``A`` or ``B`` are not tensors; if ranks are < 2 or not both 2D/3D;
        if layouts are incompatible (``A`` not COO/CSR or ``B`` not dense);
        if shapes are incompatible (batch or inner dims).
    RuntimeError
        If the underlying sparse matmul fails.

    Notes
    -----
    This avoids dense gradients for sparse matrices [1a]_ (a known issue with
    :func:`torch.sparse.mm` backprop), computing only gradients at the nonzero
    entries of :math:`A` to reduce memory use.

    See Also
    --------
    torch.sparse.mm : PyTorch's native sparse ``@`` dense.
    sparse_generic_lstsq : Sparse least-squares with sparse-aware gradients.

    References
    ----------
    .. [1a] PyTorch issue on dense gradients for sparse ops:
           https://github.com/pytorch/pytorch/issues/41128

    Examples
    --------
    ``tsgu::spmm`` (spec/commit.md Phase 3 commit 15) is CUDA-only
    (architecture.md §4) -- these examples run on a CUDA tensor.

    Basic (unbatched)::

        >>> indices = torch.tensor([[0, 0, 1, 1, 2, 2],
        ...                         [0, 2, 1, 3, 0, 2]])
        >>> values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> A = torch.sparse_coo_tensor(indices, values, (3, 4)).cuda()
        >>> B = torch.randn(4, 2).cuda()
        >>> out = sparse_mm(A, B)
        >>> out.shape
        torch.Size([3, 2])

    Batched::

        >>> A_batch = torch.stack([A, A])          # (2, 3, 4) — COO stack
        >>> B_batch = torch.randn(2, 4, 2).cuda()  # (2, 4, 2)
        >>> out = sparse_mm(A_batch, B_batch)
        >>> out.shape
        torch.Size([2, 3, 2])

    With gradients::

        >>> A.requires_grad_(True)  # doctest: +ELLIPSIS
        tensor(...)
        >>> B.requires_grad_(True)  # doctest: +ELLIPSIS
        tensor(...)
        >>> out = sparse_mm(A, B)
        >>> out.sum().backward()
        >>> A.grad.is_sparse
        True
    """

    if not isinstance(A, torch.Tensor) or not isinstance(B, torch.Tensor):
        raise ValueError("Both A and B should be instances of torch.Tensor")
    if A.dim() < 2 or B.dim() < 2:
        raise ValueError("Both A and B should be at least 2-dimensional tensors")
    if A.dim() != B.dim() or A.dim() not in (2, 3):
        raise ValueError("A and B must both be 2D or both be 3D tensors")
    if A.layout not in {torch.sparse_coo, torch.sparse_csr}:
        raise ValueError("A should be in either COO or CSR sparse format")
    if B.layout != torch.strided:
        raise ValueError("B must be a dense (strided) tensor")
    if A.dim() == 3 and A.size(0) != B.size(0):
        raise ValueError("If batched, A and B must have the same batch size")
    if A.size(-1) != B.size(-2):
        raise ValueError(f"Incompatible inner dimensions: A[..., {A.size(-1)}] vs B[..., {B.size(-2)}]")

    return _tsgu_sparse_mm(A, B)


def _tsgu_sparse_mm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """``tsgu::spmm``-backed forward (spec/commit.md Phase 3 commit 15;
    map.md routing: ``sparse_mm`` forward -> ``tsgu::spmm``, replacing
    ``_legacy_sparse_mm``/``SparseMatMul``, deleted in this commit).

    Unwraps ``A`` into its ``BatchedCSR`` descriptor at the boundary
    (architecture.md §2) and calls the kernel op directly on plain tensors
    -- no manual backward wiring is written here: ``BatchedCSR.from_torch``'s
    ``values()``/``coalesce()`` extraction is itself autograd-differentiable
    (PyTorch's built-in sparse-tensor ``values()`` adjoint reconstructs a
    sparse gradient at the same indices/layout it was called on), so
    gradients flow automatically through ``torch.ops.tsgu.spmm``'s own
    ``register_autograd`` (above, commit 9) -- which itself calls
    ``tsgu::sddmm`` for gradA and ``tsgu::spmm`` again on the cached CSC
    transpose for gradB (map.md routing) -- and back through the extraction
    chain, rewrapping gradA as a sparse tensor at A's own pattern in A's own
    layout (COO in -> COO grad out, CSR in -> CSR grad out; map.md
    invariant 3) with no explicit rewrap call needed here.

    ``B`` is unsqueezed to ``(1, m, p)`` for the unbatched case (``tsgu::spmm``
    always takes a batched ``(B, m, p)`` dense operand); ``unsqueeze``/
    ``squeeze`` are themselves differentiable views, so ``B``'s gradient
    shape is restored automatically too.
    """
    batched = A.dim() == 3
    csr = BatchedCSR.from_torch(A)
    dense = B if batched else B.unsqueeze(0)
    out = torch.ops.tsgu.spmm(csr.values, csr.rowptr, csr.col, dense, csr.shape[0], csr.shape[1], csr.shape[2])
    return out if batched else out.squeeze(0)
