import torch
from torch import Tensor

from torchsparsegradutils._batched import BatchedCSR

# ---------------------------------------------------------------------------
# tsgu::spsm — op schema, fake kernel, autograd registration.
# spec/commit.md Phase 1 #9; routing verbatim from spec/map.md "Kernel
# routing". Schema takes plain dense tensors only (architecture.md §2):
# BatchedCSR's `values`/`rowptr`/`col` fields (naming.md §2 kernel short
# names) plus shape ints `B`/`n` and the `upper`/`unitriangular`/`transpose`
# flags kept verbatim from the public contract — never a torch.sparse_*
# tensor or a BatchedCSR object crosses this boundary.
#
# spec/commit.md Phase 3 commit 16: `sparse_triangular_solve` dispatches to
# `tsgu::spsm` (forward, and its own gradB via the "transposed plan" — the
# same op with `transpose` flipped) + `tsgu::sddmm` (gradA, negate
# epilogue) below -- its pre-rewrite pure-PyTorch bodies were deleted in
# that commit. The CUDA
# implementation (cuda/csrc/kernels/spsm/spsm.cu + plan.cpp's analysis-plan
# cache, architecture.md §3) is registered.
# ---------------------------------------------------------------------------


@torch.library.custom_op("tsgu::spsm", mutates_args=())
def spsm(
    vals: Tensor,
    rowptr: Tensor,
    col: Tensor,
    rhs: Tensor,
    B: int,
    n: int,
    upper: bool,
    unitriangular: bool,
    transpose: bool,
) -> Tensor:
    r"""Batched sparse triangular solve (map.md: ``sparse_triangular_solve``
    forward; also serves its own ``gradB`` via the "transposed plan" —
    calling this same op again with ``transpose`` flipped, map.md routing).
    The analysis-plan cache (architecture.md §3) lives C++-side, keyed on
    this call's ``rowptr``/``col`` tensor identity (cuda/csrc/kernels/spsm/
    plan.h/plan.cpp) — there is no plan-tensor argument on this schema (see
    that file's module comment for why).

    Solves, per batch item ``b``, :math:`A[b]\,x[b] = \mathrm{rhs}[b]` (or
    :math:`A[b]^\top\,x[b] = \mathrm{rhs}[b]` when ``transpose=True``) for
    triangular ``A[b]``.

    Parameters
    ----------
    vals : Tensor, shape ``(nse_total,)``
        Stored values of the triangular matrix (naming.md §2 ``vals``).
    rowptr : Tensor, shape ``(B * n + 1,)``
        Absolute CSR pointer over folded rows (naming.md §2 ``rowptr``).
    col : Tensor, shape ``(nse_total,)``
        Local column indices in ``[0, n)`` (naming.md §2 ``col``) — ``A`` is
        square, ``n_rows == n_cols == n``.
    rhs : Tensor, shape ``(B, n, p)``
        Right-hand side.
    B, n : int
        ``batch_size``, ``n_rows`` (== ``n_cols``).
    upper, unitriangular, transpose : bool
        Kept verbatim from the public ``sparse_triangular_solve`` contract
        (map.md invariant 1).

    Returns
    -------
    Tensor, shape ``(B, n, p)``
        Solution ``x``, same shape as ``rhs``.
    """
    raise NotImplementedError(
        "tsgu::spsm has no Python/CPU implementation (architecture.md §4: CUDA-required at runtime) — "
        "this schema body only runs as a CPU/meta fallback; the real CUDA implementation is registered "
        "from cuda/csrc/kernels/spsm/spsm.cu."
    )


@spsm.register_fake
def _spsm_fake(
    vals: Tensor,
    rowptr: Tensor,
    col: Tensor,
    rhs: Tensor,
    B: int,
    n: int,
    upper: bool,
    unitriangular: bool,
    transpose: bool,
) -> Tensor:
    # Value-independent: output shape/dtype exactly mirror rhs.
    return rhs.new_empty(rhs.shape)


def _spsm_setup_context(ctx, inputs, output):
    vals, rowptr, col, rhs, B, n, upper, unitriangular, transpose = inputs
    ctx.save_for_backward(vals, rowptr, col, output)
    ctx.B, ctx.n = B, n
    ctx.upper, ctx.unitriangular, ctx.transpose = upper, unitriangular, transpose
    ctx.vals_requires_grad = vals.requires_grad
    ctx.rhs_requires_grad = rhs.requires_grad


def _spsm_backward(ctx, grad_output):
    vals, rowptr, col, x = ctx.saved_tensors
    B, n = ctx.B, ctx.n

    need_gradB = ctx.rhs_requires_grad or ctx.vals_requires_grad
    gradB = None
    if need_gradB:
        # gradB (map.md "transposed plan"): the same op with `transpose`
        # flipped. Unlike tsgu::spmm's gradB, spsm carries its own
        # `transpose` flag, so no separately-transposed (CSC) pattern is
        # needed — `vals`/`rowptr`/`col` are reused as-is.
        gradB = torch.ops.tsgu.spsm(
            vals, rowptr, col, grad_output, B, n, ctx.upper, ctx.unitriangular, not ctx.transpose
        )

    grad_vals = None
    if ctx.vals_requires_grad:
        # gradA (map.md): sampled outer product -gradB @ x^T at A's own
        # pattern, via tsgu::sddmm's negate epilogue.
        #
        # commit-9 flag, resolved here (commit 16): tsgu::sddmm's contract
        # is out[k] = dot(g[row(k), :], mat[col(k), :]) (negated). For
        # transpose=False this is directly gradA[i, j] = -dot(gradB[i,:],
        # x[j,:]) -- sddmm(rowptr, col, g=gradB, mat=x, negate=True).
        #
        # For transpose=True the pre-rewrite adjoint derivation
        # swaps which array A_row_idx/A_col_idx index into:
        #   mgradbselect = -gradB[A_col_idx, :]   (i.e. -gradB[j, :])
        #   xselect      =  x[A_row_idx, :]       (i.e.  x[i, :])
        #   gradA[k] = sum(mgradbselect * xselect) = -dot(gradB[j,:], x[i,:])
        # Re-derived independently from the effective system M = A^T (the
        # one actually solved when transpose=True): dL/dM = -(M^{-T} G) x^T
        # with M^{-T} = A^{-1}, so dL/dM = -gradB x^T (gradB = A^{-1} G,
        # exactly what the op call above computes for transpose=True).
        # M[p, q] = A[q, p], so dL/dA[q, p] = dL/dM[p, q] = -dot(gradB[p,:],
        # x[q,:]); for a stored entry at (row=i, col=j) — i.e. q=i, p=j —
        # this is gradA[i,j] = -dot(gradB[j,:], x[i,:]), matching the legacy
        # code exactly.
        #
        # tsgu::sddmm has no row/col-swap flag, but A is square (n_rows ==
        # n_cols == n for a triangular solve) and dot() is commutative, so
        # -dot(gradB[j,:], x[i,:]) = -dot(x[row(k),:], gradB[col(k),:]) is
        # realized by swapping the *dense operands* (g <-> mat), not the
        # pattern -- gradA stays aligned with A's own rowptr/col either way.
        g, mat = (x, gradB) if ctx.transpose else (gradB, x)
        grad_vals = torch.ops.tsgu.sddmm(rowptr, col, g, mat, B, n, n, True)

    grad_rhs = gradB if ctx.rhs_requires_grad else None

    return grad_vals, None, None, grad_rhs, None, None, None, None, None


spsm.register_autograd(_spsm_backward, setup_context=_spsm_setup_context)


# ---------------------------------------------------------------------------
# tsgu::_spsm_plan_cache_stats — test/introspection-only op (spec/commit.md
# Phase 3 commit 16 T5: "plan-cache tests: same BatchedCSR solved twice ->
# analysis computed once (assert via the lazy member's identity/a
# counter)"). The plan cache itself lives in C++ (cuda/csrc/kernels/spsm/
# plan.h/plan.cpp's module comment has the full design) -- this is the only
# way to observe its process-wide (builds, hits) counters from Python,
# since nothing else about that cache crosses the op boundary. `anchor` is
# any CUDA tensor (device/dtype-inheritance only, per the `new_empty`
# convention every other op in this file uses) -- its values are never
# read.
# ---------------------------------------------------------------------------


@torch.library.custom_op("tsgu::_spsm_plan_cache_stats", mutates_args=())
def _spsm_plan_cache_stats(anchor: Tensor) -> Tensor:
    raise NotImplementedError(
        "tsgu::_spsm_plan_cache_stats has no Python/CPU implementation -- test/introspection-only op; "
        "the real CUDA implementation reads cuda/csrc/kernels/spsm/plan.cpp's process-wide counters."
    )


@_spsm_plan_cache_stats.register_fake
def _spsm_plan_cache_stats_fake(anchor: Tensor) -> Tensor:
    return anchor.new_empty(2, dtype=torch.int64)


def sparse_triangular_solve(
    A: torch.Tensor,
    B: torch.Tensor,
    upper: bool = True,
    unitriangular: bool = False,
    transpose: bool = False,
) -> torch.Tensor:
    r"""Sparse triangular solve with memory-efficient sparse gradients.

     Solves the triangular system :math:`\mathbf{A}\,\mathbf{x} = \mathbf{B}` (or
     :math:`\mathbf{A}^{\top}\,\mathbf{x} = \mathbf{B}` if ``transpose=True``), where
     :math:`\mathbf{A} \in \mathbb{R}^{m\times m}` is sparse triangular (COO/CSR) and
     :math:`\mathbf{B} \in \mathbb{R}^{m\times p}` is dense. Gradients preserve the sparsity
     pattern of :math:`\mathbf{A}` by evaluating only at its nonzero entries. Supports
     unbatched 2D and batched 3D inputs; COO inputs are converted to CSR internally for the
     factor solve.

     Let the upstream gradient be :math:`\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}}`
     for a scalar objective :math:`\mathcal{L}` and solution :math:`\mathbf{x}`. The dense-form
     gradients are

     Gradient with respect to B (dense):

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{B}} \;=\; \mathbf{A}^{-\top} \, \mathbf{G},

     and for ``transpose=True`` replace :math:`\mathbf{A}` by :math:`\mathbf{A}^{\top}` so that
     :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{B}} = \left(\mathbf{A}^{\top}\right)^{-\top} \mathbf{G} = \mathbf{A}^{-1} \mathbf{G}`.

     Gradient with respect to A (sparse):

     .. math::
         \frac{\partial \mathcal{L}}{\partial \mathbf{A}} \;=\; -\big(\mathbf{A}^{-\top} \, \mathbf{G}\big)\, \mathbf{x}^{\top},

     and only entries at the nonzeros of :math:`\mathbf{A}` are evaluated. Equivalently,
     for a nonzero :math:`\mathbf{A}_{ij}` the contribution is

     .. math::
         \bigg[\frac{\partial \mathcal{L}}{\partial \mathbf{A}}\bigg]_{ij}
         \;=\; -\, \big(\mathbf{A}^{-\top} \, \mathbf{G}\big)_{i,:} \,\cdot\, \mathbf{x}_{j,:},

     where the dot denotes a row-wise inner product across the :math:`p` right-hand sides.

    Parameters
    ----------
    A : torch.Tensor, sparse COO or CSR, shape ``(m, m)`` or ``(b, m, m)``
        Sparse triangular coefficient matrix. Must be square per batch. All tensors must
        be on the same device.
    B : torch.Tensor, dense (strided), shape ``(m, p)`` or ``(b, m, p)``
        Right-hand side. ``B.shape[-2]`` must equal ``A.shape[-2]`` (``m``).
    upper : bool, optional
        If ``True`` (default), treat ``A`` as upper-triangular; else lower-triangular.
    unitriangular : bool, optional
        If ``True``, assume unit diagonal (implicit ones). The stored matrix must be
        strictly triangular (no explicit diagonal entries). Default: ``False``.
    transpose : bool, optional
        If ``True``, solves :math:`A^\top x = B`; otherwise :math:`A x = B`. Default: ``False``.

    Returns
    -------
    torch.Tensor
        Solution with the same shape as ``B``: ``(m, p)`` or ``(b, m, p)``.

    Raises
    ------
    ValueError
        If inputs are not tensors; ranks are < 2 or not both 2D/3D; layouts are
        incompatible (``A`` not COO/CSR or ``B`` not dense); shapes are incompatible;
        batch sizes differ; or if ``unitriangular=True`` but explicit diagonal
        entries are present.
    RuntimeError
        If the underlying triangular solve fails.

    Notes
    -----
    Backprop computes gradients only at nonzero entries of :math:`\mathbf{A}`, keeping the
    gradient sparse and reducing memory. COO inputs are converted to CSR since PyTorch's
    triangular solver requires CSR [1e]_. For autograd implementation details, see [2e]_.

    See Also
    --------
    torch.sparse.mm : Sparse ``@`` dense multiply.
    torch.linalg.solve_triangular : Dense triangular solver (modern API).

    References
    ----------
    .. [1e] PyTorch issue on sparse triangular solve:
           https://github.com/pytorch/pytorch/issues/87358
    .. [2e] PyTorch issue on autograd/triangular solve:
           https://github.com/pytorch/pytorch/issues/88890

    Examples
    --------
    ``tsgu::spsm`` (spec/commit.md Phase 3 commit 16) is CUDA-only
    (architecture.md §4) -- these examples run on a CUDA tensor.

    Upper-triangular::

        >>> import torch
        >>> from torchsparsegradutils import sparse_triangular_solve
        >>> A = torch.sparse_csr_tensor([0, 2, 3, 4], [0, 2, 1, 2],
        ...                             torch.tensor([2.0, 1.0, 3.0, 1.0]), (3, 3)).cuda()
        >>> B = torch.tensor([[1.0], [2.0], [3.0]]).cuda()
        >>> x = sparse_triangular_solve(A, B, upper=True)
        >>> x.shape
        torch.Size([3, 1])

    Lower-triangular::

        >>> A_low = torch.sparse_csr_tensor([0, 1, 3, 5], [0, 0, 1, 0, 2],
        ...                                 torch.tensor([2.0, 1.0, 3.0, 0.5, 1.0]), (3, 3)).cuda()
        >>> x = sparse_triangular_solve(A_low, B, upper=False)

    Batched::

        >>> # Convert to COO for batching (since torch.stack doesn't work with CSR)
        >>> A_coo = A.to_sparse_coo()
        >>> A_b = torch.stack([A_coo, A_coo])   # (2, 3, 3)
        >>> B_b = torch.stack([B, B])   # (2, 3, 1)
        >>> x_b = sparse_triangular_solve(A_b, B_b)
        >>> x_b.shape
        torch.Size([2, 3, 1])
    """
    # --- minimal validations to match the docstring expectations ---
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
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("A must be square on its last two dimensions")
    if A.size(-2) != B.size(-2):
        raise ValueError(f"Incompatible inner dimensions: A[..., {A.size(-2)}] vs B[..., {B.size(-2)}]")
    if A.dim() == 3 and A.size(0) != B.size(0):
        raise ValueError("If batched, A and B must have the same batch size")

    return _tsgu_sparse_triangular_solve(A, B, upper, unitriangular, transpose)


def _tsgu_sparse_triangular_solve(
    A: torch.Tensor, B: torch.Tensor, upper: bool, unitriangular: bool, transpose: bool
) -> torch.Tensor:
    """``tsgu::spsm``-backed forward (spec/commit.md Phase 3 commit 16;
    map.md routing: ``sparse_triangular_solve`` forward -> ``tsgu::spsm``,
    replacing the pre-rewrite pure-PyTorch implementation,
    deleted in this commit).

    Unwraps ``A`` into its ``BatchedCSR`` descriptor at the boundary
    (architecture.md §2; COO inputs go through ``BatchedCSR.from_torch``'s
    own coalesce+compress, matching the docstring's "COO inputs are
    converted to CSR internally") and calls the kernel op directly on plain
    tensors -- no manual backward wiring is written here, same pattern as
    ``ops/matmul.py``'s ``_tsgu_sparse_mm``: PyTorch's built-in sparse-tensor
    ``values()``/``coalesce()`` autograd reconstructs a sparse gradient at
    A's own indices/layout automatically, so gradients flow through
    ``torch.ops.tsgu.spsm``'s own ``register_autograd`` (above, commit 9;
    transpose-indexing resolved commit 16) and back through the extraction
    chain with no explicit rewrap call needed here.

    ``B`` is unsqueezed to ``(1, n, p)`` for the unbatched case (``tsgu::spsm``
    always takes a batched ``(B, n, p)`` rhs); ``unsqueeze``/``squeeze`` are
    themselves differentiable views, so ``B``'s gradient shape is restored
    automatically too.
    """
    batched = A.dim() == 3
    csr = BatchedCSR.from_torch(A)
    rhs = B if batched else B.unsqueeze(0)
    out = torch.ops.tsgu.spsm(
        csr.values, csr.rowptr, csr.col, rhs, csr.shape[0], csr.shape[1], upper, unitriangular, transpose
    )
    return out if batched else out.squeeze(0)
