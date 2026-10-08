r"""Preconditioners for the iterative solvers.

A preconditioner is any callable ``M_inv(X) -> Y`` applying an approximation :math:`\mathbf{M}^{-1}` of
:math:`\mathbf{A}^{-1}` to ``X`` of shape ``(n,)`` or ``(*batch, n, k)``, returning a tensor of the same shape,
dtype and device. The square-system solvers :func:`~torchsparsegradutils.utils.bicgstab`,
:func:`~torchsparsegradutils.utils.gmres`, :func:`~torchsparsegradutils.utils.linear_cg` and
:func:`~torchsparsegradutils.utils.minres` accept one through their ``preconditioner`` argument;
:func:`~torchsparsegradutils.utils.lsmr` does not support preconditioning yet.

The :class:`Preconditioner` base class adds what a bare callable lacks: :meth:`Preconditioner.transpose`, which
:func:`~torchsparsegradutils.sparse_generic_solve` uses to precondition the transposed system
:math:`\mathbf{A}^\top \mathbf{Y} = \mathbf{G}` solved in the backward pass, and structural flags that the
symmetric solvers rely on.

The implicit-function gradients of :func:`~torchsparsegradutils.sparse_generic_solve` do not depend on
:math:`\mathbf{M}`, so preconditioners built from :math:`\mathbf{A}` (such as :class:`JacobiPreconditioner`, or
a :class:`Preconditioner` subclass passed to :func:`~torchsparsegradutils.sparse_generic_solve`) use a detached
copy of it. :class:`MatrixPreconditioner` keeps its matrix as given: applying it directly is differentiable with
respect to that matrix.
"""

from __future__ import annotations

import abc

import torch

from .utils import sparse_diagonal

__all__ = ["Preconditioner", "JacobiPreconditioner", "MatrixPreconditioner"]


class Preconditioner(abc.ABC):
    r"""Base class for preconditioners approximating :math:`\mathbf{A}^{-1}`.

    Subclasses implement :meth:`__call__` and, unless they are symmetric, :meth:`transpose`.

    Attributes
    ----------
    shape : tuple of int
        Shape ``(n, n)`` of the operator :math:`\mathbf{M}^{-1}`.
    is_symmetric : bool
        Whether :math:`\mathbf{M}^{-1}` is symmetric, in which case :meth:`transpose` returns ``self``.
    is_positive_definite : bool
        Whether :math:`\mathbf{M}^{-1}` is symmetric positive definite, as required by
        :func:`~torchsparsegradutils.utils.linear_cg` and :func:`~torchsparsegradutils.utils.minres`.
    """

    shape: tuple[int, int]
    is_symmetric: bool = False
    is_positive_definite: bool = False

    @abc.abstractmethod
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        r"""Apply :math:`\mathbf{M}^{-1}` to ``X`` of shape ``(n,)`` or ``(*batch, n, k)``."""

    def transpose(self) -> Preconditioner:
        r"""Return the preconditioner :math:`\mathbf{M}^{-\top}` for the transposed system."""
        if self.is_symmetric:
            return self
        raise NotImplementedError(f"{type(self).__name__} does not implement transpose()")

    @property
    def T(self) -> Preconditioner:
        """Alias of :meth:`transpose`."""
        return self.transpose()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={tuple(self.shape)})"


class JacobiPreconditioner(Preconditioner):
    r"""Jacobi (diagonal) preconditioner :math:`\mathbf{M} = \operatorname{diag}(\mathbf{A})`.

    Applying it costs one elementwise product per vector and building it a single pass over the nonzeros of
    :math:`\mathbf{A}`, so it adds :math:`O(n)` memory. It is effective when :math:`\mathbf{A}` is badly scaled,
    e.g. when its rows have very different magnitudes, but does not help with conditioning that stems from the
    coupling between unknowns.

    Parameters
    ----------
    A : torch.Tensor, sparse COO or CSR, or dense (strided), shape ``(n, n)``
        Matrix to precondition. It is detached, so the preconditioner never tracks gradients.
    absolute : bool, default=False
        Use :math:`|\operatorname{diag}(\mathbf{A})|` instead of the diagonal itself. For symmetric indefinite
        matrices, this yields the symmetric positive definite preconditioner that
        :func:`~torchsparsegradutils.utils.minres` requires.

    Notes
    -----
    Zero diagonal entries are replaced by one, so :math:`\mathbf{M}^{-1}` acts as the identity on those rows and
    stays invertible. :math:`\mathbf{M}^{-1}` is always symmetric; ``is_positive_definite`` reports whether all
    (used) diagonal entries are positive, which is checked once at construction.

    Examples
    --------
    >>> import torch
    >>> from torchsparsegradutils.utils import JacobiPreconditioner, linear_cg
    >>> A = torch.tensor([[100.0, 1.0], [1.0, 0.1]]).to_sparse_csr()
    >>> M_inv = JacobiPreconditioner(A)
    >>> M_inv(torch.tensor([100.0, 0.1]))
    tensor([1., 1.])
    >>> x = linear_cg(A.matmul, torch.tensor([1.0, 2.0]), preconditioner=M_inv)
    """

    is_symmetric = True

    def __init__(self, A: torch.Tensor, *, absolute: bool = False):
        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be a square 2D matrix, got shape {tuple(A.shape)}")
        diag = sparse_diagonal(A.detach())
        if absolute:
            diag = diag.abs()
        diag = torch.where(diag == 0, torch.ones_like(diag), diag)
        self.shape = (A.shape[0], A.shape[1])
        self.inv_diag = diag.reciprocal()
        self.is_positive_definite = bool((diag > 0).all())

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        inv_diag = self.inv_diag.to(dtype=X.dtype, device=X.device)
        return inv_diag * X if X.dim() == 1 else inv_diag.unsqueeze(-1) * X


class MatrixPreconditioner(Preconditioner):
    r"""Preconditioner given by an explicit matrix :math:`\mathbf{M}^{-1}`.

    Wraps a sparse (COO/CSR) or dense matrix so that its transpose is available for the backward pass of
    :func:`~torchsparsegradutils.sparse_generic_solve`. This is how a tensor passed as ``preconditioner`` to it or
    to the solvers is interpreted. Sparse matrices are applied with :func:`~torchsparsegradutils.sparse_mm`, which
    broadcasts over vectors and batched inputs and keeps gradients sparse when a solver is differentiated directly.

    Parameters
    ----------
    M_inv : torch.Tensor, sparse COO or CSR of shape ``(n, n)``, or dense (strided) of shape ``(*batch, n, n)``
        Matrix approximating :math:`\mathbf{A}^{-1}`. A batched dense matrix is broadcast against the input as
        :func:`torch.matmul` does.
    symmetric : bool, default=False
        Declare :math:`\mathbf{M}^{-1}` symmetric, so :meth:`transpose` returns ``self`` instead of a transposed
        copy. Not checked.
    positive_definite : bool, default=False
        Declare :math:`\mathbf{M}^{-1}` symmetric positive definite. Not checked; implies ``symmetric``.

    Examples
    --------
    >>> import torch
    >>> from torchsparsegradutils.utils import MatrixPreconditioner
    >>> M_inv = MatrixPreconditioner(torch.tensor([[1.0, 2.0], [0.0, 1.0]]))
    >>> M_inv.T(torch.tensor([1.0, 1.0]))
    tensor([1., 3.])
    """

    def __init__(self, M_inv: torch.Tensor, *, symmetric: bool = False, positive_definite: bool = False):
        if M_inv.layout not in (torch.sparse_coo, torch.sparse_csr, torch.strided):
            raise TypeError(f"Unsupported layout: {M_inv.layout}. Only COO, CSR and dense (strided) are supported.")
        max_dim = None if M_inv.layout == torch.strided else 2
        if M_inv.dim() < 2 or (max_dim is not None and M_inv.dim() > max_dim) or M_inv.shape[-1] != M_inv.shape[-2]:
            expected = "(*batch, n, n)" if max_dim is None else "(n, n)"
            raise ValueError(f"M_inv must have shape {expected}, got {tuple(M_inv.shape)}")
        self.M_inv = M_inv
        self.shape = (M_inv.shape[-2], M_inv.shape[-1])
        self.is_positive_definite = positive_definite
        self.is_symmetric = symmetric or positive_definite

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        if self.M_inv.layout == torch.strided:
            return self.M_inv.matmul(X)
        # Deferred import: torchsparsegradutils.sparse_matmul imports the utils package, which imports this module
        from torchsparsegradutils.sparse_matmul import sparse_mm

        return sparse_mm(self.M_inv, X)

    def transpose(self) -> Preconditioner:
        if self.is_symmetric:
            return self
        if self.M_inv.layout == torch.sparse_csr:
            # Transposing CSR gives CSC; convert back so that matmul stays on the CSR kernels
            M_inv_t = self.M_inv.transpose(0, 1).to_sparse_csr()
        else:
            M_inv_t = self.M_inv.transpose(-2, -1)
        return MatrixPreconditioner(M_inv_t)
